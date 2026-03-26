import json

from langchain_core.prompts import ChatPromptTemplate

from agent.subgraph.pair.TL import tl_workflow, tl_graph
from agent.utils import validate_generation_tasks
from infrastructure.logger import get_logger
from prompt_template import (
    ROUTER_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GENERAL_QUERY_SYSTEM_PROMPT,
    QUERY_REWRITE_PROMPT,
    GET_IMAGE_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    RAGSEARCH_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    IMAGE_GENERATION_ENHANCE_PROMPT,
    IMAGE_GENERATION_SUCCESS_PROMPT
)
from config.settings import settings
from langchain_core.runnables import RunnableConfig
from typing import cast, Literal, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from states import AgentState, InputState, Router, GradeHallucinations

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage


from langchain_openai import ChatOpenAI
class AdditionalGuardrailsOutput(BaseModel):
    """
    格式化输出，用于判断用户的问题是否与图谱内容相关
    """
    decision: Literal["end", "proceed"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )


# 构建日志记录器
logger = get_logger(service="lg_builder")


def _ensure_router(router_obj: Any, *, fallback_question: str = "") -> Router:
    """将任意 router 结构转换为 Router 模型，保持字段访问兼容。"""
    if isinstance(router_obj, Router):
        return router_obj
    if isinstance(router_obj, dict):
        try:
            return Router.model_validate(router_obj)#对router_obj进行校验，如果符合校验，则返回Router类型
        except Exception:
            pass
    return Router(type="concept-query", logic="missing router", question=fallback_question)


def _extract_configurable(config: Any) -> Dict[str, Any]:
    """提取 LangGraph RunnableConfig 中的 configurable 字段，确保返回字典。"""
    if not config:
        return {}
    if isinstance(config, dict):
        value = config.get("configurable", {})
        return value if isinstance(value, dict) else {}
    # LangGraph RunnableConfig 支持属性访问或字典接口
    configurable = getattr(config, "configurable", None)
    if isinstance(configurable, dict):
        return configurable
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            value = getter("configurable", {})
            if isinstance(value, dict):
                return value
        except Exception:  # pragma: no cover - 容错
            pass
    return {}

def _coerce_to_bool(value: Any, *, default: bool = False) -> bool:
    """Best-effort conversion of dynamic configuration values to boolean."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


# 假设你已经定义好了前面的 QUERY_REWRITE_PROMPT
async def query_rewrite(
        state: AgentState
) -> dict:
    """
    QueryRewrite 节点：负责指代消解、去噪、以及原子化拆分。
    """

    # 1. 准备上下文：取出最近的消息和历史 slots
    query = state.messages[-1].content

    # 2. 调用 LLM (推荐使用 bind_tools 或 format_instructions 确保 JSON 输出)
    # 这里直接使用之前写的适配 PBL 的 Prompt


    model = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
        openai_api_base=settings.OPENAI_API_BASE,
        temperature=0.7,
        tags=["router"],
    )

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured for router analysis.")
    # 假设你使用的是带有 JSON Mode 的 LLM
    response = await model.ainvoke( [
                   {"role": "system", "content": QUERY_REWRITE_PROMPT},
                    {"role": "human", "content": "以下是任务：\n"+query},
               ])

    try:
        # 解析返回的 JSON
        result = json.loads(response.content)
        sub_questions = result.get("sub_questions", [])

        logger.info("成功拆分任务----"+str(sub_questions))

        # 4. 更新 State
        # 将结果存入 state，后续 Router 节点会根据 sub_questions 的动词进行判断
        return {
            "questions": sub_questions,
        }

    except Exception as e:
        # 解析失败的容错处理
        logger.error("拆分任务失败，出现错误：----"+str(e))
        return {"questions": query}

async def analyze_and_route_query(
        state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    current_task = state.current_task
    #question_text = state.messages[-1].content if state.messages else ""
    classifiers = []
    sanitized_router =None
    for  question_text in state.questions:
        # if current_task and isinstance(current_task, str):
        #     logger.info(f"------继续进行当前任务------- {current_task}")
        #     return {"router": Router.model_validate({"type": current_task}),
        #             "question": question_text}

        classier = {"question": question_text}

        model = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.7,
            tags=["router"],
        )

        # 拼接提示模版 + 用户的实时问题（包含历史上下文对话）
        messages = [
                       {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                       {"role": "human", "content": "要分类的任务："+question_text}
                   ]
        logger.info("-----Analyze user query type-----")
        logger.info(f"History messages: {question_text}")


        heuristic_router = _heuristic_router(question_text)
        fallback_router: Router = heuristic_router or Router(
            type="concept-query",
            logic="fallback: default to knowledge base routing",
            question=question_text,
        )

        allowed_types: set[str] = {
            "general-query",
            "additional-query",
            "concept-query",
            "T&L-generation",
            "project-generation",
            "rubric-generation",
            "image-query",
            "file-query",
        }
        raw_response = ""
        try:
            raw_response = await model.ainvoke(messages)
        except Exception as exc:
            logger.warning("Router LLM failed: %s. Falling back to concept-query.", exc)
            logger.error(f"！！！路由解析核心报错: {type(exc).__name__}: {exc}")

            try:
                debug_resp = await model.ainvoke(messages)
                logger.info(f"--- 模型原始回帖 (Debug) ---: {debug_resp.content}")
            except:
                logger.warning("连原始回帖都拿不到，请检查网络或 API Key")

        raw_response =  {"type":raw_response.content if raw_response.content in allowed_types else "general-query"}
        response = raw_response if isinstance(raw_response, Router) else Router.model_validate(raw_response)
        router_type = response.type
        logic = response.logic or ""

        if not router_type or router_type not in allowed_types:
            logger.warning(
                "Router returned invalid type `%s`; applying heuristic fallback.", router_type
            )
            heuristic_router = _heuristic_router(question_text)
            if heuristic_router:
                sanitized = heuristic_router
                if not sanitized.logic:
                    sanitized.logic = logic or ""
                classier["type"] = sanitized.type
            else:
                classier["type"] = "concept-query"
            # return {
            #     "router": Router(
            #         type="concept-query",
            #         logic=logic or "fallback: invalid router output",
            #         question=question_text,
            #     )
            # }

        sanitized_router = Router(
            type=router_type,
            logic=logic,
            question=response.question or question_text,
            decision=response.decision,
            confidence=response.confidence,
            reasoning=response.reasoning,
        )
        classier["type"] = sanitized_router.type
        classifiers.append(classier)

    # Heuristic router is only used when the LLM output is invalid (handled above).
    logger.info(f"Analyze user query type completed, result: {classifiers}")

    if validate_generation_tasks(classifiers)== "fail":    #TODO:此处逻辑没有写清楚
        sanitized_router.type = "general-query"
    if validate_generation_tasks(classifiers)== "T&L-generation":
        sanitized_router.type = "T&L-generation"
    if validate_generation_tasks(classifiers)== "rubric-generation":
        sanitized_router.type = "rubric-generation"

    return {
            "messages": AIMessage(content=str(classifiers)),
            "router": sanitized_router,
            "question": sanitized_router.question,}


def route_query(
        state: AgentState,
) -> Literal[
    "respond_to_general_query", "get_additional_info", "create_concept_query", "create_tl_generation","create_image_query", "create_file_query", "create_project_generation","create_rubric_generation"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"，"create_kb_query"]: 下一步操作。
    """
    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    _type = router.type or "concept-query"

    # 检查配置中是否有图片或文件路径，如果有，优先对应处理
    if hasattr(state, "config") and state.config:
        cfg = state.config.get("configurable", {})
        if cfg.get("image_path"):
            logger.info("检测到图片路径，转为图片查询处理")
            return "create_image_query"
        if cfg.get("file_path"):
            logger.info("检测到文件路径，转为文件上传处理")
            return "create_file_query"

        # 3. 根据最新的 8 种分类执行跳转映射
    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type == "concept-query":
        return "create_concept_query"
    elif _type == "T&L-generation":
        return "create_tl_generation"
    elif _type == "project-generation":
        return "create_project_generation"
    elif _type == "rubric-generation":
        return "create_rubric_generation"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"

    else:
        # 安全垫：如果出现了未定义的类型，统一降级到知识库查询，防止 Graph 崩溃
        logger.warning(f"Unexpected router type: {_type}, falling back to concept_query")
        return "create_concept_query"


async def respond_to_general_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。
    当路由器将查询分类为一般问题时，将调用此节点。
    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。
    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")

    # 使用大模型生成回复
    model = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL,
                       openai_api_base=settings.OPENAI_API_BASE, temperature=0.7,
                       tags=["general_query"])

    router = _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=router.logic
    )

    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def get_additional_info(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """
    当信息不足或需要确认业务边界时，引导老师补充 PBL 设计所需的关键维度。
    """
    logger.info("------ 开始引导老师补充教学设计信息 ------")

    # 1. 初始化模型
    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,  # 降低随机性，保证引导的专业性
        tags=["additional_info"]
    )

    # 2. 定义 PBL 专家服务范围（Guardrails 依据）
    scope_description = """
    PBL 教学辅助助手服务范围：专注于项目式学习的设计、评估与优化，包括：

    🎓 课程设计与教案生成 (T&L / Project Design)
    - 设计驱动性问题 (Driving Questions)
    - 规划预期学习成果 (ILO)
    - 安排项目实施阶段与支架教学活动

    📊 评价量表与反馈 (Rubric & Assessment)
    - 生成多维度的表现性评价量表 (Rubrics)
    - 设计符合 AAC&U VALUE 等标准的评估准则
    - 提供项目过程性评价的建议

    📖 教学理论支持
    - 提供项目制学习的核心理论支持（如金标准 PBL）
    - 协助优化教学目标与活动的一致性

    暂不支持：政治时事、娱乐八卦、天气预报等非教育教学相关内容。
    """

    # 3. 准备安全检查上下文
    # 如果你没有用到 Neo4j，这里可以直接省略 graph_context 以提升速度
    scope_context = f"参考此范围描述来决策:\n{scope_description}"
    message = scope_context + "\nQuestion: {question}"

    # 4. 运行安全拦截 (Guardrails)
    full_system_prompt = ChatPromptTemplate.from_messages([
        ("system", GUARDRAILS_SYSTEM_PROMPT),
        ("human", message),
    ])

    # 注意：这里继续使用 structured_output 是因为这是逻辑判断，通常 OpenAI/DeepSeek 兼容性较好
    # 如果 DeepSeek 报错 400，可以换成我之前教你的手动 ainvoke 解析
    try:
        guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)
        guardrails_output = await guardrails_chain.ainvoke(
            {"question": state.messages[-1].content if state.messages else ""}
        )
    except Exception:
        logger.warning("Guardrails structured output failed, defaulting to proceed")
        guardrails_output = AdditionalGuardrailsOutput(decision="proceed")

    # 5. 根据拦截结果返回
    if guardrails_output.decision == "end":
        logger.info("----- 未通过 PBL 业务边界检查 -----")
        return {"messages": [AIMessage(
            content="老师您好～抱歉哦，这个问题超出了我的 PBL 教学设计范围。我主要协助您处理教案设计、评价量表开发等教学事务，我们可以聊聊您的课程构思吗？😊")]}

    else:
        logger.info("----- 通过业务检查，开始生成追问引导 -----")

        # 确保 router 存在，提取其逻辑（为什么需要更多信息）
        # router.logic 可能包含：用户没提供学段，或没提供学科主题
        current_router = state.router
        logic_str = current_router.logic if current_router else "需要更多关于教学设计的主题或目标信息"

        # 使用我们刚才改好的专业提示词
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(logic=logic_str)

        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)

        return {"messages": [response]}

async def create_concept_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    logger.info("------ 询问概念相关问题 ------")
    return {"messages": []}


async def create_tl_generation(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    logger.info("开始生成教案")
    return {"messages": []}

async def create_image_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    return {"messages": []}
async def create_file_query(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    return {"messages": []}
async def create_project_generation(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    return {"messages": []}
async def create_rubric_generation(
        state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    return {"messages": []}

checkpointer = MemorySaver()

# 定义状态图
builder = StateGraph(AgentState)
# 添加节点
async def dummy_node(state: AgentState) -> Dict:
    """占位节点：在正式逻辑写完前，先确保图能连通"""
    last_message = state.messages[-1].content
    # 这里可以简单回一个“正在开发中”或者直接透传
    return {"messages": [AIMessage(content=f"已进入节点，正在处理：{last_message}")]}

# 在 builder 中这样注册
# Literal[
#     "respond_to_general_query", "get_additional_info", "create_concept_query", "create_tl_generation", "create_image_query", "create_file_query", "create_project_generation", "create_rubric_generation"]:
# builder.add_node("query_rewrite", query_rewrite)
# builder.add_node("router", analyze_and_route_query)
# builder.add_node("respond_to_general_query", respond_to_general_query)
# builder.add_node("create_concept_query", create_concept_query)
builder.add_node("create_tl_generation", tl_graph)
# builder.add_node("create_project_generation", dummy_node)
# builder.add_node("create_rubric_generation", dummy_node)
# builder.add_node("create_image_query", dummy_node)
# builder.add_node("create_file_query", dummy_node)
# builder.add_node("get_additional_info", get_additional_info)


# 添加边
builder.add_edge(START, "create_tl_generation")
# builder.add_edge("query_rewrite", "router")
builder.add_edge("create_tl_generation", END)

graph = builder.compile(checkpointer=checkpointer)

# png_bytes = graph.get_graph().draw_mermaid_png()
# output_path = Path(__file__).resolve().parent / "lg_builder_workflow.png"
# output_path.write_bytes(png_bytes)
# logger.info("工作流图已保存到 %s", output_path)
#
# try:
#     from IPython.display import Image as IPythonImage, display as ipython_display
# except ImportError:  # pragma: no cover - optional dependency
#     logger.info("IPython 未安装，跳过图像内联展示。")
# else:
#     ipython_display(IPythonImage(png_bytes))
from typing import Optional

def _heuristic_router(question: str) -> Optional[Router]:
    """基于 PBL 教育场景关键词的启发式路由（用于 LLM 故障时的快速响应/兜底）。"""
    if not question:
        return None

    lowered = question.lower()

    # 1. 评分量表生成 (rubric-generation) - 优先级最高，因为词簇最固定
    rubric_keywords = ["rubric", "评分标准", "评分量表", "评价量表", "量规", "评分细则"]
    if any(k in lowered for k in rubric_keywords):
        # 区分：是问“怎么写”还是“帮我写”
        if any(act in lowered for act in ["帮我", "生成", "设计", "写一个", "出个"]):
            return Router(type="rubric-generation", logic="heuristic: action-rubric", question=question)
        return Router(type="concept-query", logic="heuristic: concept-rubric", question=question)

    # 2. 教学设计生成 (T&L-generation)
    tl_keywords = ["教案", "教学设计", "教学内容", "teaching content", "ilo", "tc", "report", "课堂活动"]
    if any(k in lowered for k in tl_keywords):
        if any(act in lowered for act in ["帮我", "生成", "设计", "写一个", "做一份"]):
            return Router(type="T&L-generation", logic="heuristic: action-tl", question=question)
        return Router(type="concept-query", logic="heuristic: concept-tl", question=question)

    # 3. 项目方案生成 (project-generation)
    project_keywords = ["项目方案", "项目计划", "project plan", "大项目", "项目大纲"]
    if any(k in lowered for k in project_keywords):
        if any(act in lowered for act in ["帮我", "生成", "设计", "规划", "写一个"]):
            return Router(type="project-generation", logic="heuristic: action-project", question=question)
        return Router(type="concept-query", logic="heuristic: concept-project", question=question)

    # 4. 概念/知识咨询 (concept-query / graphrag)
    # 注意：你的 Prompt 提示词写的是 concept-query，但提示 2 却说选 graphrag-query。
    # 建议统一使用 concept-query 以对应你的 KB 检索节点。
    concept_keywords = ["什么是", "为什么", "如何", "怎么", "定义", "准则", "4c", "黄金准则", "理论"]
    if any(k in lowered for k in concept_keywords):
        return Router(type="concept-query", logic="heuristic: concept-general", question=question)

    # 5. 轻量对话 (general-query)
    greeting_keywords = ["你好", "早上好", "下午好", "嗨", "hi", "hello", "谢谢", "再见"]
    if lowered.strip() in greeting_keywords:
        return Router(type="general-query", logic="heuristic: greeting", question=question)

    return None

def build_supervisor_graph():
    """向后兼容的 Supervisor Graph 构建接口。"""
    return graph


async def safety_guardrails(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """兼容旧接口，复用 get_additional_info 的护栏逻辑。"""
    return await get_additional_info(state, config=config)