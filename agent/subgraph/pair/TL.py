import asyncio
import json
from pyexpat.errors import messages

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

from agent.subgraph.pair.state import TLCheckOutput, TLState
from agent.tool.TLtools import  tltools
from infrastructure.logger import get_logger
from agent.prompt_template import (
    GET_ADDITIONAL_SYSTEM_PROMPT,
)

from config.settings import settings
from langchain_core.runnables import RunnableConfig
from typing import cast, Literal, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from agent.states import AgentState, InputState, Router, GradeHallucinations

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

# 构建日志记录器
logger = get_logger(service="TL_builder")

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# prompt_template.py 中新增
TL_CHECKER_SYSTEM_PROMPT = """你是一名资深的 PBL 教学专家。你的任务是分析用户的对话，判断其中是否包含了清晰的 ILO（Intended Learning Outcome）。

### ILO 充足的标准：
1. **学习目标**：明确了希望学生能够学会什么样的能力/技能
2. **学习效果**：明确了希望学生掌握该能力到何种程度

### ILO 的例子
1.Deliver a concise, structured progress update for assigned tasks, clearly articulating completion status, obstacles and potential solutions in line with the group’s project timeline.
2.Identify specific interdisciplinary knowledge/resource gaps in their project team through guided discussion, and link gaps to current project progress bottlenecks.
3.Apply practical strategies for knowledge complementation and resource sharing in interdisciplinary teams, and co-create a group actionable plan for immediate implementation.

### 输出格式：
必须返回 JSON 格式，包含以下字段：
- is_info_sufficient: bool (信息是否足以直接生成详细教案)
- missing_dimensions: list (缺失的维度名称，如 ["学习目标", "学习效果"])
- logic: str (简短的理由，说明为什么不足，称呼用户为老师)
"""

TL_TC_GNERATER_SYSTEM_PROMPT = """
## Role
你是一位资深的 PBL（项目式学习）课程设计专家，擅长根据“预期学习成果 (ILO)”开发高质量的“教学内容 (Teaching Content, TC)”。

## Task
你的任务是为用户提供具体的教学内容方案。为了保证方案的专业性与合规性，你必须遵循以下流程：

### 核心工作流
1. **分析输入**：识别用户提供的 ILO 需要的学会的技能和达到何种目的

3. **生成格式**：你的输出必须是Json列表[{"ILO":"","TC":""},{"ILO":"","TC":""}]
4. **内容生成**：如果有多个ILO，请为每个ILO都写一个TC，每个ILO和TC为一个Json

### 教学内容 (TC) 编写模板如下，请严格模仿以下三个示例的格式、语气和逻辑：
### Example 1
{"ILO": "Demonstrate effective use of Feishu for tracking milestones, tasks, and progress","TC": " Progress & PM Practice Assessment- Step 1: Feishu Workspace Walkthrough. Each group shares their Feishu project workspace (screen share) to demonstrate: 1) Milestone tracking, 2) Task assignment, 3) Progress updates (e.g., completed vs. in-progress tasks from the 2-week plan). Instructor feedback focuses on Feishu optimization.- Step 2: Progress Check. Groups report on 2-week plan achievements: 1) Completed tasks, 2) Delays (if any) + reasons, 3) Alignment with revised proposal objectives. "}
### Example 2
{"ILO": "Identify key stakeholders, map them by influence/interest, and articulate their (and the group’s) potential roles, motivations, and contributions to the project.Transferable Skills: Stakeholder Analysis, Teamwork (Role Alignment)", "TC": "Map Stakeholders & Articulate Roles Identify key stakeholders (e.g., community residents, local businesses, NGOs) using Assignment 3’s guidance.Create a shared stakeholder map (categorize by influence/interest) and detail: their roles (e.g., “Residents provide usage feedback”), motivations (e.g., “Businesses want to cut waste costs”), and the group’s role in engaging them (e.g., “We’ll survey residents to validate solutions”)."}}
### Example 3
{"ILO": "Define SMART project objectives, coherent group tasks, and feasible milestones, ensuring objectives align with expected outcomes and the execution plan is resource- and time-realistic.", "TC": "SMART Objectives, Milestones & Feasibility- Workshop on Rubric 1.3 (Project Objectives & Outcomes):1. Teach SMART criteria with examples: Weak: \"Improve digitalization\" → Strong: \"Develop a free AI-driven inventory tool for 50+ small businesses in Guangzhou by May 2027 (measurable, time-bound)\".2. Link objectives to expected outcomes (e.g., \"Objective 1 → Outcome: Prototype tested with 10 entrepreneurs; 80% report time savings\").- Workshop on Rubric 1.4 (Feasibility & Milestones):1. Milestone best practices: (e.g., \"Jan 2026: Complete literature review on small business digitalization gaps\").2. Feasibility check: Analyze resources (expertise, budget) and risks (e.g., \"Lack of business user access → Mitigation: Partner with local chambers of commerce\")."}

"""
TL_TC_GENRATER_USER_PROMPT="""
### 当前任务输入
- 用户提供的 ILO: {user_ilo}

### 知识库参考案例 (RAG)
以下是从教育数据库中检索到的相似案例，仅供参考其专业术语，如果以下内容与上述高收益示例冲突，请优先遵循高收益示例的逻辑：
{retrieved_context}  

### 执行指令
请结合上述参考案例（如有）和你的专业背景，为当前任务输入生成对应的 TC。"""

# TL_REWRITE_SYSTEM_PROMPT = """
# 你是一位教育语义专家。请将用户输入的 ILO 拆解为两个互补的搜索短语，以便在向量数据库中获得最高精度的匹配：
#
# 1. **Knowledge & Action (Learn What)**: 提取具体的工具、技术或操作动作（如：Git flow, 风险矩阵, 飞书）。
# 2. **Intent & Outcome (Achieve What)**: 提取教育目标、最终达到的效果（如：团队协作, 提高代码质量, 风险缓解）。
#
# 请直接输出这两个短语，用 | 分隔。
# 示例输入：Execute a standard Git flow (branching, PRs) to manage collaborative development.
# 示例输出：Git flow branching pull requests | collaborative development quality consistency
#
# """
TL_REWRITE_SYSTEM_PROMPT = """
你是教育领域的语义拆解专家，任务是把 ILO（学习目标）拆成 **两段极短、检索友好、互不重叠** 的短语，用于向量库高精度匹配。

规则必须严格遵守：
1. 第一段：Knowledge & Action（学什么、做什么）
   - 只保留：**具体技术、工具、操作、动作、命令、概念**
   - 越短越好，用空格分隔关键词，不要长句
   - 不要目标、不要效果、不要抽象词

2. 第二段：Intent & Outcome（达到什么效果）
   - 只保留：**目标、目的、价值、质量、规范、一致性、能力**
   - 抽象、结果导向
   - 不要工具、不要操作步骤

输出格式：
【知识动作短语】|【意图效果短语】
不要解释、不要多余符号，只输出一行。

示例输入：
Execute a standard Git flow (branching, pull requests, and code reviews) to manage collaborative development, ensuring code quality.
示例输出：
Git flow branching pull requests code reviews | collaborative development code quality consistency
"""

# 2. **强制检索**：你【必须】首先调用 `search_milvus` 工具，以获取与该 ILO 相关的参考案例或知识库内容。
async def tl_analyze_node(state: AgentState):
    """
    分析节点：检测 ILO 信息，如果不足则直接生成文本回复
    """
    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,
        tags=["additional_info"]
    )
    current_count = state.current_count
    logger.info(f"Current count: {state.questions}")
    if current_count >= 2:
        return {"message":state.questions}

    # 1. 调用结构化 LLM 进行检测
    messages = [{"role": "system", "content": TL_CHECKER_SYSTEM_PROMPT},
                {"role":"human","content": "以下是用户对话："+ str(state.questions)}]
    structured_llm = model.with_structured_output(TLCheckOutput, method="json_mode")
    check_result = await structured_llm.ainvoke(messages)

    # 2. 如果信息充足，直接返回检测结果，让子图去跑生成 TC 的节点
    if check_result.is_info_sufficient:
        return {
            #"is_info_sufficient": True,
            "messages": state.messages,
            "questions": state.questions,
        }

    # 3. 如果信息不充足，根据 JSON 手动构建回复文本（不费 Token）
    tips_map = {
        "学习目标": "📍 **明确目标**：请具体描述您希望学生学会什么样的核心能力或技能（例如：阐述、设计或制作某物）。",
        "学习效果": "📍 **掌握程度**：请说明您希望学生掌握到什么程度（例如：能够独立应用，还是在引导下识别）。",
        "数量": "📍 **目标数量**：目前的学习成果条数较少，建议增加到 2-3 条，以便更全面地评价学生表现。"
    }

    missing_info = [tips_map.get(d, f"📍 补充{d}相关信息") for d in check_result.missing_dimensions]

    response_text = (
            "\n好的😊 为了能帮您生成更高质量的教学方案，我们需要再完善一下 ILO（预期学习成果）：\n\n"
            + "\n".join(missing_info)
            + f"\n\n **建议**：{check_result.logic}\n\n补充完这些，我就可以为您生成完整的方案啦！"
    )

    # 4. 更新状态：增加轮次，并把“自然语言回复”塞进 messages
    return {
        "is_info_sufficient": False,
        "missing_dimensions": check_result.missing_dimensions,
        "logic": check_result.logic,
        "current_task":"T&L-generation",
        "current_count":current_count+1,
        "messages": [AIMessage(content=response_text)],
    }

def decide_next_step(state: TLState):
    current_count = state.current_count
    # 如果 analyze 节点判定信息已经够了，或者已经强行聊了两轮，就去生成 TC
    if state.is_info_sufficient or current_count >= 2:
        return "generate"
    # 否则，因为消息已经生成在 messages 里的，直接结束本轮等用户回复
    return "wait_for_user"

async def tl_rewrite_node(state: AgentState):

    q=["Deliver a concise, structured progress update for assigned tasks, clearly articulating completion status, obstacles and potential solutions in line with the group’s project timeline.",
       "Identify specific interdisciplinary knowledge/resource gaps in their project team through guided discussion, and link gaps to current project progress bottlenecks.",
       "Apply practical strategies for knowledge complementation and resource sharing in interdisciplinary teams, and co-create a group actionable plan for immediate implementation."]
    state.questions=q
    if state.current_count >= 2:
        return {"messages": AIMessage(content="抱歉老师，录入的信息暂无法识别。已为您返回主菜单，请尝试重新描述您的的要求。"),
            "current_task": None,
            "current_count": 0,
            "is_info_sufficient":False
            }
        # 调用 LLM 进行重写

    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,  # 降低随机性，保证引导的专业性
        tags=["additional_info"])


    # 1. 定义一个内部协程函数，用于单个 question 的处理
    async def process_question(q):
        # 这里的 q 转换 str 是为了防止之前报错的 FieldInfo 问题
        q_text = str(q)
        response = await model.ainvoke([
            {"role": "system", "content": TL_REWRITE_SYSTEM_PROMPT},
            {"role": "human", "content": f"以下是用户的ILO：{q_text}"}
        ])
        logger.info(f"Rewrite response: {response.content}")
        # 根据你的原逻辑，这里返回的是 response.content 还是原 question？
        # 原逻辑 append 的是 question，这里保持一致
        return response.content

        # 2. 使用 asyncio.gather 并发执行所有请求

    # state.questions 必须是可迭代的
    tasks = [process_question(q) for q in state.questions]
    divided_questions = await asyncio.gather(*tasks)

    logger.info(f"Divided questions: {divided_questions}")

    return {
        #"is_info_sufficient": True,
        "questions": divided_questions,  # 确保是列表格式
    }


import asyncio
from concurrent.futures import ThreadPoolExecutor

# 创建一个全局线程池，用于处理 CPU 密集型的 Embedding 计算
executor = ThreadPoolExecutor(max_workers=10)


async def tl_retrieve_node(state: AgentState):
    logger.info(f"Divided questions: {state.questions}")

    # --- 1. 资源预加载（移出循环） ---
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("EducationModules")
    collection.load()

    # 模型加载在外面，避免并行时内存炸裂
    dense_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    async def single_search(query: str):
        """最底层的单个 query 查询逻辑"""
        # 使用线程池运行同步的 embed_query，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(executor, dense_embeddings.embed_query, query)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        # Milvus 的 search 方法在 python SDK 中主要是同步的，如果量大也可以放进 executor
        results = collection.search(
            data=[query_vector],
            anns_field="ilo_vector",
            param=search_params,
            limit=3,
            output_fields=["ilo", "tc"]
        )

        local_docs = []
        for hits in results:
            for hit in hits:
                if hit.distance >= 5:  # 注意：COSINE 距离 5 可能意味着不匹配，请确认你的阈值
                    continue
                local_docs.append({
                    "ILO": hit.entity.get('ilo'),
                    "TC": hit.entity.get('tc')
                })
        logger.info(f"Results: {query}的查询结果-------"+str(local_docs))
        return local_docs

    async def process_question_group(question_text: str):
        """处理单个 question 及其内部的 | 拆分并行"""
        queries = [q.strip() for q in question_text.split("|") if q.strip()]
        # --- 内层并行：针对拆分出的 queries ---
        tasks = [single_search(q) for q in queries]
        results = await asyncio.gather(*tasks)
        # 展平结果列表
        return [doc for sublist in results for doc in sublist]

    # --- 外层并行：针对 state.questions ---
    outer_tasks = [process_question_group(q_text) for q_text in state.questions]
    all_results = await asyncio.gather(*outer_tasks)

    # 汇总所有结果并去重（可选）
    documents = [doc for sublist in all_results for doc in sublist]

    logger.info(f"查到的文档总数：{len(documents)}")

    return {
        "messages": state.messages,
        "questions": state.questions,
        "documents": documents
    }



async def tl_generate_tc_node(state: AgentState):

    context_str = json.dumps(state.documents, ensure_ascii=False, indent=2)
    logger.info(f"查到的文档：Context: {context_str}")

    if not state.documents:
        context_str = "暂无相关的历史案例可供参考。"


    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,  # 降低随机性，保证引导的专业性
        tags=["additional_info"]
    )

    async def process_question(question):
        response = await model.ainvoke([{"role": "system", "content": TL_TC_GNERATER_SYSTEM_PROMPT},
                                    {"role":"human","content": "以下是用户的ILO："+ TL_TC_GENRATER_USER_PROMPT.format(user_ilo=question,
                                                                                                            retrieved_context=context_str)}] )
        return response.content

    tasks = [process_question(q) for q in state.questions]
    answer_list = await asyncio.gather(*tasks)

    answer = "\n\n".join(answer_list)

    return {"messages": answer,
            "current_task": None,
            "current_count": 0,
            "is_info_sufficient":False
            }



tl_workflow = StateGraph(AgentState)
#tl_workflow.add_node("analyze", tl_analyze_node)
tl_workflow.add_node("generate", tl_generate_tc_node)
tl_workflow.add_node("rewrite", tl_rewrite_node)
tool_node = ToolNode(tltools)
tl_workflow.add_node("retrieve", tl_retrieve_node)
tl_workflow.add_edge(START, "rewrite")

# tl_workflow.add_conditional_edges(
#     "analyze",
#     decide_next_step,
#     {
#         "generate": "rewrite",
#         "wait_for_user": END  # 这里的 END 会把 analyze 节点生成的追问发给用户
#     }
# )
tl_workflow.add_edge("rewrite", "retrieve")
tl_workflow.add_edge("retrieve", "generate")
tl_workflow.add_edge("generate", END)
tl_graph = tl_workflow.compile()