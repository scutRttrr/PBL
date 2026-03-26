import os
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.tools import tool

from config.settings import settings



# ==========================================
# 1. 定义 PBL 专业工具
# ==========================================
@tool
def search_pbl_framework(query: str):
    """当老师询问 PBL 设计框架、驱动性问题设计准则、评价量规(Rubric)编写规范时调用。"""
    # 模拟 PBL 教学法知识库（实际可接入你的工程项目中的向量库）
    pbl_kb = {
        "驱动性问题": "驱动性问题应具备：1.开放性；2.核心概念导向；3.对学生有吸引力；4.可行性。",
        "评价量规": "建议采用全量程量规（Holistic Rubrics），包含：协作能力、批判性思维、项目成果展示、自我反思四个维度。",
        "项目周期": "根据港科广 Red Bird Project 经验，一个完整的 PBL 周期通常建议设置为 4-6 周。"
    }
    for key, value in pbl_kb.items():
        if key in query: return value
    return "已记录您的教学设计需求，建议参考《黄金标准 PBL》设计指南进行细化。"


tools = [search_pbl_framework]
tool_node = ToolNode(tools)


# ==========================================
# 2. 定义状态与模型（注入教学专家 Persona）
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str


# 定义系统预设词：确保 Agent 以教育专家的身份交流
SYSTEM_PROMPT = SystemMessage(content=(
    "你是一位资深的 PBL (项目式学习) 教学顾问，专门协助老师设计高质量的教学项目。"
    "你的语气应该是：专业、鼓励探索、富有逻辑。"
    "当老师提出的设计不完整时，请引导他们思考‘驱动性问题’或‘评估标准’。"
    "在引用知识库内容时，请结合老师当前的教学情境给出建议。"
))

llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.OPENAI_MODEL,
    openai_api_base=settings.OPENAI_API_BASE,
    streaming=True
).bind_tools(tools)


# ==========================================
# 3. 节点逻辑
# ==========================================

from langchain_core.messages import SystemMessage, RemoveMessage


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    messages = state["messages"]

    # 如果对话超过 6 条（3轮），就开始压缩旧的
    if len(messages) > 6:
        # 获取需要总结的消息（除了最后 6 条以外的所有消息）
        to_summarize = messages[:-6]

        # 让 LLM 基于旧摘要和新消息生成新摘要
        summary_prompt = (
            f"当前的对话摘要为: {summary}\n\n"
            "请根据以上摘要和下面的新对话内容，更新并生成一个简炼的长期记忆摘要，"
            "重点保留老师的背景、教学目标和已确定的 PBL 方案细节："
        )
        response = llm.invoke([SystemMessage(content=summary_prompt)] + to_summarize)

        # 返回更新后的摘要，并删除已经被总结的消息
        delete_messages = [RemoveMessage(id=m.id) for m in to_summarize]
        return {"summary": response.content, "messages": delete_messages}

    return {"summary": summary}

def pbl_advisor(state: State):
    summary = state.get("summary", "")

    if summary:
        system_message = f"{SYSTEM_PROMPT}\n\n这是之前的对话要点总结：{summary}"
    else:
        system_message = SYSTEM_PROMPT

    # 将系统提示词与历史对话合并
    messages = [system_message] + state["messages"][-6:]
    return {"messages": [llm.invoke(messages)]}


def should_continue(state: State) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


# ==========================================
# 4. 构建工作流
# ==========================================
workflow = StateGraph(State)
workflow.add_node("advisor", pbl_advisor)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "advisor")
workflow.add_conditional_edges("advisor", should_continue)
workflow.add_edge("tools", "advisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ==========================================
# 5. 交互界面
# ==========================================
def run_pbl_assistant():
    config = {"configurable": {"thread_id": "teacher_consult_001"}}
    print("🎓 您好，我是您的 PBL 教学设计助手。")
    print("🍎 我们可以讨论驱动性问题设计、评价量规制定或项目进度管理。\n")

    while True:
        user_input = input("👩‍🏫 老师: ")
        if user_input.lower() in ['q', 'quit', 'exit']: break

        print("🤖 顾问: ", end="", flush=True)
        # 流式输出
        for msg, metadata in app.stream({"messages": [("user", user_input)]}, config, stream_mode="messages"):
            if isinstance(msg, AIMessageChunk) and msg.content:
                print(msg.content, end="", flush=True)
        print("\n" + "-" * 40)


if __name__ == "__main__":
    run_pbl_assistant()