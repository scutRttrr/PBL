# import sys
# import os
# from pathlib import Path
# from typing import List, Optional, Sequence
#
# # 将项目根目录添加到 Python 路径
# root_dir = Path(__file__).parent.parent.parent
# sys.path.append(str(root_dir))
#
# from states import InputState
# from utils import new_uuid
# from build import graph
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# from langgraph.graph.message import RemoveMessage
# from langgraph.types import Command
# import asyncio
# import time
# import builtins
#
# thread = {"configurable": {"thread_id": new_uuid()}}
#
#
# def _resolve_memory_turn_limit() -> Optional[int]:
#     """Resolve configured memory turns, treating non-positive values as unlimited."""
#     raw_value = os.getenv("GUSTOBOT_MEMORY_TURNS", os.getenv("GUSTOBOT_MAX_MEMORY_TURNS", "2"))
#     try:
#         value = int(raw_value)
#     except (TypeError, ValueError):
#         value = 5
#     return value if value > 0 else None
#
#
# MEMORY_TURN_LIMIT: Optional[int] = _resolve_memory_turn_limit()
#
#
# def _select_messages_to_remove(existing: Sequence[BaseMessage]) -> List[BaseMessage]:
#     """Determine which historical messages should be dropped to respect the turn limit."""
#     if not existing or MEMORY_TURN_LIMIT is None:
#         return []
#
#     humans_to_keep = max(MEMORY_TURN_LIMIT - 1, 0)
#     if humans_to_keep == 0:
#         return [msg for msg in existing if getattr(msg, "id", None)]
#
#     humans_seen = 0
#     keep_from = 0
#     for index in range(len(existing) - 1, -1, -1):
#         message = existing[index]
#         if getattr(message, "type", None) == "human":
#             humans_seen += 1
#             if humans_seen == humans_to_keep:
#                 keep_from = index
#                 break
#     else:
#         keep_from = 0
#
#     return [msg for msg in existing[:keep_from] if getattr(msg, "id", None)]
#
#
# def _stringify_content(content: object) -> str:
#     """Convert streamed message content into printable text."""
#     if content is None:
#         return ""
#     if isinstance(content, str):
#         return content
#     if isinstance(content, list):
#         parts: List[str] = []
#         for item in content:
#             if isinstance(item, str):
#                 parts.append(item)
#             elif isinstance(item, dict):
#                 parts.append(str(item.get("text", "")))
#             else:
#                 parts.append(str(item))
#         return "".join(parts)
#     return str(content)
#
#
# async def process_query(query: str) -> None:
#     state_snapshot = graph.get_state(thread)
#     existing_messages = list(state_snapshot.values.get("messages", []))#获取该线程的消息
#     messages_to_remove = _select_messages_to_remove(existing_messages)#删除该线程的消息
#
#
#     removals = [
#         RemoveMessage(id=msg.id)
#         for msg in messages_to_remove
#         if getattr(msg, "id", None)
#     ]
#     human_message = HumanMessage(content=query)
#     input_messages: List[BaseMessage] = [*removals, human_message]
#     input_state = InputState(messages=input_messages)
#
#     async for event in graph.astream(input=input_state, config=thread, stream_mode="updates"):
#         for node_name, node_update in event.items():
#             # 核心：只抓取 messages 键下的新内容
#             if "messages" in node_update:
#                 new_msg = node_update["messages"][-1]
#                 if isinstance(new_msg, AIMessage):
#                     # 这里只打印 AI 的回复文本，不会打出 JSON
#                     print(new_msg.content)
#     # async for chunk, metadata in graph.astream(
#     #     input=input_state,
#     #     stream_mode="messages",
#     #     config=thread,
#     # ):
#     #     text = _stringify_content(chunk.content)
#     #     # 💡 改进点：如果是追问引导（即文本较长且不是真正的流式块）
#     #     # 我们手动给它加一个打字机效果
#     #     if text:
#     #         # 如果是来自分析节点的完整回复（一次性给出的）
#     #         if len(text) > 10 and not hasattr(chunk, 'tool_call_chunks'):
#     #             for char in text:
#     #                 print(char, end="", flush=True)
#     #                 time.sleep(0.01)  # 模拟打字机速度
#     #         else:
#     #             # 如果是真正的模型流式块（比如 generate 节点抛出的）
#     #             print(text, end="", flush=True)
#
#     latest_snapshot = graph.get_state(thread)
#     pending_tasks = latest_snapshot.tasks
#     if pending_tasks and len(pending_tasks[0].interrupts) > 0:
#         response = input('\n响应可能包含不确定信息。重试生成？如果是，按"y"：')
#         if response.lower() == 'y':
#             async for chunk, metadata in graph.astream(
#                 Command(resume=response),
#                 stream_mode="messages",
#                 config=thread,
#             ):
#                 if chunk.additional_kwargs.get("tool_calls"):
#                     print(chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
#                 if chunk.content:
#                     time.sleep(0.05)
#                     print(chunk.content, end="", flush=True)
#
#
# async def main() -> None:
#     input_func = builtins.input
#     while True:
#         query = input_func("> ")
#         if query.strip().lower() == "q":
#             print("Exiting...")
#             break
#         await process_query(query)
#
#
# if __name__ == "__main__":
#     asyncio.run(main())

import sys
import os
from pathlib import Path
from typing import List, Optional, Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from API.tcgenerate import tc_router
from agent.build import graph
from agent.states import InputState
from agent.utils import new_uuid

# 将项目根目录添加到 Python 路径
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import RemoveMessage
from langgraph.types import Command
import asyncio
import time
import builtins
import uvicorn

thread = {"configurable": {"thread_id": new_uuid()}}


def _resolve_memory_turn_limit() -> Optional[int]:
    """Resolve configured memory turns, treating non-positive values as unlimited."""
    raw_value = os.getenv("GUSTOBOT_MEMORY_TURNS", os.getenv("GUSTOBOT_MAX_MEMORY_TURNS", "2"))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 5
    return value if value > 0 else None


MEMORY_TURN_LIMIT: Optional[int] = _resolve_memory_turn_limit()


def _select_messages_to_remove(existing: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Determine which historical messages should be dropped to respect the turn limit."""
    if not existing or MEMORY_TURN_LIMIT is None:
        return []

    humans_to_keep = max(MEMORY_TURN_LIMIT - 1, 0)
    if humans_to_keep == 0:
        return [msg for msg in existing if getattr(msg, "id", None)]

    humans_seen = 0
    keep_from = 0
    for index in range(len(existing) - 1, -1, -1):
        message = existing[index]
        if getattr(message, "type", None) == "human":
            humans_seen += 1
            if humans_seen == humans_to_keep:
                keep_from = index
                break
    else:
        keep_from = 0

    return [msg for msg in existing[:keep_from] if getattr(msg, "id", None)]


def _stringify_content(content: object) -> str:
    """Convert streamed message content into printable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


#async def process_query(query: str) -> None:
    # state_snapshot = graph.get_state(thread)
    # existing_messages = list(state_snapshot.values.get("messages", []))
    # messages_to_remove = _select_messages_to_remove(existing_messages)
    #
    # removals = [
    #     RemoveMessage(id=msg.id)
    #     for msg in messages_to_remove
    #     if getattr(msg, "id", None)
    # ]
    # human_message = HumanMessage(content=query)
    # input_messages: List[BaseMessage] = [*removals, human_message]
    # input_state = InputState(messages=input_messages)
    #
    # async for chunk, metadata in graph.astream(
    #     input=input_state,
    #     stream_mode="messages",
    #     config=thread,
    # ):
    #     if metadata.get("langgraph_node") == "query_rewrite" or metadata.get("langgraph_node") == "router":
    #         continue
    #
    #     text = _stringify_content(chunk.content)
    #     if text and "research_plan" not in metadata.get("tags", []):
    #         print(text, end="", flush=True)
    #
    # latest_snapshot = graph.get_state(thread)
    # pending_tasks = latest_snapshot.tasks
    # if pending_tasks and len(pending_tasks[0].interrupts) > 0:
    #     response = input('\n响应可能包含不确定信息。重试生成？如果是，按"y"：')
    #     if response.lower() == 'y':
    #         async for chunk, metadata in graph.astream(
    #             Command(resume=response),
    #             stream_mode="messages",
    #             config=thread,
    #         ):
    #             if chunk.additional_kwargs.get("tool_calls"):
    #                 print(chunk.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
    #             if chunk.content:
    #                 time.sleep(0.05)
    #                 print(chunk.content, end="", flush=True)


# async def main() -> None:
#     input_func = builtins.input
#     query = input_func("> ")
#     # if query.strip().lower() == "q":
#     #     print("Exiting...")
#     #     break
#     await process_query(query)

app = FastAPI(title="PBL 教案系统")
# ✅ 将接口“注册”到 app 下
app.include_router(tc_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 开发环境可以用 "*"，生产环境建议写具体地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    #asyncio.run(main())