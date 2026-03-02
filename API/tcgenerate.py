from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, END
import json

from API.Request import TCRequest
from agent.build import graph
from agent.utils import new_uuid

tc_router = APIRouter()

thread = {"configurable": {"thread_id": new_uuid()}}


import json  # 确保导入了 json


@tc_router.post("/generate-tc")
async def generate_tc(request: TCRequest):
    async def event_generator():
        config = {"configurable": {"thread_id": "teacher_001"}}

        async for event in graph.astream_events(
                {"questions": request.questions},
                version="v2",
                config=thread
        ):
            kind = event["event"]
            node_name = event["metadata"].get("langgraph_node")

            if kind == "on_chain_start" and node_name == "rewrite":
                log_data = {
                    "event_type": "on_chat_model_stream",
                    "content": "🚀 正在拆分并优化 ILOs，请稍候..."
                }
                yield f"data: {json.dumps(log_data, ensure_ascii=False)}"


            # 1. 抓取 LLM 正在蹦的词
            # if kind == "on_chat_model_stream":
            #
            #
            #     if node_name == "rewrite":
            #         content = event["data"]["chunk"].content
            #         print(content)
            #         if content:
            #             log_data = {
            #                 "event_type": "on_chat_model_stream",
            #                 "content": "正在拆分ILOs..."
            #             }
            #             yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"

            # 2. 只有当任务彻底完成时
            elif kind == "on_chain_end" and node_name == "rewrite":
                output = event["data"].get("output")
                temp_output = output["questions"]
                final_content = ""
                for line in temp_output:
                    final_content += line+"\n"
                log_data = {
                    "event_type": "on_chat_model_stream",
                    "content": "\n--- [ILO拆分完毕完毕] ---\n"+str(final_content)+"\n"+"\n"*150
                }
                yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # ✅ 显式告诉浏览器不要缓存，立刻推送
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )