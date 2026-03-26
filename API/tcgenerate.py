import asyncio
import hashlib
from typing import Optional

from fastapi import FastAPI, APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, END
import json

from API.TCRequest import TCRequest
from agent.build import graph
from agent.utils import new_uuid

tc_router = APIRouter()
running_threads = set()
lock = asyncio.Lock()


import json  # 确保导入了 json


@tc_router.post("/generate-tc")
async def generate_tc(request: TCRequest):
    content_str = "".join(request.questions) + (request.course_info or "")
    thread_id = hashlib.md5(content_str.encode()).hexdigest()
    # 幂等逻辑
    thread = {"configurable": {"thread_id": thread_id}}

    # 1. 查询记忆断点：看看之前有没有跑过
    current_state = graph.get_state(thread)
    has_memory = bool(current_state.values)  # 如果 values 不为空，说明有记忆
    is_completed = has_memory and not current_state.next  # 有记忆且没有下一步，说明已跑完

    async with lock:
        # 在这个 with 块里，同一时间只有一个请求能进来
        if thread_id in running_threads:
            is_duplicate = True
        else:
            is_duplicate = False
            running_threads.add(thread_id)

    async def event_generator():
        if is_duplicate:
            yield f"data: {json.dumps({'type': 'error', 'content': '☕ 发现相同内容的教案正在后台努力生成中，请耐心等待，不要重复点击哦~'})}\n\n"
            return

        if is_completed:
            yield f"data: {json.dumps({'event_type': 'info', 'content': '✨ 发现已生成的缓存，直接为您呈现...'})}\n\n"
            # 这里可以根据你的需求，直接 yield 之前存好的 final_markdown
            return

        try:
            # --- 关键逻辑修改点 ---
            # 如果有记忆，传入 None 表示续传；如果没有记忆，传入初始化 questions
            input_data = None if has_memory else {"questions": request.questions,"class_info": request.class_info, "course_info": request.course_info}
            print(input_data)

            if has_memory:
                yield f"data: {json.dumps({'event_type': 'info', 'content': '📂 正在恢复之前的生成进度...'})}\n\n"

            async for event in graph.astream_events(
                    input_data,
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

                if kind == "on_chain_start" and node_name == "retrieve":
                    log_data = {
                        "event_type": "on_chat_model_stream",
                        "content": "🚀 正在检索相关文档，请稍候..."
                    }
                    yield f"data: {json.dumps(log_data, ensure_ascii=False)}"

                elif kind == "on_chain_start" and node_name == "generate":
                    log_data = {
                        "event_type": "on_chat_model_stream",
                        "content": "🚀 正在结合文档生成教学task，请稍候..."
                    }
                    yield f"data: {json.dumps(log_data, ensure_ascii=False)}"


                elif kind == "on_chain_end" and node_name == "rewrite":
                    output = event["data"].get("output")
                    temp_output = output["questions"]
                    final_content = ""
                    for line in temp_output:
                        final_content += line+"\n"
                    log_data = {
                        "event_type": "on_chat_model_stream",
                        "content": "\n--- [ILO拆分完毕完毕] ---\n"+str(final_content)+"\n"
                    }
                    yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end" and node_name == "retrieve":
                    output = event["data"].get("output")
                    temp_output = output["documents"]
                    log_data = {
                        "event_type": "on_chat_model_stream",
                        "content": "\n--- [检索文档成功] ---\n共检索到"+str(len(temp_output))+"个文档\n"
                    }
                    yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end" and node_name == "generate":
                    output = event["data"].get("output")
                    # 这里的 temp_output 是你提供的那个 [{...}] 列表
                    temp_output = output.get("messages", [])

                    output_content = ""
                    print(temp_output)

                    # 1. 确保列表不为空
                    if temp_output and isinstance(temp_output, list):
                        # 2. 获取第一个元素（那个字典）
                        item = temp_output[0]

                        # 3. 提取字段并拼接成一段文字
                        # 我们优先提取 TC (教学内容)，如果没有则取 ILO，或者都取
                        tc_text = item.get("TC", "")
                        ilo_text = item.get("ILO", "")

                        if tc_text:
                            # 如果你只想返回 TC 的内容
                            output_content = f"教学活动设计：\n{tc_text}"
                        elif ilo_text:
                            output_content = f"教学目标：\n{ilo_text}"
                        else:
                            output_content = "未生成有效内容"

                    # 4. 封装并推送
                    log_data = {
                        "event_type": "on_chat_model_stream",
                        "content": f"\n--- [教学task生成成功] ---\n"
                    }

                    yield f"data: {json.dumps(log_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"❌ 业务报错: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': '生成中断'})}\n\n"
        finally:
            running_threads.remove(thread_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # ✅ 显式告诉浏览器不要缓存，立刻推送
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Task-ID": thread_id,
            "X-Accel-Buffering": "no"
        }
    )



@tc_router.get("/get-final-plan")
async def revise_tc(task_id: Optional[str] = Header(None,convert_underscores=False)):
    print(task_id)
    content_dict = {"错误": "未找到对应的教学task，请确认Task-ID是否正确，或者教学task是否已生成完成"}
    state=""
    if task_id:

        # 2. 构造 LangGraph 所需的 config
        config = {"configurable": {"thread_id": task_id}}
        # 1. 先获取快照
        snapshot = graph.get_state(config)

        # 2. 获取数据字典
        state_values = snapshot.values  # 这是一个 dict

        # 3. 使用字典方式取值
        content_dict = {
            "Scenario": state_values.get("scenario", ""),
            "Tasks": state_values.get("answer", ""),
            "Deliverables": state_values.get("deliverables", ""),

        }


    return content_dict