"""
query_server.py

一句话说明这个模块是干什么的。
"""
import json
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.requests import Request
from starlette.responses import StreamingResponse

from app.api.schema.query_schema import QueryRequestSchema, QueryStreamResponseSchema, QueryNotStreamResponseSchema, \
    ClearResponseSchema, QueryHistoryResponseSchema, QueryHistoryItemResponseSchema
from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state, QueryGraphState
from app.shared.clients import clear_history, get_recent_messages
from app.shared.runtime.logger import logger
from app.shared.utils.sse_utils import sse_generator, create_sse_queue, push_to_session, SSEEvent
from app.shared.utils.task_utils import clear_task, update_task_status, get_done_task_list

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def check_health():
    # logger.info(f"完成本次健康状态检查,{datetime.now()}")
    return {
        "ok": True
    }


@app.get("/html")
def return_html():
    html_path_obj: Path = Path(__file__).parent.parent / "htmls" / "chat.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj)[0]
    )


@app.get("/stream/{session_id}")
def stream(session_id: str, request: Request):
    logger.info(f"{datetime.now()}完成{session_id}流式响应请求")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )


def invoke_query_graph(session_id: str, is_stream: bool, original_query: str):
    """
    封装了一个函数,专门执行查询图对象!!
    :param session_id:
    :param is_stream:
    :param original_query:
    :return:
    """
    try:
        clear_task(session_id)  # 清空上游数据
        if is_stream:
            # 流式 创建一个新的队列
            create_sse_queue(session_id)

        update_task_status(session_id, "processing", is_stream)
        state: QueryGraphState = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query=original_query
        )
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        state: QueryGraphState = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        update_task_status(session_id, "completed", is_stream)
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": state.get("answer", ""),
                    "status": "completed",
                    "image_urls": state.get("image_urls", [])
                }
            )
        return state
    except Exception as e:
        update_task_status(session_id, "failed", is_stream)
        logger.exception(f"{original_query}:执行图流程报错!")
        push_to_session(
            session_id,
            SSEEvent.ERROR,
            {
                "error": str(e)
            }
        )
        return {}


@app.post("/query")
def query(param: QueryRequestSchema, tasks: BackgroundTasks):
    """
      分析: 提问 -> 触发 -> 图的执行流程
      1. 接收参数 param:QueryRequestSchema
      2. 判断是否是流式
      3. 流式->异步执行
      4. 非流式->同步执行
    :param param: QueryRequestSchema
    :return: QueryStreamResponseSchema or QueryNotStreamResponseSchema
    """
    # 1.获取参数
    query = param.query
    is_stream = param.is_stream
    session_id = param.session_id
    logger.info(f"用户发起提问: {query} 关联的session_id:{session_id},流式状态:{is_stream}")

    # 2.判断流式状态
    if is_stream:
        # 异步执行
        tasks.add_task(invoke_query_graph, session_id=session_id, is_stream=is_stream, original_query=query)
        return QueryStreamResponseSchema(
            session_id=session_id,
            message=f"{session_id}正在问题解析...."
        )
    else:
        # 同步执行
        state = invoke_query_graph(session_id=session_id, is_stream=is_stream, original_query=query)
        done_list = get_done_task_list(session_id)
        return QueryNotStreamResponseSchema(
            session_id=session_id,
            message=f"{session_id}已经完成问题解析!",
            answer=state.get("answer", ""),
            done_list=done_list,
            image_urls=state.get("image_urls", [])
        )


@app.delete("/history/{session_id}")
def delete_history(session_id: str):
    deleted_count = clear_history(session_id)
    logger.info(f"已经清空：{session_id}对应的会话历史记录,删除了 {deleted_count} 条历史记录")
    return ClearResponseSchema(
        message=f"已经清空：{session_id}对应的会话历史记录",
        deleted_count=deleted_count
    )


@app.get("/history/{session_id}")
def search_history(session_id: str, limit: int = 10):
    history_list: list[dict] = get_recent_messages(session_id=session_id, limit=limit)
    logger.info(f"查询{session_id}获取会话历史记录: {len(history_list)}")
    return QueryHistoryResponseSchema(
        session_id=session_id,
        items=[
            QueryHistoryItemResponseSchema(
                id=str(history.get("_id")),
                session_id=str(history.get("session_id")),
                role=str(history.get("role")),
                text=str(history.get("text")),
                rewritten_query=str(history.get("rewritten_query")),
                item_names=history.get("item_names", []),
                image_urls=history.get("image_urls", []),
                ts=float(history.get("ts", 0))
            ) for history in history_list
        ]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
