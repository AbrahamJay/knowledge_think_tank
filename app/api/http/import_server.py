"""
import_server.py

一句话说明这个模块是干什么的。
"""
import json
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse

from app.api.schema.import_schema import UploadResponseSchema, StatusResponseSchema
from app.process.import_.agent.main_graph import import_app
from app.process.import_.agent.state import create_default_state, ImportGraphState
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.task_utils import update_task_status, get_task_status, get_done_task_list, get_running_task_list

app = FastAPI()


def invoke_import(local_file_path: str, local_dir: str, task_id: str):
    state: ImportGraphState = create_default_state(
        task_id=task_id,
        local_file_path=local_file_path,
        local_dir=local_dir,
        is_pdf_read_enabled=True,
        is_md_read_enabled=False
    )
    logger.info(f"整体开始执行解析：{json.dumps(state, indent=4, ensure_ascii=False)}")
    try:
        update_task_status(state["task_id"], "processing")
        state = import_app.invoke(state)
        update_task_status(state["task_id"], "completed")
    except Exception as e:
        logger.error(f"整体执行解析失败：{e}")
        update_task_status(state["task_id"], "failed")
    logger.info(f"整体执行解析结束：{json.dumps(state, indent=4, ensure_ascii=False)}")


@app.post("/upload")
async def upload(tasks: BackgroundTasks, files: list[UploadFile]):
    file = files[0]
    task_id = str(uuid.uuid4())
    local_dir_obj: Path = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    if not local_dir_obj.is_dir():
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    local_file_path_obj: Path = local_dir_obj / file.filename
    data = await file.read()
    local_file_path_obj.write_bytes(data)

    tasks.add_task(
        invoke_import,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_obj),
        task_id=task_id
    )

    return UploadResponseSchema(
        message=f"{local_file_path_obj.stem}文件上传成功,已经开始解析!!",
        task_ids=[task_id]
    )


@app.get("/html")
def html():
    html_path_obj: Path = Path(__file__).parent.parent / "htmls" / "import.html"
    return FileResponse(
        path=str(html_path_obj),
        media_type=guess_type(html_path_obj.name)[0],
        # filename=
    )


@app.get("/status/{task_id}")
def status(task_id: str):
    return StatusResponseSchema(
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
