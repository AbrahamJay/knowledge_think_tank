"""
import_schema.py

一句话说明这个模块是干什么的。
"""
from pydantic import BaseModel


class UploadResponseSchema(BaseModel):
    code: int = 200
    message: str
    task_ids: list[str]


class StatusResponseSchema(BaseModel):
    code: int = 200
    task_id: str
    status: str
    done_list: list[str]
    running_list: list[str]
