"""
query_schema.py

一句话说明这个模块是干什么的。
"""
from pydantic import BaseModel


class QueryRequestSchema(BaseModel):
    query: str
    session_id: str
    is_stream: bool


class QueryStreamResponseSchema(BaseModel):
    session_id: str
    message: str


class QueryNotStreamResponseSchema(BaseModel):
    session_id: str
    message: str
    answer: str | None
    done_list: list[str]
    image_urls: list[str]


# 聊天记录对应类型
class ClearResponseSchema(BaseModel):
    message: str
    deleted_count: int


# 查询返回的item
class QueryHistoryItemResponseSchema(BaseModel):
    id: str
    session_id: str
    role: str
    text: str
    rewritten_query: str
    item_names: list[str]
    image_urls: list[str]
    ts: float


class QueryHistoryResponseSchema(BaseModel):
    session_id: str
    items: list[QueryHistoryItemResponseSchema]
