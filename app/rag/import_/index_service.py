"""
index_service.py

一句话说明这个模块是干什么的。
"""
from app.process.import_.agent.state import ImportGraphState


def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    return state