"""
embedding_service.py

一句话说明这个模块是干什么的。
"""
import json
from pathlib import Path
from typing import Any

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState) -> list[dict[str, Any]]:
    """
    验证数据并获取数据
    :param state:
    :return:
    """
    chunks: list[dict[str, Any]] = state.get("chunks", [])
    if not chunks:
        md_path: str = state.get("md_path", "")
        md_path_obj = Path(md_path)
        if md_path_obj.is_file():
            json_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk["item_name"] = state.get("item_name", "")
                state["chunks"] = chunks
            else:
                logger.error(f"chunks没有数据，json备份文件不存在")
                raise ValueError("chunks没有数据，json备份文件不存在")
        else:
            logger.error(f"chunks没有数据，同时也没有读取到对应的json备份数据")
            raise ValueError("chunks没有数据，同时也没有读取到对应的json备份数据")
    return chunks

@step_log("batch_generate_embeddings")
def batch_generate_embeddings(chunks: list[dict[str, Any]]):
    """
    批量生成向量
    为什么要批量生成向量？ 提高设备能力 | 减少通信次数
    多少为一批？ ①嵌入模型窗口的大小 8192  一个中文1.3~1.4 预估5个一批 ②显卡的显存大小
    :param chunks:
    :return:
    """
    for index in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        current_chunks = chunks[index: index + EMBEDDING_BATCH_SIZE]
        content_list: list[str] = [chunk.get('item_name', "") + "_" + chunk.get("content", "") for chunk in
                                   current_chunks]
        result: dict[str, list] = llm_provider.generate_embeddings(content_list)
        dense_list = result.get("dense", [])
        sparse_list = result.get("sparse", [])
        for i, chunk in enumerate(current_chunks):
            chunk["dense_vector"] = dense_list[i]
            chunk["sparse_vector"] = sparse_list[i]
    logger.info(f"chunks已经批量生成向量！{chunks[:2]}")

@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    # 1.获取并检验参数
    chunks: list[dict[str, Any]] = validate_and_get_data(state)
    # 2.批量chunks生成对应的向量
    batch_generate_embeddings(chunks)
    # 3.更新state embedding_content
    state["embedding_content"] = chunks
    # 4.返回即可
    return state
