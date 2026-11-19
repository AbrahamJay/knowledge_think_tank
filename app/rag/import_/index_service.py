"""
index_service.py

一句话说明这个模块是干什么的。
"""
from typing import Any

from pymilvus import MilvusClient, DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MILVUS_VECTOR_DIM
from app.shared.runtime.logger import logger


def validate_and_get_data(state: ImportGraphState) -> list[dict[str, Any]]:
    """
    验证数据并获取数据
    :param state:
    :return:
    """
    embedding_content: list[dict[str, Any]] = state.get("embedding_content", [])
    if not embedding_content:
        logger.error(f"embedding_content没有数据, 业务无法继续")
        raise ValueError("embedding_content没有数据, 业务无法继续")
    return embedding_content


def prepare_item_name_collection():
    """
    创建存储item_name需要的集合
    :return:
    """
    # 定义milvus客户端
    milvus_client: MilvusClient = milvus_gateway.client
    chunks_collection = milvus_gateway.chunks_collection

    # 检查是否存在集合
    if milvus_client.has_collection(chunks_collection):
        logger.info(f"集合{chunks_collection}已存在，无需创建")
        return
    logger.info(f"集合{chunks_collection}不存在，将创建集合")
    # 1.创建schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 2.创建索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量
    index_params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={"nlist": 128},  # IVF_FLAT 的有效参数，M/efConstruction 属于 HNSW
        # params={
        #     "M": 64,  # 相邻的最大节点数量
        #     "efConstruction": 100  # 候选的节点数量
        # },
    )
    # 稀疏向量
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},  # 跳过低分 计算高分数据集
    )
    # 3.创建集合
    milvus_client.create_collection(
        collection_name=chunks_collection,
        schema=schema,
        index_params=index_params
    )
    # 4.兜底确认集合已加载进内存（create_collection 内部会 load，但失败重试场景下状态不一定完整）
    milvus_gateway.ensure_collection_loaded(collection_name=chunks_collection)


def insert_item_name_data(embedding_content: list[dict[str, Any]]):
    """
    向集合存储item_name(item_name,稠密和稀疏,file_title)
    :param embedding_content:
    :return:
    """
    file_title: str = embedding_content[0]["file_title"] if embedding_content else ""
    # 1.先删除数据（带 filter 的删除由 QueryNode 执行，必须保证集合已加载）
    milvus_client: MilvusClient = milvus_gateway.client
    milvus_client.delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"file_title == '{file_title}'"
    )
    # 3.插入数据
    milvus_client.insert(
        collection_name=milvus_gateway.chunks_collection,
        data=embedding_content
    )


def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """
    入库服务总入口
    功能：校验切块数据 → 准备 Milvus 集合 → 清理旧数据 → 批量插入新数据
    输出：更新后的 state（chunks 中会回填 chunk_id）
    """
    # 1. 校验参数
    embedding_content = validate_and_get_data(state)
    # 2. 准备集合
    prepare_item_name_collection()
    # 3. 插入数据
    logger.info(f"{embedding_content[0]['file_title']}文档对应的数据已经完成向量数据的导入了")
    insert_item_name_data(embedding_content)
    return state
