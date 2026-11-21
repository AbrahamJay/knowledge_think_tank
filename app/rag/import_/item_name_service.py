"""
item_name_service.py

一句话说明这个模块是干什么的。
"""
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient, DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS, MILVUS_VECTOR_DIM, \
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

# item_name 集合必须存在的两个索引名：建索引与集合完整性校验共用同一份定义
DENSE_VECTOR_INDEX_NAME = "dense_vector_index"
SPARSE_VECTOR_INDEX_NAME = "sparse_vector_index"
ITEM_NAME_INDEX_NAMES = (DENSE_VECTOR_INDEX_NAME, SPARSE_VECTOR_INDEX_NAME)


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState) -> tuple[list[dict[str, Any]], str]:
    """
    验证数据并获取数据
    :param state:
    :return:
    """
    chunks: list[dict[str, Any]] = state.get("chunks", [])
    file_title: str = state.get("file_title", "")
    md_path: str = state.get("md_path", "")
    md_path_obj = Path(md_path)
    if not chunks:
        if md_path_obj.is_file():
            json_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                state["chunks"] = chunks
            else:
                logger.error(f"chunks没有数据，json备份文件不存在")
                raise ValueError("chunks没有数据，json备份文件不存在")
        else:
            logger.error(f"chunks没有数据，同时也没有读取到对应的json备份数据")
            raise ValueError("chunks没有数据，同时也没有读取到对应的json备份数据")
    if not file_title:
        file_title = md_path_obj.stem or "default_title"
        state["file_title"] = file_title
    return chunks, file_title


@step_log("call_llm_return_item_name")
def call_llm_return_item_name(chunks: list[dict[str, Any]], file_title: str) -> str:
    """
    调用LLM识别item_name
    :param chunks:
    :param file_title:
    :return:
    """
    # 1.获取模型客户端
    llm_model = llm_provider.llm_model()
    # 2.拼接模型对应提示词 file_title 和 context
    context = ""
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        context += f"标题：{chunk.get('parent_title')}, 内容：{chunk.get('content')}\n"
    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    item_name_recognition_prompt: str = load_prompt("item_name_recognition", file_title=file_title, context=context)
    message = HumanMessage(content=item_name_recognition_prompt)
    # 3.拼接执行调用链和执行
    chains = llm_model | StrOutputParser()
    item_name = chains.invoke([message])
    # 4.判断item_name是否获取成功
    if not item_name:
        item_name = file_title
        logger.warning(f"没有识别到item_name，使用file_title代替{file_title}")
    # 5.返回item_name
    return item_name


@step_log("padding_item_name_to_chunks")
def padding_item_name_to_chunks(chunks: list[dict[str, Any]], item_name: str):
    """
    完善chunks数据，添加item_name
    :param chunks:
    :param item_name:
    :return:
    """
    for chunk in chunks:
        chunk["item_name"] = item_name


@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    """
    创建存储item_name需要的集合
    :return:
    """
    # 定义milvus客户端
    milvus_client: MilvusClient = milvus_gateway.client
    collection_name = milvus_gateway.item_name_collection

    # 检查是否存在集合
    if milvus_client.has_collection(collection_name):
        # 建索引失败时集合已经创建成功且不会回滚，残留的“有集合无索引”半成品必须重建，
        # 否则后续 delete/insert 会报 code=101 collection not loaded
        exists_index_names = milvus_client.list_indexes(collection_name)
        if all(index_name in exists_index_names for index_name in ITEM_NAME_INDEX_NAMES):
            logger.info(f"集合{collection_name}已存在，无需创建")
            # 集合可能被手动 release 或服务重启后未加载，这里兼并保证处于可读写状态
            milvus_gateway.ensure_collection_loaded(collection_name)
            return
        logger.warning(f"集合{collection_name}索引不完整，已有索引：{exists_index_names}，将删除后重建")
        milvus_client.drop_collection(collection_name)
    logger.info(f"集合{collection_name}不存在，将创建集合")
    # 1.创建schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 2.创建索引
    index_params = milvus_client.prepare_index_params()
    # 稠密向量
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name=DENSE_VECTOR_INDEX_NAME,
        metric_type="COSINE",
        # params={"nlist": 128},  # IVF_FLAT 的有效参数，M/efConstruction 属于 HNSW
        params={
            "M": 64,  # 相邻的最大节点数量
            "efConstruction": 100  # 候选的节点数量
        },
    )
    # 稀疏向量
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # 倒排索引
        index_name=SPARSE_VECTOR_INDEX_NAME,
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},  # 跳过低分 计算高分数据集
    )
    # 3.创建集合
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    # 4.兜底确认集合已加载进内存（create_collection 内部会 load，但失败重试场景下状态不一定完整）
    milvus_gateway.ensure_collection_loaded(collection_name)


@step_log("insert_item_name_data")
def insert_item_name_data(item_name: str, file_title: str):
    """
    向集合存储item_name(item_name,稠密和稀疏,file_title)
    :param item_name:
    :param file_title:
    :return:
    """
    # 1.生成稠密和稀疏向量
    result: dict[str, list] = llm_provider.generate_embeddings([item_name])
    # result = {"dense": [[]], "sparse": [[]]}
    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]
    # 2.先删除数据（带 filter 的删除由 QueryNode 执行，必须保证集合已加载）
    milvus_client: MilvusClient = milvus_gateway.client
    milvus_gateway.ensure_collection_loaded(milvus_gateway.item_name_collection)
    milvus_client.delete(
        collection_name=milvus_gateway.item_name_collection,
        filter=f"file_title == '{file_title}'"
    )
    # 3.插入数据
    milvus_client.insert(
        collection_name=milvus_gateway.item_name_collection,
        data=[
            {
                "item_name": item_name,
                "file_title": file_title,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )


@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    # 1.验证数据
    chunks, file_title = validate_and_get_data(state)
    # 2.使用模型提取item_name(chunk file_title) -> item_name
    item_name: str = call_llm_return_item_name(chunks, file_title)
    # 3.先修改chunks的数据 -> item_name
    padding_item_name_to_chunks(chunks, item_name)
    # 4.创建存储item_name需要的集合
    prepare_item_name_collection()
    # 5.向集合存储item_name(item_name,稠密和稀疏,file_title)
    insert_item_name_data(item_name, file_title)
    # 6.更新 state
    state["item_name"] = item_name
    state["chunks"] = chunks
    return state
