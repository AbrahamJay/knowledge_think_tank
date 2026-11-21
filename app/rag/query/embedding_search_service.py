from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger


def validate_and_get_data(state: QueryGraphState):
    """
        验证输入数据并获取 item_names 和 rewritten_query
    :param state:
    :return:
    """
    item_names = state.get("item_names", "")
    rewritten_query = state.get("rewritten_query", "")
    if not item_names or not rewritten_query:
        logger.info(f"item_names or rewritten_query 为空，请检查输入")
        raise ValueError("item_names or rewritten_query 为空，请检查输入")
    return item_names, rewritten_query


def select_chunks_in_milvus(item_names: list[str], rewritten_query: str) -> list[dict]:
    """
        根据商品名称和改写后的问题从 Milvus 中检索相关的知识切片
    :param item_names:
    :param rewritten_query:
    :return:
    """
    # 1. 使用BGE-M3生成稠密和稀疏向量
    result = llm_provider.generate_embeddings([rewritten_query])
    dense_vector = result["dense"][0]
    sparse_vector = result["sparse"][0]
    # 2.使用封装函数,生成AnnSearchRequset列表
    reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=f"item_name in {item_names}",
        limit=5 * 2
    )
    # 3.调用混合检索函数获取结果
    response = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.5, 0.5),
        limit=5,
        norm_score=True,
        output_fields=[
            "chunk_id",
            "file_title",
            "item_name",
            "title",
            "parent_title",
            "part",
            "content"
        ]
    )
    return response[0]


def after_deal_milvus_result(real_response) -> list[dict]:
    """
        处理 Milvus 检索结果
    :param real_response:
    :return:
    """
    if not real_response:
        logger.info("Milvus 检索结果为空")
        return []
    embedding_chunks: list[dict] = [
        {
            "chunk_id": item.get("entity", {}).get("chunk_id"),
            "file_title": item.get("entity", {}).get("file_title"),
            "item_name": item.get("entity", {}).get("item_name"),
            "title": item.get("entity", {}).get("title"),
            "parent_title": item.get("entity", {}).get("parent_title"),
            "part": item.get("entity", {}).get("part"),
            "content": item.get("entity", {}).get("content"),
            "score": item.get("distance", 0.0),
            "type": "milvus"
        }
        for item in real_response
    ]
    logger.info(f"已完成问题的 Milvus 检索结果处理，共 {len(embedding_chunks)} 条结果")
    return embedding_chunks


def search_by_embedding(state: QueryGraphState) -> list[dict]:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    item_names, rewritten_query = validate_and_get_data(state)
    real_response: list[dict] = select_chunks_in_milvus(item_names, rewritten_query)
    embedding_chunks = after_deal_milvus_result(real_response)
    return embedding_chunks
