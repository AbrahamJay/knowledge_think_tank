from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import MILVUS_CHUNK_TOP_K
from app.shared.runtime.logger import logger, step_log

@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
        验证并获取数据
    :param state:
    :return:
    """
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])
    if not embedding_chunks or not hyde_embedding_chunks:
        logger.info(f"embedding_chunks or hyde_embedding_chunks 为空")
        raise ValueError("embedding_chunks or hyde_embedding_chunks 为空")
    return embedding_chunks, hyde_embedding_chunks

@step_log("use_rrf_rank")
def use_rrf_rank(data_list, k: int = 60):
    """
        使用RRF算法进行排名
    :param data_list: 数据列表
    :return: 排名结果
    """
    score_dict: dict[str, float] = {}
    chunk_dict: dict[str, dict] = {}  # 两路chunk基本属性一定一致 除了score chunk_id content
    for weight, chunks in data_list:
        for rank, chunk in enumerate(chunks, start=1):
            rrf_score = weight * (1 / (k + rank))
            chunk_id = chunk.get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + rrf_score
            # chunk_dict[chunk_id] = chunk
            chunk_dict.setdefault(chunk_id, chunk)
    for chunk_id, score in score_dict.items():
        chunk_dict[chunk_id]["score"] = score

    rrf_chunks: list[dict] = list(chunk_dict.values())
    logger.debug(f"排序之前,长度{len(rrf_chunks)},内容：{rrf_chunks}")
    rrf_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    # sorted(chunk_dict.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    logger.debug(f"排序之后,长度{len(rrf_chunks)},内容：{rrf_chunks}")

    # 截取分最高的前k个
    return rrf_chunks[:MILVUS_CHUNK_TOP_K]

@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    # 1.获取并校验参数
    embedding_chunks, hyde_embedding_chunks = validate_and_get_data(state)
    # 2.使用rrf的排名逻辑
    # 配置每路权重
    data_list: list[tuple[float, list]] = [
        (0.5, embedding_chunks),
        (0.5, hyde_embedding_chunks),
    ]
    rrf_chunks = use_rrf_rank(data_list)
    # 3.更新state
    state["rrf_chunks"] = rrf_chunks
    return state
