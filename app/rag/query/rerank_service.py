from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_SUMMARY_CHAR_RATIO, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_ABS, RERANK_GAP_RATIO
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
        验证并获取数据
    :param state:
    :return:
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    rewritten_query = state.get("rewritten_query", "")
    if not rrf_chunks or not web_search_docs or not rewritten_query:
        logger.info("rrf_chunks, web_search_docs, rewritten_query 为空，请检查输入")
        raise ValueError("rrf_chunks, web_search_docs, rewritten_query 为空，请检查输入")
    return rrf_chunks, web_search_docs, rewritten_query


@step_log("merge_list_and_web")
def merge_list_and_web(rrf_chunks: list, web_search_docs: list) -> list[dict]:
    """
        合并 RRF 和 Web Search 的文档
    :param rrf_chunks: RRF 文档列表
    :param web_search_docs: Web Search 文档列表
    :return: 合并后的文档列表
    """
    merged_list: list = []
    for chunk in rrf_chunks:
        merged_list.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("content", ""),
            "score": 0.0,
            "type": chunk.get("type", "milvus"),
            "url": ""
        })
    for doc in web_search_docs:
        merged_list.append({
            "chunk_id": "",
            "title": doc.get("title", ""),
            "text": doc.get("snippet", ""),
            "score": 0.0,
            "type": "web",
            "url": doc.get("url", "")
        })
    logger.info(f"已完成两路数据的融合, 融合后总长度为：{len(merged_list)}")
    return merged_list


@step_log("create_question_answer_pair")
def create_question_answer_pair(merged_list: list[dict], rewritten_query: str) -> list[list]:
    """
        封装问题+答案对
    :param merged_list: 合并后的文档列表
    :param rewritten_query: 重写后的查询问题
    :return: 问题+答案对列表
    """
    question_answer_pair = []
    question_len = llm_provider.compute_token_number(rewritten_query)
    for chunk in merged_list:
        # 添加长度判断
        answer = chunk.get("text", "")
        token_len = llm_provider.compute_token_number(answer)
        if token_len > RERANK_MAX_INPUT_TOKENS - 4 - question_len:
            # 压缩后的字符长度
            char_length = max(
                int((RERANK_MAX_INPUT_TOKENS - 4 - question_len) / RERANK_SUMMARY_CHAR_RATIO),
                RERANK_MIN_SUMMARY_CHARS)
            llm_model = llm_provider.llm_model()
            reranker_prompt: str = load_prompt("rerank_text_refine",
                                               question=rewritten_query,
                                               answer=answer,
                                               limit=char_length)
            message = HumanMessage(content=reranker_prompt)
            chains = llm_model | StrOutputParser()
            # 压缩后的值
            logger.debug(f"已触发压缩流程,压缩前值为：{len(answer)}:{answer[:100]}...")
            answer = chains.invoke([message])
            logger.debug(f"已触发压缩流程,压缩后值为：{len(answer)}:{answer[:100]}...")
        # text -> 答案
        question_answer_pair.append([rewritten_query, answer])

    return question_answer_pair


@step_log("list_score_and_rank")
def list_score_and_rank(merged_list: list[dict], question_answer_pair: list[list]):
    """
        对列表进行打分和排序
    :param merged_list:
    :param question_answer_pair:
    :return:
    """
    # 1.打分
    score_list = llm_provider.compute_score(question_answer_pair)
    logger.debug(f"完成打分，打分结果为：{score_list}")
    # 2.更新分
    for i, score in enumerate(score_list):
        merged_list[i]["score"] = score
    # 3.排序
    logger.debug(f"未排序之前的合并列表为：{merged_list}")
    merged_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    logger.debug(f"已排序之后的合并列表为：{merged_list}")


@step_log("dynamic_topk")
def dynamic_topk(merged_list: list[dict]):
    """
        动态TopK算法，进行动态内容截取（断崖处理）
    :param merged_list:
    :return:
    """
    # 场景2：列表长度小于max值
    max_topk = min(len(merged_list), RERANK_MAX_TOPK)
    min_topk = RERANK_MIN_TOPK
    # 定义topk
    # 场景1：一个断崖没有 top_k = max_topk
    tok_k = max_topk
    # 场景3：min > max 正常不会 列表长度小于min 直接取max
    # 循环指正
    for index in range(min_topk - 1, max_topk - 1):
        current_data = merged_list[index]
        next_data = merged_list[index + 1]
        abs = current_data.get("score", 0.0) - next_data.get("score", 0.0)
        ratio = abs / current_data.get("score", 0.0)
        if abs > RERANK_GAP_ABS or ratio > RERANK_GAP_RATIO:
            # 发生断崖
            tok_k = index + 1
            logger.debug(f"下标{tok_k}发生断崖，断崖值为：{abs}:{ratio}")
            break

    reranked_docs = merged_list[:tok_k]
    return reranked_docs


@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    # 1.获取并校验参数
    rrf_chunks, web_search_docs, rewritten_query = validate_and_get_data(state)
    # 2.对齐数据格式 rrf_chunks | web_search_docs
    merged_list: list[dict] = merge_list_and_web(rrf_chunks, web_search_docs)
    # 3.封装问题+答案列表（检查是否超过窗口/超过模型压缩）
    question_answer_pair: list[list] = create_question_answer_pair(merged_list, rewritten_query)
    # 4.调用reranker模型进行数据打分处理，并进行数据排序
    list_score_and_rank(merged_list, question_answer_pair)
    # 5.使用动态topK算法，进行动态内容截取（断崖处理）
    reranked_docs = dynamic_topk(merged_list)
    # 6.更新state
    state["reranked_docs"] = reranked_docs
    return state
