import re

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.shared.clients.mongo_history_utils import save_chat_message, get_recent_messages


def answer_exist_in_state(state: QueryGraphState) -> bool:
    """
        检查答案是否已经存在
    :param state:
    :return:
    """
    answer: str = state.get("answer", "")
    if answer:
        logger.debug(f"第一个节点没有明确的item_name, 直接输出答案：{answer}")
        return True
    else:
        logger.debug(f"没有answer,本次正常模型获取答案和图片即可")
        return False


def validate_and_get_data(state: QueryGraphState):
    """
        验证并获取数据
    :param state:
    :return:
    """
    reranked_docs = state.get("reranked_docs", [])
    session_id = state.get("session_id", "")
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query", "")
    is_stream = state.get("is_stream", False)

    if not reranked_docs or not session_id or not item_names or not rewritten_query:
        logger.error(f"核心参数为空, 业务无法继续, 提前终止!")
        raise ValueError("核心参数为空, 业务无法继续, 提前终止!")
    return reranked_docs, session_id, item_names, rewritten_query, is_stream


def get_history_by_session_id(session_id: str) -> list[dict]:
    """
      获取当前session_id对应有效的聊天记录
    :param session_id:
    :return:
    """
    # 1.调用mongo的函数获取聊天记录
    history_list: list[dict] = get_recent_messages(session_id=session_id, limit=10)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # 2.筛选有效的数据
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    # 3.返回聊天记录结果
    return history_list


def create_answer_prompt(reranked_docs: list[dict], history_message: list[dict], item_names: list[str],
                         rewritten_query: str) -> str:
    """
        拼接提示词
    :param reranked_docs: 排序后的文档列表
    :param history_list: 聊天记录列表
    :param item_names: 当前对话的item_names
    :param rewritten_query: 重写后的用户问题
    :return: 拼接后的提示词
    """
    context: str = ""
    for doc in reranked_docs:
        context += (f"标题：{doc.get('title')},数据来源：{'联网搜索' if doc.get('type') == 'web' else '向量数据库'},"
                    f"置信度：{doc.get('score')},内容：{doc.get('text')} \n")

    if history_message:
        history_text_list: list[str] = []
        for index, history in enumerate(history_message, start=1):
            if history.get("role") == 'user':
                history_text_list.append((
                    f"编号:{index},用户提问,原始问题:{history.get('text')},重写后的问题:{history.get('rewritten_query')},"
                    f"关联item_name:{','.join(history.get('item_names', []))}"))
            else:
                history_text_list.append((
                    f"编号:{index},助手回答,重写问题:{history.get('rewritten_query')},回答结果:{history.get('text')[:50]},"
                    f"关联item_name:{','.join(history.get('item_names', []))}"))
        history_text = "\n".join(history_text_list)
    else:
        history_text = "没有有效的历史聊天记录!"

    item_names_str: str = ",".join(item_names)

    answer_prompt_text: str = load_prompt("answer_out", context=context, history=history_text,
                                          item_names=item_names_str,
                                          question=rewritten_query)

    return answer_prompt_text


def call_llm_create_answer(answer_prompt: str, is_stream: bool, session_id: str) -> str:
    """
        调用 LLM 生成最终答案
    :param answer_prompt:
    :param is_stream:
    :param session_id:
    :return:
    """
    llm_model = llm_provider.llm_model()
    answer: str = ""
    if is_stream:
        stream = llm_model.stream(answer_prompt)
        for chunk in stream:
            chunk_str = chunk.content
            # 每块都需要推给前端
            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={"delta": chunk_str}
            )
            answer += chunk_str
    else:
        response = llm_model.invoke(answer_prompt)
        answer = response.content
    return answer


def extract_chunk_and_url_image(reranked_docs: list[dict]) -> list[str]:
    """
        从reranked_docs中的text中提取图片 URL
    :param reranked_docs:
    :return:
    """
    rep = re.compile(r"\!\[.*?\]\((.*?)\)")  # 匹配markdown中的图片 URL
    image_urls: list[str] = []
    for doc in reranked_docs:
        url: str = doc.get("url", "")
        text: str = doc.get("text", "")
        if url:
            if url.endswith(('.png', '.jpg', '.jpeg', '.jpeg', '.gif')):
                image_urls.append(url)
        if text:
            image_url_list: list[str] = rep.findall(text)
            if image_url_list:
                image_urls.extend(image_url_list)
    logger.info(f"已完成切片中的图片提取,提取数量:{len(image_urls)},内容:{image_urls}")
    return image_urls


def save_assistant_message(state):
    """
        保存助手消息
    :param state:
    :return:
    """
    save_chat_message(
        session_id=state.get("session_id", ""),
        role="assistant",
        text=state.get("answer", ""),
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    # 1.获取和判断answer是否存在
    has_answer: bool = answer_exist_in_state(state)
    # 2.不存在 模型回答answer以及提取image_urls的过程
    if not has_answer:
        # 2.1 获取并检验参数
        reranked_docs, session_id, item_names, rewritten_query, is_stream = validate_and_get_data(state)
        # 2.2 读取有效的聊天记录
        history_message: list[dict] = get_history_by_session_id(session_id)
        # 2.3 构建 Prompt
        answer_prompt: str = create_answer_prompt(reranked_docs, history_message, item_names, rewritten_query)
        # 2.4 调用 LLM 生成最终答案
        answer: str = call_llm_create_answer(answer_prompt, is_stream, session_id)
        # 2.5 从引用文档中提取图片 URL
        image_urls: list[str] = extract_chunk_and_url_image(reranked_docs)
        # 2.6 更新state
        state["answer"] = answer
        state["image_urls"] = image_urls
    # 3. 保存聊天记录(助手回答)
    save_assistant_message(state)
    return state
