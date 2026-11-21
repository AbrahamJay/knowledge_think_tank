from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import ITEM_NAME_CONFIRM_THRESHOLD, ITEM_NAME_CANDIDATE_THRESHOLD, ITEM_NAME_CONFIRMED_TOPK, \
    ITEM_NAME_OPTIONS_TOPK
from app.shared.clients.mongo_history_utils import save_chat_message
from app.shared.runtime.logger import logger, step_log
from app.shared.clients.mongo_history_utils import get_recent_messages
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt

@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState) -> tuple[str, str]:
    """
     获取必要参数和检验
    :param state:
    :return:
    """
    # 1.获取请求参数
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 2.非空校验
    if not session_id or not original_query:
        logger.error(f"session_id或者original_query为空,业务无法继续,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续,提前终止!")
    # 3.返结果
    return original_query, session_id

@step_log("get_history_by_session_id")
def get_history_by_session_id(session_id: str) -> list[dict]:
    """
      获取当前session_id对应有效的聊天记录
    :param session_id:
    :return:
    """
    # 1.调用mongo的函数获取聊天记录
    """
      {
        _id:ObjectId(1),
        role: user / assistant
        text: user-> original_query  assistant -> answer
        rewritten_query => 重写的问题
        item_names = [] 关联的item_name 不为空!!
        image_urls = []
        ts = 时间戳...
       }
    """
    history_list: list[dict] = get_recent_messages(session_id=session_id, limit=10)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # 2.筛选有效的数据
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    # 3.返回聊天记录结果
    return history_list

@step_log("call_llm_rewritten_and_extract_itemnames")
def call_llm_rewritten_and_extract_itemnames(history_list: list[dict], original_query: str) -> dict[str, Any]:
    """
      重写问题 + 提取item_names
    :param history_list: 有效的聊天记录
    :param original_query: 原始问题
    :return: {item_names:[],rewritten_query:str}
    """
    # 1. 获取大语言模型客户端对象
    llm_model = llm_provider.llm_model(json_mode=True)
    # 2. 拼接提示词
    """
      {
        _id:ObjectId(1),
        role: user / assistant
        text: user-> original_query  assistant -> answer
        rewritten_query => 重写的问题
        item_names = [] 关联的item_name 不为空!!
        image_urls = []
        ts = 时间戳...
       }
     问题: 不要将回答全部放入history_text 会稀释规则 / 原始问题 / json参考格式
       km3 = 角色,原始问题->重写了什么样?关联的item_name -> user [很重要]
       km4 = 助手,重写问题->什么样的回答[:50]?关联的item_name -> assistant
     问题: 建议用户当前的问题,向上提取,放在第一行! 规则/json格式一并向上!
     问题: 强调,以本次用户的问题为主,进行提取! 用户可能转变提问的主体!!

    """
    history_text: str = None
    if history_list:
        history_text_list: list[str] = []

        for index, history in enumerate(history_list, start=1):
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
    query = original_query

    prompt_text: str = load_prompt("rewritten_query_and_itemnames", query=query, history_text=history_text)

    message = HumanMessage(
        content=prompt_text
    )
    # 3. 封装调用链
    chains = llm_model | JsonOutputParser()
    # 4. 执行调用并获取结果
    json_dict: dict = chains.invoke([message])
    # 5. 处理和解析
    if "item_names" not in json_dict:
        json_dict['item_names'] = []
    if "rewritten_query" not in json_dict:
        json_dict['rewritten_query'] = original_query
    # 6. 返回结果
    return json_dict

@step_log("select_item_names_milvus")
def select_item_names_milvus(item_names: list[str]) -> dict[str, list[dict]]:
    """
        根据item_names进行向量数据检索
    :param item_names:
    :return:
    """
    milvus_search_result: dict[str, list[dict]] = {}
    # { "dense": [1, 2, 3], "sparse": [1, 2, 3] }
    result = llm_provider.generate_embeddings(item_names)
    for index, llm_item_name in enumerate(item_names):
        dense_vector = result['dense'][index]
        sparse_vector = result['sparse'][index]
        reqs_list = milvus_gateway.create_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=5 * 2
        )
        # [[ {id：xx，distance:xxx, entity:{item_name:xxx}}]]
        response = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            limit=5,
            output_fields=["item_name"]
        )
        real_response = response[0]
        if not real_response:
            logger.warning(f"根据{llm_item_name}查询Milvus无结果, 数据库中可能不存在该item_name")
            break
        llm_item_result_list: list[dict] = []
        for item in real_response:
            llm_item_result_list.append({
                "item_name": item.get("entity", {}).get("item_name", ""),
                "score": item.get("distance", 0),
            })
        milvus_search_result[llm_item_name] = llm_item_result_list
    return milvus_search_result

@step_log("select_confirmed_and_option_item_names")
def select_confirmed_and_option_item_names(milvus_dict: dict[str, list[dict]]) -> dict[str, list[str]]:
    """
        根据向量数据库检索结果评判出确定列表和可选项列表
    :param milvus_dict: 向量数据库检索结果
    :return: 确定列表和可选项列表
    """
    confirmed_list: list[str] = []
    option_list: list[str] = []
    # milvus_dict {
    # 模型item_name:[{item_name:xx, score:0.7}],
    # 模型item_name:[{item_name:xx, score:0.7}]
    # }
    for item_name, milvus_list in milvus_dict.items():
        high_list = [str(item.get("item_name")) for item in milvus_list if
                     item.get("score", 0.0) > ITEM_NAME_CONFIRM_THRESHOLD]
        middle_list = [str(item.get("item_name")) for item in milvus_list if
                       ITEM_NAME_CANDIDATE_THRESHOLD <= item.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD]
        if high_list:
            confirmed_list.extend(high_list[:ITEM_NAME_CONFIRMED_TOPK])
            logger.info(f"{item_name}已获取到可信的主题：{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue
        if middle_list:
            option_list.extend(middle_list[:ITEM_NAME_OPTIONS_TOPK])
            logger.info(f"{item_name}没有可信的主题，已获取到候选主题：{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")

    return {
        "confirmed": confirmed_list,
        "option": option_list
    }

@step_log("change_state_property")
def change_state_property(state: QueryGraphState, rewritten_query: str, confirmed_option_dict: dict[str, list[str]]):
    """
        更新state
    :param state:
    :param rewritten_query:
    :param confirmed_option_dict:
    :return:
    """
    confirmed_list = confirmed_option_dict.get("confirmed", [])
    option_list = confirmed_option_dict.get("option", [])
    if confirmed_list:
        state["item_names"] = confirmed_list
        state["rewritten_query"] = rewritten_query
        logger.info(f"已经有明确的item_name列表,已经更新state中item_names和rewritten_query")
        return None

    if option_list:
        state["answer"] = f"没有可信的item_name,但有可选的：{option_list},请从可选项中选择一个item_name进行确认"
        logger.info(f"没有可信的item_name列表,但有可选的！已经更新state中answer")
        return None

    state["answer"] = f"没有可信的item_name列表,也没有可选的！请明确以后再次提问！"
    return None

@step_log("save_user_chat_message")
def save_user_chat_message(state: QueryGraphState):
    """
        保存用户聊天消息
    :param state:
    :return:
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=[],
    )

@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    """
    confirmed_option_dict: dict[str, list[str]] = {}
    # 1. 获取参数并且校验 original_query / session_id
    original_query, session_id = validate_and_get_data(state)
    # 2. 获取[有效的]历史聊天记录 session_id -> history_list
    history_list: list[dict] = get_history_by_session_id(session_id)
    # 3. 调用模型进行问题重写和item_name提取
    llm_dict: dict[str, Any] = call_llm_rewritten_and_extract_itemnames(history_list, original_query)
    if llm_dict.get("item_names"):
        # 4.根据模型提取的item_name进行向量数据的检索
        milvus_dict: dict[str, list[dict]] = select_item_names_milvus(llm_dict.get("item_names", []))
        # 5.根据向量数据库检索的结果(分)评判出两个列表 一个确定列表 confirmed  一个可选列表 option
        confirmed_option_dict = select_confirmed_and_option_item_names(milvus_dict)
    # 6.根据确认和可选项列表进行最终的确定
    change_state_property(state, str(llm_dict.get("rewritten_query")), confirmed_option_dict)
    # 7.写入聊天记录
    save_user_chat_message(state)
    return state
