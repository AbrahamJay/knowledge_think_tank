import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
        验证输入数据并获取 rewritten_query
    :param state:
    :return:
    """
    rewritten_query = state.get("rewritten_query", "")
    if not rewritten_query:
        logger.info(f"rewritten_query 为空，请检查输入")
        raise ValueError("rewritten_query 为空，请检查输入")
    return rewritten_query


@step_log("call_mcp_tool")
async def call_mcp_tool(rewritten_query: str):
    """
        调用 MCP 工具进行搜索
        uv add openai-agents
    :param rewritten_query: 重写问题
    :return: 搜索结果
    """
    # 1.创建一个 MCP 服务器
    mcp_server = MCPServerStreamableHttp(
        name="bailian_search_mcp",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 50
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )
    # 2.主动触发连接
    await mcp_server.connect()
    try:
        # 3.查询所有工具信息
        list_tools = await mcp_server.list_tools()
        logger.info(f"查询执行服务器：{infra_config.mcp_config.mcp_base_url},获取工具列表：{list_tools}")
        # 4.调用工具
        result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 10
            }
        )
        return result
    except Exception as e:
        logger.exception(f"调用 MCP 工具出错：{str(e)}")
    finally:
        # 5.释放资源
        await mcp_server.cleanup()


@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1.检验参数
    rewritten_query: str = validate_and_get_data(state)
    # 2.调用mcp服务进行工具调用
    result = asyncio.run(call_mcp_tool(rewritten_query))
    # 3.结果解析
    text_result: str = result.content[0].text
    web_search_docs = json.loads(text_result).get("pages", [])
    return web_search_docs
