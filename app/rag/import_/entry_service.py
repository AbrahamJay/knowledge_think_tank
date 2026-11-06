"""
entry_service.py

一句话说明这个模块是干什么的。
"""
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log


@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """
    local_file_path: str = state.get("local_file_path")
    if not local_file_path:
        logger.error(f"local_file_path的参数为空, 业务无法继续进行，请检查输入参数")
        raise ValueError("local_file_path的参数为空, 业务无法继续进行，请检查输入参数")

    if local_file_path.lower().endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["is_md_read_enabled"] = False
        state["pdf_path"] = local_file_path
        state["md_path"] = ""
        logger.info(f"识别文件类型为pdf，文件路径为{local_file_path}")
    elif local_file_path.lower().endswith(".md"):
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["md_path"] = local_file_path
        state["pdf_path"] = ""
        logger.info(f"识别文件类型为md，文件路径为{local_file_path}")
    else:
        logger.error(f"{local_file_path}文件类型错误，当前项目只支持pdf和md文件")
        raise ValueError("文件类型错误，当前项目只支持pdf和md文件")

    local_file_path_obj: Path = Path(local_file_path)
    if not local_file_path_obj.is_file():
        logger.error(f"{local_file_path}文件不存在, 业务无法继续进行")
        raise ValueError(f"{local_file_path}文件不存在, 业务无法继续进行")
    state["file_title"] = local_file_path_obj.stem
    return state
