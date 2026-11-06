"""
state.py

一句话说明这个模块是干什么的。
"""
import copy
import json
from typing import TypedDict, cast


class ImportGraphState(TypedDict):
    # 任务状态
    task_id: str
    # 文件记录
    # 输入未确认文件
    local_file_path: str
    # 确认
    md_path: str
    pdf_path: str
    local_dir: str
    file_title: str

    # 文件状态
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # md内容
    md_content: str

    # 切块
    chunks: list[dict]
    item_name: str
    embedding_content: list[dict]


# 模板对象
graph_default_state = ImportGraphState(
    task_id="",
    is_md_read_enabled=False,
    is_pdf_read_enabled=False,
    local_dir="",
    local_file_path="",
    pdf_path="",
    md_path="",
    file_title="",
    md_content="",
    chunks=[],
    item_name="",
    embedding_content=[],
)


# 项目执行 => 创建state对象 => invoke‘
# 方法一：根据传入的参数创建一个对应的state
def create_default_state(**args) -> ImportGraphState:
    deep_new_state = cast(ImportGraphState, cast(object, copy.deepcopy(graph_default_state)))
    deep_new_state.update(args)
    return deep_new_state


# 方法二：获取创建好的默认空state对象
def get_default_state() -> ImportGraphState:
    return graph_default_state


if __name__ == "__main__":
    state = create_default_state(task_id="123", local_file_path="./烫金机.pdf")
    print(f"{json.dumps(state, indent=4, ensure_ascii=False)}")
