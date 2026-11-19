"""
split_service.py

一句话说明这个模块是干什么的。
"""
import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CODE_FENCE_UNCLOSED_MAX_LINES, TITLE_PATH_MAX_LEVELS, CHUNK_SIZE, CHUNK_OVERLAP, \
    CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


@step_log("validate_get_data")
def validate_get_data(state: ImportGraphState) -> tuple[str, str, str]:
    """
    获取并检验参数
    :param state:
    :return: (md_content, file_title)
    """
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")
    # 无内容时只能依靠 md_path 读取，路径不可用时直接报错，避免下一行抛出难以排查的异常
    if not md_content:
        if (not md_path) or (not Path(md_path).exists()):
            logger.error(f"md_content内容为空，md_path不存在：{md_path}")
            raise ValueError(f"md_content内容为空，md_path不存在：{md_path}")
        md_content = Path(md_path).read_text(encoding="utf-8")

    if not file_title:
        file_title = Path(md_path).stem if md_path else ""
        file_title = file_title or "default"
        logger.warning(f"file_title为空，使用默认值：{file_title}")
        state["file_title"] = file_title
    # 清洗和替换数据 统一不同系统的换行符，防止后续处理错误
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 回写清洗后的内容，保证后续节点拿到的 md_content 与切分时使用的版本一致
    state["md_content"] = md_content
    return md_content, file_title, md_path


@step_log("split_document_by_title")
def split_document_by_title(md_content: str, file_title: str) -> list[dict[str, Any]]:
    """
    根据标题进行切分（标题级一级切分）
    切分规则：
    1. 以 ATX 标题（# ~ ######）作为章节边界，同一章节的标题行与正文一起构成一个 chunk
    2. chunk 的 title 为从粗到细的标题链（如“安装指引_连接网络”），保证子章节不丢失父级语义
    3. 上一级标题下没有正文时不单独成块，而是与子标题归入同一章节（等价于标题合并）
    4. 第一个标题之前的孤儿正文独立成块，标题由 file_title 兜底，保证 title 永不为空
    5. 代码块内部的 # 开头行是注释而不是标题，不参与标题识别
    :param md_content: markdown内容
    :param file_title: 文件标题
    :return: 切分后的chunks列表
    """
    # 1. 定义存储数据的容器变量
    chunks: list[dict[str, Any]] = []
    # 标题层级链，元素为(层级, 标题文本)，用于拼出“父级_子级”的完整标题，避免子章节丢失父级语义
    title_stack: list[tuple[int, str]] = []
    # 当前章节的待落盘文本（首行为标题行）
    section_lines: list[str] = []
    # has_body 标记当前章节是否已出现正文，用它代替“行数>1”来判断，不依赖列表首行的含义
    has_body: bool = False
    # is_code 标记是否处于代码块内部；code_lines 用于兜底围栏标记缺失导致的长期不闭合
    is_code: bool = False
    code_lines: int = 0
    # 2.md_content内容切割
    md_content_lines = md_content.split("\n")
    title_reg = re.compile(r"^\s*#{1,6}\s.+$")
    # 代码块围栏（三个及以上 ` 或 ~），只在行首出现时才视为围栏
    code_fence_reg = re.compile(r"^\s*(`{3,}|~{3,})")

    def build_title() -> str:
        """
        拼接当前章节标题：无标题时（孤儿数据、纯正文文档）退化为文件标题，保证 title 字段永不为空
        """
        if not title_stack:
            return file_title
        # 标题链过长时只保留最近几级，层级越深语义越具体
        return "_".join(title_name for _, title_name in title_stack[-TITLE_PATH_MAX_LEVELS:])

    def flush_section() -> None:
        """
        当前章节落盘：只有已出现正文的章节才成块，避免产生“只有标题”的空chunk
        """
        nonlocal section_lines, has_body
        if not has_body:
            logger.debug(f"标题「{build_title()}」下没有正文，不单独成块")
            return
        chunks.append({
            "title": build_title(),
            "content": "\n".join(section_lines),
            "file_title": file_title,
        })
        section_lines, has_body = [], False

    for line in md_content_lines:
        line_strip = line.strip()
        if not line_strip:
            logger.debug(f"当前行为空行，跳过")
            continue
        # 是否进入或出去代码块：同一行内成对出现（例如 ```code```）时状态保持不变，避免is_code永久错位
        fence_match = code_fence_reg.match(line_strip)
        if fence_match and line_strip.count(fence_match.group(1)) % 2 == 1:
            is_code = not is_code
            code_lines = 0
            logger.debug(f"{'进入代码块' if is_code else '退出代码块'}")
            section_lines.append(line)
            continue
        if is_code:
            # 代码块内部一律归为正文，其中以 # 开头的是注释而不是标题
            code_lines += 1
            section_lines.append(line)
            has_body = True
            if code_lines > CODE_FENCE_UNCLOSED_MAX_LINES:
                logger.warning(f"代码块连续{code_lines}行仍未闭合，判定围栏标记缺失，强制退出代码块状态")
                is_code = False
            continue
        # 是否是有效标题[不在代码块中 | 并且符合正则规则]
        if title_reg.match(line_strip):
            # 先落盘上一章节：第一个标题之前的孤儿数据也在此处独立成块，
            # 否则会被拼进第一个章节，造成标题与正文语义不匹配
            flush_section()
            # 层级取 # 的个数，标题文本去掉 # 标记，避免markdown符号混入检索文本
            level = len(line_strip) - len(line_strip.lstrip("#"))
            title_name = line_strip.lstrip("#").strip()
            # 弹出同级及更深层级，保证标题链始终是从粗到细的单一路径
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title_name))
            # 连续标题（上一级没有正文）自然留在同一条标题链上，等价于原来的标题合并逻辑；
            # 正文只保留当前标题行，父级标题通过 title 字段体现，不再重复堆积
            section_lines = [line_strip]
            continue
        # 当前是普通行：使用原始line保留缩进，避免代码、表格、列表的层级被压平
        section_lines.append(line)
        has_body = True

    # 收尾落盘：最后一个章节不会因为缺少下一个标题而丢失，纯正文文档的标题由file_title兜底
    flush_section()
    logger.info(f"已经完成文档切分，共切分出{len(chunks)}个chunk")
    return chunks


@step_log("split_chunk_content")
def _split_chunk_content(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """
    精细切割chunk内容
    :param chunk: 待切割的chunk
    :return: 切分后的chunk列表
    """
    content = chunk.get("content")
    deal_content = content[len(chunk.get("title", "")) + 1:]
    prefix = chunk.get("title") + "\n"
    spliter_text = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    ).split_text(deal_content)
    return [
        {
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "title": f"{chunk.get('title')}_{index}",
            "part": index,
            "content": prefix + text
        }
        for index, text in enumerate(spliter_text, start=1)
    ]


@step_log("merge_chunk_content")
def _merge_chunk_content(refine_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    合并同一个parent_title下的小于CHUNK_MIN的短块；合并后总长不超过CHUNK_MAX_SIZE。
    :param refine_chunks:
    :return:
    """
    merges_refine_chunks: list[dict[str, Any]] = []
    base_chunk: dict[str, Any] = None

    def flush_base() -> None:
        """把当前 base_chunk 落盘一次；只在“决定切换”或“循环结束”时调用，避免重复 push。"""
        nonlocal base_chunk
        if base_chunk is not None:
            merges_refine_chunks.append(base_chunk)
            base_chunk = None

    for next_chunk in refine_chunks:
        if base_chunk is None:
            base_chunk = next_chunk
            continue

        base_content: str = base_chunk.get("content", "")
        parent_title: str = base_chunk.get("parent_title", "")
        next_parent_title: str = next_chunk.get("parent_title", "")
        # next 去掉自身首行标题，得到纯正文，用来精确计算合并后的长度
        next_cleard_content: str = next_chunk.get("content", "")[len(next_parent_title or "") + 1:]

        can_merge = (
                len(base_content) <= CHUNK_MIN
                and bool(parent_title)
                and parent_title == next_parent_title
                and (len(base_content) + len(next_cleard_content)) <= CHUNK_MAX_SIZE
        )

        if can_merge:
            base_chunk["content"] = base_content + '\n' + next_cleard_content
        else:
            flush_base()
            base_chunk = next_chunk

    # 循环外兜底：最后一个 base_chunk 需要落盘一次，避免“最后一组”丢失
    flush_base()
    return merges_refine_chunks


@step_log("refine_split_and_merge_chunks")
def refine_split_and_merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    精细切割和合并chunk
    :param chunks:
    :return:
    """
    refine_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if len(chunk.get("content", "")) > 600:
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)
    logger.info(f"chunks经过精细切割和合并，共{len(refine_chunks)}个chunk")
    # 短合并：将短chunk与前一个chunk合并
    refine_chunks = _merge_chunk_content(refine_chunks)
    logger.info(f"完成小于400的chunk的合并，合并后的chunk数量为{len(refine_chunks)}")
    return refine_chunks


@step_log("padding_chunks_metadata")
def padding_chunks_metadata(chunks: list[dict[str, Any]]):
    """
    补充chunks元数据
    :param chunks:
    :return:
    """
    for chunk in chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title", "")
        if "part" not in chunk:
            chunk["part"] = 1
    logger.info(f"已经完成chunks元数据的补充，所有属性已完善")


@step_log("backup_chunks_json")
def backup_chunks_json(chunks: list[dict[str, Any]], md_path: str):
    """
    备份chunks内容
    :param chunks:
    :param md_path:
    :return:
    """
    json_path_obj: Path = Path(md_path).parent / f"{Path(md_path).stem}.json"
    json_path_obj.write_text(data=json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(f"完成chunks内容的备份，备份文件路径为：{json_path_obj}")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 获取并检验参数
    2. 按照语义规则进行md_content切割
    3. 构造 chunks 列表并回写
    注：超长文本二次细切尚未实现，需要在 split_document_by_title 之后对超长chunk进一步切割
    """
    # 1.获取并检验参数
    md_content, file_title, md_path = validate_get_data(state)
    # 2.按照语义规则进行md_content切割
    chunks: list[dict[str, Any]] = split_document_by_title(md_content, file_title)
    # 3.回写chunks，供后续向量化(node_bge_embedding)与入库(node_import_milvus)节点消费
    chunks: list[dict[str, Any]] = refine_split_and_merge_chunks(chunks)
    # 4.补充parent_title和part属性
    padding_chunks_metadata(chunks)
    # 5.备份chunks的内容
    backup_chunks_json(chunks, md_path)
    # 6.更新state
    state["chunks"] = chunks
    print("chunks----", chunks)
    return state
