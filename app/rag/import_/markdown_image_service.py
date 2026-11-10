"""
markdown_image_service.py

一句话说明这个模块是干什么的。
"""
import base64
import re
from mimetypes import guess_type
from pathlib import Path
from typing import Tuple

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.infra.config.providers import infra_config
from app.infra.llm.providers import llm_provider
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS, IMAGE_CONTEXT_SUB_CHARS
from app.shared.runtime.logger import logger
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


def validate_and_data(state: ImportGraphState) -> Tuple[Path, Path, str]:
    md_path: str = state.get("md_path")
    if not md_path:
        logger.error(f"md_path变量为空，业务无法继续执行")
        raise ValueError("md_path变量为空，业务无法继续执行")
    md_path_obj = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path变量指向的{md_path}文件不存在，业务无法继续执行")
        raise ValueError(f"md_path变量指向的{md_path}文件不存在，业务无法继续执行")

    md_content = md_path_obj.read_text(encoding="utf-8")
    state["md_content"] = md_content
    md_images_dir_obj: Path = md_path_obj.parent / "images"
    return md_path_obj, md_images_dir_obj, md_content


def scan_image(md_content: str, md_images_dir_obj: Path) -> list[tuple[str, str, tuple[str, str]]]:
    """
    扫描 Markdown 中的图片
    :param md_content: Markdown 内容
    :param md_images_dir_obj: Markdown 图片目录
    :return: 图片信息列表
    """
    image_info_list: list[tuple[str, str, tuple[str, str]]] = []
    for image_file_obj in md_images_dir_obj.iterdir():
        image_name: str = image_file_obj.name
        image_path: str = str(image_file_obj)
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f"{image_path}不是支持的图片格式，已跳过")
            continue
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match = reg.search(md_content)
        if not search_match:
            logger.warning(f"{image_path}在Markdown内容中未找到，已跳过")
            continue
        start = search_match.start()
        end = search_match.end()
        pre_context = md_content[max(0, start - IMAGE_CONTEXT_SUB_CHARS): start]
        post_context = md_content[end: min(end + IMAGE_CONTEXT_SUB_CHARS, len(md_content))]
        logger.debug(f"图片信息：{image_name}, {image_path}, ({pre_context}, {post_context})")
        image_info_list.append((image_name, image_path, (pre_context, post_context)))
    logger.info(f"所有图片信息已获取，共 {len(image_info_list)} 张图片")
    return image_info_list


def summarize_images(image_info_list: list[tuple[str, str, tuple[str, str]]], root_folder: str) -> dict[str, str]:
    """
    调用视觉模型生成图片说明
    :param image_info_list: 图片信息列表
    :return: 图片说明字典
    """
    summary_image_dict: dict[str, str] = {}
    vision_model = llm_provider.vision_model(vision_model_name=infra_config.llm_config.vl_model)
    for image_name, image_path, image_content in image_info_list:
        image_prompt_text = load_prompt("image_summary", root_folder=root_folder, image_content=image_content)
        image_path = Path(image_path)
        image_base64_str: str = base64.b64encode(image_path.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_path)[0]};base64,{image_base64_str}"}
                },
                {
                    "type": "text",
                    "text": image_prompt_text
                }
            ]
        )
        chains = vision_model | StrOutputParser()
        # 添加范围内限制
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        logger.debug(f"图片说明：{image_name}, {image_summary}")
        summary_image_dict[image_name] = image_summary
    return summary_image_dict


def upload_images_get_url(image_info_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> dict[str, str]:
    """
    上传图片到 MinIO 并获取 URL
    :param image_info_list: 图片信息列表
    :param stem: 文件名称
    :return: 图片 URL 字典
    """
    # 1. 删除
    minio_client = minio_gateway.minio_client
    list_object = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        # /upload-images + '/' 前面不能 有/
        prefix=minio_gateway.image_dir[1:] + "/" + stem,
        recursive=True
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in list_object]
    errors = minio_client.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.error(f"删除图片时出错：{error}")

    # 2. 上传
    image_url_dict: dict[str, str] = {}
    for image_name, image_path, _ in image_info_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=minio_gateway.image_dir + "/" + stem + "/" + image_name,
                file_path=image_path,
                content_type=guess_type(image_path)[0]
            )
            url = minio_gateway.build_image_url(stem, image_name)
            image_url_dict[image_name] = url
            logger.debug(f"图片上传成功：{image_name}, {url}")
        except Exception as e:
            logger.error(f"图片上传失败：{image_name}, {e}")
    return image_url_dict


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1. 获取和校验参数
    md_path_obj, md_images_dir_obj, md_content = validate_and_data(state)
    # 2. 检验是否有图片
    if not md_images_dir_obj.is_dir() or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"{md_images_dir_obj}目录不存在或无图片，业务无法继续执行")
        return state
    # 3. 获取图片信息和上下文
    image_info_list: list[tuple[str, str, tuple[str, str]]] = scan_image(md_content, md_images_dir_obj)
    # 4. 交给视觉模型识别图片的含义
    summary_image_dict: dict[str, str] = summarize_images(image_info_list, md_path_obj.stem)
    # 5. 将图片信息传到minio的服务器，获取图片url
    image_url_dict: dict[str, str] = upload_images_get_url(image_info_list, md_path_obj.stem)
    print(f"图片上传完成，共 {len(image_url_dict)} 张图片")
    return state
