"""
pdf_parse_service.py

一句话说明这个模块是干什么的。
"""
import shutil
import time
from pathlib import Path
from typing import Tuple

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_POLL_TIMEOUT_SECONDS, MINERU_DOWNLOAD_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log

@step_log("validate_path_values")
def validate_path_values(state: ImportGraphState) -> Tuple[Path, Path]:
    local_dir: str = state.get("local_dir")
    pdf_path: str = state.get("pdf_path")
    if not pdf_path:
        logger.error(f"pdf_path地址为空，业务无法继续，请检查配置")
        raise ValueError("pdf_path地址为空，业务无法继续，请检查配置")

    if not local_dir:
        local_dir: Path = PROJECT_ROOT / "output"
        logger.warning(f"local_dir地址为空，不影响业务正常进行，给与默认值，默认值：{str(local_dir)}")

    pdf_path_obj: Path = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)
    if not pdf_path_obj.is_file():
        logger.error(f"pdf_path地址为{str(pdf_path)}，不存在或不是文件，业务无法继续")
        raise ValueError(f"pdf_path地址为{str(pdf_path)}，不存在或不是文件，业务无法继续")

    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir地址为{str(local_dir)}，不存在或不是文件夹，创建对应文件夹，业务继续")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    # 1. 向MinerU申请，获取上传文件地址
    token = infra_config.mineru_config.api_key
    url = f"{infra_config.mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}", "data_id": f"{pdf_path_obj.stem}"}
        ],
        "model_version": MINERU_MODEL_VERSION,
        # "extra_formats": ["markdown", "html"]
        # "callback": "https://example.com/callback" # 回调地址
    }
    response = requests.post(url=url, headers=header, json=data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"向{url}申请文件上传地址，请求失败，状态码：{status_code}")
        raise RuntimeError(f"向{url}申请文件上传地址，请求失败，状态码：{status_code}")
    response_dict = response.json()
    code = response_dict.get("code")
    msg = response_dict.get("msg")

    if code != 0:
        logger.error(f"向{url}申请文件上传地址，业务失败，错误码：{code}，错误信息：{msg}")
        raise RuntimeError(f"向{url}申请文件上传地址，业务失败，错误码：{code}，错误信息：{msg}")
    upload_file_urls: list[str] = response_dict.get("data", {}).get("file_urls", [])
    batch_id = response_dict.get("data", {}).get("batch_id")  # 用户轮询获取本次上传解析结果

    if not upload_file_urls:
        logger.error(f"向{url}申请文件上传地址，业务失败，未获取到上传文件地址")
        raise RuntimeError(f"向{url}申请文件上传地址，业务失败，未获取到上传文件地址")

    if not batch_id:
        logger.error(f"向{url}申请文件上传地址，业务失败，未获取到batch_id")
        raise RuntimeError(f"向{url}申请文件上传地址，业务失败，未获取到batch_id")
    upload_file_url: str = upload_file_urls[0]
    logger.info(f"向MinerU申请文件上传地址成功，batch_id: {batch_id}上传文件地址：{upload_file_url}")

    # 2. 向MinerU传递文件的地址，上传文件
    file_data = pdf_path_obj.read_bytes()
    with requests.Session() as session:
        # 不信任系统环境，禁用证书验证
        session.verify = False
        upload_response = session.put(upload_file_url, data=file_data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
        upload_status_code = upload_response.status_code
        if upload_status_code != 200:
            logger.error(f"上传文件到{upload_file_url}失败，状态码：{upload_status_code}")
            raise RuntimeError(f"上传文件到{upload_file_url}失败，状态码：{upload_status_code}")
        max_wait_time = MINERU_POLL_TIMEOUT_SECONDS
        poll_time = MINERU_POLL_INTERVAL_SECONDS
        start_time = time.time()
        logger.info(f"向指定：{upload_file_url}地址上传文件：{pdf_path_obj.name}上传成功，准备轮询获取文件解析结果")

        # 3. 轮询获取文件解析结果
        while True:
            if time.time() - start_time > max_wait_time:
                logger.error(f"向{batch_id}轮询获取文件解析结果，超时，超时时间：{max_wait_time}")
                raise TimeoutError(f"向{batch_id}轮询获取文件解析结果，超时，超时时间：{max_wait_time}")
            poll_url = f"{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}"
            try:
                poll_response = requests.get(url=poll_url, headers=header)
            except Exception as e:
                logger.error(f"向{batch_id}轮询获取文件解析结果，请求失败，错误信息：{e}")
                time.sleep(poll_time)
                continue
            poll_response_status_code = poll_response.status_code
            if poll_response_status_code != 200:
                if 500 <= poll_response_status_code < 600:
                    logger.error(f"向{batch_id}轮询获取文件解析结果，请求失败，状态码：{poll_response_status_code}")
                    time.sleep(poll_time)
                    continue
                else:
                    logger.error(f"向{batch_id}轮询获取文件解析结果，状态码：{poll_response_status_code}，错误无法修复")
                    raise RuntimeError(
                        f"向{batch_id}轮询获取文件解析结果，状态码：{poll_response_status_code}，错误无法修复")
            poll_response_json = poll_response.json()
            poll_code = poll_response_json.get("code")
            extract_result = poll_response_json.get("data", {}).get("extract_result", [])[0]
            # 业务失败，返回错误信息，结束轮询
            if poll_code != 0:
                logger.error(
                    f"向{batch_id}轮询获取文件解析结果，业务失败，错误码：{poll_code}，错误信息：{poll_response_json.get('msg')}")
                raise RuntimeError(
                    f"向{batch_id}轮询获取文件解析结果，业务失败，错误码：{poll_code}，错误信息：{poll_response_json.get('msg')}")
            poll_state = extract_result.get("state")
            # 查询结果完成，返回下载地址，结束轮询
            if poll_state == "done":
                full_zip_url = extract_result.get("full_zip_url")
                logger.info(f"基于:{pdf_path_obj.name}解析已完成，下载地址：{full_zip_url}")
                return full_zip_url
            # 查询结果失败，返回错误信息，结束轮询
            elif poll_state == "failed":
                logger.error(f"向{batch_id}轮询获取文件解析结果，解析失败，完整响应：{extract_result}")
                error_msg = extract_result.get("err_msg",
                                               extract_result.get("error", extract_result.get("msg", "未知错误")))
                logger.error(
                    f"向{batch_id}轮询获取文件解析结果，解析失败，状态码：{poll_response_status_code}，错误码：{poll_code}，错误信息：{error_msg}")
                raise RuntimeError(
                    f"向{batch_id}轮询获取文件解析结果，解析失败，状态码：{poll_response_status_code}，错误码：{poll_code}，错误信息：{error_msg}")
            # 查询结果进行中，返回等待信息，继续轮询
            else:
                logger.warning(f"向{batch_id}轮询获取文件解析结果，没有完成解析，等待后，重试！！")
                time.sleep(poll_time)
                continue

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_download_url: str, local_dir_obj: Path, stem: str) -> Path:
    """
    下载并解压 Markdown 文件
    
    Args:
        zip_download_url: zip 文件下载地址
        local_dir_obj: 本地目录对象
        stem: 文件名（不含扩展名）
    
    Returns:
        Path: Markdown 文件路径
    """
    # 1. 下载 zip 文件
    response = requests.get(url=zip_download_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    if response.status_code != 200:
        logger.error(f"下载文件失败，状态码：{response.status_code}")
        raise RuntimeError(f"下载文件失败，状态码：{response.status_code}")

    # 2. 保存 zip 文件（如果已存在则先删除）
    zip_file_path: Path = local_dir_obj / f"{stem}.zip"
    if zip_file_path.exists():
        logger.warning(f"zip 文件已存在，将覆盖：{zip_file_path}")
        zip_file_path.unlink()
    zip_file_path.write_bytes(data=response.content)

    # 3. 解压到对应目录（如果目录已存在则先清理）
    extract_dir_path: Path = local_dir_obj / stem
    if extract_dir_path.exists():
        logger.warning(f"解压目录已存在，将清理：{extract_dir_path}")
        shutil.rmtree(extract_dir_path)
    extract_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_file_path, extract_dir_path)

    # 4. 查找 Markdown 文件
    md_file_list: list[Path] = list(extract_dir_path.rglob("*.md"))
    if not md_file_list:
        logger.error(f"解压目录下没有找到 Markdown 文件：{extract_dir_path}")
        raise RuntimeError(f"解压目录下没有找到 Markdown 文件：{extract_dir_path}")

    # 5. 优先查找同名 md 文件，其次查找 full.md
    full_md_file_obj: Path | None = None
    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            logger.info(f"找到同名 Markdown 文件：{md_file_obj}")
            return md_file_obj
        elif md_file_obj.stem == "full":
            logger.info(f"找到 full.md 文件，将重命名为 {stem}.md：{md_file_obj}")
            full_md_file_obj = md_file_obj
            break

    # 6. 重命名 full.md 为 stem.md
    if full_md_file_obj:
        target_md_path = full_md_file_obj.with_name(f"{stem}.md")
        full_md_file_obj.rename(target_md_path)
        return target_md_path

    # 7. 如果没找到，返回第一个 md 文件作为兜底
    logger.warning(f"未找到匹配的 Markdown 文件，使用第一个文件：{md_file_list[0]}")
    return md_file_list[0]

@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1. 获取文件路径和本地目录
    pdf_path_obj, local_dir_obj = validate_path_values(state)
    # 2. 向MinerU上传PDF并轮询获取解析结果
    zip_download_url: str = upload_pdf_and_poll(pdf_path_obj)
    print(f"zip_download_url: {zip_download_url}")
    # 3. 下载并解压解析结果
    md_path_obj: Path = download_and_extract_markdown(zip_download_url, local_dir_obj, pdf_path_obj.stem)
    # 4. 更新state md_path
    state["md_path"] = str(md_path_obj)
    return state
