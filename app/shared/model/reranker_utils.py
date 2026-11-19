"""
工具模块，负责提供 reranker 相关的辅助能力。
"""
from FlagEmbedding import FlagReranker

from app.shared.config.reranker_config import reranker_config
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import resolve_model_path

_reranker_model: FlagReranker | None = None


def get_reranker_model() -> FlagReranker:
    """
    获取重排模型单例对象。

    Returns:
        FlagReranker: 初始化完成的重排模型实例。
    """
    global _reranker_model
    if _reranker_model is None:
        # 本地路径需要归一化到含 config.json 的快照目录，否则会报与真实原因无关的加载错误
        model_name_or_path = resolve_model_path(
            reranker_config.bge_reranker_large, "BGE_RERANKER_LARGE"
        )
        logger.info("开始初始化重排模型", extra={"model_name_or_path": model_name_or_path})
        _reranker_model = FlagReranker(
            model_name_or_path=model_name_or_path,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16,
        )
        logger.success("重排模型初始化成功")
    return _reranker_model
