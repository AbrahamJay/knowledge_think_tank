"""
Milvus 门面模块，统一封装向量库客户端与检索相关操作。
"""
from typing import Any

from app.shared.clients.milvus_utils import (
    create_hybrid_search_requests,
    ensure_collection_loaded,
    get_milvus_client,
    hybrid_search,
)
from app.infra.config.providers import infra_config


class MilvusGateway:
    @property
    def chunks_collection(self) -> str:
        """
        获取文档切块集合名称。

        Returns:
            str: Milvus 中存放知识切块的集合名。
        """
        return infra_config.milvus_config.chunks_collection

    @property
    def item_name_collection(self) -> str:
        """
        获取主体名称集合名称。

        Returns:
            str: Milvus 中存放主体名称向量的集合名。
        """
        return infra_config.milvus_config.item_name_collection

    @property
    def client(self):
        """
        获取 Milvus 客户端实例。

        Returns:
            Any: 底层 Milvus 客户端对象。

        Raises:
            RuntimeError: 客户端创建失败（返回 None）时给出可读错误，
                避免后续退化成 'NoneType' object has no attribute ... 之类的迷惑报错。
        """
        client = get_milvus_client()
        if client is None:
            raise RuntimeError("Milvus 客户端不可用，请检查 MILVUS_URL 配置以及 Milvus 服务是否启动")
        return client

    def ensure_collection_loaded(self, collection_name: str) -> bool:
        """
        确保目标集合已加载进内存。

        Args:
            collection_name: 目标集合名称。

        Returns:
            bool: True 表示本次触发了加载，False 表示集合已经处于加载状态。
        """
        return ensure_collection_loaded(client=self.client, collection_name=collection_name)

    def create_requests(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            *,
            expr: str = None,
            limit: int = 5,
    ):
        """
        创建 Milvus 混合检索请求对象。

        Args:
            dense_vector: 稠密向量表示。
            sparse_vector: 稀疏向量表示。
            expr: 可选过滤表达式，用于限定检索范围。
            limit: 单路检索返回条数上限。

        Returns:
            Any: 底层 Milvus 混合检索请求列表。
        """
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )

    def hybrid_search(
            self,
            *,
            collection_name: str,
            reqs: list[Any],
            ranker_weights: tuple[float, float] = (0.5, 0.5),
            norm_score: bool = False,
            limit: int = 5,
            output_fields: list[str] | None = None,
            search_params: dict | None = None,
    ):
        """
        执行 Milvus 混合检索。

        Args:
            collection_name: 目标集合名称。
            reqs: 检索请求列表。
            ranker_weights: 稠密与稀疏路召回结果的融合权重。
            norm_score: 是否对分数做归一化。
            limit: 最终返回条数上限。
            output_fields: 需要返回的字段列表。
            search_params: 额外检索参数。

        Returns:
            Any: Milvus 返回的原始检索结果。
        """
        return hybrid_search(
            client=self.client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )


milvus_gateway = MilvusGateway()
