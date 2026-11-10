"""
minio_gateway.py

一句话说明这个模块是干什么的。
"""
from app.infra.config.providers import infra_config
from app.shared.clients import get_minio_client


class MinIOGateWay:
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    @property
    def image_dir(self):
        return infra_config.minio_config.minio_img_dir

    @property
    def minio_client(self):
        return get_minio_client()

    def build_image_url(self, file_name: str, object_name: str):
        prefix = "https://" if infra_config.minio_config.minio_secure else "http://"
        return f"{prefix}{infra_config.minio_config.endpoint}/{self.bucket_name}{self.image_dir}/{file_name}/{object_name}"


minio_gateway = MinIOGateWay()
