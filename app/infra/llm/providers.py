"""
providers.py

一句话说明这个模块是干什么的。
"""
from app.shared.model import get_llm_client, generate_embeddings


class LLMProvider:
    def vision_model(self, vision_model_name: str):
        return get_llm_client(model=vision_model_name)

    def llm_model(self, llm_model_name: str | None = None, json_mode: bool = False):
        return get_llm_client(model=llm_model_name, json_mode=json_mode)

    def generate_embeddings(self, texts: list[str]) -> dict[str, list]:
        return generate_embeddings(texts)


llm_provider = LLMProvider()
