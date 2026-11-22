"""
providers.py

一句话说明这个模块是干什么的。
"""
from app.shared.model import get_llm_client, generate_embeddings, get_reranker_model


class LLMProvider:
    def vision_model(self, vision_model_name: str):
        return get_llm_client(model=vision_model_name)

    def llm_model(self, llm_model_name: str | None = None, json_mode: bool = False):
        return get_llm_client(model=llm_model_name, json_mode=json_mode)

    def generate_embeddings(self, texts: list[str]) -> dict[str, list]:
        return generate_embeddings(texts)

    def reranker_model(self):
        return get_reranker_model()

    def compute_score(self, question_answer_pair: list[list[str]]):
        """
            使用函数进行打分
        :param question_answer_pair:
        :return:
        """
        reranker_model = get_reranker_model()
        return reranker_model.compute_score(question_answer_pair, normalize=True)

    def compute_token_number(self, content: str):
        """
            计算token数量
        :param content:
        :return:
        """
        reranker_model = get_reranker_model()
        tokenizer = reranker_model.tokenizer
        return len(tokenizer.encode(content, add_special_tokens=False))


llm_provider = LLMProvider()
