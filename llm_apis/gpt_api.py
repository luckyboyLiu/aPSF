import openai
from typing import List
from .base_api import BaseLLM

class GPT_API(BaseLLM):
    """
    OpenAI GPT 模型的包装器 (例如, gpt-4, gpt-3.5-turbo)。
    也兼容任何与 OpenAI API 兼容的端点 (例如, vLLM)。
    """

    def __init__(self, model_name: str, api_key: str, api_base: str = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        # 为该实例创建一个独立的客户端，以支持自定义 api_base
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=api_base
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用 ChatCompletions 端点生成单个文本补全。
        """
        request_kwargs = {**self.model_kwargs, **kwargs}

        try:
            # 使用实例客户端进行 API 调用
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI 兼容 API 发生错误: {e}")
            return f"Error: {e}"