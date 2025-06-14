import openai
from typing import List
from .base_api import BaseLLM

class GPT_API(BaseLLM):
    """
    OpenAI GPT 模型的包装器 (例如, gpt-4, gpt-3.5-turbo)。
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用 OpenAI ChatCompletions 端点生成单个文本补全。
        """
        request_kwargs = {**self.model_kwargs, **kwargs}

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API 发生错误: {e}")
            return f"Error: {e}"