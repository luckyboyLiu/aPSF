# 这是 Google PaLM 2 API (text-bison@001) 的实现。
# 需要安装 `google-generativeai` 库。
from typing import List
from .base_api import BaseLLM
import google.generativeai as genai

class Google_API(BaseLLM):
    """
    Google Generative AI 模型的包装器 (例如, text-bison@001)。
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        genai.configure(api_key=self.api_key)
        
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=self.model_kwargs.get('temperature', 0.0),
            top_p=self.model_kwargs.get('top_p'),
        )
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        使用 Google Generative AI SDK 生成单个文本补全。
        """
        try:
            response = self.model.generate_content(prompt)
            if response.parts:
                return response.text
            else:
                reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
                return f"Error: 模型未返回内容。原因: {reason}"
        except Exception as e:
            print(f"Google API 发生错误: {e}")
            return f"Error: {e}"