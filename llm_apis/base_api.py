from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLM(ABC):
    """
    所有大语言模型 API 的抽象基类。
    它为调用不同 LLM 提供商定义了一个统一的接口。
    """

    def __init__(self, model_name: str, api_key: str, **kwargs):
        """
        初始化 LLM API 包装器。

        Args:
            model_name (str): 要使用的模型名称。
            api_key (str): 用于身份验证的 API 密钥。
            **kwargs: 其他特定于模型的参数 (例如, temperature, top_p)。
        """
        self.model_name = model_name
        self.api_key = api_key
        self.model_kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        为给定提示生成单个文本补全。

        Args:
            prompt (str): 模型的输入提示。
            **kwargs: 提供商特定的覆盖参数。

        Returns:
            str: 模型生成的文本。
        """
        pass

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        为一批提示生成文本补全。
        如果提供商支持高效批处理，子类应实现此方法。
        否则，它可以默认迭代调用 `generate`。
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        允许像函数一样调用对象。
        """
        return self.generate(prompt, **kwargs)