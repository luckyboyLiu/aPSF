from .gpt_api import GPT_API
from .google_api import Google_API
from .llama_api import Llama_API
from .base_api import BaseLLM
from ..config import MODELS, API_KEYS

def get_llm(model_id: str) -> BaseLLM:
    """
    获取 LLM API 包装器实例的工厂函数。

    Args:
        model_id (str): 配置文件中的模型标识符 (例如, "architect", "worker")。

    Returns:
        BaseLLM: 相应 LLM 包装器的实例。
    """
    if model_id not in MODELS:
        raise ValueError(f"模型 ID '{model_id}' 在 config.py 中未找到。")

    model_config = MODELS[model_id]
    provider = model_config.get("provider")
    model_name = model_config.get("model_name")
    
    model_kwargs = {k: v for k, v in model_config.items() if k not in ["provider", "model_name"]}

    if provider == "openai":
        api_key = API_KEYS.get("openai")
        if not api_key or "YOUR" in api_key:
            raise ValueError("OpenAI API 密钥未在 config.py 中设置。")
        return GPT_API(model_name=model_name, api_key=api_key, **model_kwargs)
    
    elif provider == "google":
        api_key = API_KEYS.get("google")
        if not api_key or "YOUR" in api_key:
            raise ValueError("Google API 密钥未在 config.py 中设置。")
        return Google_API(model_name=model_name, api_key=api_key, **model_kwargs)

    elif provider == "llama_local":
        return Llama_API(model_name=model_name, api_key=None, **model_kwargs)

    else:
        raise ValueError(f"提供商 '{provider}' (模型 '{model_id}') 不被支持。")