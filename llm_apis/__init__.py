from .gpt_api import GPT_API
from .google_api import Google_API
from .llama_api import Llama_API
from .base_api import BaseLLM
from ..config import MODELS, API_KEYS, API_BASE_URLS

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
    
    model_kwargs = {
        k: v for k, v in model_config.items() 
        if k not in ["provider", "model_name", "api_base_id"]
    }

    if provider == "openai":
        api_key = API_KEYS.get("openai")
        if not api_key or "YOUR" in api_key:
            print("警告: OpenAI API 密钥未设置。对于 vLLM 等本地服务，这可能是正常的。")
            api_key = "N/A"
        
        api_base_id = model_config.get("api_base_id")
        api_base_url = API_BASE_URLS.get(api_base_id) if api_base_id else None
        
        return GPT_API(
            model_name=model_name, 
            api_key=api_key, 
            api_base=api_base_url, 
            **model_kwargs
        )
    
    elif provider == "google":
        api_key = API_KEYS.get("google")
        if not api_key or "YOUR" in api_key:
            raise ValueError("Google API 密钥未在 config.py 中设置。")
        return Google_API(model_name=model_name, api_key=api_key, **model_kwargs)

    elif provider == "llama_local":
        return Llama_API(model_name=model_name, api_key=None, **model_kwargs)

    else:
        raise ValueError(f"提供商 '{provider}' (模型 '{model_id}') 不被支持。")