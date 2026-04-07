from .gpt_api import GPT_API
from .google_api import Google_API
from .llama_api import Llama_API
from .base_api import BaseLLM
from ..config import MODELS, API_KEYS, API_BASE_URLS

def get_llm(model_id: str) -> BaseLLM:
    """
    Factory function to get an LLM API wrapper instance.

    Args:
        model_id (str): Model identifier from configuration file (e.g., "architect", "worker").

    Returns:
        BaseLLM: Instance of the corresponding LLM wrapper.
    """
    if model_id not in MODELS:
        raise ValueError(f"Model ID '{model_id}' not found in config.py.")

    model_config = MODELS[model_id]
    provider = model_config.get("provider")
    model_name = model_config.get("model_name")
    
    model_kwargs = {
        k: v for k, v in model_config.items()
        if k not in ["provider", "model_name", "api_base_id", "api_key"] # Add "api_key" here
    }

    if provider == "openai":
        # Prefer api_key identifier from model config, then look up from API_KEYS dict
        api_key_id = model_config.get("api_key")
        api_key = API_KEYS.get(api_key_id) if api_key_id else API_KEYS.get("openai")

        if not api_key or "YOUR" in api_key:
            print("Warning: OpenAI API key not set. This may be normal for local services like vLLM.")
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
            raise ValueError("Google API key not set in config.py.")
        return Google_API(model_name=model_name, api_key=api_key, **model_kwargs)

    elif provider == "llama_local":
        return Llama_API(model_name=model_name, api_key=None, **model_kwargs)

    else:
        raise ValueError(f"Provider '{provider}' (model '{model_id}') is not supported.")