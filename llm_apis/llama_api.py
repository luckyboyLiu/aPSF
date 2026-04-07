# This is a placeholder for local LLaMA models.
# Actual implementation depends on the serving framework used, e.g., Hugging Face Transformers, vLLM or Ollama.
# Below is an example using the Hugging Face Transformers library.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from .base_api import BaseLLM

class Llama_API(BaseLLM):
    """
    Wrapper for loading and running local models using Hugging Face Transformers.
    """
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        
        # Automatically detect available device (prefer GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local model on {self.device}: {self.model_name}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16, # Use float16 to save GPU memory
                device_map="auto" # Automatically shard model across available GPUs
            )
            print(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model path is correct and you have installed `torch` and `transformers`.")
            raise

    @torch.no_grad() # Disable gradient computation during inference to save resources
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a single text completion using a local Hugging Face model.
        """
        # Merge model default parameters with runtime parameters
        generation_kwargs = {**self.model_kwargs, **kwargs}

        # Core fix: Disable sampling (greedy decoding) when temperature is 0
        if generation_kwargs.get("temperature") == 0.0:
            generation_kwargs["do_sample"] = False

        # Encode input as tensors and move to appropriate device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text
        outputs = self.model.generate(**inputs, **generation_kwargs)

        # Decode generated tokens, skipping input portion tokens
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Usually, the result includes the original prompt; we need to remove it.
        # This is a simple implementation; for more complex prompt structures, adjustment may be needed.
        if result.startswith(prompt):
            return result[len(prompt):].strip()
        else:
            return result.strip()