import openai
from typing import List
from .base_api import BaseLLM

class GPT_API(BaseLLM):
    """
    Wrapper for OpenAI GPT models (e.g., gpt-4, gpt-3.5-turbo).
    Also compatible with any OpenAI API-compatible endpoint (e.g., vLLM).
    """

    def __init__(self, model_name: str, api_key: str, api_base: str = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        # Create a separate client for this instance to support custom api_base
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=api_base
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a single text completion using the ChatCompletions endpoint.
        For Qwen3 architect model, automatically extract formal output content after the think process.
        """
        request_kwargs = {**self.model_kwargs, **kwargs}

        try:
            # Use instance client for API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **request_kwargs
            )
            raw_content = response.choices[0].message.content.strip()

            # Record token statistics
            self.api_calls += 1
            if hasattr(response, 'usage') and response.usage:
                self.prompt_tokens += response.usage.prompt_tokens or 0
                self.completion_tokens += response.usage.completion_tokens or 0
                self.total_tokens += response.usage.total_tokens or 0

            # Check if this is a thinking model (Qwen3, gpt-oss-120b, etc.), if so extract content after think
            if self._is_thinking_model():
                return self._extract_content_after_think(raw_content)

            return raw_content
        except Exception as e:
            print(f"OpenAI compatible API error: {e}")
            return f"Error: {e}"
    
    def _is_qwen3_architect_model(self) -> bool:
        """Check if current model is Qwen3 architect model"""
        # Check if model name contains Qwen3-related identifiers
        qwen3_indicators = ["qwen3", "Qwen3", "QWEN3"]
        return any(indicator in self.model_name.lower() for indicator in qwen3_indicators)

    def _is_thinking_model(self) -> bool:
        """Check if current model outputs thinking process (Qwen3, gpt-oss-120b, etc.)"""
        thinking_indicators = ["qwen3", "gpt-oss", "oss-120b", "deepseek-r1", "o1", "o3"]
        model_lower = self.model_name.lower()
        return any(indicator in model_lower for indicator in thinking_indicators)
    
    def _extract_content_after_think(self, content: str) -> str:
        """
        Extract formal content after the think process from Qwen3 architect model output.

        Supported think tag formats:
        1. <think>...</think>
        2. <thinking>...</thinking>
        3. Think: ... (paragraphs starting with Think:)
        4. Thought: ... (alternative thinking markers)
        """
        import re
        
        # Method 1: Extract content after <think>...</think> tags (supports full-width and half-width brackets)
        think_patterns = [
            r'[<＜]think[>＞].*?[<＜]/think[>＞]\s*(.*?)$',  # <think>...</think> or ＜think＞...＜/think＞
            r'[<＜]thinking[>＞].*?[<＜]/thinking[>＞]\s*(.*?)$',  # <thinking>...</thinking>
            r'[<＜]thought[>＞].*?[<＜]/thought[>＞]\s*(.*?)$',  # <thought>...</thought>
        ]
        
        for pattern in think_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:  # Ensure extracted content is not empty
                    return extracted

        # Method 2: Find content after "Think:" or "Thinking:"
        think_start_patterns = [
            r'(?:Think)[:]\s*(.*?)$',
            r'(?:Thinking)[:]\s*(.*?)$',
        ]
        
        for pattern in think_start_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        
        # Method 3: Find content after think process ends (supports full-width and half-width brackets)
        # Try to find the end position of think process, then extract content after it
        think_end_markers = [
            r'[<＜]/think[>＞]\s*(.*?)$',
            r'[<＜]/thinking[>＞]\s*(.*?)$', 
            r'[<＜]/thought[>＞]\s*(.*?)$',
            r'(?:end\s+think|think\s+end)\s*(.*?)$',
        ]
        
        for pattern in think_end_markers:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        
        # Method 4: If no clear think markers found, try intelligent splitting
        # Find possible think process end position
        lines = content.split('\n')
        think_end_line = -1

        # Find possible think process end line
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Find indicators of think process end
            if any(end_word in line_lower for end_word in [
                'end think', 'think end',
                'now i will', 'let me', 'based on', 'therefore',
            ]):
                think_end_line = i
                break
        
        # If think end line found, extract content after it
        if think_end_line >= 0 and think_end_line < len(lines) - 1:
            remaining_lines = lines[think_end_line + 1:]
            remaining_content = '\n'.join(remaining_lines).strip()
            if remaining_content:
                return remaining_content
        
        # If all methods fail, return original content
        print(f" Could not extract content after think from Qwen3 output, returning original content")
        return content