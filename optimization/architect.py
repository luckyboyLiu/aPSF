import re
from ..llm_apis import get_llm, BaseLLM
from .prompt_object import PromptStructure

class Architect:
    """
    使用一个强大的 LLM (Architect) 来自动发现给定任务的最佳提示结构。
    """
    def __init__(self, architect_llm_id: str = "architect"):
        """
        初始化 Architect。

        Args:
            architect_llm_id (str): 在 config.py 中定义的 Architect LLM 的 ID。
        """
        self.architect_llm: BaseLLM = get_llm(architect_llm_id)

    def _create_discovery_prompt(self, task_description: str, example: str) -> str:
        """为结构发现创建一个元提示 (meta-prompt)。"""
        return f"""
You are an expert in prompt engineering. Your task is to decompose a complex task into a series of clear, modular, and effective prompt components (which we call 'factors').

Task Description: "{task_description}"

Here is an example instance of the task:
--- EXAMPLE ---
{example}
--- END EXAMPLE ---

Based on the task, identify the essential factors that a language model would need to generate a high-quality response. For each factor, provide a descriptive name and a brief, initial placeholder text.

The output should be in a structured format, like this:

[FACTOR_NAME_1]
[Initial placeholder content for factor 1.]

[FACTOR_NAME_2]
[Initial placeholder content for factor 2.]

Common factors include, but are not limited to: 'Instruction', 'Rationale', 'Examples', 'Output_Format', 'Constraints', 'Persona'. Choose the most relevant ones for this specific task.
"""

    def discover_structure(self, task_description: str, example_data: str) -> PromptStructure:
        """
        为给定任务发现一个新的提示结构。

        Args:
            task_description (str): 任务的描述。
            example_data (str): 一个或多个任务样本，用于帮助 LLM 理解任务。

        Returns:
            PromptStructure: 一个包含已发现因子和初始内容的新 PromptStructure 对象。
        """
        meta_prompt = self._create_discovery_prompt(task_description, example_data)
        response = self.architect_llm.generate(meta_prompt)
        
        factors = self._parse_structure_response(response)
        
        if not factors:
            print("Warning: Architect did not return a valid structure. Using a default fallback.")
            factors = {
                "Instruction": f"Solve the following task: {task_description}",
                "Input": "{input}",
                "Rationale": "Think step by step to reach the solution.",
            }
            
        return PromptStructure(task_description=task_description, factors=factors)

    def _parse_structure_response(self, response: str) -> dict:
        """从 LLM 的响应中解析出因子。"""
        factors = {}
        # 使用正则表达式匹配 [FACTOR_NAME] 和其后的内容
        pattern = re.compile(r"\[([^\]]+)\]\n(.*?)(?=\n\[[^\]]+\]|\Z)", re.DOTALL)
        matches = pattern.findall(response)
        
        for name, content in matches:
            factors[name.strip()] = content.strip()
            
        return factors 