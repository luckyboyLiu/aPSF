import evaluate
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class PassAtKEvaluator(BaseEvaluator):
    """
    使用 `evaluate` 库中的 `code_eval` 来计算 Pass@k 的评测器。
    这对于 HumanEval 任务是必需的。
    """
    def __init__(self, k: int = 1):
        self.k = k
        self.code_eval = evaluate.load("code_eval")

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估代码生成任务的 Pass@k。

        Args:
            predictions (List[str]): 模型生成的代码补全列表。
            references (List[Dict[str, Any]]): 包含 'test' (单元测试) 和 'entry_point' 的样本列表。
        """
        # predictions 的每个元素是一个代码字符串
        # humaneval 数据集中的每个样本包含 'test' 和 'entry_point'
        # 我们需要将它们格式化为 code_eval 所需的格式
        
        # 假设 predictions 已经是完整的代码
        test_cases = [ref["test"] for ref in references]
        
        # code_eval 需要一个二维列表作为 candidates
        # [[candidate_1_for_problem_1, ...], [candidate_1_for_problem_2, ...]]
        candidates = [[pred] for pred in predictions]

        pass_at_k, results = self.code_eval.compute(
            references=test_cases,
            predictions=candidates,
            k=[self.k]
        )
        
        return {f"pass@{self.k}": pass_at_k[f"pass@{self.k}"]} 