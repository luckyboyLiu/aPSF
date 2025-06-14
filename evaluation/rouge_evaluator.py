import evaluate
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class RougeEvaluator(BaseEvaluator):
    """
    使用 `evaluate` 库计算 ROUGE 分数的评测器，用于摘要任务。
    """
    def __init__(self):
        self.rouge = evaluate.load('rouge')

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估摘要任务的 ROUGE 分数。

        Args:
            predictions (List[str]): 模型生成的摘要列表。
            references (List[Dict[str, Any]]): 包含 'summary' 字段的参考摘要。
        """
        gold_summaries = [ref["summary"] for ref in references]
        
        results = self.rouge.compute(
            predictions=predictions,
            references=gold_summaries,
            use_stemmer=True
        )
        
        # 我们主要关心 ROUGE-L
        return {"rouge-L": results["rougeL"]} 