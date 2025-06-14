from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEvaluator(ABC):
    """
    所有评测器的抽象基类。
    定义了一个用于评估模型预测的标准接口。
    """

    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估模型预测的性能。

        Args:
            predictions (List[str]): 模型生成的输出列表。
            references (List[Dict[str, Any]]): 黄金标准参考数据列表，每个元素是一个样本字典。

        Returns:
            Dict[str, float]: 包含评测分数的字典，例如 {'accuracy': 0.85}。
        """
        pass

    def __call__(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """允许像函数一样调用对象。"""
        return self.evaluate(predictions, references) 