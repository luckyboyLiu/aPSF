from ..config import DATASET_CONFIG
from .base_evaluator import BaseEvaluator
from .accuracy_evaluator import AccuracyEvaluator
from .pass_at_k_evaluator import PassAtKEvaluator
from .rouge_evaluator import RougeEvaluator
from .execution_evaluator import ExecutionEvaluator

EVALUATOR_MAPPING = {
    "AccuracyEvaluator": AccuracyEvaluator,
    "PassAtKEvaluator": PassAtKEvaluator,
    "RougeEvaluator": RougeEvaluator,
    "ExecutionEvaluator": ExecutionEvaluator,
}

def get_evaluator(dataset_name: str) -> BaseEvaluator:
    """
    获取评测器实例的工厂函数。

    Args:
        dataset_name (str): 配置文件中的数据集名称 (例如, "gsm8k")。

    Returns:
        BaseEvaluator: 相应评测器的实例。
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"数据集 '{dataset_name}' 在 config.py 中未找到。")
    
    config = DATASET_CONFIG[dataset_name]
    evaluator_class_name = config.get("evaluator")
    
    if evaluator_class_name not in EVALUATOR_MAPPING:
        raise ValueError(f"评测器类 '{evaluator_class_name}' 未在 EVALUATOR_MAPPING 中定义。")
        
    evaluator_class = EVALUATOR_MAPPING[evaluator_class_name]
    
    return evaluator_class() 