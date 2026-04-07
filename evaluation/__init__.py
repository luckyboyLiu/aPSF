from ..config import DATASET_CONFIG, USE_LLM_ANSWER_COMPARISON, ANSWER_EXTRACTOR_LLM
from .base_evaluator import BaseEvaluator
from .accuracy_evaluator import AccuracyEvaluator
from .execution_evaluator import ExecutionEvaluator
from .gsm8k_evaluator import GSM8KEvaluator
from .aqua_evaluator import AQuAEvaluator
from .web_of_lies_evaluator import WebOfLiesEvaluator
from .multiarith_evaluator import MultiArithEvaluator
from .gsm_hard_evaluator import GSMHardEvaluator
from .mmlu_evaluator import MMLUEvaluator
from .aime2025_evaluator import AIME2025Evaluator
from .competition_math_evaluator import CompetitionMathEvaluator

EVALUATOR_MAPPING = {
    "AccuracyEvaluator": AccuracyEvaluator,
    "ExecutionEvaluator": ExecutionEvaluator,
    "GSM8KEvaluator": GSM8KEvaluator,
    "GSMHardEvaluator": GSMHardEvaluator,
    "AQuAEvaluator": AQuAEvaluator,
    "WebOfLiesEvaluator": WebOfLiesEvaluator,
    "MultiArithEvaluator": MultiArithEvaluator,
    "MMLUEvaluator": MMLUEvaluator,
    "AIME2025Evaluator": AIME2025Evaluator,
    "CompetitionMathEvaluator": CompetitionMathEvaluator,
}

def get_evaluator(dataset_name: str) -> BaseEvaluator:
    """
    Get the corresponding evaluator instance based on dataset name

    Args:
        dataset_name: Dataset name (e.g., 'aqua', 'gsm8k')

    Returns:
        BaseEvaluator: Evaluator instance
    """
    # Get the evaluator name for this dataset from config
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    evaluator_name = DATASET_CONFIG[dataset_name]["evaluator"]

    if evaluator_name not in EVALUATOR_MAPPING:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")

    # If AccuracyEvaluator, pass LLM comparison config and answer extractor model
    if evaluator_name == "AccuracyEvaluator":
        return EVALUATOR_MAPPING[evaluator_name](
            extractor_llm_id=ANSWER_EXTRACTOR_LLM,
            use_llm_comparison=USE_LLM_ANSWER_COMPARISON
        )

    # If CompetitionMathEvaluator, pass LLM comparison config
    if evaluator_name == "CompetitionMathEvaluator":
        return EVALUATOR_MAPPING[evaluator_name](
            use_llm_comparison=USE_LLM_ANSWER_COMPARISON
        )

    return EVALUATOR_MAPPING[evaluator_name]() 