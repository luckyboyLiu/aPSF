import numpy as np
from typing import List, Dict, Any

from ..llm_apis import BaseLLM
from ..evaluation import BaseEvaluator
from ..config import OPTIMIZATION_PARAMS
from .opro_baseline import _evaluate_prompt # 复用评估函数

def _create_protegi_meta_prompt(current_prompt: str, task_desc: str, last_score: float, num_candidates: int) -> str:
    """为 ProTeGi 创建一个更复杂的元提示，包含对过去性能的反思。"""
    return f"""
You are a world-class prompt engineering system called ProTeGi. Your task is to refine the given prompt to maximize its performance.

The task is: {task_desc}

The current prompt is:
--- CURRENT PROMPT ---
{current_prompt}
--- END CURRENT PROMPT ---

This prompt achieved a validation score of {last_score:.4f}. While good, it can be better. Think about potential weaknesses in the current prompt. Is it unclear? Is it missing context? Could the structure be improved?

Based on this reflection, generate {num_candidates} improved versions of the entire prompt. The new versions should be thoughtful and diverse attempts to fix potential issues.

Output each new version separated by '--- CANDIDATE ---'.
"""

def run_protegi(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    worker_llm: BaseLLM,
    architect_llm: BaseLLM
) -> Dict[str, Any]:
    """运行 ProTeGi 基线。"""
    print("\n--- Running ProTeGi Baseline ---")
    
    max_steps = OPTIMIZATION_PARAMS.get("total_optimization_steps", 100)
    num_candidates = OPTIMIZATION_PARAMS.get("candidates_per_step", 5)
    
    best_prompt = "Task: {input}\nLet's think step by step."
    best_score = _evaluate_prompt(best_prompt, worker_llm, eval_data, evaluator)
    print(f"Initial ProTeGi prompt score: {best_score:.4f}")

    for step in range(max_steps // 10): # 为加速演示，减少步数
        print(f"ProTeGi Step {step + 1}/{max_steps // 10}...")
        meta_prompt = _create_protegi_meta_prompt(best_prompt, task_desc, best_score, num_candidates)
        response = architect_llm.generate(meta_prompt)
        candidates = [c.strip() for c in response.split("--- CANDIDATE ---") if c.strip()]

        if not candidates:
            continue

        scores = [_evaluate_prompt(c, worker_llm, eval_data, evaluator) for c in candidates]
        
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > best_score:
            best_score = scores[current_best_idx]
            best_prompt = candidates[current_best_idx]
            print(f"  -> New best prompt found with validation score: {best_score:.4f}")

    print("\nProTeGi optimization finished. Evaluating on test set...")
    final_score = _evaluate_prompt(best_prompt, worker_llm, test_data, evaluator)
    print(f"ProTeGi final test score: {final_score:.4f}")

    return {"score": final_score, "prompt": best_prompt} 