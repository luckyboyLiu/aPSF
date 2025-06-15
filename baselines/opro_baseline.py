# OPRO: Optimization by PROmpting 的简化实现占位符
# OPRO 相当于 aPSF 的 "无因子化" 版本

import numpy as np
from typing import List, Dict, Any
from ..llm_apis import BaseLLM
from ..evaluation import BaseEvaluator
from ..config import OPTIMIZATION_PARAMS

def _create_opro_meta_prompt(current_prompt: str, task_desc: str, num_candidates: int) -> str:
    """为 OPRO 创建元提示，用于生成对整个提示的改进版本。"""
    return f"""
You are an expert prompt optimizer. Your goal is to iteratively refine a given prompt to improve its performance on a specific task.

Task Description: {task_desc}

Here is the current prompt that needs improvement:
--- CURRENT PROMPT ---
{current_prompt}
--- END CURRENT PROMPT ---

Please generate {num_candidates} new versions of this entire prompt. The new versions should be diverse and aim to be more effective for the task. Each version should be a complete prompt.

Output each new version separated by '--- CANDIDATE ---'.
"""

def _evaluate_prompt(prompt: str, worker_llm: BaseLLM, eval_data: List[Dict[str, Any]], evaluator: BaseEvaluator) -> float:
    """评估单个完整提示在验证集上的性能。"""
    predictions = []
    for item in eval_data:
        # 简化版输入格式化
        formatted_prompt = prompt.format(input=item.get('question') or item.get('prompt', ''))
        prediction = worker_llm.generate(formatted_prompt)
        predictions.append(prediction)
    
    # 假设评测器的第一个返回值就是主要指标
    metric_key = list(evaluator.evaluate([], [{}])[0].keys())[0]
    score = evaluator.evaluate(predictions, eval_data)[metric_key]
    return score

def run_opro(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    worker_llm: BaseLLM,
    architect_llm: BaseLLM
) -> Dict[str, Any]:
    """
    运行 OPRO 基线。

    Returns:
        A dictionary containing the final score and the best prompt.
    """
    print("\n--- Running OPRO Baseline ---")
    
    max_steps = OPTIMIZATION_PARAMS.get("total_optimization_steps", 100)
    num_candidates = OPTIMIZATION_PARAMS.get("candidates_per_step", 5)
    
    # 从一个非常简单的提示开始
    best_prompt = "Instruction: Please solve the following task.\nInput: {input}\nAnswer:"
    best_score = _evaluate_prompt(best_prompt, worker_llm, eval_data, evaluator)
    print(f"Initial OPRO prompt score: {best_score:.4f}")

    for step in range(max_steps // 10): # 为加速演示，减少步数
        print(f"OPRO Step {step + 1}/{max_steps // 10}...")
        meta_prompt = _create_opro_meta_prompt(best_prompt, task_desc, num_candidates)
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

    print("\nOPRO optimization finished. Evaluating on test set...")
    final_score = _evaluate_prompt(best_prompt, worker_llm, test_data, evaluator)
    print(f"OPRO final test score: {final_score:.4f}")

    return {"score": final_score, "prompt": best_prompt} 