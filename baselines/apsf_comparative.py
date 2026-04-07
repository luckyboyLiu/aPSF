"""
aPSF Comparative Experiments Implementation
Contains various comparative experiments:
1. Prompt structure transfer - Transfer structure from one task to another
2. Worker LLM comparison - Test prompts with different LLM capabilities
3. Manual few-shot comparison - Compare with human-written prompts
4. Prompt stability testing - Run multiple optimizations to check consistency
"""

import logging
import random
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from ..optimization import Architect, Optimizer
from ..optimization.prompt_object import PromptStructure
from ..llm_apis import get_llm
from ..evaluation import BaseEvaluator
from ..config import RESULTS_DIR


def run_apsf_prompt_transfer(
    source_task: str,
    target_task: str,
    source_eval_data: List[Dict[str, Any]],
    target_eval_data: List[Dict[str, Any]],
    target_test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prompt structure transfer experiment - Transfer structure from one task to another
    """
    logging.info(f"Running prompt structure transfer experiment: {source_task} -> {target_task}")
    
    # Step 1: Learn structure on source task
    logging.info(f" Step 1: Learning prompt structure on source task {source_task}")
    architect = Architect()
    source_examples = _construct_examples(source_eval_data[:5])
    source_structure = architect.discover_structure(f"Task: {source_task}", source_examples)
    
    logging.info(f"Source task structure: {source_structure.factors}")
    
    # Step 2: Transfer structure to target task
    logging.info(f" Step 2: Transferring structure to target task {target_task}")
    
    # Create target task structure but keep source task factor framework
    target_structure = PromptStructure(
        task_description=f"Task: {target_task}",
        factors=source_structure.factors.copy(),  # Copy source task factors
        fusion_prompt=source_structure.fusion_prompt  # Use source task fusion template
    )
    
    # Fine-tune on target task
    optimizer = Optimizer(
        target_structure, target_eval_data, evaluator, dataset_config,
        method_name="aPSF-Transfer"
    )
    
    transfer_steps = max(3, dataset_config.get("total_optimization_steps", 10) // 3)
    for step in range(transfer_steps):
        logging.info(f"Transfer optimization step {step + 1}/{transfer_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(target_test_data)
    
    return {
        "final_score": test_score,
        "source_task": source_task,
        "target_task": target_task,
        "source_structure": source_structure.to_dict(),
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "transfer_success": test_score > 0.5  # Simple success judgment
    }


def run_apsf_worker_llm_comparison(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    worker_llm_configs: List[str] = None
) -> Dict[str, Any]:
    """
    Worker LLM comparison experiment - Test prompt performance across different LLMs
    """
    if worker_llm_configs is None:
        worker_llm_configs = ["worker", "small_worker", "large_worker"]  # Default test three scales
    
    logging.info(f"Running Worker LLM comparison experiment, testing {len(worker_llm_configs)} models")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    results = {}
    
    for worker_config in worker_llm_configs:
        logging.info(f"\nTesting Worker LLM: {worker_config}")
        
        # Use different Worker LLM for optimization
        optimizer = Optimizer(
            prompt_struct.copy(), eval_data, evaluator, dataset_config,
            worker_llm_id=worker_config,
            method_name=f"aPSF-Worker-{worker_config}"
        )
        
        # Run optimization
        total_steps = dataset_config.get("total_optimization_steps", 10)
        for step in range(total_steps):
            logging.info(f"{worker_config} optimization step {step + 1}/{total_steps}")
            optimizer.step()
        
        # Test set evaluation
        test_score = optimizer.evaluate_on_test_set(test_data)
        
        results[worker_config] = {
            "final_score": test_score,
            "optimized_prompt": optimizer.get_optimized_prompt()
        }
    
    # Analyze results
    best_worker = max(results.keys(), key=lambda k: results[k]["final_score"])
    
    return {
        "worker_results": results,
        "best_worker": best_worker,
        "best_score": results[best_worker]["final_score"],
        "performance_variance": _calculate_variance([r["final_score"] for r in results.values()])
    }


def run_apsf_vs_manual_fewshot(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    manual_prompt: str = None,
    few_shot_examples: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Manual few-shot comparison experiment - Compare with human-written prompts
    """
    logging.info("Running manual few-shot comparison experiment")
    
    # 1. Run aPSF optimization
    logging.info("Step 1: Running aPSF optimization")
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, dataset_config,
        method_name="aPSF-AutoPrompt"
    )
    
    total_steps = dataset_config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        optimizer.step()
    
    apsf_score = optimizer.evaluate_on_test_set(test_data)
    apsf_prompt = optimizer.get_optimized_prompt()
    
    # 2. Evaluate manual few-shot
    logging.info("Step 2: Evaluating manual few-shot")
    
    if manual_prompt is None:
        # Default manual few-shot template
        manual_prompt = _create_manual_fewshot_prompt(few_shot_examples or eval_data[:3])
    
    # Evaluate manual prompt
    worker_llm = get_llm("worker")
    manual_predictions = []
    
    for item in test_data:
        input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
        question = item.get(input_key, '')
        formatted_prompt = manual_prompt.format(input=question) if "{input}" in manual_prompt else f"{manual_prompt}\n\n{question}"
        prediction = worker_llm.generate(formatted_prompt)
        manual_predictions.append(prediction)
    
    manual_results = evaluator.evaluate(manual_predictions, test_data)
    manual_score = manual_results.get(dataset_config.get("metric", "accuracy"), 0.0)
    
    # 3. Comparative analysis
    improvement = ((apsf_score - manual_score) / manual_score * 100) if manual_score > 0 else float('inf')
    
    return {
        "apsf_score": apsf_score,
        "manual_score": manual_score,
        "improvement_percentage": improvement,
        "apsf_prompt": apsf_prompt,
        "manual_prompt": manual_prompt,
        "winner": "aPSF" if apsf_score > manual_score else "Manual"
    }


def run_apsf_stability_test(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Prompt stability testing - Run multiple optimizations to check consistency
    """
    logging.info(f"Running prompt stability testing, total {num_runs} runs")
    
    run_results = []
    all_prompts = []
    all_scores = []
    
    for run_idx in range(num_runs):
        logging.info(f"\nRun {run_idx + 1}/{num_runs}")
        
        # Set different random seed
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'] + run_idx)
        
        # Use Architect to discover structure
        architect = Architect()
        example_data = _construct_examples(eval_data[:5])
        prompt_struct = architect.discover_structure(task_desc, example_data)
        
        # Run optimization
        optimizer = Optimizer(
            prompt_struct, eval_data, evaluator, dataset_config,
            method_name=f"aPSF-Stability-Run{run_idx+1}"
        )
        
        total_steps = dataset_config.get("total_optimization_steps", 10)
        for step in range(total_steps):
            optimizer.step()
        
        # Test set evaluation
        test_score = optimizer.evaluate_on_test_set(test_data)
        optimized_prompt = optimizer.get_optimized_prompt()
        
        run_results.append({
            "run": run_idx + 1,
            "score": test_score,
            "prompt": optimized_prompt
        })
        
        all_scores.append(test_score)
        all_prompts.append(optimized_prompt)
    
    # Calculate stability metrics
    score_mean = sum(all_scores) / len(all_scores)
    score_std = _calculate_std(all_scores)
    score_variance = _calculate_variance(all_scores)
    
    # Calculate prompt similarity
    prompt_similarity = _calculate_prompt_similarity(all_prompts)
    
    return {
        "num_runs": num_runs,
        "run_results": run_results,
        "score_statistics": {
            "mean": score_mean,
            "std": score_std,
            "variance": score_variance,
            "min": min(all_scores),
            "max": max(all_scores),
            "range": max(all_scores) - min(all_scores)
        },
        "prompt_similarity": prompt_similarity,
        "stability_rating": _rate_stability(score_std, prompt_similarity)
    }


# Helper functions
def _construct_examples(samples: List[Dict[str, Any]]) -> str:
    """Construct example strings"""
    example_strings = []
    for i, sample in enumerate(samples, 1):
        input_text = sample.get('input', sample.get('prompt', sample.get('question', '')))
        output_text = sample.get('target', sample.get('answer', sample.get('output', '')))
        example_strings.append(
            f"--- Example {i} ---\nInput: {input_text}\nExpected Output: {output_text}"
        )
    return "\n\n".join(example_strings)


def _create_manual_fewshot_prompt(examples: List[Dict[str, Any]]) -> str:
    """Create manual few-shot prompt"""
    prompt_parts = ["Please solve the following problem. Here are some examples:"]
    
    for i, example in enumerate(examples, 1):
        input_text = example.get('input', example.get('prompt', example.get('question', '')))
        output_text = example.get('target', example.get('answer', example.get('output', '')))
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Input: {input_text}")
        prompt_parts.append(f"Output: {output_text}")
    
    prompt_parts.append("\nNow solve this problem:\n{input}")
    
    return "\n".join(prompt_parts)


def _calculate_variance(values: List[float]) -> float:
    """Calculate variance"""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation"""
    return _calculate_variance(values) ** 0.5


def _calculate_prompt_similarity(prompts: List[str]) -> float:
    """Calculate similarity between prompts (simple version)"""
    if len(prompts) < 2:
        return 1.0
    
    # Simple word-level similarity
    similarities = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            words1 = set(prompts[i].lower().split())
            words2 = set(prompts[j].lower().split())
            if words1 or words2:
                similarity = len(words1 & words2) / len(words1 | words2)
                similarities.append(similarity)
    
    return sum(similarities) / len(similarities) if similarities else 1.0


def _rate_stability(score_std: float, prompt_similarity: float) -> str:
    """Rate stability level"""
    if score_std < 0.02 and prompt_similarity > 0.8:
        return "Very Stable"
    elif score_std < 0.05 and prompt_similarity > 0.6:
        return "Stable"
    elif score_std < 0.1 and prompt_similarity > 0.4:
        return "Moderately Stable"
    else:
        return "Unstable"
