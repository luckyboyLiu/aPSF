import os
import argparse
import json
import logging
import random
import glob
from datetime import datetime
from .optimization import Architect, Optimizer 
from .data_loader import get_loader  # 
from .evaluation import get_evaluator, BaseEvaluator  # 
from .llm_apis import get_llm  # 
from .config import DATASET_CONFIG, OPTIMIZATION_PARAMS, RESULTS_DIR, DATA_PATHS, DATA_SPLIT_CONFIG  # 
from .baselines import (
    run_apsf_nostructure,
    run_apsf_nofactor,
    run_apsf_nodap,
    run_apsf_randselect,
    run_apsf_feedback,
    run_apsf_smallarchitect,
    run_apsf_thompson,
    run_apsf_roundrobin,
    run_apsf_greedy,
    run_apsf_prompt_transfer,
    run_apsf_worker_llm_comparison,
    run_apsf_vs_manual_fewshot,
    run_apsf_stability_test
)
from typing import List, Dict, Any, Optional
from .checkpoint_manager import CheckpointManager, BBHAllCheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_logs.log'),
        logging.StreamHandler()
    ]
)

def set_random_seed(seed=None):
    """Set random seed for reproducibility"""
    if seed is None:
        seed = DATA_SPLIT_CONFIG['random_seed']
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logging.info(f"Random seed set to: {seed}")

def run_apsf_pipeline(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    enable_feedback: bool = False,
    step: Optional[int] = None,
    initial_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Run the complete aPSF pipeline

    Args:
        task_desc: Task description
        eval_data: Validation data
        test_data: Test data
        evaluator: Evaluator instance
        dataset_config: Dataset configuration
        enable_feedback: Whether to enable feedback mechanism
        step: Number of optimization steps
        initial_prompt: Initial prompt (optional, e.g., "Let's think step by step")
                       If provided, optimization starts from this prompt instead of scratch

    Returns:
        Dictionary containing optimization results
    """
    # Ensure random seed consistency for reproducible data splits
    set_random_seed()

    logging.info(" Starting aPSF pipeline...")

    # Log initial prompt if provided
    if initial_prompt:
        logging.info(f" Using initial prompt: '{initial_prompt}'")
        logging.info(" aPSF will perform factor discovery and optimization on this basis")

    # Merge optimization params into dataset config
    merged_config = dataset_config.copy()
    merged_config.update(OPTIMIZATION_PARAMS)

    # Override config if step parameter is specified
    if step is not None:
        merged_config['total_optimization_steps'] = step
        logging.info(f"Optimization steps set to: {step}")

    # Use Architect to discover prompt structure
    architect = Architect()

    # Construct example data for structure discovery - using dataset-specific format (with solution steps)
    dataset_name = dataset_config.get('dataset', '')
    example_data = _construct_universal_discovery_examples(dataset_name, eval_data)

    # Pass initial_prompt parameter
    prompt_struct = architect.discover_structure(task_desc, example_data, initial_prompt=initial_prompt)

    # Use standard optimizer, pass feedback parameter
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, merged_config,
        enable_feedback=enable_feedback
    )
    logging.info(" Using Standard aPSF Optimizer")

    # Run optimization steps iteratively
    total_steps = merged_config.get("total_optimization_steps", 10)
    logging.info(f" Starting optimization for {total_steps} steps...")
    
    for step_num in range(total_steps):
        logging.info(f" Optimization Step {step_num + 1}/{total_steps}")
        optimizer.step()
        
        # Check for early stopping condition
        if optimizer.current_optimization_step >= total_steps:
            break

    # Reflection optimization phase
    enable_reflection = merged_config.get("enable_reflection", True)  # Reflection enabled by default
    if enable_reflection:
        logging.info(" Starting reflection optimization phase...")
        reflection_improved = optimizer.reflection_optimization(eval_data)
        if reflection_improved:
            logging.info(" Reflection optimization succeeded, prompt updated")
        else:
            logging.info(" Reflection optimization did not improve performance, keeping original prompt")
    else:
        logging.info(" Skipping reflection optimization phase")

    # Evaluate on test set
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    # Print optimization summary statistics (including regression rate)
    optimizer.print_optimization_summary()

    # Also call print_final_summary if available
    if hasattr(optimizer, 'print_final_summary'):
        optimizer.print_final_summary(test_score)

    # Save factor analysis report
    dataset_name = dataset_config.get('dataset', 'unknown')
    factor_analysis_path = optimizer.save_factor_analysis(dataset_name)
    logging.info(f" Factor analysis report saved to: {factor_analysis_path}")

    # Serialize PromptStructure object to dict
    best_structure = optimizer.get_best_structure()
    best_structure_dict = None
    if best_structure:
        try:
            best_structure_dict = best_structure.to_dict()
        except Exception as e:
            logging.warning(f" Cannot serialize PromptStructure: {e}")
            best_structure_dict = {
                "task_description": getattr(best_structure, 'task_description', ''),
                "fusion_prompt": getattr(best_structure, 'fusion_prompt', ''),
                "factors": getattr(best_structure, 'factors', {}),
                "serialization_error": str(e)
            }

    # Get regression rate statistics
    regression_rate = optimizer.get_regression_rate()
    regression_stats_summary = {
        'regression_rate': regression_rate,
        'total_accepted_updates': optimizer.regression_stats['total_accepted_updates'],
        'total_regressions': optimizer.regression_stats['total_regressions'],
        'per_factor_regressions': optimizer.regression_stats['per_factor_regressions']
    }

    # Final output (ensure printed at the end)
    print(f"\n{'='*80}", flush=True)
    print(f" aPSF complete, test set final score: {test_score:.4f}", flush=True)
    print(f" Best prompt obtained at optimization step {optimizer.global_best_step}", flush=True)
    # Token statistics
    worker_stats = optimizer.worker_llm.get_token_stats()
    architect_stats = optimizer.architect_llm.get_token_stats()
    total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
    print(f" TOKENS AT BEST STEP: {optimizer.global_best_tokens:,} tokens")
    print(f" TOTAL TOKENS: {total_tokens:,} tokens")
    print(f"{'='*80}\n", flush=True)

    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": best_structure_dict,  # Returns serializable dict
        "factor_analysis_path": factor_analysis_path,  # Factor analysis file path
        "regression_stats": regression_stats_summary  # Regression rate statistics
    }

def save_results(method_name: str, dataset_name: str, results_data: dict):
    """Unified function for saving experiment results to file."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{method_name}_{dataset_name}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f" Results saved to: {filepath}")

def _construct_universal_discovery_examples(dataset_name: str, eval_data: List[Dict[str, Any]]) -> str:
    """Construct structure discovery examples - dataset-agnostic version"""

    if not eval_data:
        return "No examples available for structure discovery."

    # Randomly select 3-5 samples for structure discovery
    num_samples = min(5, len(eval_data))
    selected_samples = random.sample(eval_data, num_samples)

    # Dataset-agnostic mode: use generic format only, let algorithm discover patterns
    if not dataset_name:  # Empty string indicates dataset-agnostic mode
        return _construct_generic_examples(selected_samples)

    # Compatibility: keep original logic for non-blind scenarios
    if "gsm" in dataset_name.lower():
        return _construct_gsm_examples(selected_samples)
    elif "bbh" in dataset_name.lower() or "colored_objects" in dataset_name.lower():
        return _construct_bbh_examples(selected_samples, dataset_name)
    elif "aqua" in dataset_name.lower():
        return _construct_aqua_examples(selected_samples)
    elif "multiarith" in dataset_name.lower():
        return _construct_multiarith_examples(selected_samples)
    elif "humaneval" in dataset_name.lower():
        return _construct_humaneval_examples(selected_samples)
    elif "mmlu" in dataset_name.lower():
        return _construct_mmlu_examples(selected_samples)
    elif "squad" in dataset_name.lower():
        return _construct_squad_examples(selected_samples)
    else:
        # Generic format
        return _construct_generic_examples(selected_samples)

def _construct_generic_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct generic format examples - includes input, reasoning process, output"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        # Try different key names
        input_text = sample.get('input', sample.get('prompt', sample.get('question', '')))
        output_text = sample.get('target', sample.get('answer', sample.get('output', '')))
        reasoning = sample.get('solution', sample.get('rationale', sample.get('explanation', '')))

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {input_text}")
        if reasoning:
            example_parts.append(f"Reasoning: {reasoning}")
        example_parts.append(f"Expected Output: {output_text}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_gsm_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct GSM8K-specific example format - includes question, solution steps, answer"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        question = sample.get('question', sample.get('input', ''))
        answer = sample.get('answer', sample.get('target', ''))
        solution = sample.get('solution', '')  # Get solution steps

        # Build example: question + solution process + answer
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {question}")
        if solution:
            # Clean solution (remove possible b' ' wrapper)
            if isinstance(solution, str) and solution.startswith("b'"):
                solution = solution[2:-1].replace('\\n', '\n')
            example_parts.append(f"Solution Steps: {solution}")
        example_parts.append(f"Expected Output: {answer}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_bbh_examples(selected_samples: List[Dict[str, Any]], dataset_name: str) -> str:
    """Construct BBH-specific example format - includes question, reasoning process, answer"""
    # Extract task type
    task_type = "BBH Reasoning"
    if "colored_objects" in dataset_name:
        task_type = "Colored Objects Reasoning"
    elif "web_of_lies" in dataset_name:
        task_type = "Web of Lies"
    elif "movie_recommendation" in dataset_name:
        task_type = "Movie Recommendation"
    
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        input_text = sample.get('input', '')
        output_text = sample.get('target', '')
        reasoning = sample.get('rationale', sample.get('explanation', ''))

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {input_text}")
        if reasoning:
            example_parts.append(f"Reasoning: {reasoning}")
        example_parts.append(f"Expected Output: {output_text}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_aqua_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct AQuA-specific example format - includes question, solution process, answer"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('correct', sample.get('target', ''))
        rationale = sample.get('rationale', sample.get('solution', ''))

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {question}")
        if rationale:
            example_parts.append(f"Solution Steps: {rationale}")
        example_parts.append(f"Expected Output: {answer}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_multiarith_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct MultiArith-specific example format - includes question, solution steps, answer"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        question = sample.get('question', sample.get('input', ''))
        answer = sample.get('answer', sample.get('target', ''))
        solution = sample.get('solution', sample.get('rationale', ''))

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {question}")
        if solution:
            example_parts.append(f"Solution Steps: {solution}")
        example_parts.append(f"Expected Output: {answer}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_humaneval_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct HumanEval-specific example format - includes problem, solution"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        prompt = sample.get('prompt', '')
        solution = sample.get('canonical_solution', sample.get('target', ''))
        docstring = sample.get('docstring', '')

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {prompt}")
        if docstring:
            example_parts.append(f"Description: {docstring}")
        example_parts.append(f"Expected Output: {solution}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_mmlu_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct MMLU-specific example format - includes question, reasoning, answer"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        question = sample.get('input', sample.get('question', ''))
        answer = sample.get('target', sample.get('answer', ''))
        explanation = sample.get('explanation', sample.get('rationale', ''))

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Input: {question}")
        if explanation:
            example_parts.append(f"Reasoning: {explanation}")
        example_parts.append(f"Expected Output: {answer}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def _construct_squad_examples(selected_samples: List[Dict[str, Any]]) -> str:
    """Construct SQuAD 2.0-specific example format - includes passage context, question, answer"""
    example_strings = []

    for i, sample in enumerate(selected_samples, 1):
        context = sample.get('context', '')
        question = sample.get('question', sample.get('input', ''))
        answer = sample.get('answer', sample.get('target', ''))
        is_impossible = sample.get('is_impossible', False)

        # Build example
        example_parts = [f"--- Example {i} ---"]
        example_parts.append(f"Passage: {context[:300]}...")  # Show first 300 chars of context
        example_parts.append(f"Question: {question}")

        if is_impossible:
            example_parts.append(f"Expected Output: The answer cannot be found in the passage.")
        else:
            example_parts.append(f"Expected Output: {answer}")

        example_strings.append("\n".join(example_parts))

    return "\n\n".join(example_strings)

def run_baseline_method(method_name: str, task_desc: str, val_data: List[Dict],
                       test_data: List[Dict], evaluator, config: Dict, step: Optional[int] = None,
                       resume: bool = False) -> Dict[str, Any]:
    """Run baseline method - display full process for all samples"""
    from .evaluation.unified_scoring import evaluate_with_unified_scoring, UnifiedScorer

    # Ensure random seed consistency for reproducible data splits
    set_random_seed()

    worker_llm = get_llm("worker")
    architect_llm = get_llm("architect")

    print(f" Running baseline method: {method_name}")
    print(f" Data size: validation {len(val_data)}, test {len(test_data)}")

    # Unified display format for validation and test evaluation
    def detailed_evaluation_with_display(data, data_type, prompt_template, method_display_name, do_scoring=False):
        """Generate responses, optionally with scoring"""
        predictions = []
        correct_count = 0

        print(f"\n Starting {method_display_name} {data_type} detailed evaluation", flush=True)
        print(f" Sample count: {len(data)}", flush=True)

        for i, item in enumerate(data):
            input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
            question = item.get(input_key, '')

            # Format prompt
            if '{input}' in prompt_template:
                formatted_prompt = prompt_template.format(input=question)
            else:
                # Combine instruction with question directly for pure instructions
                formatted_prompt = f"{prompt_template}\n\n{question}"

            # Display question and current template
            print(f"\n{'='*80}", flush=True)
            print(f" {method_display_name} {data_type} sample {i+1}/{len(data)}", flush=True)
            print(f"{'='*80}", flush=True)
            print(f" Question:", flush=True)
            print(f"   {question}", flush=True)
            print(f"\n Current {method_display_name} template:", flush=True)
            print(f"   {prompt_template}", flush=True)
            print(f"\n {method_display_name} response:", flush=True)

            # Generate response
            prediction = worker_llm.generate(formatted_prompt)
            predictions.append(prediction)
            print(f"   {prediction}", flush=True)

            # Score if needed (only for test set)
            if do_scoring:
                from .evaluation.unified_scoring import UnifiedScorer
                scorer = UnifiedScorer(worker_llm, f"bbh_{data_type}")
                extracted_answer, target_answer, is_correct = scorer.extract_and_score(
                    prediction, item, evaluator
                )

                print(f"\n Extracted answer: '{extracted_answer}'", flush=True)
                print(f" Target answer: '{target_answer}'", flush=True)
                verdict = " Correct" if is_correct else "Incorrect"
                print(f"Answer match: {verdict}", flush=True)

                if is_correct:
                    correct_count += 1

        # Return accuracy if scoring was performed
        if do_scoring:
            accuracy = correct_count / len(data) if len(data) > 0 else 0.0

            # Check for MMLU task and output category results
            if hasattr(evaluator, 'subject_mapping'):
                print("\n" + "="*80, flush=True)
                print(" Using MMLU evaluator to generate subject category results", flush=True)
                print("="*80, flush=True)
                mmlu_results = evaluator.evaluate(predictions, data)
                accuracy = mmlu_results.get("Average", accuracy)
                print(f"\n MMLU average accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)
            else:
                print(f"\n {data_type} final results:", flush=True)
                print(f"  Correct: {correct_count}/{len(data)}", flush=True)
                print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", flush=True)

            return accuracy, predictions
        else:
            return predictions
    # def detailed_evaluation_with_display(data, data_type, prompt_template, method_display_name):
    #     """Detailed evaluation with full sample display"""
    #     predictions = []
    #     correct_count = 0
    #
    #     print(f"\n Starting {method_display_name} {data_type} detailed evaluation")
    #     print(f" Sample count: {len(data)}")
    #
    #     for i, item in enumerate(data):
    #         input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
    #         question = item.get(input_key, '')
    #
    #         if '{input}' in prompt_template:
    #             formatted_prompt = prompt_template.format(input=question)
    #         else:
    #             formatted_prompt = f"{prompt_template}\n\n{question}"
    #
    #         print(f"\n{'='*80}")
    #         print(f"{method_display_name} {data_type} sample {i+1}/{len(data)}")
    #         print(f"{'='*80}")
    #         print(f" Question:")
    #         print(f"   {question}")
    #         print(f"\n Current {method_display_name} template:")
    #         print(f"   {prompt_template}")
    #         print(f"\n {method_display_name} response:")
    #
    #         prediction = worker_llm.generate(formatted_prompt)
    #         predictions.append(prediction)
    #
    #         print(f"   {prediction}")
    #
    #         scorer = UnifiedScorer(worker_llm, f"bbh_{task_desc.split()[3] if 'task' in task_desc else 'unknown'}")
    #         extracted_answer, target_answer, is_correct = scorer.extract_and_score(prediction, item, evaluator)
    #
    #         print(f"\n Extracted answer: '{extracted_answer}'")
    #         print(f" Target answer: '{target_answer}'")
    #         verdict = "Correct" if is_correct else "Wrong"
    #         print(f" Answer match: {verdict}")
    #
    #         if is_correct:
    #             correct_count += 1
    #
    #         print(f"{'='*80}")
    #
    #     eval_results = evaluate_with_unified_scoring(
    #         predictions=predictions,
    #         eval_data=data,
    #         evaluator=evaluator,
    #         llm=worker_llm,
    #         verbose=False,
    #         dataset_name=f"bbh_{data_type}_{task_desc.split()[3] if 'task' in task_desc else 'unknown'}"
    #     )
        
    #     accuracy = correct_count / len(data) if len(data) > 0 else 0.0
    #     return accuracy, predictions
    
    # Process based on method
    dataset_name = config.get('dataset', 'unknown')
    if dataset_name == 'unknown':
        logging.warning(
            "Baseline config missing 'dataset'; OPRO checkpoints may collide and prompt optimization may appear stuck. "
            "Pass a config dict with config['dataset']=<dataset_id>."
        )

    if method_name == 'opro':
        print(" Starting OPRO optimization...")

        # Run OPRO first to get optimized prompt (without showing detailed process)
        print("  OPRO internal optimization in progress...")
        opro_results = run_opro(
            task_desc=task_desc,
            eval_data=val_data,
            test_data=None,  # Don't pass test_data yet to avoid duplicate evaluation
            evaluator=evaluator,
            worker_llm=worker_llm,
            architect_llm=architect_llm,
            use_llm_extraction=True,
            step=step,  # Pass step parameter
            dataset=dataset_name,  # Pass dataset name
            resume=resume
        )

        best_prompt = opro_results["prompt"]
        best_step = opro_results.get("best_step", "unknown")
        print(f" OPRO optimized best prompt: {best_prompt}")
        print(f" Best prompt obtained at step {best_step}", flush=True)

        # Print OPRO regression rate statistics
        if 'regression_stats' in opro_results:
            regression_stats = opro_results['regression_stats']
            print(f"\n{'='*80}", flush=True)
            print(f" OPRO Regression Rate Statistics", flush=True)
            print(f"{'='*80}", flush=True)
            print(f" Total accepted updates: {regression_stats['total_accepted_updates']}", flush=True)
            print(f" Total regressions: {regression_stats['total_regressions']}", flush=True)
            print(f" Regression rate: {regression_stats['regression_rate']:.2%} "
                  f"({regression_stats['total_regressions']}/{regression_stats['total_accepted_updates']})", flush=True)
            print(f"{'='*80}\n", flush=True)

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "OPRO", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "OPRO", do_scoring=True)

        print(f"\n{'='*80}", flush=True)
        print(f" OPRO complete, test set final score: {test_score:.4f}", flush=True)
        print(f" Best prompt obtained at optimization step {best_step}", flush=True)
        # Token statistics
        worker_stats = worker_llm.get_token_stats()
        architect_stats = architect_llm.get_token_stats()
        total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
        best_step_tokens = opro_results.get("best_step_tokens", 0)
        print(f" TOKENS AT BEST STEP: {best_step_tokens:,} tokens")
        print(f" TOTAL TOKENS: {total_tokens:,} tokens")
        print(f"{'='*80}\n", flush=True)

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "regression_stats": opro_results.get('regression_stats', {}),  # Add regression stats
            "status": "success"
        }

    elif method_name == 'protegi':
        print(" Starting ProTeGi optimization...")

        # Run ProTeGi first to get optimized prompt
        print("  ProTeGi internal optimization in progress...")
        protegi_results = run_protegi(task_desc, val_data, None, evaluator, worker_llm, architect_llm, step=step)

        best_prompt = protegi_results["prompt"]
        best_step = protegi_results.get("best_achieved_at_step", "unknown")
        print(f" ProTeGi optimized best prompt: {best_prompt}")
        print(f" Best prompt obtained at step {best_step}", flush=True)

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "ProTeGi", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "ProTeGi", do_scoring=True)

        print(f"\n{'='*80}", flush=True)
        print(f" ProTeGi complete, test set final score: {test_score:.4f}", flush=True)
        print(f" Best prompt obtained at optimization step {best_step}", flush=True)
        # Token statistics
        worker_stats = worker_llm.get_token_stats()
        architect_stats = architect_llm.get_token_stats()
        total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
        best_step_tokens = protegi_results.get("best_step_tokens", 0)
        print(f" TOKENS AT BEST STEP: {best_step_tokens:,} tokens")
        print(f" TOTAL TOKENS: {total_tokens:,} tokens")
        print(f"{'='*80}\n", flush=True)

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    elif method_name == 'ape':
        print(" Starting APE optimization...")

        # Run APE first to get optimized prompt
        print("  APE internal optimization in progress...")
        ape_results = run_ape(task_desc, val_data, evaluator, worker_llm, architect_llm, step=step)

        best_prompt = ape_results["prompt"]
        best_step = ape_results.get("best_achieved_at_iteration", "unknown")
        print(f" APE optimized best prompt: {best_prompt}")
        print(f" Best prompt obtained at iteration {best_step}", flush=True)

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "APE", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "APE", do_scoring=True)

        print(f"\n{'='*80}", flush=True)
        print(f" APE complete, test set final score: {test_score:.4f}", flush=True)
        print(f" Best prompt obtained at optimization iteration {best_step}", flush=True)
        # Token statistics
        worker_stats = worker_llm.get_token_stats()
        architect_stats = architect_llm.get_token_stats()
        total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
        best_step_tokens = ape_results.get("best_step_tokens", 0)
        print(f" TOKENS AT BEST STEP: {best_step_tokens:,} tokens")
        print(f" TOTAL TOKENS: {total_tokens:,} tokens")
        print(f"{'='*80}\n", flush=True)

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    elif method_name == 'grips':
        print(" Starting GRIPS optimization...")

        print("  GRIPS internal optimization in progress...")
        grips_results = run_grips(task_desc, val_data, evaluator, worker_llm, architect_llm, step=step)

        best_prompt = grips_results["prompt"]
        best_step = grips_results.get("best_achieved_at_iteration", "unknown")
        print(f" GRIPS optimized best prompt: {best_prompt}")
        print(f" Best prompt obtained at step {best_step}", flush=True)

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "GRIPS", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "GRIPS", do_scoring=True)

        print(f"\n{'='*80}", flush=True)
        print(f" GRIPS complete, test set final score: {test_score:.4f}", flush=True)
        print(f" Best prompt obtained at optimization step {best_step}", flush=True)
        # Token statistics
        worker_stats = worker_llm.get_token_stats()
        architect_stats = architect_llm.get_token_stats()
        total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
        best_step_tokens = grips_results.get("best_step_tokens", 0)
        print(f" TOKENS AT BEST STEP: {best_step_tokens:,} tokens")
        print(f" TOTAL TOKENS: {total_tokens:,} tokens")
        print(f"{'='*80}\n", flush=True)

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    elif method_name == 'empty_cot':
        print(" Starting Empty CoT evaluation...")

        print("  Empty CoT internal optimization in progress...")
        empty_cot_results = run_empty_cot(task_desc, val_data, evaluator, worker_llm, architect_llm)

        best_prompt = empty_cot_results["prompt"]
        print(f" Empty CoT best prompt: {best_prompt}")

        # Validation set: generate responses only, no scoring
        #val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "Empty CoT", do_scoring=False)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "Empty CoT", do_scoring=True)

        print(f" Empty CoT complete, test set final score: {test_score:.4f}")

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    elif method_name == 'dspy':
        print(" Starting DSPy optimization...")

        print("  DSPy internal optimization in progress...")
        optimization_steps = step if step is not None else OPTIMIZATION_PARAMS.get("total_optimization_steps", 10)
        dspy_results = run_dspy(task_desc, val_data, evaluator, worker_llm, architect_llm=architect_llm, optimization_steps=optimization_steps)

        best_prompt = dspy_results.get("prompt", "Let's think step by step.")
        best_step = dspy_results.get("best_achieved_at_step", "unknown")
        print(f" DSPy optimized best prompt: {best_prompt}")
        print(f" Best prompt obtained at step {best_step}", flush=True)

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", best_prompt, "DSPy", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", best_prompt, "DSPy", do_scoring=True)

        print(f"\n{'='*80}", flush=True)
        print(f" DSPy complete, test set final score: {test_score:.4f}", flush=True)
        print(f" Best prompt obtained at optimization step {best_step}", flush=True)
        print(f"{'='*80}\n", flush=True)

        return {
            "final_score": test_score,
            "optimized_prompt": best_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    elif method_name == 'qwen3_direct':
        print(" Starting Qwen3 direct inference...")

        # Qwen3 direct inference uses fixed prompt
        direct_prompt = "Please analyze the question and provide your answer."
        print(f" Qwen3 direct inference prompt: {direct_prompt}")

        # Validation set: generate responses and show answer matching
        val_accuracy, val_predictions = detailed_evaluation_with_display(val_data, "validation", direct_prompt, "Qwen3Direct", do_scoring=True)

        # Test set: generate responses with scoring, output final score when done
        test_score, test_predictions = detailed_evaluation_with_display(test_data, "test", direct_prompt, "Qwen3Direct", do_scoring=True)

        print(f" Qwen3 direct inference complete, test set final score: {test_score:.4f}")

        return {
            "final_score": test_score,
            "optimized_prompt": direct_prompt,
            "test_predictions": test_predictions,
            "status": "success"
        }

    # ============ aPSF Ablation Experiments ============
    elif method_name == 'apsf_nostructure':
        from .baselines.apsf_ablation import run_apsf_nostructure
        print(" Starting aPSF-NoStructure ablation experiment...")
        return run_apsf_nostructure(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_notone':
        from .baselines.apsf_ablation import run_apsf_nofactor
        print(" Starting aPSF-NoTone ablation experiment...")
        return run_apsf_nofactor(task_desc, val_data, test_data, evaluator, config, "tone", step)

    elif method_name == 'apsf_noformat':
        from .baselines.apsf_ablation import run_apsf_nofactor
        print(" Starting aPSF-NoFormat ablation experiment...")
        return run_apsf_nofactor(task_desc, val_data, test_data, evaluator, config, "format", step)

    elif method_name == 'apsf_noperspective':
        from .baselines.apsf_ablation import run_apsf_nofactor
        print(" Starting aPSF-NoPerspective ablation experiment...")
        return run_apsf_nofactor(task_desc, val_data, test_data, evaluator, config, "perspective", step)

    elif method_name == 'apsf_nodap':
        from .baselines.apsf_ablation import run_apsf_nodap
        print(" Starting aPSF-NoDAP ablation experiment...")
        return run_apsf_nodap(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_randselect':
        from .baselines.apsf_ablation import run_apsf_randselect
        print(" Starting aPSF-RandSelect ablation experiment...")
        return run_apsf_randselect(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_feedback':
        from .baselines.apsf_ablation import run_apsf_feedback
        print(" Starting aPSF-Feedback ablation experiment...")
        return run_apsf_feedback(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_smallarchitect':
        from .baselines.apsf_ablation import run_apsf_smallarchitect
        print(" Starting aPSF-SmallArchitect ablation experiment...")
        return run_apsf_smallarchitect(task_desc, val_data, test_data, evaluator, config, step)

    # ============ Multi-Armed Bandit Algorithm Comparison ============
    elif method_name == 'apsf_thompson':
        from .baselines.apsf_ablation import run_apsf_thompson
        print(" Starting aPSF-Thompson Sampling comparison...")
        return run_apsf_thompson(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_roundrobin':
        from .baselines.apsf_ablation import run_apsf_roundrobin
        print(" Starting aPSF-Round-robin comparison...")
        return run_apsf_roundrobin(task_desc, val_data, test_data, evaluator, config, step)

    elif method_name == 'apsf_greedy':
        from .baselines.apsf_ablation import run_apsf_greedy
        print(" Starting aPSF-Greedy-best comparison...")
        return run_apsf_greedy(task_desc, val_data, test_data, evaluator, config, step)

    # ============ aPSF Comparative Experiments ============
    elif method_name == 'apsf_transfer':
        from .baselines.apsf_comparative import run_apsf_prompt_transfer
        print(" Starting aPSF-Transfer comparison experiment...")
        # Simplified handling for source and target tasks
        source_task = "mathematical_reasoning"
        target_task = task_desc
        return run_apsf_prompt_transfer(
            source_task, target_task, val_data[:5], val_data[5:], test_data, evaluator, config
        )

    elif method_name == 'apsf_worker_llm':
        from .baselines.apsf_comparative import run_apsf_worker_llm_comparison
        print(" Starting aPSF-WorkerLLM comparison experiment...")
        return run_apsf_worker_llm_comparison(task_desc, val_data, test_data, evaluator, config)

    elif method_name == 'apsf_vs_manual':
        from .baselines.apsf_comparative import run_apsf_vs_manual_fewshot
        print(" Starting aPSF vs Manual Few-shot comparison experiment...")
        return run_apsf_vs_manual_fewshot(task_desc, val_data, test_data, evaluator, config)

    elif method_name == 'apsf_stability':
        from .baselines.apsf_comparative import run_apsf_stability_test
        print(" Starting aPSF stability test...")
        return run_apsf_stability_test(task_desc, val_data, test_data, evaluator, config, num_runs=3)

    else:
        raise ValueError(f"Unsupported method: {method_name}")

def create_experiment_id(method: str, dataset: str, args) -> str:
    """Create unique experiment ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_suffix = "_feedback" if getattr(args, 'feedback', False) else ""
    return f"{method}_{dataset}{feedback_suffix}_{timestamp}"

def should_resume_experiment(args) -> bool:
    """Check whether to resume experiment"""
    return getattr(args, 'resume', False)

def find_resumable_checkpoint(method: str, dataset: str) -> Optional[str]:
    """Find resumable checkpoint"""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()

    # Find matching checkpoints
    matching_checkpoints = [
        cp for cp in checkpoints
        if cp['method'] == method and cp['dataset'] == dataset and cp['status'] != 'completed'
    ]

    if not matching_checkpoints:
        return None

    # Return most recent checkpoint
    latest_checkpoint = max(matching_checkpoints, key=lambda x: x['timestamp'])
    return latest_checkpoint['filename'].replace('_checkpoint.json', '')

def run_bbh_all_tasks_evaluation(method_name: str, config: Dict[str, Any], args, step: Optional[int] = None) -> Dict[str, Any]:
    """
    Run evaluation on all BBH tasks - supports resume
    """
    print(f"\n Starting BBH full task evaluation - Method: {method_name.upper()}")

    # Initialize checkpoint manager
    bbh_checkpoint_manager = BBHAllCheckpointManager()

    # Check if resume needed
    if should_resume_experiment(args):
        print(" Checking for resumable checkpoint...")
        experiment_id = find_resumable_checkpoint(method_name, "bbh_all")
        if experiment_id:
            print(f" Found resumable experiment: {experiment_id}")
            checkpoint = bbh_checkpoint_manager.load_checkpoint(method_name, "bbh_all", experiment_id)
            if checkpoint and checkpoint.get("status") == "completed":
                print(f" Experiment already completed, returning saved results")
                return checkpoint.get("final_results", {})
        else:
            print(" No resumable checkpoint found, starting new experiment")

    print("=" * 80)

    # Load data
    loader = get_loader("bbh_all")
    evaluator = get_evaluator("bbh_all")

    # Get all task names
    all_task_names = loader.get_all_task_names()

    if not all_task_names:
        logging.error("Failed to load any BBH tasks")
        return {}

    # Get remaining incomplete tasks (supports resume)
    if should_resume_experiment(args):
        remaining_tasks = bbh_checkpoint_manager.get_remaining_tasks(method_name, all_task_names)
        if not remaining_tasks:
            print(" All tasks completed!")
            # Load and return final results
            checkpoint = bbh_checkpoint_manager.load_checkpoint(method_name, "bbh_all")
            if checkpoint:
                return checkpoint.get("final_results", {})
    else:
        remaining_tasks = all_task_names

    print(f" Will evaluate {len(remaining_tasks)} BBH tasks:")
    for i, task_name in enumerate(remaining_tasks, 1):
        print(f"  {i:2d}. {task_name}")
    print("-" * 80)

    # Load existing results (if resuming)
    task_results = {}
    successful_tasks = 0
    total_score = 0.0

    if should_resume_experiment(args):
        checkpoint = bbh_checkpoint_manager.load_checkpoint(method_name, "bbh_all")
        if checkpoint:
            task_results = checkpoint.get("completed_tasks", {})
            successful_tasks = len(task_results)
            total_score = sum(result.get("final_score", 0.0) for result in task_results.values())
            print(f" Resumed from checkpoint: {successful_tasks} tasks completed")

    # Evaluate remaining tasks one by one
    for i, task_name in enumerate(remaining_tasks, 1):
        current_index = len(all_task_names) - len(remaining_tasks) + i
        print(f"\n [{current_index}/{len(all_task_names)}] Evaluating task: {task_name}")
        print("-" * 60)

        try:
            # Get validation and test data for this task
            val_data = loader.get_task_data(task_name, 'validation')
            test_data = loader.get_task_data(task_name, 'test')

            if not val_data:
                print(f" Task {task_name} validation data is empty, skipping")
                task_result = {
                    "status": "failed",
                    "error": "Empty validation data",
                    "final_score": 0.0
                }
                bbh_checkpoint_manager.save_task_result(method_name, task_name, task_result)
                continue

            print(f" Data summary: validation {len(val_data)}, test {len(test_data)}")

            # Build task description
            task_desc = ""
            print(f" Task description: {task_desc}")

            # Run evaluation based on method
            if method_name == 'apsf':
                task_config = config.copy()
                task_config['dataset'] = task_name
                # Read initial_prompt from config
                initial_prompt = task_config.get('initial_prompt', OPTIMIZATION_PARAMS.get('initial_prompt'))
                result = run_apsf_pipeline(task_desc, val_data, test_data, evaluator, task_config,
                                          args.feedback, step, initial_prompt=initial_prompt)
                result['status'] = 'success'
            else:
                task_config = config.copy()
                task_config['dataset'] = task_name
                result = run_baseline_method(method_name, task_desc, val_data, test_data, evaluator, task_config, step, resume=should_resume_experiment(args))

            # Save single task result to checkpoint
            bbh_checkpoint_manager.save_task_result(method_name, task_name, result)

            # Update overall results
            task_results[task_name] = result

            if result.get("status") == "success":
                score = result.get("final_score", 0.0)
                total_score += score
                successful_tasks += 1
                print(f" Task completed: score {score:.4f}")
            else:
                print(f" Task failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            error_msg = f"Evaluation error: {str(e)}"
            print(f" {error_msg}")
            logging.error(f"Task {task_name} failed: {error_msg}")
            task_result = {
                "status": "failed",
                "error": error_msg,
                "final_score": 0.0
            }
            bbh_checkpoint_manager.save_task_result(method_name, task_name, task_result)
    
    # Calculate final results
    average_score = total_score / successful_tasks if successful_tasks > 0 else 0.0

    final_results = {
        "method": method_name,
        "total_tasks": len(all_task_names),
        "successful_tasks": successful_tasks,
        "failed_tasks": len(all_task_names) - successful_tasks,
        "average_score": average_score,
        "total_score": total_score,
        "task_results": task_results,
        "status": "success"
    }

    # Save final results
    bbh_checkpoint_manager.finalize_experiment(method_name, final_results)

    # Print summary
    print("\n" + "=" * 80)
    print(f" BBH Full Task Evaluation Complete - {method_name.upper()}")
    print("=" * 80)
    print(f" Overall Statistics:")
    print(f"   Total tasks: {final_results['total_tasks']}")
    print(f"   Successful tasks: {final_results['successful_tasks']}")
    print(f"   Failed tasks: {final_results['failed_tasks']}")
    print(f"   Average score: {final_results['average_score']:.4f}")
    print(f"   Total score: {final_results['total_score']:.4f}")

    print(f"\n Detailed Results by Task:")
    print("-" * 80)
    print(f"{'Task Name':<40} {'Score':<10} {'Status':<10}")
    print("-" * 80)

    for task_name, task_result in task_results.items():
        if task_result.get("status") == "success":
            score = f"{task_result['final_score']:.4f}"
            status = " Success"
        else:
            score = "N/A"
            status = f" Failed"

        print(f"{task_name:<40} {score:<10} {status:<10}")

    print("=" * 80)

    return final_results

def run_mmlu_all_subjects_evaluation(method_name: str, config: Dict[str, Any], args, step: Optional[int] = None) -> Dict[str, Any]:
    """
    Run evaluation on all MMLU subjects - by category (faster)
    """
    print(f"\n Starting MMLU Full Subject Evaluation (by category) - Method: {method_name.upper()}")

    # Initialize checkpoint manager (using BBH checkpoint manager, which is generic)
    from .checkpoint_manager import BBHAllCheckpointManager
    mmlu_checkpoint_manager = BBHAllCheckpointManager()

    # Check if resume is needed
    if should_resume_experiment(args):
        print(" Checking for resumable checkpoint...")
        experiment_id = find_resumable_checkpoint(method_name, "mmlu_all")
        if experiment_id:
            print(f" Found resumable experiment: {experiment_id}")
            checkpoint = mmlu_checkpoint_manager.load_checkpoint(method_name, "mmlu_all", experiment_id)
            if checkpoint and checkpoint.get("status") == "completed":
                print(f" Experiment already completed, returning saved results")
                return checkpoint.get("final_results", {})
        else:
            print("ℹ No resumable checkpoint found, starting new experiment")

    print("=" * 80)

    # Load data
    loader = get_loader("mmlu_all")
    evaluator = get_evaluator("mmlu_all")

    # Get all category names (not all subject names)
    all_category_names = loader.get_all_category_names()

    if not all_category_names:
        logging.error(" No MMLU categories loaded successfully")
        return {}

    # Get remaining incomplete categories (supports resume)
    if should_resume_experiment(args):
        remaining_categories = mmlu_checkpoint_manager.get_remaining_tasks(method_name, all_category_names)
        if not remaining_categories:
            print(" All categories completed!")
            # Load and return final results
            checkpoint = mmlu_checkpoint_manager.load_checkpoint(method_name, "mmlu_all")
            if checkpoint:
                return checkpoint.get("final_results", {})
    else:
        remaining_categories = all_category_names

    print(f" Will evaluate {len(remaining_categories)} MMLU categories:")
    from .config import MMLU_CATEGORIES
    for i, category_name in enumerate(remaining_categories, 1):
        num_subjects = len(MMLU_CATEGORIES[category_name])
        print(f"  {i:2d}. {category_name} ({num_subjects} subjects)")
    print("-" * 80)

    # Load existing results (if resuming)
    category_results = {}
    successful_categories = 0
    total_score = 0.0

    if should_resume_experiment(args):
        checkpoint = mmlu_checkpoint_manager.load_checkpoint(method_name, "mmlu_all")
        if checkpoint:
            category_results = checkpoint.get("completed_tasks", {})
            successful_categories = len(category_results)
            total_score = sum(result.get("final_score", 0.0) for result in category_results.values())
            print(f" Resumed from checkpoint: {successful_categories} categories completed")

    # Evaluate remaining categories one by one
    for i, category_name in enumerate(remaining_categories, 1):
        current_index = len(all_category_names) - len(remaining_categories) + i
        num_subjects = len(MMLU_CATEGORIES[category_name])
        print(f"\n [{current_index}/{len(all_category_names)}] Evaluating category: {category_name} (contains {num_subjects} subjects)")
        print("-" * 60)

        try:
            # Get validation and test data for all subjects in this category (merged)
            val_data = loader.get_category_data(category_name, 'validation')
            test_data = loader.get_category_data(category_name, 'test')

            if not val_data:
                print(f" Category {category_name} has empty validation data, skipping")
                category_result = {
                    "status": "failed",
                    "error": "Empty validation data",
                    "final_score": 0.0
                }
                mmlu_checkpoint_manager.save_task_result(method_name, category_name, category_result)
                continue

            print(f" Data overview: {len(val_data)} validation, {len(test_data)} test samples")

            # Build category description
            category_desc = f"MMLU {category_name.replace('_', ' ')} - Multiple choice questions covering {num_subjects} subjects in this category."
            print(f" Category description: {category_desc}")

            # Run evaluation based on method
            if method_name == 'apsf':
                category_config = config.copy()
                category_config['dataset'] = f"mmlu_{category_name}"
                # Read initial_prompt from config
                initial_prompt = category_config.get('initial_prompt', OPTIMIZATION_PARAMS.get('initial_prompt'))
                result = run_apsf_pipeline(category_desc, val_data, test_data, evaluator, category_config,
                                          args.feedback, step, initial_prompt=initial_prompt)
                result['status'] = 'success'
            else:
                category_config = config.copy()
                category_config['dataset'] = f"mmlu_{category_name}"
                result = run_baseline_method(method_name, category_desc, val_data, test_data, evaluator, category_config, step, resume=should_resume_experiment(args))

            # Save individual category result to checkpoint
            mmlu_checkpoint_manager.save_task_result(method_name, category_name, result)

            # Update overall results
            category_results[category_name] = result

            if result.get("status") == "success":
                score = result.get("final_score", 0.0)
                total_score += score
                successful_categories += 1
                print(f" Category complete: score {score:.4f}")
            else:
                print(f" Category failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            error_msg = f"Evaluation error: {str(e)}"
            print(f" {error_msg}")
            logging.error(f"Category {category_name} failed: {error_msg}")
            category_result = {
                "status": "failed",
                "error": error_msg,
                "final_score": 0.0
            }
            mmlu_checkpoint_manager.save_task_result(method_name, category_name, category_result)

    # Calculate final results
    average_score = total_score / successful_categories if successful_categories > 0 else 0.0

    final_results = {
        "method": method_name,
        "total_categories": len(all_category_names),
        "successful_categories": successful_categories,
        "failed_categories": len(all_category_names) - successful_categories,
        "average_score": average_score,
        "total_score": total_score,
        "category_results": category_results,
        "status": "success"
    }

    # Save final results
    mmlu_checkpoint_manager.finalize_experiment(method_name, final_results)

    # Print summary
    print("\n" + "=" * 80)
    print(f" MMLU Full Category Evaluation Complete - {method_name.upper()}")
    print("=" * 80)
    print(f" Overall Statistics:")
    print(f"   Total categories: {final_results['total_categories']}")
    print(f"   Successful categories: {final_results['successful_categories']}")
    print(f"   Failed categories: {final_results['failed_categories']}")
    print(f"   Average score: {final_results['average_score']:.4f}")
    print(f"   Total score: {final_results['total_score']:.4f}")

    print(f"\n Detailed Results by Category:")
    print("-" * 80)
    print(f"{'Category Name':<30} {'Subjects':<10} {'Score':<10} {'Status':<10}")
    print("-" * 80)

    for category_name, category_result in category_results.items():
        num_subjects = len(MMLU_CATEGORIES[category_name])
        if category_result.get("status") == "success":
            score = f"{category_result['final_score']:.4f}"
            status = " Success"
        else:
            score = "N/A"
            status = f" Failed"

        print(f"{category_name:<30} {num_subjects:<10} {score:<10} {status:<10}")

    print("=" * 80)

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run aPSF and baseline experiments.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=list(DATASET_CONFIG.keys()),
        help="The name of the dataset to run the experiment on."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            'apsf', 'opro', 'protegi', 'dspy', 'ape', 'empty_cot', 'grips', 'qwen3_direct',
            # Ablation experiments
            'apsf_nostructure', 'apsf_notone', 'apsf_noformat', 'apsf_noperspective',
            'apsf_nodap', 'apsf_randselect', 'apsf_feedback', 'apsf_smallarchitect',
            # MAB algorithm comparison
            'apsf_thompson', 'apsf_roundrobin', 'apsf_greedy',
            # Comparison experiments
            'apsf_transfer', 'apsf_worker_llm', 'apsf_vs_manual', 'apsf_stability'
        ],
        help="The optimization method to run."
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable feedback mechanism to improve prompt based on wrong answers in the first stage."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint if available."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Number of optimization steps (default: use config value)."
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Initial prompt to start optimization from (e.g., 'Let\\'s think step by step'). "
             "You can also use presets: 'cot', 'cot_zh', 'analytical', 'expert'. "
             "If not specified, aPSF will generate prompt from scratch."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    method_name = args.method

    logging.info(f"========== Starting Experiment: METHOD={method_name.upper()}, DATASET={dataset_name.upper()} ==========")

    if args.step:
        logging.info(f"Optimization steps: {args.step}")

    if args.resume:
        logging.info("Resume mode enabled")

    # Handle initial_prompt argument
    initial_prompt_arg = getattr(args, 'initial_prompt', None)
    if initial_prompt_arg:
        # Check if using preset
        if initial_prompt_arg in OPTIMIZATION_PARAMS.get('initial_prompt_presets', {}):
            initial_prompt = OPTIMIZATION_PARAMS['initial_prompt_presets'][initial_prompt_arg]
            logging.info(f"Using initial prompt preset '{initial_prompt_arg}': {initial_prompt}")
        else:
            initial_prompt = initial_prompt_arg
            logging.info(f"Using custom initial prompt: {initial_prompt}")
        # Override config setting
        OPTIMIZATION_PARAMS['initial_prompt'] = initial_prompt

    # Special handling for bbh_all dataset
    if dataset_name == "bbh_all":
        config = DATASET_CONFIG[dataset_name]
        results = run_bbh_all_tasks_evaluation(method_name, config, args, args.step)
    # Special handling for mmlu_all dataset
    elif dataset_name == "mmlu_all":
        config = DATASET_CONFIG[dataset_name]
        results = run_mmlu_all_subjects_evaluation(method_name, config, args, args.step)
    else:
        # Original single dataset processing logic
        checkpoint_manager = CheckpointManager()

        # Check if resume is needed
        if args.resume:
            checkpoint = checkpoint_manager.load_checkpoint(method_name, dataset_name)
            if checkpoint and checkpoint.get("status") == "completed":
                logging.info(" Found completed experiment results")
                results = checkpoint.get("final_results", {})
                save_results(method_name, dataset_name, results)
                logging.info(f"========== Experiment for {method_name.upper()} on {dataset_name.upper()} Already Completed ==========\n")
                exit(0)
            elif checkpoint:
                logging.info(" Found incomplete checkpoint, but step-level resume not supported for single task experiments")

        logging.info("Step 1: Loading data and evaluator...")
        config = DATASET_CONFIG[dataset_name]
        loader = get_loader(dataset_name)
        evaluator = get_evaluator(dataset_name)

        # Ensure validation and test sets do not overlap
        val_size = config["val_size"]
        val_data = loader.get_split(config["val_split"], num_samples=val_size, offset=0)

        test_size = config.get("test_size", None)
        if test_size == 0:
            test_data = []
        else:
            # The following datasets have validation/test already split in loader, no offset needed
            # - aime2025: AIME2025Loader already splits internally
            # - gpqa series: GPQALoader already splits validation and test internally
            if dataset_name == "aime2025" or dataset_name.startswith("gpqa"):
                test_data = loader.get_split(config["test_split"], num_samples=test_size, offset=0)
            else:
                # Other datasets: test set starts after validation set to avoid overlap
                test_data = loader.get_split(config["test_split"], num_samples=test_size, offset=val_size)

        if not val_data:
            logging.error(f"Validation data for dataset '{dataset_name}' is empty. aPSF pipeline requires validation examples to proceed.")
            exit(1)

        # Run method
        results = {}
        task_config = config.copy()
        task_config['dataset'] = dataset_name
        task_desc = "Analyze the given problem and provide the correct answer."

        if method_name == 'apsf':
            # Read initial_prompt from config
            initial_prompt = task_config.get('initial_prompt', OPTIMIZATION_PARAMS.get('initial_prompt'))
            results = run_apsf_pipeline(task_desc, val_data, test_data, evaluator, task_config,
                                       args.feedback, args.step, initial_prompt=initial_prompt)
        else:
            # All non-aPSF methods go through baseline router, internally handles empty_cot/opro/protegi/dspy/ape/grips/qwen3_direct
            results = run_baseline_method(method_name, task_desc, val_data, test_data, evaluator, task_config, args.step, resume=args.resume)

        # Save checkpoint
        if results:
            results['status'] = 'completed'
            checkpoint_manager.save_checkpoint(method_name, dataset_name, {
                "final_results": results,
                "status": "completed"
            })

    if results:
        save_results(method_name, dataset_name, results)

        # If bbh_all, print detailed task results
        if dataset_name == "bbh_all" and "task_results" in results:
            print("\n" + "="*80)
            print(f" BBH Multi-Task Evaluation Results - {method_name.upper()}")
            print("="*80)
            print(f" Overall Statistics:")
            print(f"   Total tasks: {results['total_tasks']}")
            print(f"   Successful tasks: {results['successful_tasks']}")
            print(f"   Average score: {results['average_score']:.4f}")

            print(f"\n Detailed Results by Task:")
            print("-" * 80)
            print(f"{'Task Name':<30} {'Score':<10} {'Status':<10}")
            print("-" * 80)

            for task_name, task_result in results['task_results'].items():
                if task_result.get("status") == "success":
                    score = f"{task_result['final_score']:.4f}"
                    status = " Success"
                else:
                    score = "N/A"
                    status = f" {task_result.get('error', 'Failed')}"

                print(f"{task_name:<30} {score:<10} {status:<10}")

            print("="*80)

        # If mmlu_all, print detailed category results
        if dataset_name == "mmlu_all" and "category_results" in results:
            from .config import MMLU_CATEGORIES
            print("\n" + "="*80)
            print(f" MMLU Multi-Category Evaluation Results - {method_name.upper()}")
            print("="*80)
            print(f" Overall Statistics:")
            print(f"   Total categories: {results['total_categories']}")
            print(f"   Successful categories: {results['successful_categories']}")
            print(f"   Average score: {results['average_score']:.4f}")

            print(f"\n Detailed Results by Category:")
            print("-" * 80)
            print(f"{'Category Name':<30} {'Subjects':<10} {'Score':<10} {'Status':<10}")
            print("-" * 80)

            for category_name, category_result in results['category_results'].items():
                num_subjects = len(MMLU_CATEGORIES.get(category_name, []))
                if category_result.get("status") == "success":
                    score = f"{category_result['final_score']:.4f}"
                    status = " Success"
                else:
                    score = "N/A"
                    status = f" {category_result.get('error', 'Failed')}"

                print(f"{category_name:<30} {num_subjects:<10} {score:<10} {status:<10}")

            print("="*80)

    logging.info(f"========== Experiment for {method_name.upper()} on {dataset_name.upper()} Finished ==========\n") 
