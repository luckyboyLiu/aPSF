import argparse
import json
import os
import logging
from datetime import datetime

from config import DATASET_CONFIG, OPTIMIZATION_PARAMS, RESULTS_DIR
from data_loader import get_loader
from evaluation import get_evaluator
from optimization import Architect, Optimizer
from llm_apis import get_llm
from baselines import run_opro, run_protegi, run_dspy
# from baselines import run_opro # 导入基线

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_apsf_pipeline(dataset_name: str, val_data, test_data, evaluator, task_desc: str) -> dict:
    """
    运行完整的 aPSF 优化流程和评估。
    """
    logging.info("Step 2: Discovering prompt structure with Architect...")
    architect = Architect(architect_llm_id="architect")
    
    example_input = json.dumps(val_data[0], indent=2)
    prompt_struct = architect.discover_structure(task_desc, example_input)
    logging.info(f"Discovered structure: {prompt_struct}")

    logging.info("Step 3: Optimizing prompt content with Optimizer...")
    optimizer = Optimizer(
        prompt_struct=prompt_struct,
        eval_data=val_data,
        evaluator=evaluator
    )

    max_steps = OPTIMIZATION_PARAMS.get("total_optimization_steps", 100)
    for step in range(max_steps // 10): # 为加速演示，减少步数
        optimizer.step()
    
    optimized_prompt = optimizer.prompt_struct
    logging.info(f"Optimization finished. Final structure: {optimized_prompt.render()}")

    logging.info("Step 4: Performing final evaluation on test set...")
    worker_llm = get_llm("worker")
    predictions = []
    for item in test_data:
        prompt = optimized_prompt.render().format(input=item.get('question') or item.get('prompt', ''))
        prediction = worker_llm.generate(prompt)
        predictions.append(prediction)

    final_score = evaluator.evaluate(predictions, test_data)
    logging.info(f"Final aPSF score on test set: {final_score}")

    return {
        "final_score": final_score,
        "optimized_prompt": optimized_prompt.factors,
    }

def save_results(method_name: str, dataset_name: str, results_data: dict):
    """一个统一的函数，用于将实验结果保存到文件中。"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"results_{method_name}_{dataset_name}_{timestamp}.json")
    
    # 构造完整的保存对象
    output = {
        "dataset": dataset_name,
        "method": method_name.upper(),
        **results_data
    }

    with open(result_file, 'w') as f:
        json.dump(output, f, indent=4)
        
    logging.info(f"Results saved to {result_file}")


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
        choices=['apsf', 'opro', 'protegi', 'dspy'],
        help="The optimization method to run."
    )
    args = parser.parse_args()
    
    dataset_name = args.dataset
    method_name = args.method
    
    logging.info(f"========== Starting Experiment: METHOD={method_name.upper()}, DATASET={dataset_name.upper()} ==========")

    # --- 通用设置 ---
    logging.info("Step 1: Loading data and evaluator...")
    config = DATASET_CONFIG[dataset_name]
    loader = get_loader(dataset_name)
    evaluator = get_evaluator(dataset_name)
    val_data = loader.get_split(config["val_split"], num_samples=config["val_size"])
    test_data = loader.get_split(config["test_split"])
    task_desc = f"This is a {dataset_name} task. The goal is to produce a correct answer based on the input."
    logging.info(f"Loaded {len(val_data)} samples for validation and {len(test_data)} for final testing.")

    results = {}

    if method_name == 'apsf':
        results = run_apsf_pipeline(dataset_name, val_data, test_data, evaluator, task_desc)
    
    elif method_name == 'opro':
        worker_llm = get_llm("worker")
        architect_llm = get_llm("architect")
        opro_results = run_opro(task_desc, val_data, test_data, evaluator, worker_llm, architect_llm)
        results = {"final_score": opro_results["score"], "optimized_prompt": opro_results["prompt"]}

    elif method_name == 'protegi':
        worker_llm = get_llm("worker")
        architect_llm = get_llm("architect")
        protegi_results = run_protegi(task_desc, val_data, test_data, evaluator, worker_llm, architect_llm)
        results = {"final_score": protegi_results["score"], "optimized_prompt": protegi_results["prompt"]}

    elif method_name == 'dspy':
        dspy_results = run_dspy(dataset_name)
        results = {"final_score": dspy_results["score"], "optimized_prompt": dspy_results["prompt"]}

    if results:
        save_results(method_name, dataset_name, results)
        
    logging.info(f"========== Experiment for {method_name.upper()} on {dataset_name.upper()} Finished ==========\n") 