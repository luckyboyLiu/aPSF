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
# from baselines import run_opro # 导入基线

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single_experiment(dataset_name: str):
    """
    为一个指定的数据集运行完整的 aPSF 优化流程和评估。
    """
    logging.info(f"========== Starting Experiment for: {dataset_name.upper()} ==========")

    # 1. 加载数据和评测器
    logging.info("Step 1: Loading data and evaluator...")
    config = DATASET_CONFIG[dataset_name]
    loader = get_loader(dataset_name)
    evaluator = get_evaluator(dataset_name)

    # 从训练/验证集中划分出用于优化的验证集 D_val
    val_data = loader.get_split(config["val_split"], num_samples=config["val_size"])
    test_data = loader.get_split(config["test_split"])
    logging.info(f"Loaded {len(val_data)} samples for validation and {len(test_data)} for final testing.")

    # 2. 自动发现结构 (Architect)
    logging.info("Step 2: Discovering prompt structure with Architect...")
    architect = Architect(architect_llm_id="architect")
    
    # 使用第一个验证样本来帮助发现结构
    example_input = json.dumps(val_data[0], indent=2)
    task_description = f"This is a {dataset_name} task. The goal is to produce a correct answer."
    
    prompt_struct = architect.discover_structure(task_description, example_input)
    logging.info(f"Discovered structure: {prompt_struct}")

    # 3. 自动优化内容 (Optimizer)
    logging.info("Step 3: Optimizing prompt content with Optimizer...")
    optimizer = Optimizer(
        prompt_struct=prompt_struct,
        eval_data=val_data,
        evaluator=evaluator
    )

    max_steps = OPTIMIZATION_PARAMS.get("total_optimization_steps", 100)
    for step in range(max_steps):
        optimizer.step()
    
    optimized_prompt = optimizer.prompt_struct
    logging.info(f"Optimization finished. Final structure: {optimized_prompt.render()}")

    # 4. 在测试集上进行最终评估
    logging.info("Step 4: Performing final evaluation on test set...")
    worker_llm = get_llm("worker")
    predictions = []
    for item in test_data:
        # 这是一个简化的 input 格式化，实际需要根据任务调整
        prompt = optimized_prompt.render().format(input=item.get('question') or item.get('prompt'))
        prediction = worker_llm.generate(prompt)
        predictions.append(prediction)

    final_score = evaluator.evaluate(predictions, test_data)
    logging.info(f"Final score on test set: {final_score}")

    # 5. (可选) 运行基线进行比较
    # opro_score = run_opro(...)
    
    # 6. 保存结果
    results = {
        "dataset": dataset_name,
        "aPSF_score": final_score,
        "optimized_prompt_structure": optimized_prompt.factors,
        # "opro_score": opro_score,
    }
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"results_{dataset_name}_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Results saved to {result_file}")
    logging.info(f"========== Experiment for {dataset_name.upper()} Finished ==========\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run aPSF experiments.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=list(DATASET_CONFIG.keys()),
        help="The name of the dataset to run the experiment on."
    )
    args = parser.parse_args()
    
    run_single_experiment(args.dataset) 