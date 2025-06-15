import math
import random
from typing import List, Dict, Any

from ..llm_apis import get_llm, BaseLLM
from ..evaluation import BaseEvaluator
from .prompt_object import PromptStructure
from ..config import OPTIMIZATION_PARAMS

class Optimizer:
    """
    实现了 aPSF 的核心优化循环，用于迭代地改进 PromptStructure 的内容。
    """
    def __init__(self,
                 prompt_struct: PromptStructure,
                 eval_data: List[Dict[str, Any]],
                 evaluator: BaseEvaluator,
                 architect_llm_id: str = "architect",
                 worker_llm_id: str = "worker"):
        
        self.prompt_struct = prompt_struct
        self.eval_data = eval_data
        self.evaluator = evaluator
        self.architect_llm = get_llm(architect_llm_id)
        self.worker_llm = get_llm(worker_llm_id)
        
        self.total_steps = 0
        self.ucb_c = OPTIMIZATION_PARAMS.get("ucb1_exploration_constant", 2.0)
        self.candidates_per_step = OPTIMIZATION_PARAMS.get("candidates_per_step", 5)
        
        self.dap_patience_M = OPTIMIZATION_PARAMS.get("dap_patience_M", 4)
        self.dap_improvement_delta = OPTIMIZATION_PARAMS.get("dap_improvement_delta", 0.005) # 0.5%

    def _select_factor_to_optimize(self) -> str:
        """使用 DAP-UCB 算法选择下一个要优化的、未被冻结的因子。"""
        active_factors = {
            name: stats for name, stats in self.prompt_struct.factor_stats.items()
            if not stats["is_frozen"]
        }

        if not active_factors:
            return None # 所有因子都被冻结

        # 在初始阶段，确保每个活动因子至少被选择一次
        unexplored_factors = [name for name, stats in active_factors.items() if stats["selections"] == 0]
        if unexplored_factors:
            return random.choice(unexplored_factors)

        best_factor = None
        max_ucb_score = -1

        for name, stats in active_factors.items():
            avg_score = stats["score"] / stats["selections"]
            exploration_bonus = self.ucb_c * math.sqrt(math.log(self.total_steps) / stats["selections"])
            ucb_score = avg_score + exploration_bonus
            
            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_factor = name
        
        return best_factor

    def _generate_candidates(self, factor_to_optimize: str) -> List[str]:
        """为选定的因子生成 N 个候选内容。"""
        current_prompt_render = self.prompt_struct.render()
        current_factor_content = self.prompt_struct.factors[factor_to_optimize]

        meta_prompt = f"""
You are a prompt engineering expert. You will be given a complete prompt that is structured into several factors. Your task is to rewrite ONE specific factor to improve the overall prompt's performance on its task.

The task is: {self.prompt_struct.task_description}

Here is the current full prompt:
--- FULL PROMPT ---
{current_prompt_render}
--- END FULL PROMPT ---

You must rewrite ONLY the factor named '{factor_to_optimize}'. The current content of this factor is:
--- CURRENT FACTOR CONTENT ---
{current_factor_content}
--- END CURRENT FACTOR CONTENT ---

Generate {self.candidates_per_step} diverse and creative alternative versions for the '{factor_to_optimize}' factor. Each new version should be a potential improvement.

Output each new version separated by '--- CANDIDATE ---'. Do not include the factor name in your output.
"""
        response = self.architect_llm.generate(meta_prompt)
        candidates = response.split("--- CANDIDATE ---")
        return [c.strip() for c in candidates if c.strip()]

    def _evaluate_candidate(self, factor_name: str, candidate_content: str) -> float:
        """使用 worker LLM 和评测集来评估单个候选内容的分数。"""
        # 创建一个临时的 PromptStructure 用于测试
        temp_struct = PromptStructure(self.prompt_struct.task_description, self.prompt_struct.factors.copy())
        temp_struct.update_factor(factor_name, candidate_content)
        
        predictions = []
        for item in self.eval_data:
            # 这是一个简化的 input 格式化，实际需要根据任务调整
            prompt = temp_struct.render().format(input=item.get('question') or item.get('prompt'))
            prediction = self.worker_llm.generate(prompt)
            predictions.append(prediction)
            
        # 假设评测器的第一个返回值就是主要指标
        metric_key = list(self.evaluator.evaluate([], [{}])[0].keys())[0]
        score = self.evaluator.evaluate(predictions, self.eval_data)[metric_key]
        return score

    def step(self):
        """执行 aPSF 的一个完整优化步骤，包含 DAP-UCB 逻辑。"""
        # 1. 选择因子
        factor_to_optimize = self._select_factor_to_optimize()
        
        if factor_to_optimize is None:
            print("All factors are frozen. Optimization complete.")
            return "COMPLETED"

        print(f"Step {self.total_steps + 1}: Optimizing factor '{factor_to_optimize}'...")

        # 2. 生成候选
        candidates = self._generate_candidates(factor_to_optimize)
        if not candidates:
            print("Warning: No candidates were generated. Skipping step.")
            self.total_steps += 1
            return

        # 3. 评估候选
        best_candidate_content = self.prompt_struct.factors[factor_to_optimize]
        best_score_this_step = -1

        for candidate in candidates:
            score = self._evaluate_candidate(factor_to_optimize, candidate)
            print(f"  - Candidate for '{factor_to_optimize}' got score: {score:.4f}")
            if score > best_score_this_step:
                best_score_this_step = score
                best_candidate_content = candidate

        # 4. 更新 DAP-UCB 统计数据
        stats = self.prompt_struct.factor_stats[factor_to_optimize]
        previous_best_score = stats["best_score"]

        # 检查是否有改进
        if best_score_this_step > previous_best_score:
            improvement = best_score_this_step - previous_best_score
            stats["max_improvement"] = max(stats["max_improvement"], improvement)
            stats["best_score"] = best_score_this_step
            stats["patience_counter"] = 0
            print(f"  -> Improvement found for '{factor_to_optimize}'. New best score: {best_score_this_step:.4f}")
        else:
            stats["patience_counter"] += 1
            print(f"  -> No improvement. Patience: {stats['patience_counter']}/{self.dap_patience_M}")

        # 更新 UCB1 基础统计
        stats["score"] += best_score_this_step
        stats["selections"] += 1
        self.total_steps += 1

        # 检查是否需要冻结
        if stats["patience_counter"] >= self.dap_patience_M and stats["max_improvement"] < self.dap_improvement_delta:
            stats["is_frozen"] = True
            print(f"!!! Factor '{factor_to_optimize}' has been frozen due to stagnation.")

        # 如果找到了更好的版本，则更新提示结构
        self.prompt_struct.update_factor(factor_to_optimize, best_candidate_content)
        print(f"-> Updated '{factor_to_optimize}' content (best score this step: {best_score_this_step:.4f}).\n") 