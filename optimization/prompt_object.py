from typing import List, Dict, Optional

class PromptStructure:
    """
    一个将提示分解为多个可独立优化因子的数据结构。
    """
    def __init__(self, task_description: str, factors: Optional[Dict[str, str]] = None):
        """
        初始化提示结构。

        Args:
            task_description (str): 对任务的简要描述。
            factors (Optional[Dict[str, str]]): 一个字典，键是因子名称 (例如 "Instruction")，
                                             值是该因子的初始内容。
        """
        self.task_description = task_description
        self.factors = factors if factors else {}
        # 为 UCB 和 DAP-UCB 算法初始化统计数据
        self.factor_stats = {
            name: self._create_initial_stats() for name in self.factors
        }

    def _create_initial_stats(self) -> Dict:
        """为单个因子创建初始统计信息字典。"""
        return {
            "selections": 0,          # (UCB1) 被选择次数
            "score": 0.0,             # (UCB1) 累积得分
            "best_score": -1.0,       # (DAP-UCB) 历史最佳得分
            "max_improvement": 0.0,   # (DAP-UCB) 最大提升幅度 Δ_k
            "patience_counter": 0,    # (DAP-UCB) 停滞计数器
            "is_frozen": False        # (DAP-UCB) 是否被冻结
        }

    def add_factor(self, name: str, content: str):
        """添加一个新的因子。"""
        if name not in self.factors:
            self.factors[name] = content
            self.factor_stats[name] = self._create_initial_stats()

    def update_factor(self, name: str, new_content: str):
        """更新一个已存在因子的内容。"""
        if name in self.factors:
            self.factors[name] = new_content
        else:
            raise ValueError(f"因子 '{name}' 未找到。")

    def render(self, factor_order: Optional[List[str]] = None) -> str:
        """
        将所有因子组合成一个完整的、可执行的提示字符串。

        Args:
            factor_order (Optional[List[str]]): 指定因子在最终提示中出现的顺序。
                                             如果为 None，则按默认顺序组合。

        Returns:
            str: 组合后的完整提示。
        """
        prompt = ""
        
        ordered_factors = factor_order if factor_order else self.factors.keys()
        
        for name in ordered_factors:
            if name in self.factors:
                # 添加标题以明确区分各部分
                prompt += f"--- {name.upper()} ---\n"
                prompt += self.factors[name]
                prompt += "\n\n"
        
        return prompt.strip()

    def get_factor_names(self) -> List[str]:
        """返回所有因子的名称列表。"""
        return list(self.factors.keys())

    def __str__(self):
        return f"PromptStructure(task='{self.task_description}', factors={list(self.factors.keys())})" 