from datasets import load_dataset
from .base_loader import BaseLoader

class HumanEvalLoader(BaseLoader):
    """加载 HumanEval 数据集。"""
    def _load_data(self):
        dataset = load_dataset('openai_humaneval')
        # HumanEval 只有一个 'test' 切片
        self.data = {
            "test": list(dataset['test'])
        } 