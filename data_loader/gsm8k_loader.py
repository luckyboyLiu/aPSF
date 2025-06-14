from datasets import load_dataset
from .base_loader import BaseLoader

class GSM8KLoader(BaseLoader):
    """加载 GSM8K 数据集。"""
    def _load_data(self):
        # GSM8K 有两个子集: 'main' 和 'socratic'
        dataset = load_dataset('gsm8k', 'main')
        self.data = {
            "train": list(dataset['train']),
            "test": list(dataset['test'])
        } 