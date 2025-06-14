from datasets import load_dataset
from .base_loader import BaseLoader

class SpiderLoader(BaseLoader):
    """加载 Spider Text-to-SQL 数据集。"""
    def _load_data(self):
        dataset = load_dataset('spider')
        self.data = {
            "train": list(dataset['train']),
            "validation": list(dataset['validation']) # spider的测试集是validation
        } 