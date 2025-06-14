from datasets import load_dataset
from .base_loader import BaseLoader

class BillSumLoader(BaseLoader):
    """加载 BillSum 数据集。"""
    def _load_data(self):
        dataset = load_dataset('billsum')
        self.data = {
            "train": list(dataset['train']),
            "test": list(dataset['test']),
            "ca_test": list(dataset['ca_test']) # 还有一个加州测试集
        } 