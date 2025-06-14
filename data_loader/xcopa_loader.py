from datasets import load_dataset
from .base_loader import BaseLoader

class XCOPALoader(BaseLoader):
    """加载 XCOPA 数据集。"""
    def _load_data(self):
        # 我们以英语（en）为例
        dataset = load_dataset('xcopa', 'en')
        self.data = {
            "validation": list(dataset['validation']),
            "test": list(dataset['test'])
        } 