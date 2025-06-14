from datasets import load_dataset
from .base_loader import BaseLoader

class BBHLoader(BaseLoader):
    """加载 Big-Bench Hard (BBH) 数据集。"""
    def _load_data(self):
        # BBH 包含 23 个任务。加载脚本需要知道具体加载哪个任务。
        # 这里我们仅作一个示例，实际使用时可能需要动态传入任务名。
        # 此处我们加载 'causal_judgement' 任务作为代表。
        # 在实际的实验脚本中，我们会迭代所有 BBH 任务。
        dataset = load_dataset('lukaemon/bbh', 'causal_judgement')
        self.data = {
            # BBH 通常只有 test 切片
            "test": list(dataset['test'])
        } 