from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseLoader(ABC):
    """
    所有数据集加载器的抽象基类。
    它定义了一个用于加载、切分和抽样数据的标准接口。
    """
    def __init__(self, path: Optional[str] = None):
        """
        初始化加载器。

        Args:
            path (Optional[str]): 数据集的本地路径。如果为 None，则尝试从 Hub 加载。
        """
        self.path = path
        self.data = None
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """
        从源（本地或 Hub）加载数据集。
        子类必须实现此方法，将数据加载到 self.data 中。
        self.data 的格式通常是一个字典，键是 'train', 'test' 等，值是数据列表。
        """
        pass

    def get_split(self, split: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取指定的数据切片。

        Args:
            split (str): 要获取的切片名称 (例如, 'train', 'test', 'validation')。
            num_samples (Optional[int]): 要从此切片中抽样的样本数。如果为 None，则返回所有样本。

        Returns:
            List[Dict[str, Any]]: 一个字典列表，每个字典代表一个数据样本。
        """
        if self.data is None or split not in self.data:
            raise ValueError(f"切片 '{split}' 在数据集中未找到或数据未加载。")
        
        dataset_split = self.data[split]
        
        if num_samples and num_samples > 0 and num_samples < len(dataset_split):
            # 为了可复现性，这里可以添加随机种子
            # random.seed(42)
            # return random.sample(dataset_split, num_samples)
            # 但为了简单起见，我们只取前 n 个
            return dataset_split[:num_samples]
        
        return dataset_split 