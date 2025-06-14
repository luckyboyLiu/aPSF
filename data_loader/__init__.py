from ..config import DATASET_CONFIG, DATA_PATHS
from .base_loader import BaseLoader
from .gsm8k_loader import GSM8KLoader
from .bbh_loader import BBHLoader
from .humaneval_loader import HumanEvalLoader
from .xcopa_loader import XCOPALoader
from .billsum_loader import BillSumLoader
from .spider_loader import SpiderLoader

# 将字符串名称映射到加载器类
LOADER_MAPPING = {
    "GSM8KLoader": GSM8KLoader,
    "BBHLoader": BBHLoader,
    "HumanEvalLoader": HumanEvalLoader,
    "XCOPALoader": XCOPALoader,
    "BillSumLoader": BillSumLoader,
    "SpiderLoader": SpiderLoader,
}

def get_loader(dataset_name: str) -> BaseLoader:
    """
    获取数据集加载器实例的工厂函数。

    Args:
        dataset_name (str): 配置文件中的数据集名称 (例如, "gsm8k")。

    Returns:
        BaseLoader: 相应数据集加载器的实例。
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"数据集 '{dataset_name}' 在 config.py 中未找到。")
    
    config = DATASET_CONFIG[dataset_name]
    loader_class_name = config.get("loader")
    
    if loader_class_name not in LOADER_MAPPING:
        raise ValueError(f"加载器类 '{loader_class_name}' 未在 LOADER_MAPPING 中定义。")
        
    loader_class = LOADER_MAPPING[loader_class_name]
    dataset_path = DATA_PATHS.get(dataset_name)
    
    return loader_class(path=dataset_path) 