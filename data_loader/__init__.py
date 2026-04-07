import json
import os
import glob
import random
from typing import Dict, List, Any
from .base_loader import BaseLoader
from ..config import DATA_PATHS

class BBHMultiTaskLoader(BaseLoader):
    """Load all tasks from Big-Bench Hard (BBH) dataset, supports multi-task evaluation."""
    
    def __init__(self, path: str = None):
        """
        Initialize BBH multi-task loader
        Args:
            path: Data path, use default path if None
        """
        self.path = path or DATA_PATHS.get("bbh_hard", "data/BIG-Bench-Hard-data")
        super().__init__()
    
    def _load_data(self):
        """Load all BBH task data"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"BBH data path does not exist: {self.path}")
        
        # Get all JSON files
        json_files = glob.glob(os.path.join(self.path, "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.path}")
        
        task_data_dict = {}  # Store data grouped by task
        
        for json_file in json_files:
            # Extract task name (filename without .json extension)
            task_name = os.path.basename(json_file).replace('.json', '')
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                task_examples = []
                # Check data format and convert
                if 'examples' in task_data:
                    for example in task_data['examples']:
                        # Convert input to prompt, target to answer
                        processed_example = {
                            'prompt': example.get('input', ''),
                            'answer': example.get('target', ''),
                            'task': task_name  # Add task identifier
                        }
                        task_examples.append(processed_example)
                
                task_data_dict[task_name] = task_examples
                        
            except Exception as e:
                print(f"Error loading file {json_file}: {e}")
                continue
        
        # Extract samples for each task
        all_data = []
        samples_per_task = 10  # Take 10 samples per task
        
        print(f"  Found {len(task_data_dict)} BBH tasks:")
        for task_name, task_examples in task_data_dict.items():
            if len(task_examples) < samples_per_task:
                print(f"    {task_name}: Only {len(task_examples)} samples, less than {samples_per_task}")
                selected_samples = task_examples
            else:
                # Randomly sample data
                random.shuffle(task_examples)
                selected_samples = task_examples[:samples_per_task]
            
            all_data.extend(selected_samples)
            print(f"   {task_name}: {len(selected_samples)} samples")
        
        # Shuffle all data
        random.shuffle(all_data)
        
        # Split data: 30% for validation, 70% for testing
        split_idx = int(len(all_data) * 0.3)
        
        self.data = {
            "train": all_data[:split_idx],
            "test": all_data[split_idx:]
        }
        
        print(f" Successfully loaded BBH multi-task dataset: {len(all_data)} samples")
        print(f" Train/validation set: {len(self.data['train'])} samples")
        print(f" Test set: {len(self.data['test'])} samples")
        
        # Display task distribution statistics
        train_tasks = {}
        test_tasks = {}
        
        for sample in self.data['train']:
            task = sample.get('task', 'unknown')
            train_tasks[task] = train_tasks.get(task, 0) + 1
        
        for sample in self.data['test']:
            task = sample.get('task', 'unknown')
            test_tasks[task] = test_tasks.get(task, 0) + 1
        
        print(f" Training set task distribution: {len(train_tasks)} tasks")
        for task, count in sorted(train_tasks.items()):
            print(f"    {task}: {count} samples")

        print(f" Test set task distribution: {len(test_tasks)} tasks")
        for task, count in sorted(test_tasks.items()):
            print(f"    {task}: {count} samples")

from ..config import DATASET_CONFIG, DATA_PATHS
from .base_loader import BaseLoader
from .gsm8k_loader import GSM8KLoader
from .bbh_loader import BBHLoader
from .bbh_multi_task_loader import BBHMultiTaskLoader
from .aqua_loader import AQuALoader
from .humaneval_loader import HumanEvalLoader
from .bbh_single_task_loader import BBHSingleTaskLoader
from .multiarith_loader import MultiArithLoader
from .gsm_hard_loader import GSMHardLoader
from .mmlu_loader import MMLULoader
from .mmlu_single_subject_loader import MMLUSingleSubjectLoader
from .mmlu_all_subjects_loader import MMLUAllSubjectsLoader
from .bbh_all_tasks_loader import BBHAllTasksLoader
from .aime2025_loader import AIME2025Loader
from .competition_math_loader import CompetitionMathLoader
from .gpqa_loader import GPQALoader

# Map string names to loader classes
LOADER_MAPPING = {
    "GSM8KLoader": GSM8KLoader,
    "BBHLoader": BBHLoader,
    "BBHMultiTaskLoader": BBHMultiTaskLoader,
    "BBHSingleTaskLoader": BBHSingleTaskLoader,
    "AQuALoader": AQuALoader,
    "HumanEvalLoader": HumanEvalLoader,
    "GSMHardLoader": GSMHardLoader,
    "MultiArithLoader": MultiArithLoader,
    "MMLULoader": MMLULoader,
    "MMLUSingleSubjectLoader": MMLUSingleSubjectLoader,
    "MMLUAllSubjectsLoader": MMLUAllSubjectsLoader,
    "BBHAllTasksLoader": BBHAllTasksLoader,
    "AIME2025Loader": AIME2025Loader,
    "CompetitionMathLoader": CompetitionMathLoader,
    "GPQALoader": GPQALoader,
}

def get_loader(dataset_name: str) -> BaseLoader:
    """
    Factory function to get dataset loader instances.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.py.")
    
    config = DATASET_CONFIG[dataset_name]
    loader_class_name = config.get("loader")
    
    if loader_class_name not in LOADER_MAPPING:
        raise ValueError(f"Loader class '{loader_class_name}' not defined in LOADER_MAPPING.")
        
    loader_class = LOADER_MAPPING[loader_class_name]
    dataset_path = DATA_PATHS.get(dataset_name)
    
    if loader_class_name == "BBHSingleTaskLoader":
        task_file = config.get("task_file", "reasoning_about_colored_objects.json")
        # Fallback to BBH general data directory, ensure single tasks have data directory
        if not dataset_path:
            dataset_path = DATA_PATHS.get("bbh_all") or DATA_PATHS.get("bbh_hard") or "data/BIG-Bench-Hard-data"
        return BBHSingleTaskLoader(data_path=dataset_path, task_file=task_file)
    
    if loader_class_name == "MMLUSingleSubjectLoader":
        subject = config.get("subject")
        if not dataset_path:
            dataset_path = DATA_PATHS.get("mmlu") or "data/MMLU-data"
        return MMLUSingleSubjectLoader(data_path=dataset_path, subject=subject)
    
    # Support GPQALoader domain_filter and val_size parameters
    if loader_class_name == "GPQALoader":
        domain_filter = config.get("domain_filter", None)  # Get subject filter config
        val_size = config.get("val_size", None)  # Get validation set size config
        return GPQALoader(path=dataset_path, domain_filter=domain_filter, val_size=val_size)
    
    return loader_class(path=dataset_path)
# def get_loader(dataset_name: str) -> BaseLoader:
#     """
#     Factory function to get dataset loader instances.

#     Args:
#         dataset_name (str): Dataset name in config file (e.g., "gsm8k").

#     Returns:
#         BaseLoader: Instance of the corresponding dataset loader.
#     """
#     if dataset_name not in DATASET_CONFIG:
#         raise ValueError(f"Dataset '{dataset_name}' not found in config.py.")
    
#     config = DATASET_CONFIG[dataset_name]
#     loader_class_name = config.get("loader")
    
#     if loader_class_name not in LOADER_MAPPING:
#         raise ValueError(f"Loader class '{loader_class_name}' not defined in LOADER_MAPPING.")
        
#     loader_class = LOADER_MAPPING[loader_class_name]
#     dataset_path = DATA_PATHS.get(dataset_name)
    
#     if loader_class_name == "BBHSingleTaskLoader":
#         task_file = config.get("task_file", "reasoning_about_colored_objects.json")
#         return BBHSingleTaskLoader(data_path=dataset_path, task_file=task_file)
    
#     return loader_class(path=dataset_path) 