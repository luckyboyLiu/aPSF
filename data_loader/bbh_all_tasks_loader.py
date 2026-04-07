import json
import os
import random
from typing import Dict, List, Any
from .base_loader import BaseLoader
from ..config import DATA_PATHS, BBH_ALL_TASKS, DATA_SPLIT_CONFIG

class BBHAllTasksLoader(BaseLoader):
    """
    Data loader for all BBH tasks, supports individual evaluation of each task
    """
    
    def __init__(self, path: str = None):
        """
        Initialize BBH all tasks loader
        Args:
            path: Data path, use default path if None
        """
        # Fix: Determine path first, then pass to parent class
        if path is None:
            path = DATA_PATHS.get("bbh_all", "data/BIG-Bench-Hard-data")
        
        self.task_data = {}  # Store data for each task
        super().__init__(path)  # Pass path to parent class
    
    def _load_data(self):
        """Load all BBH task data, store separately by task"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"BBH data path does not exist: {self.path}")
        
        print(f" Starting to load all BBH task data...")
        print(f" Data path: {self.path}")
        
        # Set random seed for reproducibility
        random.seed(DATA_SPLIT_CONFIG['random_seed'])
        
        self.task_data = {}
        successful_tasks = 0
        
        for task_name in BBH_ALL_TASKS:
            json_file = os.path.join(self.path, f"{task_name}.json")
            
            if not os.path.exists(json_file):
                print(f" Task file does not exist: {json_file}")
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                if 'examples' not in task_data:
                    print(f" Task {task_name} data format error: missing examples field")
                    continue
                
                # Convert data format
                examples = []
                for example in task_data['examples']:
                    processed_example = {
                        'input': example.get('input', ''),
                        'target': example.get('target', ''),
                        'task': task_name
                    }
                    examples.append(processed_example)
                
                # Shuffle data
                if DATA_SPLIT_CONFIG['shuffle_before_split']:
                    random.shuffle(examples)
                
                # Adaptive data split: validation set fixed at 50, test set is the rest
                val_size = 50
                total_available = len(examples)
                
                if total_available < val_size:
                    print(f" Task {task_name} insufficient samples: {len(examples)} < {val_size}")
                    # If total samples less than 50, adjust split ratio
                    val_size = max(1, total_available // 2)  # Keep at least 1 validation sample
                    test_size = total_available - val_size
                else:
                    # Normal case: validation set 50, test set is the rest
                    test_size = total_available - val_size
                
                validation_data = examples[:val_size]
                test_data = examples[val_size:val_size + test_size]
                
                self.task_data[task_name] = {
                    'validation': validation_data,
                    'test': test_data,
                    'all': examples
                }
                
                successful_tasks += 1
                print(f"{task_name}: validation {len(validation_data)} samples, test {len(test_data)} samples (total {total_available})")
                
            except Exception as e:
                print(f" Error loading task {task_name}: {e}")
                continue
        
        print(f" Successfully loaded {successful_tasks}/{len(BBH_ALL_TASKS)} BBH tasks")
        
        # Create a comprehensive data dict for compatibility with existing interface
        all_validation = []
        all_test = []
        
        for task_name, task_data in self.task_data.items():
            all_validation.extend(task_data['validation'])
            all_test.extend(task_data['test'])
        
        self.data = {
            'validation': all_validation,
            'test': all_test,
            'task_data': self.task_data  # Keep data grouped by task
        }
        
        print(f" Total: validation {len(all_validation)} samples, test {len(all_test)} samples")
    
    def get_split(self, split_name: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get data for specified split

        Args:
            split_name: Split name ('validation', 'test')
            num_samples: Number of samples to return
            offset: Starting position to get samples (default 0)
        """
        if self.data is None:
            self._load_data()
        
        if split_name in ['validation', 'test']:
            data = self.data.get(split_name, [])
        else:
            data = []
        
        # Apply offset and num_samples
        if offset > 0:
            data = data[offset:]
        
        if num_samples is not None and len(data) > num_samples:
            data = data[:num_samples]
        
        return data
    
    def get_task_data(self, task_name: str, split_name: str) -> List[Dict[str, Any]]:
        """Get data for specific task"""
        if self.data is None:
            self._load_data()
        
        if task_name not in self.task_data:
            return []
        
        return self.task_data[task_name].get(split_name, [])
    
    def get_all_task_names(self) -> List[str]:
        """Get all successfully loaded task names"""
        if self.data is None:
            self._load_data()
        
        return list(self.task_data.keys()) 