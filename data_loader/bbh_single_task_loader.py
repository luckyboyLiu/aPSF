import json
import os
import random
from typing import List, Dict, Any
from .base_loader import BaseLoader

class BBHSingleTaskLoader(BaseLoader):
    """
    Data loader for loading single BBH task with non-overlapping data splitting
    """
    
    def __init__(self, data_path: str, task_file: str = "reasoning_about_colored_objects.json"):
        """
        Initialize BBH single task loader

        Args:
            data_path: Path to data folder
            task_file: Name of the specific task file to load
        """
        self.task_file = task_file
        # Extract task name from task_file (remove .json suffix)
        self.task_name = os.path.splitext(task_file)[0]
        super().__init__(data_path)  # This calls _load_data()
        
    def _load_data(self):
        """
        Implement abstract method from BaseLoader, load data and perform non-overlapping split
        """
        file_path = os.path.join(self.path, self.task_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Task file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        
        # Convert data format
        examples = []
        if 'examples' in task_data:
            for example in task_data['examples']:
                processed_example = {
                    'input': example.get('input', ''),
                    'target': example.get('target', ''),
                    'task': self.task_name  # Dynamically set task name
                }
                examples.append(processed_example)

        print(f"Loaded {len(examples)} samples from {self.task_name}")
        
        # Use unified random seed for data splitting
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'])

        if DATA_SPLIT_CONFIG['shuffle_before_split']:
            random.shuffle(examples)

        # Create non-overlapping validation and test sets
        self.data = self._create_non_overlapping_splits(examples)
    
    def _create_non_overlapping_splits(self, examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create non-overlapping data splits
        """
        total_samples = len(examples)

        # Fix validation set at 50, remaining all as test set (if total not enough, validation takes all)
        val_size = min(50, total_samples)
        test_size = max(0, total_samples - val_size)

        # Create non-overlapping split
        validation_data = examples[:val_size]  # First val_size as validation
        test_data = examples[val_size:val_size + test_size]  # Next test_size as test

        print(f"Data split complete: validation {len(validation_data)}, test {len(test_data)}, non-overlapping")
        
        return {
            'validation': validation_data,
            'test': test_data,
            'examples': examples  # Keep original data for backward compatibility
        }
    
    def get_split(self, split_name: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get data for specified split

        Args:
            split_name: Name of split ('validation', 'test', 'examples')
            num_samples: Number of samples to return
            offset: Starting position for sampling (default: 0)
        """
        if self.data is None:
            self._load_data()

        # Support new split names
        if split_name == 'validation':
            data = self.data.get('validation', [])
        elif split_name == 'test':
            data = self.data.get('test', [])
        elif split_name == 'examples':
            # Backward compatibility: if requesting examples, determine validation or test based on num_samples
            if num_samples is not None:
                data = self.data.get('validation', [])
            else:
                data = self.data.get('test', [])
        else:
            data = []

        # Apply offset and num_samples
        if offset > 0:
            data = data[offset:]

        if num_samples is not None and len(data) > num_samples:
            data = data[:num_samples]

        return data