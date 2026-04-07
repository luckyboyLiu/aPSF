import json
import os
import glob
import random
from .base_loader import BaseLoader
from ..config import DATA_PATHS

class BBHLoader(BaseLoader):
    """Load Big-Bench Hard (BBH) dataset."""
    def _load_data(self):
        # Load all JSON files from local BBH data directory
        bbh_data_path = DATA_PATHS.get("bbh_hard", "aPSF/data/BIG-Bench-Hard-data")
        
        # Get all JSON files
        json_files = glob.glob(os.path.join(bbh_data_path, "*.json"))
        
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
        
        # Extract 5 samples per task
        all_data = []
        samples_per_task = 5  # Take 5 samples per task
        
        print(f"  Found {len(task_data_dict)} BBH tasks:")
        for task_name, task_examples in task_data_dict.items():
            if len(task_examples) < samples_per_task:
                print(f"    {task_name}: Only {len(task_examples)} samples, less than {samples_per_task}")
                selected_samples = task_examples
            else:
                # Randomly sample 5 examples
                random.shuffle(task_examples)
                selected_samples = task_examples[:samples_per_task]
            
            all_data.extend(selected_samples)
            print(f"   {task_name}: {len(selected_samples)} samples")
        
        # Shuffle all data
        random.shuffle(all_data)
        
        # Don't split test set, all data used for training/validation
        self.data = {
            "train": all_data,  # All data used for training
            "test": []  # Empty test set
        }
        
        print(f" Successfully loaded BBH dataset: {len(all_data)} samples from {len(task_data_dict)} tasks")
        print(f" Train/validation set: {len(self.data['train'])} samples")
        print(f" Test set: {len(self.data['test'])} samples (no test set)")
        
        # Display task distribution statistics
        train_tasks = {}
        for sample in self.data['train']:
            task = sample.get('task', 'unknown')
            train_tasks[task] = train_tasks.get(task, 0) + 1
        
        print(f" Task distribution statistics: {len(train_tasks)} tasks")
        for task, count in sorted(train_tasks.items()):
            print(f"    {task}: {count} samples")