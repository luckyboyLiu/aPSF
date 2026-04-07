import csv
import os
import random
from typing import Dict, List, Any
from .base_loader import BaseLoader
from ..config import DATA_PATHS, MMLU_ALL_SUBJECTS, MMLU_CATEGORIES, DATA_SPLIT_CONFIG

class MMLUAllSubjectsLoader(BaseLoader):
    """
    Data loader for all MMLU subjects, supports individual evaluation of each subject
    """
    
    def __init__(self, path: str = None):
        """
        Initialize MMLU all subjects loader
        Args:
            path: Data path, use default path if None
        """
        if path is None:
            path = DATA_PATHS.get("mmlu", "data/MMLU-data")
        
        self.subject_data = {}  # Store data for each subject
        super().__init__(path)
    
    def _load_data(self):
        """Load all MMLU subject data, store separately by subject"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"MMLU data path does not exist: {self.path}")
        
        print(f" Starting to load all MMLU subject data...")
        print(f" Data path: {self.path}")
        
        # Set random seed for reproducibility
        random.seed(DATA_SPLIT_CONFIG['random_seed'])
        
        self.subject_data = {}
        successful_subjects = 0
        
        for subject in MMLU_ALL_SUBJECTS:
            test_file = os.path.join(self.path, "test", f"{subject}_test.csv")
            
            if not os.path.exists(test_file):
                print(f" Subject file does not exist: {test_file}")
                continue
            
            try:
                # Load subject data
                subject_data_list = self._load_subject_data(test_file, subject)
                
                if not subject_data_list:
                    print(f" Subject {subject} data is empty")
                    continue
                
                # Shuffle data
                if DATA_SPLIT_CONFIG['shuffle_before_split']:
                    random.shuffle(subject_data_list)
                
                # Split data: validation set 50, test set is the rest
                val_size = 50
                total_available = len(subject_data_list)
                
                if total_available < val_size:
                    # If total samples less than 50, adjust split ratio
                    val_size = max(1, total_available // 2)
                    test_size = total_available - val_size
                else:
                    test_size = total_available - val_size
                
                validation_data = subject_data_list[:val_size]
                test_data = subject_data_list[val_size:val_size + test_size]
                
                self.subject_data[subject] = {
                    'validation': validation_data,
                    'test': test_data,
                    'all': subject_data_list
                }
                
                successful_subjects += 1
                print(f" {subject}: validation {len(validation_data)} samples, test {len(test_data)} samples (total {total_available})")
                
            except Exception as e:
                print(f" Error loading subject {subject}: {e}")
                continue
        
        print(f"\n Successfully loaded {successful_subjects}/{len(MMLU_ALL_SUBJECTS)} MMLU subjects")
        
        # Create a comprehensive data dict for compatibility with existing interface
        all_validation = []
        all_test = []
        
        for subject, subject_data in self.subject_data.items():
            all_validation.extend(subject_data['validation'])
            all_test.extend(subject_data['test'])
        
        self.data = {
            'validation': all_validation,
            'test': all_test,
            'subject_data': self.subject_data  # Keep data grouped by subject
        }
        
        print(f" Total: validation {len(all_validation)} samples, test {len(all_test)} samples")
    
    def _load_subject_data(self, file_path: str, subject: str) -> List[Dict[str, Any]]:
        """
        Load CSV data for a single subject

        Args:
            file_path: CSV file path
            subject: Subject name

        Returns:
            List[Dict[str, Any]]: Data list for this subject
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader):
                    if len(row) < 6:
                        continue
                    
                    question = row[0].strip()
                    option_a = row[1].strip()
                    option_b = row[2].strip()
                    option_c = row[3].strip()
                    option_d = row[4].strip()
                    answer = row[5].strip().upper()
                    
                    # Ensure answer format is correct
                    if answer not in ['A', 'B', 'C', 'D']:
                        continue
                    
                    # Construct options dictionary
                    options = {
                        'A': option_a,
                        'B': option_b,
                        'C': option_c,
                        'D': option_d
                    }
                    
                    # Construct input text (multiple choice format)
                    input_text = f"Question: {question}\n\n"
                    input_text += f"A. {option_a}\n"
                    input_text += f"B. {option_b}\n"
                    input_text += f"C. {option_c}\n"
                    input_text += f"D. {option_d}\n\n"
                    input_text += "Answer:"
                    
                    # Create data item
                    item = {
                        # Standard fields
                        'input': input_text,
                        'question': question,
                        'options': options,
                        'answer': f"({answer})",  # Format as (A) form
                        'target': f"({answer})",
                        'subject': subject,
                        
                        # Compatibility fields
                        'prompt': input_text,
                        'output': f"({answer})",
                        'label': answer,
                        'correct_answer': answer,
                        
                        # Metadata
                        'task_type': 'multiple_choice',
                        'dataset': 'mmlu',
                        'id': f"{subject}_{row_idx}"
                    }
                    
                    data.append(item)
                    
        except Exception as e:
            print(f" Error loading {file_path}: {e}")
            return []
        
        return data
    
    def get_split(self, split_name: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get data for specified split

        Args:
            split_name: Split name ('validation', 'test')
            num_samples: Number of samples to return
            offset: Starting position to get samples (default 0)
        """
        if self.data is None:
            self._load_data()
        
        if split_name in ['validation', 'test', 'dev']:
            data = self.data.get(split_name, [])
            if split_name == 'dev' and not data:
                data = self.data.get('validation', [])
        else:
            data = []
        
        # Apply offset and num_samples
        if offset > 0:
            data = data[offset:]
        
        if num_samples is not None and len(data) > num_samples:
            data = data[:num_samples]
        
        return data
    
    def get_subject_data(self, subject: str, split_name: str) -> List[Dict[str, Any]]:
        """Get data for specific subject"""
        if self.data is None:
            self._load_data()
        
        if subject not in self.subject_data:
            return []
        
        return self.subject_data[subject].get(split_name, [])
    
    def get_all_subject_names(self) -> List[str]:
        """Get all successfully loaded subject names"""
        if self.data is None:
            self._load_data()
        
        return list(self.subject_data.keys())
    
    def get_category_data(self, category: str, split_name: str) -> List[Dict[str, Any]]:
        """
        Get data for all subjects in a specific category (merge then split)

        Note: This method first merges all raw data from subjects in the category,
        then extracts a fixed number (50) as validation set, and the rest as test set.
        This way each category has only 50 training samples, fitting the few-shot setting.

        Args:
            category: Subject category name (e.g., 'Mathematics', 'Science', 'Computer_Science', etc.)
            split_name: Data split name ('validation' or 'test')

        Returns:
            Validation set (50 samples) or test set (remaining all) for this category
        """
        if self.data is None:
            self._load_data()
        
        if category not in MMLU_CATEGORIES:
            print(f" Category '{category}' does not exist")
            return []
        
        category_subjects = MMLU_CATEGORIES[category]
        
        # First merge all raw data from subjects in this category (using 'all' field)
        merged_all_data = []
        for subject in category_subjects:
            if subject in self.subject_data:
                subject_all_data = self.subject_data[subject].get('all', [])
                merged_all_data.extend(subject_all_data)
        
        # Already shuffled using seed during loading, no need to shuffle again
        # But for category-level shuffling, we shuffle once more
        random.seed(DATA_SPLIT_CONFIG['random_seed'])
        random.shuffle(merged_all_data)
        
        # Fixed split: first 50 as validation set, rest as test set
        val_size = 50
        if len(merged_all_data) < val_size:
            val_size = max(1, len(merged_all_data) // 2)
        
        if split_name == 'validation':
            return merged_all_data[:val_size]
        elif split_name == 'test':
            return merged_all_data[val_size:]
        else:
            return []
    
    def get_all_category_names(self) -> List[str]:
        """Get all subject category names"""
        return list(MMLU_CATEGORIES.keys())

