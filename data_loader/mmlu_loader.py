import csv
import os
import glob
import random
from typing import List, Dict, Any, Tuple
from .base_loader import BaseLoader

class MMLULoader(BaseLoader):
    """MMLU dataset loader"""
    
    def __init__(self, path: str):
        """
        Initialize MMLU loader

        Args:
            path: MMLU dataset path
        """
        self.subjects = []
        super().__init__(path)
    
    def _load_data(self):
        """Load data from local MMLU files"""
        print(f"Loading MMLU data from: {self.path}")

        # Load all subjects
        self._load_subjects()

        # Load test data
        test_data = self._load_split_data("test")

        # If dev data exists, load dev data as validation set
        dev_data = self._load_split_data("dev")

        # If no dev data, split validation set from test data by subject
        if not dev_data and test_data:
            print("No dev data found, splitting validation set from test data by subject")
            dev_data, test_data = self._split_data_by_subject(test_data)

            print(f"Data split: validation set {len(dev_data)} samples, test set {len(test_data)} samples")

        self.data = {
            "train": [],  # MMLU typically has no training data
            "test": test_data,
            "dev": dev_data,
            "validation": dev_data  # alias
        }

        print(f"Loading complete: {len(self.subjects)} subjects")
        print(f"   - Validation set (for prompt optimization): {len(dev_data)} samples")
        print(f"   - Test set (for final evaluation): {len(test_data)} samples")
        print(f"   - Subject list: {', '.join(self.subjects[:5])}{'...' if len(self.subjects) > 5 else ''}")
    
    def _load_subjects(self):
        """Load all MMLU subjects"""
        test_dir = os.path.join(self.path, "test")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"MMLU test directory does not exist: {test_dir}")

        # Get all test files
        test_files = glob.glob(os.path.join(test_dir, "*_test.csv"))

        if not test_files:
            raise FileNotFoundError(f"No *_test.csv files found in {test_dir}")

        self.subjects = []
        for file_path in test_files:
            subject = os.path.basename(file_path).replace("_test.csv", "")
            self.subjects.append(subject)

        self.subjects.sort()  # Sort to ensure consistency
        print(f"Found {len(self.subjects)} MMLU subjects")
    
    def _load_split_data(self, split: str) -> List[Dict[str, Any]]:
        """
        Load data for specified split

        Args:
            split: Split name ("test", "dev", "val")

        Returns:
            List[Dict[str, Any]]: List of data
        """
        all_data = []

        for subject in self.subjects:
            file_path = os.path.join(self.path, split, f"{subject}_{split}.csv")

            if not os.path.exists(file_path):
                # If dev split, try val split
                if split == "dev":
                    file_path = os.path.join(self.path, "val", f"{subject}_val.csv")
                    if not os.path.exists(file_path):
                        continue
                else:
                    continue

            subject_data = self._load_subject_data(file_path, subject)
            all_data.extend(subject_data)

        return all_data
    
    def _split_data_by_subject(self, test_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split data by subject, taking 50 samples from each subject as validation set, remaining as test set

        Args:
            test_data: Test data from all subjects

        Returns:
            tuple: (dev_data, test_data) validation and test sets
        """
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'])  # Ensure reproducibility

        dev_data = []
        remaining_test_data = []

        # Group data by subject
        subject_data = {}
        for item in test_data:
            subject = item['subject']
            if subject not in subject_data:
                subject_data[subject] = []
            subject_data[subject].append(item)

        print(f"Split data by subject (50 validation samples per subject):")

        for subject, data in subject_data.items():
            # Shuffle data for this subject
            random.shuffle(data)

            total_samples = len(data)
            val_samples_per_subject = 50  # Fix 50 validation samples per subject

            if total_samples <= 50:
                # If samples <= 50, all as test set
                remaining_test_data.extend(data)
                print(f"   - {subject}: {total_samples} samples all as test set (less than 50 samples)")
            else:
                # Split validation and test set
                subject_dev = data[:val_samples_per_subject]
                subject_test = data[val_samples_per_subject:]

                dev_data.extend(subject_dev)
                remaining_test_data.extend(subject_test)

                print(f"   - {subject}: {len(subject_dev)} validation, {len(subject_test)} test")

        print(f"Total: {len(dev_data)} validation samples, {len(remaining_test_data)} test samples")

        return dev_data, remaining_test_data
    
    def _load_subject_data(self, file_path: str, subject: str) -> List[Dict[str, Any]]:
        """
        Load data for single subject

        Args:
            file_path: Path to CSV file
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
                        print(f"Skip malformed row: {file_path}:{row_idx+1}")
                        continue

                    question = row[0].strip()
                    option_a = row[1].strip()
                    option_b = row[2].strip()
                    option_c = row[3].strip()
                    option_d = row[4].strip()
                    answer = row[5].strip().upper()

                    # Ensure answer format is correct
                    if answer not in ['A', 'B', 'C', 'D']:
                        print(f"Abnormal answer format: {answer} in {file_path}:{row_idx+1}")
                        continue

                    # Construct options dict
                    options = {
                        'A': option_a,
                        'B': option_b,
                        'C': option_c,
                        'D': option_d
                    }

                    # Construct input text (similar to multiple choice format)
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
                        'answer': f"({answer})",  # Format as (A)
                        'target': f"({answer})",  # Target answer
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
            print(f"Error loading {file_path}: {e}")
            return []

        return data
    
    def get_subjects(self) -> List[str]:
        """Get all subject list"""
        return self.subjects.copy()

    def get_subject_data(self, subject: str, split: str = "test") -> List[Dict[str, Any]]:
        """
        Get data for specific subject

        Args:
            subject: Subject name
            split: Split name

        Returns:
            List[Dict[str, Any]]: Data for this subject
        """
        if split not in self.data:
            return []

        return [item for item in self.data[split] if item.get('subject') == subject] 