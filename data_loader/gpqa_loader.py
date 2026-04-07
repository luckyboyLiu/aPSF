import csv
import os
import random
from typing import List, Dict, Any
from .base_loader import BaseLoader

class GPQALoader(BaseLoader):
    """
    Loader for GPQA (Google-Proof Q&A) Diamond dataset
    GPQA is a high-difficulty scientific Q&A dataset designed by experts, covering physics, chemistry, biology and other fields
    """
    
    def __init__(self, path: str = None, subset: str = "diamond", domain_filter: str = None, val_size: int = None):
        """
        Initialize GPQA loader
        
        Args:
            path: GPQA dataset root directory path
            subset: Which subset to use - "diamond", "main", "extended", or "experts"
            domain_filter: Optional domain filter - "Chemistry", "Physics", "Biology", or None for all
            val_size: Optional validation set size. If None, defaults to 30 for diamond, 50 for others
        """
        self.subset = subset
        self.domain_filter = domain_filter
        self.custom_val_size = val_size  # Store custom val_size
        super().__init__(path)  # This will call _load_data()
        
    def _load_data(self):
        """
        Load GPQA data and perform non-overlapping split
        """
        # Build file path
        data_file = os.path.join(self.path, f"gpqa_{self.subset}.csv")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"GPQA file not found: {data_file}")
        
        domain_info = f" - {self.domain_filter}" if self.domain_filter else " (all domains)"
        print(f" Loading GPQA {self.subset.upper()} subset{domain_info}...")
        
        # Load data
        gpqa_data = self._load_gpqa_data(data_file)
        
        # Filter by domain if specified
        if self.domain_filter:
            original_count = len(gpqa_data)
            gpqa_data = [item for item in gpqa_data if item.get('domain', '') == self.domain_filter]
            print(f" Domain filtering: {self.domain_filter} - {len(gpqa_data)}/{original_count} questions")
        
        if not gpqa_data:
            raise ValueError(f"GPQA {self.subset} has no valid data")
        
        print(f" Successfully loaded {len(gpqa_data)} samples")
        
        # Use unified random seed for data split
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'])
        
        # Use stratified sampling (by subdomain) to ensure validation and test sets cover all subdomain types
        from collections import defaultdict
        subdomain_groups = defaultdict(list)
        for item in gpqa_data:
            subdomain = item.get('subdomain', 'Unknown')
            subdomain_groups[subdomain].append(item)
        
        # Shuffle each subdomain group
        for subdomain in subdomain_groups:
            random.shuffle(subdomain_groups[subdomain])
        
        # Split each subdomain proportionally
        # Use custom val_size if provided, otherwise use defaults
        if self.custom_val_size is not None:
            val_size = self.custom_val_size
        else:
            val_size = 30 if self.subset == "diamond" else 50
        
        # Ensure val_size doesn't exceed total data
        val_size = min(val_size, len(gpqa_data))
        
        validation_data = []
        test_data = []
        
        for subdomain, items in subdomain_groups.items():
            # Calculate split point (proportional to overall val_size)
            subdomain_val_size = int(len(items) * val_size / len(gpqa_data))
            
            # Special handling: ensure validation and test sets each have at least 1 sample (if subdomain has >=2 questions)
            if len(items) >= 2:
                if subdomain_val_size == 0:
                    subdomain_val_size = 1  # At least 1 question for validation
                elif subdomain_val_size == len(items):
                    subdomain_val_size = len(items) - 1  # Leave at least 1 question for test
            else:
                # Subdomains with only 1 question: prioritize allocation to validation set to let aPSF learn more diverse subdomains
                # This improves validation set subdomain coverage
                subdomain_val_size = 1  # Allocate to validation set
            
            validation_data.extend(items[:subdomain_val_size])
            test_data.extend(items[subdomain_val_size:])
        
        # Shuffle to mix subdomains
        random.shuffle(validation_data)
        random.shuffle(test_data)
        
        self.data = {
            "train": [],  # GPQA has no training data
            "validation": validation_data,
            "test": test_data,
            "dev": validation_data  # Alias
        }
        
        print(f" Data split completed:")
        print(f"   - Validation set (prompt optimization): {len(validation_data)} samples")
        print(f"   - Test set (final evaluation): {len(test_data)} samples")
        
        # Show domain distribution
        domains = {}
        for item in gpqa_data:
            domain = item.get('domain', 'Unknown')
            domains[domain] = domains.get(domain, 0) + 1
        
        print(f"\n Domain distribution:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            print(f"   {domain}: {count} questions")
    
    def _load_gpqa_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load CSV data for GPQA
        
        Args:
            file_path: CSV file path
            
        Returns:
            List[Dict[str, Any]]: Data list
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    # Extract question and answers
                    question = row.get('Question', '').strip()
                    correct_answer = row.get('Correct Answer', '').strip()
                    incorrect_1 = row.get('Incorrect Answer 1', '').strip()
                    incorrect_2 = row.get('Incorrect Answer 2', '').strip()
                    incorrect_3 = row.get('Incorrect Answer 3', '').strip()
                    explanation = row.get('Explanation', '').strip()
                    
                    if not question or not correct_answer:
                        print(f"  Skipping incomplete row: {file_path}:{row_idx+1}")
                        continue
                    
                    # Randomly assign answers to options A, B, C, D
                    # This prevents position bias
                    all_answers = [
                        correct_answer,
                        incorrect_1,
                        incorrect_2,
                        incorrect_3
                    ]
                    
                    # Create a fixed random permutation based on row index
                    # This ensures reproducibility
                    temp_random = random.Random(42 + row_idx)
                    temp_random.shuffle(all_answers)
                    
                    # Find which option is correct
                    correct_letter = chr(65 + all_answers.index(correct_answer))  # A, B, C, or D
                    
                    # Construct options dictionary
                    options = {
                        'A': all_answers[0],
                        'B': all_answers[1],
                        'C': all_answers[2],
                        'D': all_answers[3]
                    }
                    
                    # Construct input text (multiple choice format)
                    input_text = f"Question: {question}\n\n"
                    input_text += f"A. {all_answers[0]}\n"
                    input_text += f"B. {all_answers[1]}\n"
                    input_text += f"C. {all_answers[2]}\n"
                    input_text += f"D. {all_answers[3]}\n\n"
                    input_text += "Please provide your reasoning and conclude with the answer in the format: Answer: (X), where X is one of A, B, C, or D.\n\n"
                    input_text += "Your response:"
                    
                    # Get domain information
                    domain = row.get('High-level domain', 'Unknown').strip()
                    subdomain = row.get('Subdomain', '').strip()
                    
                    # Create data item
                    item = {
                        # Standard fields
                        'input': input_text,
                        'question': question,
                        'options': options,
                        'answer': f"({correct_letter})",  # Format as (A)
                        'target': f"({correct_letter})",
                        
                        # Reference explanation (for analysis)
                        'explanation': explanation,
                        'reference_solution': explanation,
                        
                        # Domain information
                        'domain': domain,
                        'subdomain': subdomain,
                        
                        # Compatibility fields
                        'prompt': input_text,
                        'output': f"({correct_letter})",
                        'label': correct_letter,
                        'correct_answer': correct_letter,
                        
                        # Metadata
                        'task_type': 'multiple_choice',
                        'dataset': 'gpqa',
                        'subset': self.subset,
                        'id': f"gpqa_{self.subset}_{row_idx}"
                    }
                    
                    data.append(item)
                    
        except Exception as e:
            print(f" Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return data

