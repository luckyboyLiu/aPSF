import json
import os
from typing import List, Dict, Any
from .base_loader import BaseLoader

class AIME2025Loader(BaseLoader):
    """Loader for AIME2025 mathematics competition dataset"""
    
    def _load_data(self):
        """Load AIME2025 data from JSONL files"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"AIME2025 data path does not exist: {self.path}")
        
        all_data = []
        
        # Load all JSONL files
        jsonl_files = [f for f in os.listdir(self.path) if f.endswith('.jsonl')]
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {self.path}")
        
        print(f" Loading AIME2025 dataset...")
        
        for jsonl_file in sorted(jsonl_files):
            file_path = os.path.join(self.path, jsonl_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            item = json.loads(line)
                            processed_item = self._process_item(item, jsonl_file, line_num)
                            all_data.append(processed_item)
                        except json.JSONDecodeError as e:
                            print(f" Skipping {jsonl_file}:{line_num} - JSON parsing error: {e}")
                            continue
                            
                print(f" Loaded {len([item for item in all_data if item.get('source_file') == jsonl_file])} samples from {jsonl_file}")
                        
            except Exception as e:
                print(f" Error reading file {jsonl_file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No AIME2025 samples were successfully loaded")
        
        # AIME dataset is small, we split data into validation and test sets
        # Validation set: 6 samples; Test set: 24 samples
        total_samples = len(all_data)
        val_size = 6
        test_size = 24
        
        val_data = all_data[:val_size]
        test_data = all_data[val_size:val_size + test_size]
        
        self.data = {
            "train": val_data,  # AIME uses validation set as training set for prompt optimization
            "validation": val_data,
            "test": test_data
        }
        
        print(f" Successfully loaded AIME2025 dataset:")
        print(f"   Total samples: {total_samples}")
        print(f"   Validation set: {len(val_data)} samples (for prompt optimization)")
        print(f"   Test set: {len(test_data)} samples (for final evaluation)")
        print(f"   Source files: {', '.join(jsonl_files)}")
    
    def _process_item(self, item: Dict[str, Any], source_file: str, line_num: int) -> Dict[str, Any]:
        """Process single AIME data item, convert to unified format"""
        
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        
        if not question or not answer:
            raise ValueError(f"Question or answer is empty: question={question}, answer={answer}")
        
        # Try to convert answer to numerical value (AIME answers are usually integers)
        numerical_answer = self._extract_numerical_answer(answer)
        
        processed_item = {
            # AIME2025 standard fields
            "question": question,
            "answer": answer,

            # Compatibility fields (adapt to different evaluators)
            "prompt": question,
            "target": answer,
            "input": question,
            "output": answer,

            # Numerical answer (for mathematical evaluation)
            "numerical_answer": numerical_answer,

            # Metadata
            "source_file": source_file,
            "line_number": line_num,
            "dataset_type": "AIME2025"
        }
        
        return processed_item
    
    def _extract_numerical_answer(self, answer_str: str) -> float:
        """Extract numerical value from answer string

        AIME answer format is usually pure numbers, like "70", "588", "16", etc.
        """
        try:
            # Directly convert to number
            return float(answer_str.strip())
        except ValueError:
            # If contains non-numeric characters, try to extract numbers
            import re
            numbers = re.findall(r'-?\d+(?:\.\d+)?', answer_str)
            if numbers:
                try:
                    return float(numbers[0])  # Take first number
                except ValueError:
                    pass
            
            print(f" Unable to extract numerical value from answer: '{answer_str}'")
            return 0.0
    
    def get_sample_info(self) -> Dict[str, Any]:
        """Get detailed information about the dataset"""
        if not hasattr(self, 'data') or not self.data:
            return {}
        
        info = {
            "dataset_name": "AIME2025",
            "total_samples": len(self.data.get('train', [])) + len(self.data.get('test', [])),
            "validation_samples": len(self.data.get('validation', [])),
            "test_samples": len(self.data.get('test', [])),
            "source_files": [],
            "sample_questions": []
        }
        
        # Count source files
        all_samples = self.data.get('train', []) + self.data.get('test', [])
        source_files = set()
        for sample in all_samples:
            source_files.add(sample.get('source_file', 'unknown'))
        info["source_files"] = list(source_files)
        
        # Get first 3 questions as examples
        if all_samples:
            info["sample_questions"] = [
                {
                    "question": sample["question"][:100] + "..." if len(sample["question"]) > 100 else sample["question"],
                    "answer": sample["answer"]
                }
                for sample in all_samples[:3]
            ]
        
        return info
