import json
import os
from .base_loader import BaseLoader

class GSMHardLoader(BaseLoader):
    """Load GSM-hard dataset (JSONL format)."""
    
    def _load_data(self):
        """Load GSM-hard data from local JSONL file"""

        # GSM-hard data file path
        jsonl_file = os.path.join(self.path, "gsmhardv2.jsonl")

        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"GSM-hard data file not found: {jsonl_file}")

        print(f"Loading GSM-hard data: {jsonl_file}")
        
        all_data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    processed_item = self._process_item(item, line_num)
                    all_data.append(processed_item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} JSON parsing failed: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Line {line_num} processing failed: {e}")
                    continue

        # Split train and test data (20% test, 80% train)
        split_idx = int(len(all_data) * 0.2)
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]

        self.data = {
            "train": train_data,
            "test": test_data
        }

        print(f"GSM-hard loading complete: {len(train_data)} train samples, {len(test_data)} test samples")
    
    def _process_item(self, item, line_num):
        """Process single data item"""

        # Get original fields
        input_text = item.get('input', '').strip()
        code = item.get('code', '').strip()
        target = item.get('target', 0)

        # Ensure target is numeric
        if isinstance(target, str):
            try:
                target = float(target)
            except ValueError:
                print(f"Warning: Line {line_num} target cannot be converted to numeric: {target}")
                target = 0

        # Extract numerical answer (for evaluation)
        numerical_answer = float(target) if target is not None else 0

        processed_item = {
            # GSM-hard original fields
            "input": input_text,
            "code": code,
            "target": target,

            # Compatibility fields (for different evaluators)
            "question": input_text,
            "answer": str(target),
            "solution": code,
            "prompt": input_text,
            "output": str(target),

            # Numerical answer (for math evaluation)
            "numerical_answer": numerical_answer,

            # Metadata
            "line_number": line_num
        }

        return processed_item 