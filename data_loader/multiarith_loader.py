import json
import os
import random
from .base_loader import BaseLoader

class MultiArithLoader(BaseLoader):
    """Load MultiArith math problem dataset"""
    
    def _load_data(self):
        """Load MultiArith data from local JSON file"""

        json_file = os.path.join(self.path, "MultiArith.json")

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"MultiArith.json not found at {json_file}")

        print(f"Loading MultiArith data: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Process data
        all_data = self._process_data(raw_data)

        # Set random seed to ensure data split consistency
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'])
        random.shuffle(all_data)

        # Split train and test data (80/20 split)
        split_idx = int(len(all_data) * 0.2)
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]

        self.data = {
            "train": train_data,
            "test": test_data
        }

        print(f"Loading complete MultiArith: {len(train_data)} train samples, {len(test_data)} test samples")
    
    def _process_data(self, raw_data):
        """Process MultiArith data format"""
        processed_data = []

        for item in raw_data:
            question = item['sQuestion'].strip()
            # Take first solution as answer
            answer = item['lSolutions'][0] if item['lSolutions'] else None
            # Take first equation as solution process
            equation = item['lEquations'][0] if item['lEquations'] else ""

            processed_item = {
                # MultiArith original fields
                "question": question,
                "answer": str(answer) if answer is not None else "",
                "equation": equation,
                "iIndex": item['iIndex'],

                # Compatibility fields (for different evaluators)
                "prompt": question,
                "target": str(answer) if answer is not None else "",
                "input": question,
                "output": str(answer) if answer is not None else "",

                # Numerical answer (for math evaluation)
                "numerical_answer": float(answer) if answer is not None else None
            }

            processed_data.append(processed_item)

        return processed_data
    
    def get_validation_data(self, split="train", size=50):
        """Get validation dataset"""
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in data")

        data = self.data[split]
        return data[:min(size, len(data))]

    def get_test_data(self, split="test", size=None):
        """Get test dataset"""
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in data")

        data = self.data[split]
        if size is None:
            return data
        return data[:min(size, len(data))] 