import json
import os
import random
import re
from .base_loader import BaseLoader

class AQuALoader(BaseLoader):
    """Load AQuA mathematical reasoning dataset (JSON Lines format)."""""
    
    def _load_data(self):
        """Load AQuA data from local JSON file"""
        
        aqua_file = os.path.join(self.path, "AQuA.json")
        
        if not os.path.exists(aqua_file):
            raise FileNotFoundError(f"AQuA data file not found: {aqua_file}")
        
        # Read JSON lines format data
        all_data = []
        try:
            with open(aqua_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data_item = json.loads(line)
                            all_data.append(data_item)
                        except json.JSONDecodeError as e:
                            print(f"WARNING: Line {line_num} JSON parsing error: {e}")
                            continue
        except Exception as e:
            print(f" Error reading AQuA data: {e}")
            raise
        
        # Process data format
        processed_data = self._process_aqua_data(all_data)
        
        # Shuffle data but don't split, let run_experiments.py split dynamically based on config
        from ..config import DATA_SPLIT_CONFIG
        random.seed(DATA_SPLIT_CONFIG['random_seed'])  # For reproducibility
        random.shuffle(processed_data)
        
        # Put all data in both train and test, let caller sample as needed
        self.data = {
            "train": processed_data,
            "test": processed_data
        }
        
        print(f" Successfully loaded AQuA dataset: {len(processed_data)} samples total")
        print(f" First sample check: target='{processed_data[0].get('target', 'MISSING')}', correct='{processed_data[0].get('correct', 'MISSING')}'")
    
    def _process_aqua_data(self, raw_data):
        """Process AQuA data format"""
        processed_data = []
        
        for idx, item in enumerate(raw_data):
            try:
                question = item.get("question", "").strip()
                options = item.get("options", [])
                rationale = item.get("rationale", "").strip()
                correct_answer = item.get("correct", "").strip()
                
                # Validate data integrity
                if not question or not options or not correct_answer:
                    print(f"  Item {idx} data incomplete, skipping")
                    continue
                
                # Format options text
                options_text = "\n".join(options) if options else ""
                
                # Build complete question text (including options) - optimized for multiple choice
                full_question = f"{question}\n\n{options_text}\n\nPlease select the correct answer (A, B, C, D, or E):"
                
                # Clean correct answer - ensure it's a single letter
                clean_answer = self._clean_answer(correct_answer)
                
                if not clean_answer:
                    print(f"  Item {idx} answer format error: {correct_answer}, skipping")
                    continue
                
                # Build standardized input-output format
                processed_item = {
                    # AQuA original fields
                    "question": question,
                    "options": options,
                    "rationale": rationale,
                    "correct": clean_answer,  # Use cleaned answer

                    # Standardized fields (compatible with different evaluators) - critical fix
                    "prompt": full_question,
                    "input": full_question,
                    "target": clean_answer,      # Ensure target field is correct
                    "output": clean_answer,
                    "answer": clean_answer,      # Additional answer field

                    # Additional information fields
                    "question_type": "multiple_choice",
                    "domain": "mathematical_reasoning",
                    "options_list": options,
                    "explanation": rationale,
                    "full_text": full_question,
                    
                    # Mapping for answer matching
                    "answer_choices": self._extract_answer_choices(options),
                    "raw_correct": correct_answer  # Keep raw answer for debugging
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"  Error processing item {idx}: {e}")
                continue
        
        return processed_data
    
    def _clean_answer(self, answer):
        """Clean answer format, extract standard letter"""
        if not answer:
            return ""
        
        # Remove extra spaces and convert to uppercase
        cleaned = answer.strip().upper()
        
        # If answer is A), B) format, extract letter
        if len(cleaned) >= 2 and cleaned[1] == ')':
            letter = cleaned[0]
            if letter.isalpha():
                return letter
        
        # If answer is just a letter (supports all A-Z options)
        if len(cleaned) == 1 and cleaned.isalpha():
            return cleaned
        
        # Try to extract first letter from string
        match = re.search(r'[A-Za-z]', cleaned)
        if match:
            return match.group().upper()
        
        print(f"  Unable to clean answer format: {answer}")
        return ""
    
    def _extract_answer_choices(self, options):
        """Extract answer choice mapping from options (supports all A-Z options)"""
        choices = {}
        for option in options:
            # Extract option letter, e.g., extract "A" from "A)xxx"
            match = re.match(r'^([A-Za-z])\)', option.strip())
            if match:
                letter = match.group(1).upper()
                content = option[2:].strip()  # Remove "A)" part
                choices[letter] = content
        return choices
    
    def get_formatted_sample(self, sample):
        """
        Get formatted sample for display and debugging

        Args:
            sample: Data sample

        Returns:
            str: Formatted string
        """
        question = sample.get("question", "")
        options = sample.get("options", [])
        rationale = sample.get("rationale", "")
        correct = sample.get("correct", "")
        target = sample.get("target", "")
        
        formatted = f"Question: {question}\n\n"
        formatted += "Options:\n"
        for option in options:
            formatted += f"  {option}\n"
        formatted += f"\nCorrect Answer: {correct}\n"
        formatted += f"Target Field: {target}\n"
        formatted += f"\nExplanation: {rationale}"
        
        return formatted
    
    def validate_answer(self, predicted_answer, correct_answer):
        """
        Validate if answer is correct

        Args:
            predicted_answer: Predicted answer
            correct_answer: Correct answer

        Returns:
            bool: Whether correct
        """
        if not predicted_answer or not correct_answer:
            return False
        
        # Clean both answers
        pred_clean = self._clean_answer(str(predicted_answer))
        correct_clean = self._clean_answer(str(correct_answer))
        
        return pred_clean == correct_clean 