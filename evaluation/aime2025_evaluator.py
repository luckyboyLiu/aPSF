import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class AIME2025Evaluator(BaseEvaluator):
    """
    AIME2025 Evaluator - Specialized for handling AIME mathematics competition answer formats
    AIME answers are usually pure integers, range 0-999, no need for #### [number] format
    """

    def _get_target_answer(self, item: Dict[str, Any]) -> str:
        """Get target answer"""
        answer = item.get('answer', item.get('target', ''))
        return str(answer).strip()

    def _extract_final_answer(self, text: str) -> str:
        """
        Extract AIME answer from text
        AIME answer characteristics:
        1. Usually integers from 0-999
        2. May appear anywhere in the text
        3. Prioritize extracting explicitly marked answers
        """
        if not text:
            return ""
        
        text = str(text).strip()
        
        # Method 1: Look for common answer marking patterns
        answer_patterns = [
            r'answer(?:\s*is|:)?\s*(\d+)',
            r'the\s+answer\s+is\s+(\d+)',
            r'final\s+answer\s*:?\s*(\d+)',
            r'result\s*(?:is|:)?\s*(\d+)',
            r'therefore.*?(\d+)\s*$',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1)
                if self._is_valid_aime_answer(answer):
                    return answer
        
        # Method 2: GSM8K style #### number format
        gsm_pattern = re.search(r'####\s*(\d+)', text)
        if gsm_pattern:
            answer = gsm_pattern.group(1)
            if self._is_valid_aime_answer(answer):
                return answer
        
        # Method 3: Look for numbers at the end of sentences (usually the answer)
        end_number_patterns = [
            r'[.]\s*(\d+)\s*[.]?\s*$',
            r'is\s*(\d+)\s*[.]?\s*$',
            r'equals?\s*(\d+)\s*[.]?\s*$',
        ]
        
        for pattern in end_number_patterns:
            match = re.search(pattern, text)
            if match:
                answer = match.group(1)
                if self._is_valid_aime_answer(answer):
                    return answer
        
        # Method 4: Extract all numbers, select the most likely answer
        numbers = re.findall(r'\b(\d+)\b', text)
        
        if numbers:
            # Filter numbers within valid AIME answer range
            valid_numbers = [n for n in numbers if self._is_valid_aime_answer(n)]
            
            if valid_numbers:
                # Prefer the last valid number that appears (usually the final answer)
                return valid_numbers[-1]
        
        # Method 5: Fallback strategy - return the last number (even if it might be out of range)
        if numbers:
            return numbers[-1]
        
        return ""

    def _is_valid_aime_answer(self, answer_str: str) -> bool:
        """
        Check if it's a valid AIME answer
        AIME answer range is usually integers from 0-999
        """
        try:
            num = int(answer_str)
            return 0 <= num <= 999
        except (ValueError, TypeError):
            return False

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate AIME2025 prediction results
        """
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

        eval_data = references
        correct_count = 0
        total_count = len(predictions)
        
        print(f"\n=== AIME2025 Evaluation ===")
        print(f"Total samples: {total_count}")
        print(f"Evaluation criteria: Pure number answer matching (0-999 range)")
        
        detailed_results = []
        extracted_count = 0  # Count successfully extracted answers
        
        for i, prediction in enumerate(predictions):
            if i >= len(eval_data):
                break
                
            target_answer = self._get_target_answer(eval_data[i])
            predicted_answer = self._extract_final_answer(prediction)
            
            is_correct = False
            has_extraction = bool(predicted_answer)
            
            if has_extraction:
                extracted_count += 1
                
                try:
                    pred_num = int(predicted_answer)
                    target_num = int(target_answer)
                    if pred_num == target_num:
                        is_correct = True
                except (ValueError, TypeError):
                    # If conversion fails, try string matching
                    if predicted_answer.strip() == target_answer.strip():
                        is_correct = True
            
            if is_correct:
                correct_count += 1
            
            # Show detailed results (first 20 samples)
            if i < 20:
                status = "PASS" if is_correct else "FAIL"
                extraction_status = "Extracted" if has_extraction else "Not extracted"
                print(f"  Sample{i+1:3d} {status} Predicted: '{predicted_answer}' | Ground truth: '{target_answer}' [{extraction_status}]")

                # For the first few incorrect samples, show more information
                if not is_correct and i < 5:
                    print(f"    Question: {eval_data[i].get('question', '')[:100]}...")
                    print(f"    Full response: {prediction[:300]}...")
            
            detailed_results.append({
                'sample_id': i,
                'question': eval_data[i].get('question', ''),
                'prediction': prediction,
                'extracted_answer': predicted_answer,
                'gold_answer': target_answer,
                'correct': is_correct,
                'has_extraction': has_extraction
            })

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        extraction_rate = extracted_count / total_count if total_count > 0 else 0.0
        
        print(f"\n=== AIME2025 Final Results ===")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Answer extraction rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
        print(f"Successfully extracted: {extracted_count}/{total_count}")
        
        # AIME-specific statistics
        if detailed_results:
            # Count answer distribution
            predicted_answers = [r['extracted_answer'] for r in detailed_results if r['extracted_answer']]
            if predicted_answers:
                print(f"Predicted answer range: {min(predicted_answers)} - {max(predicted_answers)}")
        
        return {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "extracted_samples": extracted_count,
            "extraction_rate": extraction_rate,
            "detailed_results": detailed_results,
            "evaluation_type": "AIME2025_math"
        }
