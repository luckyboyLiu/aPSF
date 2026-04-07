import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator
from ..llm_apis import get_llm  # New import, assumes get_llm is defined in llm_apis for getting optimized LLM

class GSMHardEvaluator(BaseEvaluator):
    """
    GSM-hard evaluator - supports code execution and numerical comparison
    """

    def __init__(self, llm_id: str = "architect"):  # New init parameter, use optimized LLM (like architect)
        self.extractor_llm = get_llm(llm_id)  # Initialize LLM instance for answer extraction

    def _get_target_answer(self, item: Dict[str, Any]) -> float:
        """Get target answer"""
        target = item.get('target', 0)
        if isinstance(target, str):
            try:
                return float(target)
            except ValueError:
                return 0.0
        return float(target) if target is not None else 0.0

    def _extract_final_answer(self, text: str) -> str:
        """Extract final numerical answer from model output (using optimized LLM for intelligent extraction)."""
        # Optimized LLM extraction prompt template, focused on answer extraction rather than recalculation
        extraction_prompt = f"""You are a specialized answer extractor. Please find and extract the final numerical answer from the following mathematical solution.

Important notes:
- Extract answer only from the given solution, absolutely do not recalculate
- Return only pure numbers, do not include any text, currency symbols, units, or punctuation
- If answer is an integer, return integer form; if decimal, keep decimal point

Mathematical solution:
{text}

Please carefully read the above solution, find the final answer and return only the number:"""

        try:
            # Call optimized LLM extraction
            llm_response = self.extractor_llm.generate(extraction_prompt)

            # Clean LLM response, extract pure numbers
            cleaned_response = llm_response.strip()

            # Use more precise regex to match numbers
            number_pattern = r'([+-]?\d+(?:\.\d+)?)'
            match = re.search(number_pattern, cleaned_response)

            if match:
                extracted = match.group(1)
                print(f" LLM extracted answer: '{cleaned_response}' -> number: '{extracted}'")
                return extracted
            else:
                print(f" No number found in LLM response: '{cleaned_response}'")

        except Exception as e:
            print(f" LLM extraction failed: {e}, fallback to regex extraction")

        # If LLM fails, use multiple regex patterns for fallback extraction
        fallback_patterns = [
            r'total of \$?([+-]?\d+(?:\.\d+)?)',  # "total of $123" or "total of 123"
            r'(?:answer|result)\s*(?:is|:)?\s*\$?([+-]?\d+(?:\.\d+)?)',
            r'(?:total|sum)\s*(?:is|:)?\s*\$?([+-]?\d+(?:\.\d+)?)',
            r'\$?([+-]?\d+(?:\.\d+)?)(?:\s*dollars?)?$',
            r'([+-]?\d+(?:\.\d+)?)(?=\s*$)',      # number at end of text
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1)
                print(f" Regex extracted number: '{extracted}' (pattern: {pattern})")
                return extracted

        # Finally try to extract all numbers from text, take the last one
        all_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if all_numbers:
            last_number = all_numbers[-1]
            print(f" Extracted last number: '{last_number}'")
            return last_number

        print(" Unable to extract any numbers")
        return ""
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer to float string (remove extra characters, ensure float format)."""
        try:
            # Convert to float and return string representation (keep .0 if needed)
            normalized = str(float(answer))
            print(f" Normalized number: '{normalized}'")
            return normalized
        except ValueError:
            print(" Normalization failed, return original")
            return answer

    def _execute_code_answer(self, text: str) -> float:
        """
        Try to execute Python code in text to get answer
        """
        try:
            # Find Python code blocks
            code_patterns = [
                r'```python\s*(.*?)\s*```',
                r'```\s*(.*?)\s*```',
                r'def solution\(\):(.*?)return result',
            ]

            for pattern in code_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for code in matches:
                    try:
                        # Execute code safely
                        local_vars = {}
                        exec(code.strip(), {}, local_vars)

                        # Look for possible answer variables
                        for var_name in ['result', 'answer', 'final_answer']:
                            if var_name in local_vars:
                                return float(local_vars[var_name])

                        # If there's a solution function, call it
                        if 'solution' in local_vars:
                            result = local_vars['solution']()
                            return float(result)

                    except Exception as e:
                        continue
            
        except Exception:
            pass
        
        return None

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate prediction results - supports numerical comparison and code execution
        """
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

        eval_data = references
        correct_count = 0
        total_count = len(predictions)
        
        print(f"\n=== GSM-hard Evaluation ===")
        print(f"Total samples: {total_count}")
        print(f"Evaluation criteria: Exact numerical match (error < 1e-6)")

        detailed_results = []
        valid_extraction_count = 0  # Count successful answer extractions
        
        for i, prediction in enumerate(predictions):
            if i >= len(eval_data):
                break
                
            target_answer = self._get_target_answer(eval_data[i])
            predicted_answer = self._extract_final_answer(prediction)
            
            is_correct = False
            has_extraction = False
            
            if predicted_answer:  # If answer was extracted
                has_extraction = True
                valid_extraction_count += 1
                
                try:
                    pred_num = float(predicted_answer)
                    if abs(pred_num - target_answer) < 1e-6:
                        is_correct = True
                except (ValueError, TypeError):
                    pass
            
            if is_correct:
                correct_count += 1
            
            # Show detailed results (enhanced logging)
            if i < 20:
                status = "PASS" if is_correct else "FAIL"
                extraction_status = "Extraction successful" if has_extraction else "Extraction failed"
                print(f"  Sample{i+1:3d} {status} Predicted: '{predicted_answer}' | Actual: '{target_answer}' [{extraction_status}]")
                print(f"    Original output snippet: {prediction[-100:]}...")  # Show output end to track extraction
                if not is_correct and i < 5:
                    print(f"    Full response: {prediction[:200]}...")
            
            detailed_results.append({
                'sample_id': i,
                'prediction': prediction,
                'extracted_answer': predicted_answer,
                'gold_answer': target_answer,
                'correct': is_correct,
                'has_extraction': has_extraction,
                'input': eval_data[i].get('input', ''),
                'code': eval_data[i].get('code', '')
            })

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        extraction_rate = valid_extraction_count / total_count if total_count > 0 else 0.0
        
        print(f"\n=== Final Results (GSM-hard) ===")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Answer extraction success rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
        print(f"Successfully extracted samples: {valid_extraction_count}/{total_count}")

        return {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "valid_extraction_count": valid_extraction_count,
            "extraction_rate": extraction_rate,
            "detailed_results": detailed_results
        } 