import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class MultiArithEvaluator(BaseEvaluator):
    """
    MultiArith evaluator - mathematical problem solving evaluation
    supports numerical answer comparison and equation solving
    """

    def _get_target_answer(self, item: Dict[str, Any]) -> float:
        """Get target answer (numerical)"""
        if 'numerical_answer' in item and item['numerical_answer'] is not None:
            return float(item['numerical_answer'])
        
        answer = item.get('answer', '')
        try:
            return float(answer)
        except (ValueError, TypeError):
            return None

    def _extract_numerical_answer(self, text: str) -> float:
        """
        Extract numerical answer from prediction text (traditional regex method)
        Supports multiple formats:
        1. Direct number
        2. "answer is X"
        3. "= X"
        4. Last appearing number
        """
        if not text:
            return None
        
        text = str(text).strip()
        
        # 1. Look for explicit answer expressions
        answer_patterns = [
            r'answer\s*[:\s]?\s*([+-]?\d+(?:\.\d+)?)',
            r'result\s*[:\s]?\s*([+-]?\d+(?:\.\d+)?)',
            r'equals?\s*([+-]?\d+(?:\.\d+)?)',
            r'=\s*([+-]?\d+(?:\.\d+)?)',
            r'final\s+answer\s*:?\s*([+-]?\d+(?:\.\d+)?)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # 2. Extract all numbers, take the last one
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None

    def _llm_extract_numerical_answer(self, prediction: str, llm=None) -> str:
        """
        Use LLM to intelligently extract MultiArith numerical answers
        Specifically optimized for multi-step arithmetic reasoning problems
        """
        if not prediction.strip() or not llm:
            return ""
        
        # Build LLM extraction prompt specifically for MultiArith
        extraction_prompt = f"""You are an expert at reading mathematical problem solutions and extracting the final numerical answer.

Below is a response to a multi-step arithmetic word problem:

RESPONSE:
{prediction}

Your task: Read the response carefully and determine what the final numerical answer is. Look for:
- Numbers in formats like "#### 25" or "The answer is 25"
- The final calculated result after all arithmetic operations
- Numbers that represent the solution to the original question
- Consider the context: this is a multi-step arithmetic problem requiring basic operations (+, -, ×, ÷)

IMPORTANT: 
- Output ONLY the number (integer or decimal) that represents the final answer
- Do not include any units, explanations, formatting, or additional text
- The answer should be a simple number like: 42 or 15.5

Answer:"""

        try:
            # Use LLM to extract answer
            llm_response = llm.generate(extraction_prompt)
            extracted = llm_response.strip()

            # Extract numbers from response, with stricter validation
            import re
            # Prioritize matching complete numbers (including decimals)
            number_match = re.search(r'^([+-]?\d+(?:\.\d+)?)$', extracted) or \
                          re.search(r'([+-]?\d+(?:\.\d+)?)', extracted)
            
            if number_match:
                number = number_match.group(1)
                # Verify if it's a valid number
                try:
                    float(number)  # Verify it can be converted to float
                    print(f"     MultiArith LLM extraction successful: {number}")
                    return number
                except ValueError:
                    print(f"     MultiArith LLM extracted invalid value: '{number}'")
                    return ""
            else:
                print(f"     MultiArith LLM failed to extract number: '{extracted}'")
                return ""

        except Exception as e:
            print(f"     MultiArith LLM extraction error: {e}")
            return ""

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate prediction results
        """
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

        eval_data = references
        correct_count = 0
        total_count = len(predictions)
        
        print(f"\n=== MultiArith Mathematical Problem Evaluation ===")
        print(f"Total samples: {total_count}")
        print(f"Evaluation criteria: Accurate numerical answer matching (tolerance: 1e-6)")

        detailed_results = []
        valid_extraction_count = 0  # Count successful answer extractions
        
        for i, prediction in enumerate(predictions):
            if i >= len(eval_data):
                break
                
            target_answer = self._get_target_answer(eval_data[i])
            predicted_answer = self._extract_numerical_answer(prediction)
            
            is_correct = False
            extraction_success = predicted_answer is not None
            
            if extraction_success:
                valid_extraction_count += 1
                
                if target_answer is not None:
                    try:
                        # Use small tolerance for floating point comparison
                        if abs(predicted_answer - target_answer) < 1e-6:
                            is_correct = True
                    except (ValueError, TypeError):
                        pass
            
            if is_correct:
                correct_count += 1
            
            # Show detailed results (first 20 samples)
            if i < 20:
                status = "PASS" if is_correct else "FAIL"
                extraction_status = "Extraction successful" if extraction_success else "Extraction failed"
                print(f"  Sample{i+1:3d} {status} Predicted: {predicted_answer} | Actual: {target_answer} [{extraction_status}]")
                if not is_correct and i < 5:  # Show first 5 incorrect questions and answers
                    question = eval_data[i].get('question', '')[:100]
                    print(f"    Question: {question}...")
                    print(f"    Answer: {prediction[:150]}...")
            
            detailed_results.append({
                'sample_id': i,
                'question': eval_data[i].get('question', ''),
                'prediction': prediction,
                'extracted_answer': predicted_answer,
                'target_answer': target_answer,
                'correct': is_correct,
                'extraction_success': extraction_success
            })

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        extraction_rate = valid_extraction_count / total_count if total_count > 0 else 0.0
        
        print(f"\n=== MultiArith Evaluation Results ===")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Answer extraction success rate: {extraction_rate:.4f} ({extraction_rate*100:.2f}%)")
        print(f"Successfully extracted answer samples: {valid_extraction_count}/{total_count}")

        return {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "valid_extraction_count": valid_extraction_count,
            "extraction_rate": extraction_rate,
            "detailed_results": detailed_results
        } 