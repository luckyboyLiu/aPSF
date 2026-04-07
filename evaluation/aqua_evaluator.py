import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class AQuAEvaluator(BaseEvaluator):
    """
    AQuA Evaluator - Uses LLM intelligent answer extraction, simplified logic
    """
    
    def __init__(self):
        super().__init__()
        self.metric_name = "accuracy"
    
    def _extract_choice_answer(self, text: str, item: Dict[str, Any] = None) -> str:
        """
        Extract multiple choice answer letter - return empty directly, force LLM extraction
        """
        return ""
    
    def _match_number_to_option(self, prediction: str, item: Dict[str, Any]) -> str:
        """
        Method retained for optimizer compatibility - now returns empty directly, forces LLM usage
        """
        return ""
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer format - extract letter and convert to uppercase
        """
        if not answer:
            return ""
        
        # Extract letter and normalize format (supports all A-Z options)
        letter_match = re.search(r'([A-Za-z])', answer)
        if letter_match:
            return letter_match.group(1).upper()
        
        return answer.upper().strip()
    
    def _compare_answers(self, predicted: str, target: str) -> bool:
        """
        Compare predicted answer and target answer
        """
        pred_normalized = self._normalize_answer(predicted)
        target_normalized = self._normalize_answer(target)
        
        return pred_normalized == target_normalized
    
    def _llm_extract_choice_answer(self, prediction: str, item: Dict[str, Any], llm=None) -> str:
        """
        Use LLM to extract multiple choice answer - improved English meta prompt
        """
        if not prediction.strip() or not llm:
            return ""
        
        # Get options information for context
        options = item.get('options', [])
        options_text = "\n".join(options) if options else ""
        
        # Build improved LLM extraction prompt
        extraction_prompt = f"""You are an expert at reading mathematical problem solutions and identifying the final answer choice.

Below is a response to a multiple choice question:

RESPONSE:
{prediction}

OPTIONS:
{options_text}

Your task: Read the response carefully and determine which option letter was selected as the final answer. Look for:
- Direct statements like "The answer is D"
- Mathematical calculations that match a specific option value
- Any clear indication of the chosen option

IMPORTANT: Output ONLY the single letter (A, B, C, D, E, F, G, etc.) that represents the final answer choice. Do not include any explanation or additional text.

Answer:"""

        try:
            # Use LLM to extract answer
            llm_response = llm.generate(extraction_prompt)
            extracted = llm_response.strip().upper()
            
            # Validate if extracted answer is a valid letter option
            if len(extracted) == 1 and extracted.isalpha():
                print(f"     LLM extraction successful: {extracted}")
                return extracted
            else:
                # Try to extract letter from response (supports all A-Z options)
                letter_match = re.search(r'([A-Za-z])', llm_response)
                if letter_match:
                    extracted = letter_match.group(1).upper()
                    print(f"     LLM extraction successful (regex assisted): {extracted}")
                    return extracted
                else:
                    print(f"     LLM extraction invalid: '{extracted}'")
                    return ""
            
        except Exception as e:
            print(f"     LLM extraction error: {e}")
            return ""

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
        """
        Evaluate prediction results - completely dependent on LLM-driven answer extraction
        """
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

        correct_count = 0
        total_count = len(predictions)
        
        print(f"\n=== AQuA Evaluator (Pure LLM Intelligent Extraction) ===")
        print(f"Total samples: {total_count}")
        print("="*60)
        
        for i, prediction in enumerate(predictions):
            if i >= len(references):
                break
                
            # Get target answer
            target_answer = references[i].get('correct', '').strip()
            
            # Use LLM to extract answer (LLM must be provided)
            if llm:
                predicted_answer = self._llm_extract_choice_answer(prediction, references[i], llm)
            else:
                print(f"     Warning: No LLM provided, cannot extract answer")
                predicted_answer = ""
            
            # Standardize comparison
            is_correct = self._compare_answers(predicted_answer, target_answer)
            
            if is_correct:
                correct_count += 1
            
            # Show details for first 10 samples
            if i < 10:
                status = "Yes" if is_correct else "No"
                print(f"\nSample {i+1} {status}")
                print(f"   Original response: {prediction[:200]}{'...' if len(prediction) > 200 else ''}")
                print(f"   Extracted answer: '{predicted_answer}'")
                print(f"   Target answer: '{target_answer}'")
                
                pred_norm = self._normalize_answer(predicted_answer)
                target_norm = self._normalize_answer(target_answer)
                print(f"   Standardized comparison: '{pred_norm}' vs '{target_norm}'")
                print(f"   Answer match: {'Correct' if is_correct else 'Wrong'}")
                print("-" * 50)
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        print(f"\n=== Final Evaluation Results ===")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count
        }

    def _match_numeric_answer_to_option(self, prediction: str, item: Dict[str, Any]) -> str:
        """
        Keep this method for backward compatibility, but now returns empty directly, forces LLM usage
        """
        return ""