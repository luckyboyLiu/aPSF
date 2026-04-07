import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    """
    GSM8K evaluator - uses LLM for intelligent numerical answer extraction
    """

    def _get_target_answer(self, item: Dict[str, Any]) -> str:
        """Get target answer"""
        answer = item.get('answer', '')
        if isinstance(answer, str) and '####' in answer:
            return answer.split('####')[-1].strip()
        return str(answer).strip()

    def _extract_final_answer(self, text: str) -> str:
        """
        Extract numerical answer - return empty directly, force LLM extraction
        """
        return ""
    
    def _llm_judge_answer(self, prediction: str, target_answer: str, llm=None) -> bool:
        """
        Use LLM to directly judge if model response is correct

        Returns:
            bool: whether correct
        """
        if not prediction.strip() or not llm:
            return False

        # Build LLM judgment prompt
        judge_prompt = f"""You are a math teacher grading a student's answer.

STUDENT'S RESPONSE:
{prediction}

CORRECT ANSWER: {target_answer}

Is the student's final answer correct?
- Ignore units, formatting differences (e.g., 56 = $56 = 56.0 = 56.00)
- Focus on the final answer, not intermediate steps

Respond with only one word: YES or NO"""

        try:
            llm_response = llm.generate(judge_prompt).strip().upper()
            is_correct = 'YES' in llm_response
            print(f"     LLM judgment: {llm_response} -> {'Correct' if is_correct else 'Incorrect'}")
            return is_correct

        except Exception as e:
            print(f"     LLM judgment error: {e}")
            return False

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]], llm=None) -> Dict[str, Any]:
        """
        Evaluate prediction results - use LLM to directly judge correctness
        """
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0, "correct_samples": 0}

        eval_data = references
        correct_count = 0
        total_count = len(predictions)

        print(f"\n=== GSM8K Evaluator (LLM Direct Judgment Mode) ===")
        print(f"Total samples: {total_count}")
        print("="*60)

        detailed_results = []

        for i, prediction in enumerate(predictions):
            if i >= len(eval_data):
                break

            target_answer = self._get_target_answer(eval_data[i])

            # Use LLM to directly judge correctness
            if llm:
                is_correct = self._llm_judge_answer(prediction, target_answer, llm)
            else:
                print(f"     Warning: No LLM provided, cannot evaluate answers")
                is_correct = False

            if is_correct:
                correct_count += 1

            # Show details for first 10 samples
            if i < 10:
                status = "Correct" if is_correct else "Incorrect"
                print(f"\nSample {i+1} [{status}]")
                print(f"   Response: {prediction[:300]}{'...' if len(prediction) > 300 else ''}")
                print(f"   Target: {target_answer}")
                print("-" * 50)

            detailed_results.append({
                'sample_id': i,
                'prediction': prediction,
                'gold_answer': target_answer,
                'correct': is_correct
            })

        accuracy = correct_count / total_count if total_count > 0 else 0.0

        print(f"\n=== Final Evaluation Results ===")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return {
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "detailed_results": detailed_results
        }