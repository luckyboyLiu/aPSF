import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator
from ..llm_apis import get_llm, BaseLLM

class AccuracyEvaluator(BaseEvaluator):
    """
    Accuracy evaluator that uses LLM for semantic understanding and answer extraction
    """

    def __init__(self, extractor_llm_id: str = "worker", use_llm_comparison: bool = True):
        """
        Initialize evaluator

        Args:
            extractor_llm_id: LLM ID for answer extraction, default use worker
            use_llm_comparison: Whether to use LLM for answer comparison, default enabled
        """
        try:
            self.extractor_llm: BaseLLM = get_llm(extractor_llm_id)
            self.use_llm_extraction = True
            self.use_llm_comparison = use_llm_comparison
            comparison_mode = "LLM semantic comparison" if use_llm_comparison else "rule-based comparison"
            print(f"Enable LLM semantic answer extraction (using {extractor_llm_id})")
            print(f"Answer matching mode: {comparison_mode}")
        except Exception as e:
            print(f"LLM answer extractor initialization failed, fallback to rule-based method: {e}")
            self.use_llm_extraction = False
            self.use_llm_comparison = False

    def _detect_task_type(self, response: str, reference: Dict[str, Any]) -> str:
        """Auto detect task type"""

        # Check if multiple choice, support multiple answer field names and all letter options (A-Z)
        answer_field = (reference.get('answer') or
                       reference.get('target') or
                       reference.get('label') or
                       reference.get('output'))

        if (answer_field and
            isinstance(answer_field, str) and
            re.match(r'^\([A-Za-z]\)$', str(answer_field).strip())):
            return "multiple_choice"

        # Check if numerical question (GSM8K style)
        if (answer_field and
            '####' in str(answer_field)):
            return "numerical"

        # Check if response has multiple choice indicators (support all letter options)
        if re.search(r'\([A-Za-z]\)', response):
            return "multiple_choice"

        # Default to text generation
        return "text_generation"

    def _llm_extract_answer(self, response: str, task_type: str, original_question: str = "", options: Dict = None) -> str:
        """Use LLM to directly extract answer - simplest method"""

        if task_type == "multiple_choice":
            # For MMLU: if options provided, include option info to help LLM match content to letter
            options_info = ""
            if options:
                options_info = "\n\nAvailable Options:\n"
                for key, value in sorted(options.items()):
                    options_info += f"{key}. {value}\n"
                options_info += "\nIMPORTANT: If the response only gives the option content (e.g., '0', '-1', 'True') without the letter, match it to the corresponding letter above.\n"

            # Use English prompt extract - improved version: explicitly require finding final answer
            extraction_prompt = f"""Extract the FINAL chosen answer option letter from the response below.

Response:
{response}
{options_info}
IMPORTANT INSTRUCTIONS:
- The response may discuss multiple options (A, B, C, etc.) - IGNORE these discussions
- Look for the FINAL CONCLUSION or FINAL ANSWER at the end
- Look for emphasized answers (e.g., **A**, **B**, etc.)
- Look for phrases like "the answer is", "the correct answer is", "therefore", "thus"
- Focus on the LAST part of the response where the conclusion is stated
- If the response gives option content instead of letter, match it using the Available Options above
- Output ONLY the final option letter (A, B, C, D, E, F, G, H, etc.), nothing else

Final Answer Letter:"""

        elif task_type == "numerical":
            extraction_prompt = f"""Extract the final numerical answer from the response below.

Response:
{response}

Output only the number, nothing else.

Answer:"""

        else:  # text_generation
            extraction_prompt = f"""Extract the core answer from the response below.

Response:
{response}

Requirements:
- For boolean judgments (Yes/No questions), output ONLY "Yes" or "No" (one word, no punctuation).
- For numerical answers, output only the number.
- For other answers, output only the core phrase or word, no explanation.

Core answer:"""
        
        try:
            extracted = self.extractor_llm.generate(extraction_prompt).strip()

            if task_type == "multiple_choice":
                # Simple processing: extract first letter and format (support A-Z all options)
                letter_match = re.search(r'([A-Za-z])', extracted)
                if letter_match:
                    return f"({letter_match.group(1).upper()})"

                # If extraction fails, use rule-based fallback
                print(f"LLM extraction failed: '{extracted}', use rule-based fallback")
                return self._rule_based_extract(response, task_type)

            elif task_type == "numerical":
                # Extract number
                num_match = re.search(r'([+-]?\d+(?:\.\d+)?)', extracted)
                if num_match:
                    return num_match.group(1)
                else:
                    return self._rule_based_extract(response, task_type)

            # Remove possible prompt prefix and markdown format
            extracted = re.sub(r'^(core answer|answer|Answer|Final answer|Core answer)\s*[:：]?\s*', '', extracted, flags=re.IGNORECASE)
            extracted = extracted.strip('`"\'')  # Remove backticks and quotes
            extracted = extracted.strip()

            # Special handling for boolean values (English)
            bool_match = re.search(r'\b(True|False|Yes|No)\b', extracted, re.IGNORECASE)
            if bool_match:
                return bool_match.group(1).capitalize()

            # Normalize common boolean synonyms to Yes/No
            bool_synonyms = {
                r'^\s*correct\b': 'Yes',
                r'^\s*incorrect\b': 'No',
                r'^\s*wrong\b': 'No',
                r'^\s*right\b': 'Yes',
            }
            for pattern, english in bool_synonyms.items():
                if re.search(pattern, extracted, re.IGNORECASE):
                    return english

            return extracted

        except Exception as e:
            print(f"LLM answer extraction failed: {e}")
            return self._rule_based_extract(response, task_type)

    def _rule_based_extract(self, response: str, task_type: str) -> str:
        """Enhanced rule-based answer extraction"""
        
        # For text_generation type, first try to extract common answer formats
        if task_type == "text_generation":
            # Try to extract boolean values
            bool_patterns = [
                r'(?:answer|result)\s*(?:is|:)\s*[`"\']*\s*(True|False|Yes|No)\b',
                r'(?:therefore|thus|so)[,，]?\s*(?:the\s+)?(?:answer\s+is\s+)?[`"\']*\s*(True|False|Yes|No)\b',
                r'\b(True|False|Yes|No)\s*[.。]?\s*$',  # Boolean values at end of sentence
            ]

            for pattern in bool_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).capitalize()

            # If no clear boolean value is found, try to extract the content of the last line
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            if lines:
                last_line = lines[-1]
                # Try again to find boolean values in the last line
                bool_match = re.search(r'\b(True|False|Yes|No)\b', last_line, re.IGNORECASE)
                if bool_match:
                    return bool_match.group(1).capitalize()
                # Return the cleaned last line
                return re.sub(r'^[`"\']+|[`"\']+$', '', last_line).strip()
        
        if task_type == "multiple_choice":
            # More strict multiple choice pattern matching - sorted by priority (support all A-Z options)
            patterns = [
                r'answer\s+is\s*\(([A-Za-z])\)',                    # "answer is (B)"
                r'answer:\s*\(([A-Za-z])\)',                        # "Answer: (B)"
                r'answer:\s*([A-Za-z])\b',                          # "Answer: B"
                r'correct\s+(?:answer|option)\s+is\s*\(([A-Za-z])\)',  # "correct answer is (B)"
                r'therefore.*?(?:answer|option)\s*is\s*\(([A-Za-z])\)',  # "Therefore, the answer is (B)"
                r'(?:^|\n)\s*\(([A-Za-z])\)\s*(?:\n|$)',           # Standalone line (B)
                r'\*\*\(([A-Za-z])\)[^*]*\*\*',                    # **(B) silver**
                r'\*\*([A-Za-z])\*\*',                             # **B**
                r'option\s+([A-Za-z])\b',                          # "option B"
                r'choice\s+([A-Za-z])\b',                          # "choice B"
                r'choice\s*([A-Za-z])',                            # "choice B"
                r'answer\s*is\s*([A-Za-z])',                       # "answer is B"
                r'answer\s*:\s*([A-Za-z])',                        # "answer: B"
            ]

            # Try matching in priority order
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Return the last match (final answer)
                    return f"({matches[-1].upper()})"

            # Last resort: find any single letter, but be cautious
            # Avoid false matches, only match in clear contexts
            context_patterns = [
                r'(?:answer|option|choice).*?([A-Za-z])',
                r'([A-Za-z])\..*?(?:is|correct)',
                r'\b([A-Za-z])\b(?=\s*$)',  # Single letter at end of line
            ]

            for pattern in context_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    return f"({matches[-1].upper()})"
            
        elif task_type == "numerical":
            # GSM8K style numerical extraction
            gsm_pattern = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', response)
            if gsm_pattern:
                return gsm_pattern.group(1).strip()

            # Find other numerical formats
            number_patterns = [
                r'answer\s*is\s*([+-]?\d+(?:\.\d+)?)',
                r'result\s*is\s*([+-]?\d+(?:\.\d+)?)',
                r'=\s*([+-]?\d+(?:\.\d+)?)(?:\s|$)',
                r'answer\s*is\s*([+-]?\d+(?:\.\d+)?)',
                r'answer\s*:\s*([+-]?\d+(?:\.\d+)?)',
            ]

            for pattern in number_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        # Final fallback: return the last line of the response or the entire response
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        if lines:
            return lines[-1]
        return response.strip()

    def _extract_answer_with_context(self, prediction: str, reference: Dict[str, Any]) -> str:
        """Answer extraction with context"""

        if self.use_llm_extraction:
            # Detect task type
            task_type = self._detect_task_type(prediction, reference)

            # Fix: use enhanced question extraction method
            original_question = self._extract_question_from_item(reference)

            # Extract option information (for MMLU and other multiple choice datasets)
            options = reference.get('options', None)

            # Intelligently extract answers, pass option information
            return self._llm_extract_answer(prediction, task_type, original_question, options)
        else:
            # Fallback to rule method
            task_type = self._detect_task_type(prediction, reference)
            return self._rule_based_extract(prediction, task_type)

    def _extract_answer(self, prediction: str) -> str:
        """Extract answer - compatibility method (old interface)"""
        if self.use_llm_extraction:
            # Simple task type detection (support all A-Z options)
            if re.search(r'\([A-Za-z]\)', prediction):
                return self._llm_extract_answer(prediction, "multiple_choice")
            elif '####' in prediction:
                return self._llm_extract_answer(prediction, "numerical")
            else:
                return self._llm_extract_answer(prediction, "text_generation")
        else:
            return self._rule_based_extract(prediction, "multiple_choice")

    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate prediction accuracy - using intelligent semantic answer extraction
        """
        correct = 0
        total = len(predictions)

        extraction_method = "LLM semantic extraction" if self.use_llm_extraction else "Rule-based extraction"
        print(f"\n=== {extraction_method} - Evaluating {total} samples ===")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Use intelligent extraction with context
            extracted_pred = self._extract_answer_with_context(pred, ref)

            # Fix: get target answer, support multiple field names
            gold_answer = str(ref.get('answer') or
                             ref.get('target') or
                             ref.get('label') or
                             ref.get('output') or '').strip()

            # Detect task type for comparison
            task_type = self._detect_task_type(pred, ref)

            # Standardized comparison
            is_correct = self._compare_answers(extracted_pred, gold_answer, task_type)

            if is_correct:
                correct += 1

            # Debug output for first 10 samples (increase sample count for debugging)
            if i < 10:
                status = "Correct" if is_correct else "Incorrect"
                print(f"\nSample {i+1} [{task_type}] {status}")
                print(f"   Original response: {pred[:200]}{'...' if len(pred) > 200 else ''}")
                print(f"   Extracted answer: '{extracted_pred}'")
                print(f"   Target answer: '{gold_answer}'")

                if self.use_llm_comparison:
                    print(f"   LLM semantic comparison: {'Same' if is_correct else 'Different'}")
                else:
                    pred_norm = self._normalize_answer(extracted_pred, task_type)
                    gold_norm = self._normalize_answer(gold_answer, task_type)
                    print(f"   Normalized comparison: '{pred_norm}' vs '{gold_norm}'")

                print(f"   Answer match: {'Correct' if is_correct else 'Incorrect'}")
                print("=" * 50)
        
        accuracy = correct / total if total > 0 else 0.0

        print(f"\n=== Final Evaluation Results ===")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Extraction method: {extraction_method}")

        return {"accuracy": accuracy}
    
    def _llm_compare_answers(self, pred: str, gold: str) -> bool:
        """Use LLM for semantic-level answer comparison"""

        comparison_prompt = f"""Determine if the following two answers express the same meaning.

Extracted answer: {pred}
Correct answer: {gold}

If the two answers express the same meaning (even if formatted differently), respond with "same".
If the two answers express different meanings, respond with "different".

Note:
- True/true/TRUE should all be considered the same
- False/false/FALSE should all be considered the same
- Yes/yes/YES should all be considered the same
- No/no/NO should all be considered the same
- Numbers 42 and 42.0 should be considered the same
- Options (A) and A should be considered the same

Result:"""

        try:
            result = self.extractor_llm.generate(comparison_prompt).strip()
            result_lower = result.lower()

            # Check if result contains expressions of "same"
            if any(keyword in result_lower for keyword in ['same', 'match', 'equal', 'correct', 'yes']):
                # But exclude negative expressions like "not same", "different", etc.
                if any(negative in result_lower for negative in ['not same', 'not match', 'not equal', 'different', 'no']):
                    return False
                return True
            return False

        except Exception as e:
            print(f"LLM answer comparison failed, fallback to rule-based comparison: {e}")
            # Fallback to rule-based comparison
            pred_norm = self._normalize_answer(pred, "text_generation")
            gold_norm = self._normalize_answer(gold, "text_generation")
            return pred_norm == gold_norm
    
    def _compare_answers(self, pred: str, gold: str, task_type: str) -> bool:
        """Compare if answers match"""

        # If LLM comparison is enabled, use LLM for semantic comparison
        if self.use_llm_comparison and hasattr(self, 'extractor_llm'):
            return self._llm_compare_answers(pred, gold)

        # Otherwise use rule-based comparison
        pred_norm = self._normalize_answer(pred, task_type)
        gold_norm = self._normalize_answer(gold, task_type)
        return pred_norm == gold_norm
    
    def _normalize_answer(self, answer: str, task_type: str) -> str:
        """Normalize answer format"""

        if task_type == "multiple_choice":
            # Strengthen multiple choice answer normalization (support all A-Z options)
            # Method 1: directly match bracket format
            bracket_match = re.search(r'\(([A-Za-z])\)', answer)
            if bracket_match:
                return bracket_match.group(1).upper()

            # Method 2: extract any letter
            letter_match = re.search(r'([A-Za-z])', answer)
            if letter_match:
                return letter_match.group(1).upper()

            # Method 3: if no letter found, return first character of original answer (if it's a letter)
            clean_answer = answer.upper().strip()
            if clean_answer and clean_answer[0].isalpha():
                return clean_answer[0]

            return clean_answer
        
        elif task_type == "numerical":
            # Handle GSM8K format
            if '####' in answer:
                answer = answer.split('####')[-1].strip()

            # Extract numbers
            num_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
            if num_match:
                try:
                    num_val = float(num_match.group(1))
                    if num_val.is_integer():
                        return str(int(num_val))
                    else:
                        return str(num_val)
                except:
                    return num_match.group(1).strip()

            return answer.strip()

        else:
            # Special handling for boolean values to ensure unified format
            answer_stripped = answer.strip()

            # Prioritize matching English boolean values
            bool_match = re.search(r'\b(True|False|Yes|No)\b', answer_stripped, re.IGNORECASE)
            if bool_match:
                return bool_match.group(1).capitalize()

            # Chinese boolean value mapping (consistent with extractor)
            zh_mappings = {
                r'^\s*Yes\s*[-—–]': 'Yes',  # "Yes——..." or "Yes—..."
                r'^\s*No\s*[-—–]': 'No',
                r'^\s*Yes\s*[.,]': 'Yes',  # "Yes." or "Yes,"
                r'^\s*No\s*[.,]': 'No',
                r'^\s*Yes\s*$': 'Yes',  # Only "Yes"
                r'^\s*No\s*$': 'No',  # Only "No"
                r'^\s*Correct\s*[-—–.,]?': 'Yes',
                r'^\s*Wrong\s*[-—–.,]?': 'No',
                r'^\s*Right\s*[-—–.,]?': 'Yes',
                r'^\s*Wrong\s*[-—–.,]?': 'No',
            }
            for pattern, english in zh_mappings.items():
                if re.search(pattern, answer_stripped):
                    return english

            return answer_stripped.lower()

    def debug_single_sample(self, prediction: str, reference: Dict[str, Any]) -> Dict[str, Any]:
        """Debug the extraction process for a single sample"""

        task_type = self._detect_task_type(prediction, reference)

        # Try two extraction methods
        llm_result = None
        rule_result = None

        if self.use_llm_extraction:
            # Fix: get original question information, support multiple field names
            original_question = (reference.get('question') or
                               reference.get('prompt') or
                               reference.get('input') or
                               reference.get('text') or
                               str(reference.get('examples', [{}])[0].get('input', '')) if reference.get('examples') else '')
            llm_result = self._llm_extract_answer(prediction, task_type, original_question)

        rule_result = self._rule_based_extract(prediction, task_type)

        # Fix: support multiple answer field names
        gold_answer = str(reference.get('answer') or
                         reference.get('target') or
                         reference.get('label') or
                         reference.get('output') or '').strip()

        return {
            "task_type": task_type,
            "original_response": prediction,
            "llm_extracted": llm_result,
            "rule_extracted": rule_result,
            "gold_answer": gold_answer,
            "llm_correct": self._compare_answers(llm_result, gold_answer, task_type) if llm_result else None,
            "rule_correct": self._compare_answers(rule_result, gold_answer, task_type) if rule_result else None,
        }

    def evaluate_single_prediction(self, prediction: str, reference: Dict[str, Any], show_detail: bool = False) -> Dict[str, Any]:
        """Evaluate single prediction result - unified interface for baseline calls"""
        extracted_pred = self._extract_answer_with_context(prediction, reference)

        # Fix: support multiple answer field names
        gold_answer = str(reference.get('answer') or
                         reference.get('target') or
                         reference.get('label') or
                         reference.get('output') or '').strip()

        task_type = self._detect_task_type(prediction, reference)
        is_correct = self._compare_answers(extracted_pred, gold_answer, task_type)
        
        result = {
            "is_correct": is_correct,
            "extracted_answer": extracted_pred,
            "target_answer": gold_answer,
            "task_type": task_type,
            "prediction": prediction
        }
        
        if show_detail:
            pred_norm = self._normalize_answer(extracted_pred, task_type)
            gold_norm = self._normalize_answer(gold_answer, task_type)
            result.update({
                "extracted_normalized": pred_norm,
                "target_normalized": gold_norm
            })
        
        return result

    def evaluate_batch_with_details(self, predictions: List[str], references: List[Dict[str, Any]],
                                   show_progress: bool = True) -> Dict[str, Any]:
        """Batch evaluate prediction results, return detailed information - for baseline calls"""
        detailed_results = []
        correct_count = 0
        total_count = len(predictions)

        extraction_method = "LLM semantic extraction" if self.use_llm_extraction else "Rule-based extraction"

        if show_progress:
            print(f"\n=== {extraction_method} - Evaluating {total_count} samples ===")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            result = self.evaluate_single_prediction(pred, ref, show_detail=True)
            detailed_results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
            
            # Show detailed information for the first few samples
            if show_progress and i < 3:
                status = "Correct" if result["is_correct"] else "Incorrect"
                print(f"\nSample {i+1} [{result['task_type']}] {status}")
                print(f"   Original response: {pred[:150]}{'...' if len(pred) > 150 else ''}")
                print(f"   Extracted answer: '{result['extracted_answer']}'")
                print(f"   Target answer: '{result['target_answer']}'")
                print(f"   Normalized comparison: '{result['extracted_normalized']}' vs '{result['target_normalized']}'")
                print(f"   Answer match: {'Correct' if result['is_correct'] else 'Incorrect'}")
                print("=" * 50)
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        if show_progress:
            print(f"\n=== Final Evaluation Results ===")
            print(f"Correct: {correct_count}/{total_count}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Extraction method: {extraction_method}")
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "detailed_results": detailed_results
        }

    def evaluate_prompt_unified(self, prompt: str, worker_llm, eval_data: List[Dict[str, Any]],
                               method_name: str = "Baseline", show_samples: int = 2) -> float:
        """Unified prompt evaluation function - for all baseline calls"""
        predictions = []

        print(f"\nStarting {method_name} evaluation (total {len(eval_data)} samples)...")
        print(f"Prompt content: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        # Generate all predictions
        for i, item in enumerate(eval_data):
            # Fix: enhance data field extraction logic, support more BBH data formats
            question = self._extract_question_from_item(item)

            # Verify if question extraction was successful
            if not question.strip():
                print(f"Sample {i+1} question field is empty, data structure: {list(item.keys())}")
                # Try to output complete data structure for debugging
                print(f"Data content: {str(item)[:200]}...")
                continue

            # Format prompt
            if '{input}' in prompt:
                formatted_prompt = prompt.format(input=question)
            else:
                formatted_prompt = f"{prompt}\n\n{question}"

            try:
                prediction = worker_llm.generate(formatted_prompt)
                predictions.append(prediction)

                # Show basic information for the first few samples
                if i < show_samples:
                    print(f"\nSample {i+1}: {question[:80]}...")
                    print(f"{method_name} response: {prediction[:100]}...")

            except Exception as e:
                print(f"Sample {i+1} generation failed: {e}")
                predictions.append("")
        
        # Use unified evaluation
        result = self.evaluate_batch_with_details(
            predictions, eval_data, show_progress=True
        )
        
        return result["accuracy"]

    def _extract_question_from_item(self, item: Dict[str, Any]) -> str:
        """Enhanced question field extraction method - support multiple data formats"""

        # Common field names in BBH dataset
        question_fields = [
            'question', 'prompt', 'input', 'text', 'query',
            'problem', 'content', 'statement', 'task'
        ]

        # Try direct field matching
        for field in question_fields:
            if field in item and item[field]:
                return str(item[field]).strip()

        # Try nested structure (e.g., examples array)
        if 'examples' in item and item['examples']:
            for example in item['examples']:
                if isinstance(example, dict):
                    for field in question_fields:
                        if field in example and example[field]:
                            return str(example[field]).strip()

        # Try other possible nested structures
        nested_keys = ['data', 'item', 'sample', 'instance']
        for key in nested_keys:
            if key in item and isinstance(item[key], dict):
                for field in question_fields:
                    if field in item[key] and item[key][field]:
                        return str(item[key][field]).strip()

        # Final fallback: if there is a string value and length is reasonable, it might be question content
        for key, value in item.items():
            if isinstance(value, str) and 20 <= len(value) <= 1000:
                # Might be question content
                return value.strip()

        # If none found, return empty string
        print(f"Unable to extract question from data item: {list(item.keys())}")
        return ""

    def debug_data_structure(self, eval_data: List[Dict[str, Any]], num_samples: int = 3) -> None:
        """Debug data structure - help understand data format"""
        print(f"\nDebugging data structure (first {num_samples} samples)...")

        for i, item in enumerate(eval_data[:num_samples]):
            print(f"\n--- Sample {i+1} ---")
            print(f"All fields: {list(item.keys())}")

            for key, value in item.items():
                if isinstance(value, str):
                    preview = value[:100] + '...' if len(value) > 100 else value
                    print(f"  {key}: {repr(preview)}")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

        print("\n" + "="*50)