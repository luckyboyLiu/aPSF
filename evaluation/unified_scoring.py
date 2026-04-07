"""
Unified scoring mechanism - Use LLM for intelligent answer extraction
"""
import re
import logging
from typing import Dict, Any, List
from ..llm_apis import BaseLLM
from .aqua_evaluator import AQuAEvaluator

class UnifiedScorer:
    """Unified answer extraction and scoring mechanism using LLM intelligent extraction"""
    
    def __init__(self, llm: BaseLLM, dataset_name: str = ""):
        self.llm = llm
        self.dataset_name = dataset_name.lower()
    
    def extract_and_score(self, prediction: str, item: Dict[str, Any], evaluator) -> tuple:
        """
        Intelligently extract answers and score - select appropriate extraction method based on task type
        """
        target_answer = self._get_target_answer(item)
        task_type = self._identify_task_type(item, target_answer, evaluator)
        
        # Use specialized answer extraction method based on evaluator type
        evaluator_name = evaluator.__class__.__name__
        
        if evaluator_name == "WebOfLiesEvaluator":
            # WebOfLies: Yes/No tasks
            if hasattr(evaluator, '_extract_yes_no_answer'):
                original_question = item.get('input', item.get('question', ''))
                extracted_answer = evaluator._extract_yes_no_answer(prediction, original_question)
            else:
                extracted_answer = self._extract_yes_no_answer_regex(prediction)
                
        elif evaluator_name == "AQuAEvaluator":
            # AQuA: Force use LLM intelligent answer extraction
            if hasattr(evaluator, '_llm_extract_choice_answer') and self.llm:
                extracted_answer = evaluator._llm_extract_choice_answer(prediction, item, self.llm)
            else:
                # Fallback: use unified scorer LLM extraction
                extracted_answer = self._extract_with_llm_only(prediction)
                
        elif evaluator_name == "GSM8KEvaluator":
            # GSM8K: Use LLM to directly judge correctness
            if hasattr(evaluator, '_llm_judge_answer') and self.llm:
                is_correct = evaluator._llm_judge_answer(prediction, target_answer, self.llm)
                # Return judgment result directly, no need to extract answer
                return "", target_answer, is_correct
            else:
                # Fallback: use unified scorer LLM extraction
                extracted_answer = self._extract_with_llm_only(prediction)
                
        elif evaluator_name == "GSMHardEvaluator":
            # GSM-Hard: LLM intelligent numerical extraction
            if hasattr(evaluator, '_extract_final_answer'):
                extracted_answer = evaluator._extract_final_answer(prediction)
            else:
                extracted_answer = self._extract_math_answer_regex(prediction)
                
        elif evaluator_name == "MultiArithEvaluator":
            # MultiArith: Force use LLM intelligent numerical extraction (consistent with GSM8K)
            if hasattr(evaluator, '_llm_extract_numerical_answer') and self.llm:
                extracted_answer = evaluator._llm_extract_numerical_answer(prediction, self.llm)
            else:
                # Fallback: use unified scorer LLM extraction
                extracted_answer = self._extract_with_llm_only(prediction)
                if not extracted_answer:
                    # Final fallback: use regular expression
                    numeric_answer = evaluator._extract_numerical_answer(prediction) if hasattr(evaluator, '_extract_numerical_answer') else None
                    extracted_answer = str(numeric_answer) if numeric_answer is not None else ""
                
        elif evaluator_name == "AIME2025Evaluator":
            # AIME2025: Numerical answer extraction + LLM intelligent extraction
            if hasattr(evaluator, '_extract_final_answer'):
                extracted_answer = evaluator._extract_final_answer(prediction)
            else:
                # Backup: Use LLM intelligent numerical extraction
                extracted_answer = self._extract_aime_answer_with_llm(prediction)
                
        elif evaluator_name == "CompetitionMathEvaluator":
            # CompetitionMath: Directly use LLM to judge, do not extract answer
            if hasattr(evaluator, '_judge_answer_with_llm') and hasattr(evaluator, 'extractor_llm') and evaluator.extractor_llm:
                # Use new direct judgment method
                question = item.get('problem') or item.get('question') or item.get('input') or ''
                gold_answer = self._get_target_answer(item)
                gold_solution = item.get('solution') or item.get('explanation') or ''
                is_correct = evaluator._judge_answer_with_llm(prediction, question, gold_answer, gold_solution)
                # Still extract answer for display, but use LLM judgment for result
                if hasattr(evaluator, '_extract_answer'):
                    extracted_answer = evaluator._extract_answer(prediction)
                else:
                    extracted_answer = self._extract_with_llm_only(prediction)
                # Return extracted answer, target answer and LLM judgment result
                return extracted_answer, gold_answer, is_correct
            else:
                # Fallback: extract answer
                if hasattr(evaluator, '_extract_answer'):
                    extracted_answer = evaluator._extract_answer(prediction)
                else:
                    extracted_answer = self._extract_with_llm_only(prediction)
                
        elif evaluator_name in ["MMLUEvaluator", "AccuracyEvaluator"]:
            # MMLU/Accuracy: Intelligent context extraction
            if hasattr(evaluator, '_extract_answer_with_context'):
                extracted_answer = evaluator._extract_answer_with_context(prediction, item)
            elif hasattr(evaluator, '_extract_answer'):
                extracted_answer = evaluator._extract_answer(prediction)
            else:
                extracted_answer = self._extract_with_llm_only(prediction)
        
        elif evaluator_name == "SquadV2Evaluator":
            # SQuAD 2.0: Extractive QA, need to extract text spans or identify unanswerable
            extracted_answer = self._extract_squad_answer(prediction)
                
        else:
            # Default processing: select method based on task type
            if task_type == "yes_no":
                extracted_answer = self._extract_yes_no_answer_regex(prediction)
            elif task_type == "math":
                extracted_answer = self._extract_math_answer_regex(prediction)
            elif task_type == "multiple_choice":
                extracted_answer = self._extract_choice_answer_regex(prediction)
            else:
                # General LLM extraction
                extracted_answer = self._extract_with_llm_only(prediction)
                if not extracted_answer:
                    extracted_answer = self._fallback_extraction(prediction, task_type)
        
        # Standardized comparison
        if task_type == "yes_no":
            # Yes/No tasks: direct comparison
            norm_extracted = str(extracted_answer).strip().upper()
            norm_target = str(target_answer).strip().upper()
        else:
            # Other tasks: numerical/choice standardization
            def normalize_for_comparison(ans):
                ans_str = str(ans).strip().upper().replace('(', '').replace(')', '')
                ans_str = ans_str.replace(',', '')
                
                try:
                    if '.' in ans_str:
                        num = float(ans_str)
                        if num.is_integer():
                            return str(int(num))
                        else:
                            return str(num)
                    else:
                        num = int(ans_str)
                        return str(num)
                except ValueError:
                    return ans_str
            
            norm_extracted = normalize_for_comparison(extracted_answer)
            norm_target = normalize_for_comparison(target_answer)
        
        is_correct = (norm_extracted == norm_target)
        
        return extracted_answer, target_answer, is_correct
    
    def _extract_yes_no_answer(self, prediction: str) -> str:
        """Extract Yes/No answer from response - specifically optimized for Web of Lies"""
        if not prediction:
            return "No"
        
        prediction_lower = prediction.lower()
        import re
        
        # 1. Direct Yes/No patterns
        direct_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(yes|no)\b',
            r'(?:therefore|so|thus|hence),?\s+(?:the\s+answer\s+is\s+)?(yes|no)\b',
            r'\b(yes|no)\s*[.,!]?\s*$',
        ]
        for pattern in direct_patterns:
            match = re.search(pattern, prediction_lower, re.MULTILINE)
            if match:
                return "Yes" if match.group(1) == "yes" else "No"

        # 2. Specific negation patterns (highest priority)
        negation_patterns = [
            r'\w+\s+does\s+not\s+tell\s+the\s+truth',
            r'\w+\s+do\s+not\s+tell\s+the\s+truth',
            r'\w+\s+is\s+not\s+telling\s+the\s+truth',
            r'\w+\s+are\s+not\s+telling\s+the\s+truth',
            r'\w+\s+does\s+not\s+tell\s+truth',
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, prediction_lower):
                return "No"

        # 3. Positive truth patterns
        truth_patterns = [
            r'\w+\s+(?:does\s+)?tell(?:s)?\s+the\s+truth',
            r'\w+\s+(?:is\s+)?telling\s+the\s+truth',
        ]
        
        for pattern in truth_patterns:
            if re.search(pattern, prediction_lower):
                return "Yes"

        # 4. Lie patterns
        lie_patterns = [
            r'\w+\s+tells?\s+(?:a\s+)?lie',
            r'\w+\s+lies?\b',
            r'\w+\s+(?:is\s+)?lying\b',
            r'\w+\s+(?:is\s+)?(?:a\s+)?liar\b',
        ]
        
        for pattern in lie_patterns:
            if re.search(pattern, prediction_lower):
                return "No"

        return "No"  # Default to No
    
    def debug_web_of_lies_extraction(self, prediction: str) -> str:
        """Debug Web of Lies extraction"""
        print(f"\n Debug extraction process:")
        print(f"Original: {prediction}")

        prediction_lower = prediction.lower()
        print(f"Lowercase: {prediction_lower}")
        
        import re
        
        # Test negation patterns
        negation_patterns = [
            r'\w+\s+does\s+not\s+tell\s+the\s+truth',
            r'\w+\s+do\s+not\s+tell\s+the\s+truth',
        ]

        for i, pattern in enumerate(negation_patterns):
            match = re.search(pattern, prediction_lower)
            print(f"Negation pattern {i+1} '{pattern}': {' Match' if match else ' No match'}")
            if match:
                print(f"Matched content: '{match.group()}'")
                return "No"
        
        # Test truth patterns
        truth_patterns = [
            r'\w+\s+(?:does\s+)?tell(?:s)?\s+the\s+truth',
            r'\w+\s+(?:is\s+)?telling\s+the\s+truth',
        ]

        for i, pattern in enumerate(truth_patterns):
            match = re.search(pattern, prediction_lower)
            print(f"Truth pattern {i+1} '{pattern}': {' Match' if match else ' No match'}")
            if match:
                print(f"Matched content: '{match.group()}'")
                return "Yes"

        print("No matches, return default value No")
        return "No"
    
    def _extract_answer_smart(self, prediction: str, task_type: str) -> str:
        """
        Smart answer extraction: prioritize rule extraction, use LLM when necessary
        """
        if not prediction:
            return ""

        # First try rule extraction
        rule_extracted = self._fallback_extraction(prediction, task_type)

        # If rule extraction succeeds, return directly
        if rule_extracted:
            return rule_extracted

        # If rule extraction fails, then use LLM extraction
        if self.llm:
            try:
                return self._extract_answer_with_llm(prediction, task_type)
            except Exception as e:
                logging.warning(f"LLM answer extraction failed: {e}")
                return rule_extracted  # Fallback to rule extraction result
        
        return rule_extracted
    
    def _identify_task_type(self, item: Dict[str, Any], target_answer: str, evaluator) -> str:
        """Identify task type to select appropriate processing method"""

        # Prioritize judgment by evaluator type
        evaluator_name = evaluator.__class__.__name__
        if evaluator_name == "WebOfLiesEvaluator":
            return "yes_no"
        elif evaluator_name == "AQuAEvaluator":
            return "multiple_choice"  # AQuA is math multiple choice
        elif evaluator_name in ["GSM8KEvaluator", "GSMHardEvaluator", "MultiArithEvaluator", "AIME2025Evaluator"]:
            return "math"  # Mathematical reasoning tasks
        elif evaluator_name in ["MMLUEvaluator", "AccuracyEvaluator"]:
            return "multiple_choice"  # MMLU/Accuracy are usually multiple choice

        # Judge by target answer format
        if target_answer.lower() in ['true', 'false']:
            return "boolean"
        elif target_answer.lower() in ['yes', 'no']:
            return "yes_no"
        elif re.match(r'^\([A-Za-z]\)$', target_answer.strip()):
            return "multiple_choice"

        # Judge by question content
        input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
        question = item.get(input_key, '').lower()

        # Web of Lies specific pattern recognition
        if ('tell the truth' in question or 'tells the truth' in question or
            'lies' in question or 'lying' in question):
            return "yes_no"

        # Boolean logic expression questions
        if ('true' in question and 'false' in question and
            any(op in question for op in ['and', 'or', 'not']) and question.endswith(' is')):
            return "boolean"

        # Yes/No judgment questions
        if 'options:' in question and '- yes' in question and '- no' in question:
            return "yes_no"

        # Traditional multiple choice questions
        if re.search(r'options:\s*\([a-z]\)', question) or re.search(r'\([a-z]\)', question):
            return "multiple_choice"

        # Mathematical reasoning tasks
        evaluator_type = str(type(evaluator)).lower()
        if ('gsm8k' in evaluator_type or 'multiarith' in evaluator_type
            or 'gsmhard' in evaluator_type or 'gsm_hard' in evaluator_type):
            return "math"

        # Default to multiple choice
        return "multiple_choice"
    
    def _extract_answer_with_llm(self, prediction: str, task_type: str) -> str:
        """Use LLM to intelligently extract answers (as backup method)"""
        if not prediction or not self.llm:
            return ""
        
        try:
            if task_type == "boolean":
                extraction_prompt = f"""Please extract the final boolean logic answer from the following response.

Response:
{prediction}

Please output only True or False, without any other content.

Answer:"""
                
            elif task_type == "yes_no":
                extraction_prompt = f"""You are a precise answer extraction specialist. Extract the final Yes/No judgment from this response.

Response:
{prediction}

Guidelines:
- Look for the ultimate conclusion or final answer
- The answer might be in various formats: Yes, No, **Yes**, **No**, or embedded in sentences
- Focus on the FINAL judgment, ignore intermediate reasoning
- Handle any formatting (markdown, punctuation, etc.)

Output exactly "Yes" or "No" - nothing else.

Answer:"""
                
            elif task_type == "math":
                extraction_prompt = f"""Please extract the final numerical answer from the following response.

Response:
{prediction}

Please output only the number, without any other content.

Answer:"""
                
            else:  # multiple_choice
                extraction_prompt = f"""Read the following response and output the chosen option letter:

{prediction}

Output only A, B, C, D, or E:"""
            
            # Call LLM to extract answer
            extracted = self.llm.generate(extraction_prompt).strip()

            # Post-process extracted answers
            if task_type == "boolean":
                # Standardize boolean answers
                if extracted.lower() in ['true', 'false']:
                    return extracted.lower().capitalize()  # True or False

            elif task_type == "yes_no":
                # Standardize Yes/No answers
                if extracted.lower() in ['yes', 'no']:
                    return extracted.lower().capitalize()  # Yes or No

            elif task_type == "math":
                # Extract numbers
                numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', extracted)
                if numbers:
                    return numbers[0]

            else:  # multiple_choice
                # Extract option letters
                letter_match = re.search(r'([A-Za-z])', extracted)
                if letter_match:
                    return f"({letter_match.group(1).upper()})"
            
            return extracted
            
        except Exception as e:
            logging.warning(f"LLM answer extraction failed: {e}")
            return ""
    
    def _fallback_extraction(self, prediction: str, task_type: str) -> str:
        """Rule extraction method (now as main extraction method)"""
        if task_type == "boolean":
            return self._extract_boolean_answer_regex(prediction)
        elif task_type == "yes_no":
            return self._extract_yes_no_answer_regex(prediction)
        elif task_type == "math":
            return self._extract_math_answer_regex(prediction)
        else:  # multiple_choice
            return self._extract_choice_answer_regex(prediction)
    
    def _extract_boolean_answer_regex(self, prediction: str) -> str:
        """Extract boolean answers using regex"""
        if not prediction:
            return ""
        
        prediction_lower = prediction.lower().strip()
        
        # Modified: First check if it's a simple single letter answer
        if prediction_lower in ['t', 'true']:
            return "True"
        elif prediction_lower in ['f', 'false']:
            return "False"

        # Find boolean answer patterns
        boolean_patterns = [
            r'(?:answer is|final answer is|result is|therefore|thus|so|simplifies to)\s*[:,]?\s*`?(true|false)`?',
            r'(?:answer is|answer is|result is|therefore|thus|so)\s*[:,]?\s*`?(true|false)`?',
            r'\b(true|false)\s*[.\s]*$',  # True/False at end of sentence
            r'is\s*[:,]?\s*`?(true|false)`?',
        ]

        for pattern in boolean_patterns:
            match = re.search(pattern, prediction_lower)
            if match:
                return match.group(1).capitalize()  # True or False

        # Find last appearing True/False
        all_matches = re.findall(r'\b(true|false)\b', prediction_lower)
        if all_matches:
            return all_matches[-1].capitalize()
        
        return ""
    
    def _extract_yes_no_answer_regex(self, prediction: str) -> str:
        """Extract Yes/No answers using regex - specifically optimized for Web of Lies"""
        if not prediction:
            return ""
        
        # New: Remove Markdown formatting symbols, keep original for debugging
        prediction_clean = re.sub(r'\*+', '', prediction)  # Remove all asterisks
        prediction_clean = re.sub(r'_+', '', prediction_clean)  # Remove underscores
        prediction_lower = prediction_clean.lower().strip()

        # Modified: First check if it's a simple single letter answer
        if prediction_lower in ['y', 'yes']:
            return "Yes"
        elif prediction_lower in ['n', 'no']:
            return "No"

        # New: Handle Markdown formatted Yes/No (highest priority)
        markdown_patterns = [
            r'\*\*\s*(yes|no)\s*\*\*',  # **Yes** or **No**
            r'\*\s*(yes|no)\s*\*',      # *Yes* or *No*
            r'__\s*(yes|no)\s*__',      # __Yes__ or __No__
            r'_\s*(yes|no)\s*_',        # _Yes_ or _No_
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, prediction.lower(), re.MULTILINE)
            if match:
                return "Yes" if match.group(1) == "yes" else "No"
        
        # Enhanced: Find Yes/No answer patterns (more comprehensive)
        yes_no_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+(?:is\s+)?:?\s*(yes|no)\b',
            r'(?:therefore|thus|so|hence),?\s+(?:the\s+answer\s+(?:is\s+)?)?(yes|no)\b',
            r'(?:conclusion|result):\s*(yes|no)\b',
            r'\b(yes|no)\s*[.,!]?\s*$',  # Yes/No at end of sentence
            r'^\s*(yes|no)\b',  # Yes/No at beginning of line
        ]

        for pattern in yes_no_patterns:
            match = re.search(pattern, prediction_lower, re.MULTILINE)
            if match:
                return "Yes" if match.group(1) == "yes" else "No"

        # New: Web of Lies specific logic pattern recognition
        # Analyze last few lines, look for judgments about final character
        lines = prediction_lower.split('\n')
        for line in reversed(lines[-5:]):  # Only check last 5 lines
            line = line.strip()
            if not line:
                continue

            # Look for negative expressions like "does not tell the truth", "lies"
            if any(neg_phrase in line for neg_phrase in [
                'does not tell the truth', 'do not tell the truth',
                'is not telling the truth', 'are not telling the truth',
                'tells a lie', 'tell lies', 'is lying', 'lies',
                'is a liar', 'are liars'
            ]):
                return "No"

            # Look for positive expressions like "tells the truth"
            if any(pos_phrase in line for pos_phrase in [
                'tells the truth', 'tell the truth',
                'is telling the truth', 'are telling the truth',
                'is truthful', 'are truthful'
            ]):
                return "Yes"

        # Improved: Find all Yes/No, return the last one
        all_matches = re.findall(r'\b(yes|no)\b', prediction_lower)
        if all_matches:
            return "Yes" if all_matches[-1] == "yes" else "No"

        # Default to No (because Web of Lies usually has more negative answers)
        return "No"
    
    def _extract_math_answer_regex(self, prediction: str) -> str:
        """Extract mathematical answers using regex"""
        if not prediction:
            return ""
        
        # 1. First try to extract GSM8K format answers (#### number)
        gsm_pattern = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', prediction)
        if gsm_pattern:
            return gsm_pattern.group(1).strip()

        # 2. Extract all numbers, return the last one
        numbers = self._extract_all_numbers(prediction)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def _extract_choice_answer_regex(self, prediction: str) -> str:
        """Extract multiple choice answers using regex"""
        if not prediction:
            return ""
        
        prediction_stripped = prediction.strip()
        
        # First check if it's a simple single letter answer (supports all A-Z options)
        if len(prediction_stripped) == 1 and prediction_stripped.isalpha():
            return f"({prediction_stripped.upper()})"

        # Directly search for (A), (B), (C), (D), (E) format
        choice_pattern = re.search(r'\(([A-Za-z])\)', prediction)
        if choice_pattern:
            return f"({choice_pattern.group(1).upper()})"

        # Search for Answer: A format
        answer_pattern = re.search(r'(?:Answer|answer):\s*([A-Za-z])', prediction)
        if answer_pattern:
            return f"({answer_pattern.group(1).upper()})"

        # Find letters from end of text forward
        letters = re.findall(r'([A-Za-z])', prediction)
        if letters:
            return f"({letters[-1].upper()})"
        
        return ""
    
    def _is_answer_correct_by_task_type(self, extracted_answer: str, target_answer: str, evaluator, task_type: str) -> bool:
        """Judge if answer is correct based on task type"""
        
        if task_type == "boolean":
            return self._compare_boolean_answers(extracted_answer, target_answer)
        elif task_type == "yes_no":
            return self._compare_yes_no_answers(extracted_answer, target_answer)
        elif task_type == "math":
            return self._compare_math_answers(extracted_answer, target_answer)
        else:  # multiple_choice
            return self._compare_choice_answers(extracted_answer, target_answer)
    
    def _compare_boolean_answers(self, extracted: str, target: str) -> bool:
        """Compare boolean logic answers"""
        if not extracted or not target:
            return False
        return extracted.lower() == target.lower()
    
    def _compare_yes_no_answers(self, extracted: str, target: str) -> bool:
        """Compare Yes/No answers"""  
        if not extracted or not target:
            return False
        return extracted.lower() == target.lower()
    
    def _compare_math_answers(self, extracted: str, target: str) -> bool:
        """Compare mathematical answers"""
        if not extracted or not target:
            return False
        try:
            return abs(float(extracted) - float(target)) < 1e-6
        except (ValueError, TypeError):
            return False
    
    def _compare_choice_answers(self, extracted: str, target: str) -> bool:
        """Compare multiple choice answers"""
        normalized_extracted = self._normalize_choice_answer_apsf_style(extracted)
        normalized_target = self._normalize_choice_answer_apsf_style(target)
        return normalized_extracted == normalized_target
    
    def _normalize_choice_answer_apsf_style(self, answer: str) -> str:
        """Improved multiple choice answer normalization"""
        if not answer:
            return ""
        
        answer = str(answer).strip().upper()
        
        # Extract letters, whether with or without parentheses
        import re

        # Match (A), A, (a), a etc. formats
        letter_patterns = [
            r'\(([A-Z])\)',  # (A) format
            r'^([A-Z])$',    # Simple letter A format
            r'([A-Z])',      # Any format containing letters
        ]

        for pattern in letter_patterns:
            match = re.search(pattern, answer)
            if match:
                return match.group(1)  # Return pure letter, e.g. 'B'
        
        return answer
    
    def _get_target_answer(self, item: Dict[str, Any]) -> str:
        """Get target answer"""
        # Try multiple possible answer keys, including output
        answer_keys = ['answer', 'target', 'correct', 'label', 'output']
        
        for key in answer_keys:
            if key in item:
                answer = item[key]
                # Handle GSM8K format answers
                if isinstance(answer, str) and '####' in answer:
                    return answer.split('####')[-1].strip()
                return str(answer).strip()
        
        # If none found, log warning
        logging.warning(f"Target answer key not found, available keys: {list(item.keys())}")
        return ""
    
    def _extract_aime_answer_with_llm(self, prediction: str) -> str:
        """
        Use LLM to intelligently extract AIME2025 numerical answers
        AIME answer characteristics: integers 0-999, may appear in $\boxed{number}$ format
        """
        if not prediction or not self.llm:
            return ""
        
        extraction_prompt = f"""Please extract the final numerical answer from the following mathematical reasoning process.

Mathematical reasoning process:
{prediction}

Extraction requirements:
- AIME answers are usually integers between 0 and 999
- Answers may appear in $\\boxed{{number}}$ format
- Answers may appear after expressions like "answer is", "final answer", "result is"
- Output only pure numbers, no other content

Please extract the final answer:"""

        try:
            extracted = self.llm.generate(extraction_prompt).strip()
            
            # Extract numbers from LLM response
            import re
            number_match = re.search(r'(\d+)', extracted)
            if number_match:
                number = number_match.group(1)
                # Verify if within valid AIME range
                try:
                    num_val = int(number)
                    if 0 <= num_val <= 999:
                        return number
                    else:
                        # Even if out of range, return the extracted number
                        return number
                except ValueError:
                    pass

            # If LLM doesn't return valid numbers, use regex backup method
            return self._extract_aime_answer_regex(prediction)

        except Exception as e:
            logging.warning(f"AIME LLM answer extraction failed: {e}")
            return self._extract_aime_answer_regex(prediction)
    
    def _extract_aime_answer_regex(self, prediction: str) -> str:
        """
        Backup method to extract AIME answers using regex
        """
        if not prediction:
            return ""
        
        # 1. Prioritize extracting $\boxed{number}$ format
        boxed_pattern = re.search(r'\$\\boxed\{(\d+)\}\$', prediction)
        if boxed_pattern:
            return boxed_pattern.group(1)

        # 2. Extract other boxed formats
        boxed_patterns = [
            r'\\boxed\{(\d+)\}',
            r'\$\\boxed\{(\d+)\}',
            r'boxed\{(\d+)\}',
        ]

        for pattern in boxed_patterns:
            match = re.search(pattern, prediction)
            if match:
                return match.group(1)

        # 3. Extract explicit answer markers
        answer_patterns = [
            r'answer(?:is|is|:)\s*(\d+)',
            r'final answer(?:is|is|:)?\s*(\d+)',
            r'final\s+answer\s*:?\s*(\d+)',
            r'the\s+answer\s+is\s+(\d+)',
            r'result\s*:?\s*(\d+)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1)

        # 4. Extract all numbers, return the last one
        numbers = re.findall(r'\b(\d+)\b', prediction)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def _extract_all_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        if not text:
            return []
        
        pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        
        # Remove commas and return pure number strings
        cleaned_numbers = []
        for match in matches:
            cleaned_num = match.replace(',', '')
            if cleaned_num:
                cleaned_numbers.append(cleaned_num)
        
        return cleaned_numbers
    
    def _extract_squad_answer(self, prediction: str) -> str:
        """
        Extract SQuAD 2.0 answers
        - Detect "unanswerable" indicators
        - Extract text span answers
        """
        if not prediction:
            return ""
        
        prediction_lower = prediction.lower().strip()
        
        # 1. Detect unanswerable indicators
        no_answer_patterns = [
            'no answer', 'unanswerable', 'cannot answer',
            "can't answer", "can not answer",
            'no information', 'not mentioned', 'not provided',
            'unknown', 'cannot answer', 'no answer', 'no answer available',
            '<no_answer', 'no_answer', '[no answer]'
        ]
        
        for pattern in no_answer_patterns:
            if pattern in prediction_lower:
                return ""  # Return empty string to indicate unanswerable
        
        # 2. Extract <answer>...</answer> tags
        answer_tag_match = re.search(r'<answer>(.*?)</answer>', prediction, re.IGNORECASE | re.DOTALL)
        if answer_tag_match:
            return answer_tag_match.group(1).strip()

        # 3. Extract answers in quotes
        # Prioritize extracting content in quotes after "answer is" or "answer:"
        answer_quote_patterns = [
            r'(?:answer is|answer:|named their interactive service)\s*["\']([^"\']+)["\']',
            r'(?:called|named)\s*["\']([^"\']+)["\']',
            r'["\']([^"\']{1,50})["\'](?:\s*(?:is the answer|is correct))',
        ]
        
        for pattern in answer_quote_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 4. Extract "answer is X" or "answer: X" format
        answer_is_patterns = [
            r'(?:final |the )?answer (?:is|was|would be)\s*[:\-]?\s*(.+?)(?:\.|$|\n)',
            r'(?:called|named)\s+(.+?)(?:\.|$|\n)',
            r'(?:answer|result)\s*[:\-]\s*(.+?)(?:\.|$|\n)',
        ]

        for pattern in answer_is_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                ans = match.group(1).strip()
                # Clean answer, remove extra punctuation and quotes
                ans = ans.strip('"\'.,:;!? ')
                if len(ans) > 0 and len(ans) < 100:  # Reasonable length answers
                    return ans

        # 5. As last resort, use LLM extraction (if available)
        if self.llm:
            try:
                llm_prompt = f"""Extract the final answer from this response. Return ONLY the answer text, nothing else.

Response: {prediction[:500]}

Rules:
- If the response says the question is unanswerable, return: NO_ANSWER
- Otherwise, return only the short answer phrase (usually 1-5 words)
- Do not include explanations or quotes

Answer:"""
                llm_answer = self.llm.generate(llm_prompt).strip()
                if llm_answer and 'NO_ANSWER' not in llm_answer.upper():
                    return llm_answer.strip('"\'')
                elif 'NO_ANSWER' in llm_answer.upper():
                    return ""
            except:
                pass

        # 6. Final backup: return key phrases from last sentence
        sentences = [s.strip() for s in prediction.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            # Remove common connecting words and redundancy
            words = last_sentence.split()
            if len(words) <= 10:  # Short sentences may be the answer
                return last_sentence.strip()

        return prediction[:100].strip()  # Return first 100 characters as last resort

    def _extract_with_llm_only(self, prediction: str) -> str:
        """
        Use LLM to intelligently extract answers - simplified stable version
        For Qwen3 architect model, prioritize extracting content after think process
        """
        if not prediction or not self.llm:
            return ""
        
        # Check if it's Qwen3 architect model output, if so extract content after think first
        processed_prediction = self._preprocess_qwen3_output(prediction)
        
        # Use more explicit answer extraction prompt
        prompt = f"""Read this response and tell me what the final answer is:

{processed_prediction}

For multiple choice questions: Output ONLY the letter (A, B, C, D, or E) - NOT the number or calculation
For Yes/No questions: Output only Yes or No
For True/False questions: Output only True or False
For numerical questions: Output only the number

Answer:"""

        try:
            extracted = self.llm.generate(prompt).strip()
            
            # Post-processing: Remove common prefixes and suffixes
            prefixes_to_remove = [
                "Answer:", "Answer is:", "Final answer:", "Result:", "Result is:",
                "Answer:", "Final answer:", "Result:", "The answer is:",
                "Answer", "Final answer", "Result"
            ]
            
            for prefix in prefixes_to_remove:
                if extracted.startswith(prefix):
                    extracted = extracted[len(prefix):].strip()
            
            # Remove common suffixes
            suffixes_to_remove = ["。", ".", "!", "！"]
            for suffix in suffixes_to_remove:
                if extracted.endswith(suffix):
                    extracted = extracted[:-len(suffix)].strip()

            # New: Remove thousands separator (comma)
            extracted = extracted.replace(',', '')
            
            return extracted
            
        except Exception as e:
            logging.warning(f"LLM answer extraction failed: {e}")
            return ""
    
    def _preprocess_qwen3_output(self, prediction: str) -> str:
        """
        Preprocess Qwen3 architect model output, extract content after think process
        This method is consistent with _extract_content_after_think method in GPT_API
        """
        import re
        
        # Check if think process markers are included
        think_indicators = ['<think>', '<thinking>', '<thought>', 'think:', 'thinking:', 'thinking:']
        has_think_process = any(indicator.lower() in prediction.lower() for indicator in think_indicators)

        if not has_think_process:
            return prediction  # If no think process, return original content directly
        
        # Method 1: Extract content after <think>...</think> tags
        think_patterns = [
            r'<think>.*?</think>\s*(.*?)$',  # <think>...</think>
            r'<thinking>.*?</thinking>\s*(.*?)$',  # <thinking>...</thinking>
            r'<thought>.*?</thought>\s*(.*?)$',  # <thought>...</thought>
        ]

        for pattern in think_patterns:
            match = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:  # Ensure extracted content is not empty
                    logging.info(" Successfully extracted content after think from Qwen3 output")
                    return extracted
        
        # Method 2: Find content after "Think:" or "Thinking:"
        think_start_patterns = [
            r'(?:Think|Thinking)[:：]\s*(.*?)$',  # Think: or Thinking:
            r'(?:Thinking|Thinking)[:：]\s*(.*?)$',  # Thinking: or Thinking:
        ]

        for pattern in think_start_patterns:
            match = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    logging.info(" Successfully extracted content after think from Qwen3 output")
                    return extracted
        
        # Method 3: Find content after think process ends
        think_end_markers = [
            r'</think>\s*(.*?)$',
            r'</thinking>\s*(.*?)$',
            r'</thought>\s*(.*?)$',
            r'(?:end\s+think|think\s+end|end\s+thinking)\s*(.*?)$',
        ]

        for pattern in think_end_markers:
            match = re.search(pattern, prediction, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    logging.info(" Successfully extracted content after think from Qwen3 output")
                    return extracted
        
        # Method 4: Smart segmentation - find think process end position
        lines = prediction.split('\n')
        think_end_line = -1

        # Find possible think process end lines
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Look for think process end indicator words
            if any(end_word in line_lower for end_word in [
                'end think', 'think end', 'end thinking', 'thinking end',
                'now i will', 'let me', 'based on', 'therefore',
                'so', 'therefore', 'based on', 'according to'
            ]):
                think_end_line = i
                break

        # If think end line found, extract content after it
        if think_end_line >= 0 and think_end_line < len(lines) - 1:
            remaining_lines = lines[think_end_line + 1:]
            remaining_content = '\n'.join(remaining_lines).strip()
            if remaining_content:
                logging.info(" Successfully extracted content after think from Qwen3 output")
                return remaining_content

        # If all methods fail, log warning and return original content
        logging.warning(" Unable to extract content after think from Qwen3 output, using original content")
        return prediction

def _identify_bbh_task_type(self, prediction: str) -> str:
    """
    Intelligently identify BBH task type
    """
    prediction_lower = prediction.lower()
    
    # Boolean logic expressions
    if any(word in prediction_lower for word in ['true', 'false', 'boolean', 'expression']):
        if any(op in prediction_lower for op in ['and', 'or', 'not']):
            return "boolean_logic"

    # New: Argument validity questions (formal fallacies)
    if any(word in prediction_lower for word in ['valid', 'invalid', 'fallacy', 'fallacies', 'argument', 'logical']):
        return "formal_fallacies"

    # Yes/No judgment questions
    if any(word in prediction_lower for word in ['yes', 'no']) and \
       any(word in prediction_lower for word in ['therefore', 'conclude', 'answer']):
        return "yes_no"

    # Multiple choice questions
    if any(pattern in prediction_lower for pattern in [
        'option', 'choice',
        '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
        '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)',
        '(u)', '(v)', '(w)', '(x)', '(y)', '(z)'
    ]):
        return "multiple_choice"
    
    # Navigation questions
    if any(word in prediction_lower for word in ['north', 'south', 'east', 'west', 'direction']):
        return "navigation"

    # Counting questions
    if any(word in prediction_lower for word in ['count', 'total', 'number', 'how many']):
        return "counting"

    # Sorting questions
    if any(word in prediction_lower for word in ['sort', 'order', 'arrange', 'sequence']):
        return "sorting"
    
    return "general"


def evaluate_with_unified_scoring(
    predictions: List[str],
    eval_data: List[Dict[str, Any]],
    evaluator,
    llm: BaseLLM,
    dataset_name: str = "unknown",
    verbose: bool = True  # New parameter to control output detail level
) -> Dict[str, Any]:
    """
    Unified scoring function - use LLM for intelligent answer extraction
    """
    scorer = UnifiedScorer(llm, dataset_name)
    
    correct_count = 0
    detailed_results = []
    
    if verbose:
        print(f"\n Starting intelligent answer extraction scoring - Dataset: {dataset_name}")
        print(f" Sample count: {len(predictions)}")
    
    for i, (prediction, item) in enumerate(zip(predictions, eval_data)):
        extracted_answer, target_answer, is_correct = scorer.extract_and_score(
            prediction=prediction,
            item=item,
            evaluator=evaluator
        )
        
        if is_correct:
            correct_count += 1
            
        detailed_results.append({
            'sample_id': i,
            'prediction': prediction,
            'extracted_answer': extracted_answer,
            'target_answer': target_answer,
            'correct': is_correct
        })
        
        # Modified: Only show details when verbose=True
        if verbose and i < 10:  # Only show first 10 samples to avoid excessive output
            # Modified: Choose appropriate normalization method based on task type
            task_type = scorer._identify_task_type(item, target_answer, evaluator)
            if task_type == "yes_no":
                normalized_extracted = extracted_answer.lower() if extracted_answer else ""
                normalized_target = target_answer.lower() if target_answer else ""
            elif task_type == "boolean":
                normalized_extracted = extracted_answer.lower() if extracted_answer else ""
                normalized_target = target_answer.lower() if target_answer else ""
            else:
                normalized_extracted = scorer._normalize_choice_answer_apsf_style(extracted_answer)
                normalized_target = scorer._normalize_choice_answer_apsf_style(target_answer)

            print(f"\n Validating sample {i+1}/{len(predictions)}")
            print(f" Standardized comparison: '{normalized_extracted}' vs '{normalized_target}'")
            print(f" Intelligently extracted answer: '{extracted_answer}'")
            print(f" Target answer: '{target_answer}'")
            verdict = " Correct" if is_correct else " Incorrect"
            print(f" Answer match: {verdict}")
    
    accuracy = correct_count / len(predictions) if predictions else 0.0
    
    if verbose:
        print(f"\n Intelligent extraction scoring results:")
        print(f"  Correct: {correct_count}/{len(predictions)}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(predictions),
        "detailed_results": detailed_results
    } 