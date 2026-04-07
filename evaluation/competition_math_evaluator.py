import re
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator
from ..llm_apis import get_llm, BaseLLM
from ..config import ANSWER_EXTRACTOR_LLM

class CompetitionMathEvaluator(BaseEvaluator):
    """
    Dedicated evaluator for CompetitionMath dataset
    - Supports mathematical problem accuracy evaluation
    - Uses solution field for step-level error analysis
    - Supports various answer formats (numbers, algebraic expressions, sets, etc.)
    """
    
    def __init__(self, use_llm_comparison: bool = True):
        """
        Initialize evaluator

        Args:
            use_llm_comparison: Whether to use LLM for answer comparison
        """
        try:
            self.extractor_llm: BaseLLM = get_llm(ANSWER_EXTRACTOR_LLM)
            self.use_llm_comparison = use_llm_comparison
            comparison_mode = "LLM Direct Judgment" if use_llm_comparison else "Rule-based"
            print(f"CompetitionMath Evaluator Initialized")
            print(f"   Model: {ANSWER_EXTRACTOR_LLM}")
            print(f"   Mode: {comparison_mode}")
        except Exception as e:
            print(f"LLM initialization failed, fallback to rule-based: {e}")
            self.use_llm_comparison = False
            self.extractor_llm = None
    
    def _extract_boxed_answer(self, text: str) -> str:
        """Extract content from \boxed{}, handle nested braces"""
        # Find position of \boxed{
        start = text.find(r'\boxed{')
        if start == -1:
            return None
        
        # Start counting braces after \boxed{
        start_pos = start + len(r'\boxed{')
        brace_count = 1
        pos = start_pos
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            # Successful match
            return text[start_pos:pos-1].strip()
        
        return None
    
    def _llm_extract_answer(self, prediction: str) -> str:
        """Use LLM to intelligently extract mathematical answers"""
        
        extraction_prompt = f"""Extract the FINAL ANSWER from the mathematical solution below.

Response:
{prediction}

IMPORTANT INSTRUCTIONS:
- Look for answers in \\boxed{{}} format - extract ONLY the content inside the braces
- If there's \\(\\boxed{{answer}}\\), extract the answer
- Ignore intermediate calculations and steps
- Focus on the FINAL CONCLUSION at the end
- Output ONLY the final answer value (number, expression, formula, or option letter), nothing else

SPECIAL CASES:
- For multiple choice questions:
  * If you see "\\boxed{{\\text{{C}}}}" or "\\boxed{{C}}", output: C
  * If you see "\\boxed{{\\text{{(A)}}}}" or "\\boxed{{(A)}}", output: (A)
  * Output the LETTER or (LETTER), not the option content
- For numerical answers:
  * If you see "\\boxed{{352}}", output: 352
  * If you see "\\boxed{{\\sqrt{{10}}}}", output: \\sqrt{{10}}
  * If you see "\\boxed{{-5}}", output: -5
- For formula answers:
  * If you see "\\boxed{{\\frac{{bx}}{{h}}(h - x)}}", output: \\frac{{bx}}{{h}}(h - x)

Final Answer:"""
        
        try:
            extracted = self.extractor_llm.generate(extraction_prompt).strip()
            
            # Clean possible LaTeX wrapping
            extracted = extracted.strip('\\(\\)').strip('\\[\\]').strip()
            
            # If LLM returns an option letter, try to find actual answer from original text
            if re.match(r'^\([A-Za-z]\)$', extracted) or re.match(r'^[A-Za-z]$', extracted):
                # Fallback to rule-based extraction
                return self._rule_based_extract(prediction)
            
            return extracted
            
        except Exception as e:
            print(f"  LLM answer extraction failed: {e}, using rule-based method")
            return self._rule_based_extract(prediction)
    
    def _rule_based_extract(self, prediction: str) -> str:
        """Rule-based method to extract answers (as backup for LLM method)"""
        
        # Method 1: Extract content from \boxed{} (supports nested braces and LaTeX format)
        boxed_answer = self._extract_boxed_answer(prediction)
        if boxed_answer:
            return boxed_answer
        
        # Method 2: Look for patterns like "answer is", "result is", etc.
        answer_patterns = [
            r'(?:answer|result|final answer)\s*(?:is|=)\s*([^\n.]+)',
            r'(?:answer|result)\s*(?:is|=)\s*([^\n.]+)',
            r'=\s*([^\n]+?)(?:\n|$)',
            r'\*\*\s*([^\*]+?)\s*\*\*',  # **answer** format
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                ans = match.group(1).strip()
                if ans and len(ans) < 300:  # Reasonable length
                    return ans
        
        # Method 3: Extract non-empty content from last line
        lines = [l.strip() for l in prediction.split('\n') if l.strip()]
        if lines:
            last_line = lines[-1]
            # Remove possible sentence-ending punctuation
            cleaned = re.sub(r'[。\.\!\?]+$', '', last_line).strip()
            if cleaned and len(cleaned) < 300:
                return cleaned
        
        # Fallback: return the last part of original prediction
        return prediction.strip()[-200:] if len(prediction.strip()) > 200 else prediction.strip()
    
    def _extract_answer(self, prediction: str) -> str:
        """Extract answer from model prediction (main entry point)"""
        
        # Prefer LLM extraction
        if hasattr(self, 'extractor_llm') and self.extractor_llm:
            return self._llm_extract_answer(prediction)
        else:
            return self._rule_based_extract(prediction)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer format for comparison"""
        
        if not answer:
            return ""
        
        # Clean spaces
        normalized = answer.strip()
        
        # Remove \boxed{} wrapper (supports nested braces)
        if '\\boxed{' in normalized:
            boxed_content = self._extract_boxed_answer(normalized)
            if boxed_content:
                normalized = boxed_content
        
        # Remove boxed wrapper without backslash (handle extraction errors)
        # Also need to support nested structures
        if 'boxed{' in normalized and '\\boxed{' not in normalized:
            # Use the same brace matching algorithm
            start = normalized.find('boxed{')
            if start != -1:
                start_pos = start + len('boxed{')
                brace_count = 1
                pos = start_pos
                
                while pos < len(normalized) and brace_count > 0:
                    if normalized[pos] == '{':
                        brace_count += 1
                    elif normalized[pos] == '}':
                        brace_count -= 1
                    pos += 1
                
                if brace_count == 0:
                    # Successful match, extract content
                    content = normalized[start_pos:pos-1].strip()
                    # Replace entire boxed{...} part
                    normalized = normalized[:start] + content + normalized[pos:]
        
        # Remove LaTeX text wrapper: \text{...}
        normalized = re.sub(r'\\text\{([^}]+)\}', r'\1', normalized)
        
        # Remove LaTeX format symbols
        normalized = re.sub(r'\\left\(', '(', normalized)
        normalized = re.sub(r'\\right\)', ')', normalized)
        normalized = re.sub(r'\\left\[', '[', normalized)
        normalized = re.sub(r'\\right\]', ']', normalized)
        
        # Standardize spaces (keep for later processing)
        normalized = normalized.strip()
        
        # Special handling: normalize option letters to lowercase
        # Convert (A), A, (a), a all to a
        normalized_no_space = re.sub(r'\s+', '', normalized)
        option_match = re.match(r'^\(?([a-zA-Z])\)?$', normalized_no_space)
        if option_match:
            return option_match.group(1).lower()
        
        # Standardize LaTeX commands: ensure all commands have backslash
        # Fix "frac" → "\frac", "sqrt" → "\sqrt", etc.
        for cmd in ['frac', 'dfrac', 'tfrac', 'sqrt', 'pi', 'sin', 'cos', 'tan', 'log', 'ln']:
            # If command doesn't have backslash, add it
            normalized = re.sub(rf'(?<!\\)\b{cmd}\b', rf'\\{cmd}', normalized)
        
        # Unify frac variants: \dfrac, \tfrac all converted to \frac
        normalized = re.sub(r'\\[dt]frac', r'\\frac', normalized)
        
        # Handle coordinate format: convert "15, -29" to "(15,-29)"
        # Match patterns like "number, number"
        coord_pattern = r'^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$'
        coord_match = re.match(coord_pattern, normalized.strip())
        if coord_match:
            normalized = f"({coord_match.group(1)},{coord_match.group(2)})"
        
        # Remove extra backslashes (if duplicated)
        normalized = re.sub(r'\\\\+', r'\\', normalized)
        
        # Now remove all spaces
        normalized = re.sub(r'\s+', '', normalized)
        
        # Standardize parentheses: handle mismatched parentheses
        open_count = normalized.count('(')
        close_count = normalized.count(')')
        
        if open_count > close_count:
            # Add missing closing parentheses
            normalized += ')' * (open_count - close_count)
        elif close_count > open_count:
            # Remove extra closing parentheses (from end)
            diff = close_count - open_count
            for _ in range(diff):
                idx = normalized.rfind(')')
                if idx != -1:
                    normalized = normalized[:idx] + normalized[idx+1:]
        
        # Convert to lowercase for text comparison
        normalized_lower = normalized.lower()
        
        return normalized_lower
    
    def _judge_answer_with_llm(self, model_response: str, question: str, ground_truth_answer: str, ground_truth_solution: str = "") -> bool:
        """Use LLM to directly judge if the model's answer is correct"""
        
        judgment_prompt = f"""You are a professional mathematics answer evaluator. Your task is to determine if the model's final answer is mathematically correct.

**Problem:**
{question}

**Ground Truth Answer:**
{ground_truth_answer}

**Model's Complete Response:**
{model_response}

**Evaluation Guidelines:**
1. **Focus ONLY on the final answer** - ignore the reasoning process quality
2. **Check mathematical equivalence** - the final numerical/symbolic result must match
3. **Ignore ALL formatting differences**:
   - LaTeX syntax: \\boxed{{}}, \\frac{{}}{{}}, \\text{{}}
   - Spacing and whitespace
   - Parentheses placement
   - Mathematical notation variants
4. **Consider mathematical equivalence**:
   - Different forms of the same expression (e.g., 2/4 = 0.5 = 1/2)
   - Algebraic commutativity (e.g., a+b = b+a)
   - Simplified vs unsimplified forms
5. **For choice questions**: (A), A, and "Option A" are all equivalent
6. **Extract the final answer yourself** from the model's response and compare

**CRITICAL**: If the final answer is mathematically correct, output CORRECT even if:
- The reasoning has minor issues
- The formatting is different
- The answer is expressed differently but is mathematically equivalent

Output EXACTLY one word: CORRECT or WRONG

Your judgment:"""
        
        try:
            result = self.extractor_llm.generate(judgment_prompt).strip().upper()
            # More strict judgment logic
            if 'CORRECT' in result:
                # Ensure it's not "INCORRECT" or "NOT CORRECT"
                if 'INCORRECT' in result or 'NOT CORRECT' in result or 'WRONG' in result:
                    return False
                return True
            return False
        except Exception as e:
            print(f"  LLM judgment failed: {e}")
            return False
    
    def _compare_answers(self, pred: str, gold: str) -> bool:
        """Compare if two mathematical answers are equal"""
        
        # First normalize answers
        pred_norm = self._normalize_answer(pred)
        gold_norm = self._normalize_answer(gold)
        
        # Directly compare normalized answers
        if pred_norm == gold_norm:
            return True
        
        # If LLM comparison is enabled and rule comparison fails, use LLM as fallback
        if self.use_llm_comparison and hasattr(self, 'extractor_llm') and self.extractor_llm:
            return self._judge_answer_with_llm(pred, "", gold)
        
        return False
    
    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate accuracy of predictions

        Args:
            predictions: List of model-generated answers
            references: List of dicts with ground truth answers and step information

        Returns:
            Dict containing accuracy
        """
        correct = 0
        total = len(predictions)
        
        print(f"\n{'='*60}")
        print(f" CompetitionMath Accuracy Evaluation")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"Answer comparison mode: {'LLM Semantic Comparison' if self.use_llm_comparison else 'Rule-based Comparison'}")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Get ground truth information
            question = ref.get('problem') or ref.get('question') or ref.get('input') or ''
            gold_answer = str(ref.get('answer') or ref.get('target') or ref.get('output') or '').strip()
            gold_solution = ref.get('solution') or ref.get('explanation') or ''
            
            # Use LLM judgment directly (without extracting answer)
            if self.use_llm_comparison and hasattr(self, 'extractor_llm') and self.extractor_llm:
                is_correct = self._judge_answer_with_llm(pred, question, gold_answer, gold_solution)
            else:
                # Fallback: extract answer then compare
                extracted_pred = self._extract_answer(pred)
                is_correct = self._compare_answers(extracted_pred, gold_answer)
            
            if is_correct:
                correct += 1
            
            # Show detailed information for first 5 samples
            if i < 5:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"\n[Sample {i+1}] {status}")
                print(f"   Problem: {question[:100]}...")
                print(f"   Level: {ref.get('level', 'N/A')} | Type: {ref.get('type', 'N/A')}")
                print(f"   Model Response: {pred[:150]}...")
                print(f"   Ground Truth: {gold_answer[:80]}...")
        
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f" Evaluation Results")
        print(f"{'='*60}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*60}\n")
        
        return {"accuracy": accuracy}
    
    def evaluate_with_step_analysis(self, predictions: List[str], references: List[Dict[str, Any]], 
                                   llm=None, factor_names: List[str] = None,
                                   current_prompt: str = None, factors_dict: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Evaluation with step-level error analysis

        Utilize CompetitionMath's solution field for fine-grained error localization
        """
        
        if llm is None:
            try:
                llm = get_llm("worker")
            except:
                print("  Cannot get LLM, skipping step-level analysis")
                return self.evaluate(predictions, references)
        
        # Use base_evaluator's step-level analysis
        result = self.collect_errors_with_step_analysis(
            predictions, references, 
            llm=llm,
            factor_names=factor_names,
            current_prompt=current_prompt,
            factors_dict=factors_dict
        )
        
        # Calculate accuracy
        accuracy = 1.0 - (result['total_errors'] / len(predictions)) if len(predictions) > 0 else 0.0
        result['accuracy'] = accuracy
        
        print(f"\n Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return result
