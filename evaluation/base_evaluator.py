from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Defines a standard interface for evaluating model predictions.
    """

    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model prediction performance.

        Args:
            predictions (List[str]): List of model generated outputs.
            references (List[Dict[str, Any]]): List of gold standard reference data, each element is a sample dict.

        Returns:
            Dict[str, float]: Dictionary containing evaluation scores, e.g., {'accuracy': 0.85}.
        """
        pass

    def __call__(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """Allow calling the object as a function."""
        return self.evaluate(predictions, references)

    def collect_errors_with_step_analysis(self, predictions: List[str], references: List[Dict[str, Any]],
                                         llm=None, factor_names: List[str] = None,
                                         current_prompt: str = None, factors_dict: Dict[str, str] = None,
                                         factor_selection_history: str = None,
                                         full_reasoning: List[str] = None,
                                         evaluation_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Collect and analyze error samples - fine-grained error localization based on reasoning steps

        Args:
            predictions: List of model generated predictions (extracted answers for correctness check)
            references: List of ground truth answers and step information
            llm: LLM for error analysis (skip LLM analysis if None)
            factor_names: List of actual factor names (for LLM analysis suggestions)
            current_prompt: Current complete prompt being used
            factors_dict: Factor dictionary {factor_name: factor_description}
            factor_selection_history: Factor selection history
            full_reasoning: List of model's full reasoning process (for error analysis, use predictions if None)
            evaluation_results: Existing evaluation results list (containing correct field), avoid duplicate judgment

        Returns:
            Dictionary containing error analysis results, including error type distribution and factor improvement suggestions
        """
        import re
        import json

        errors = []
        factor_effectiveness = {}

        print(f"\n[Collecting Errors] Total samples: {len(predictions)}, LLM analysis: {'enabled' if llm else 'disabled'}")

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            gold_answer = str(ref.get('answer') or ref.get('target') or ref.get('label') or '').strip()

            # Prefer using existing evaluation results (consistent with normal validation flow, including LLM judgment)
            if evaluation_results and i < len(evaluation_results):
                is_correct = evaluation_results[i].get('correct', False)
            else:
                # Fallback to simple comparison
                is_correct = self._check_correctness(str(pred), gold_answer)

            if not is_correct:
                ground_truth_steps = ref.get('solution') or ref.get('rationale') or ref.get('explanation') or ''

                # Get full reasoning process: prefer full_reasoning, fallback to pred
                model_full_output = full_reasoning[i] if full_reasoning and i < len(full_reasoning) else pred

                error_info = {
                    'sample_id': i,
                    'question': ref.get('question') or ref.get('input') or ref.get('prompt') or '',
                    'model_output': model_full_output,  # Full reasoning process for error analysis
                    'extracted_answer': pred,  # Extracted answer for display
                    'target_answer': gold_answer,
                    'ground_truth_steps': ground_truth_steps,
                }

                if llm:
                    # Analyze regardless of whether steps exist, pass steps if available, otherwise pass empty
                    has_steps = bool(ground_truth_steps)
                    analysis_type = "step-level analysis" if has_steps else "question-answer based analysis"
                    print(f"  [Sample {i}] Calling LLM for {analysis_type}...")
                    print(f"    Model answer: {pred}")
                    print(f"    Gold answer: {gold_answer}")
                    
                    step_analysis = self._do_step_analysis(
                        llm, error_info['question'], error_info['model_output'],
                        error_info['ground_truth_steps'], error_info['target_answer'],
                        factor_names=factor_names,
                        current_prompt=current_prompt,
                        factors_dict=factors_dict,
                        factor_selection_history=factor_selection_history
                    )
                    error_info['step_analysis'] = step_analysis

                    suggested_factor = step_analysis.get('suggested_factor')
                    if suggested_factor:  # As long as not None/empty, use directly
                        factor_effectiveness[suggested_factor] = factor_effectiveness.get(suggested_factor, 0) + 1

                    print(f"    Error description: {step_analysis.get('error_description', 'N/A')[:80]}")
                    print(f"    Suggested factor: {suggested_factor}, confidence: {step_analysis.get('confidence', 0.0):.2f}")

                errors.append(error_info)

        total_errors = len(errors)
        print(f"\n[Collection Complete] Total error samples: {total_errors}")
        print(f"           Factor effectiveness: {factor_effectiveness}\n")

        factor_priorities = {}
        if total_errors > 0 and factor_effectiveness:
            for factor, count in factor_effectiveness.items():
                factor_priorities[factor] = count / total_errors

        return {
            'total_errors': total_errors,
            'errors': errors,
            'factor_priorities': factor_priorities,
            'factor_effectiveness': factor_effectiveness
        }

    def _check_correctness(self, prediction: str, gold_answer: str) -> bool:
        """Basic correctness check (can be overridden by subclasses)"""
        if not gold_answer:
            return False

        pred_norm = str(prediction).strip().lower()
        gold_norm = str(gold_answer).strip().lower()

        if pred_norm == gold_norm:
            return True

        try:
            pred_num = float(pred_norm)
            gold_num = float(gold_norm)
            return abs(pred_num - gold_num) < 1e-6
        except:
            pass

        return False

    def _do_step_analysis(self, llm, question: str, model_output: str,
                          ground_truth_steps: str, target_answer: str,
                          factor_names: List[str] = None,
                          current_prompt: str = None,
                          factors_dict: Dict[str, str] = None,
                          factor_selection_history: str = None) -> Dict[str, Any]:
        """Use LLM for step-level error analysis"""

        # Build factor information
        if factor_names and factors_dict:
            factor_list = ", ".join(factor_names)
            factors_info = "\n".join([f"- **{name}**: {desc}" for name, desc in factors_dict.items()])
        elif factor_names:
            factor_list = ", ".join(factor_names)
            factors_info = "\n".join([f"- {name}" for name in factor_names])
        else:
            factor_list = "Task, Reasoning, Format"
            factors_info = "- Task\n- Reasoning\n- Format"
        
        # Build current prompt information
        prompt_context = ""
        if current_prompt:
            prompt_context = f"""
=== Current Prompt Being Used ===
{current_prompt}

=== Prompt Factors Structure ===
{factors_info}

"""
        
        # Add factor selection history information
        history_context = ""
        if factor_selection_history:
            history_context = f"""
=== Factor Selection History ===
{factor_selection_history}
"""
        
        # Build different analysis tasks depending on whether steps exist
        has_steps = bool(ground_truth_steps and str(ground_truth_steps).strip())

        if has_steps:
            # Has steps: show steps and require comparison
            steps_section = f"""
**Correct Solution Steps:**
{ground_truth_steps}
"""
            task_instruction = "Compare the model's output with the correct solution steps, then **choose which factor should be improved** to fix this error."
        else:
            # No steps: explain no steps available, analyze based on question and answer only
            steps_section = """
**Correct Solution Steps:** (Not available for this dataset)
"""
            task_instruction = "Based on the question, model's output, and correct answer, analyze what went wrong and **choose which factor should be improved** to fix this error."
        
        analysis_prompt = f"""You are an expert in prompt optimization and error analysis. 

{prompt_context}{history_context}
=== Error Analysis Task ===

The model was given the following problem and produced an incorrect answer. Please analyze:

**Question:**
{question}
{steps_section}
**Model's Output:**
{model_output}

**Correct Answer:** {target_answer}

**Your Task:**
{task_instruction}

**Available Factors:** {factor_list}

**Instructions:**
- You MUST select exactly ONE factor from the list above
- Choose the factor most relevant to fixing this specific error
- Output ONLY valid JSON, no explanation

**Output Format:**
Return a JSON object with these fields:
{{
    "error_description": "Brief description of what went wrong",
    "root_cause": "Why this error happened",
    "suggested_factor": "ONE factor name from [{factor_list}]",
    "confidence": "How likely this factor is the root cause of the error (0.0=unlikely, 1.0=very likely, must be between 0.0-1.0)"
}}

**Important:** The confidence score should reflect how strongly you believe THIS specific factor is responsible for the error. Higher confidence means this factor is more likely the root cause.

JSON Output:"""

        try:
            response = llm.generate(analysis_prompt)
            parsed_result = self._parse_analysis(response, factor_names)
            return parsed_result
        except Exception as e:
            print(f"     Step-level error analysis failed: {e}")
            # Even if analysis fails, return first factor instead of None
            fallback_factor = factor_names[0] if factor_names else None
            return {
                'error_description': 'Analysis failed',
                'root_cause': 'Unable to determine',
                'suggested_factor': fallback_factor,
                'confidence': 0.0  # Analysis failed, no model score
            }

    def _parse_analysis(self, response: str, factor_names: List[str] = None) -> Dict[str, Any]:
        """Parse LLM's step-level error analysis response"""
        import json
        import re

        result = {
            'error_description': '',
            'root_cause': '',
            'suggested_factor': None,
            'confidence': 0.0
        }

        # Try JSON parsing
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                result.update(parsed)
                
                # Fix suggested_factor matching
                if factor_names:
                    result['suggested_factor'] = self._match_factor_name(
                        result.get('suggested_factor'), factor_names
                    )

                # Ensure confidence is in [0.0, 1.0] range (if model provided it)
                if 'confidence' in result:
                    result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))

                return result
        except Exception as e:
            print(f"    JSON parsing failed: {e}")

        # Fallback to line-by-line parsing
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(x in line.lower() for x in ['error_description', 'error description']):
                result['error_description'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif any(x in line.lower() for x in ['root_cause', 'root cause']):
                result['root_cause'] = line.split(':', 1)[1].strip() if ':' in line else line
            elif any(x in line.lower() for x in ['suggested_factor', 'suggested factor']):
                factor = line.split(':', 1)[1].strip() if ':' in line else line
                if factor_names:
                    result['suggested_factor'] = self._match_factor_name(factor, factor_names)
            elif any(x in line.lower() for x in ['confidence']):
                try:
                    conf_str = line.split(':', 1)[1].strip() if ':' in line else line
                    confidence_val = float(re.search(r'[\d.]+', conf_str).group()) if re.search(r'[\d.]+', conf_str) else 0.0
                    # Ensure confidence is in [0.0, 1.0] range
                    result['confidence'] = max(0.0, min(1.0, confidence_val))
                except:
                    pass

        # Final validation: ensure suggested_factor is not None
        if result['suggested_factor'] is None and factor_names:
            result['suggested_factor'] = factor_names[0]

        # Ensure confidence is in reasonable range (if model provided it)
        if 'confidence' in result:
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))

        return result
    
    def _match_factor_name(self, suggested: str, factor_names: List[str]) -> str:
        """Loose matching of factor names"""
        if not factor_names:
            return None

        if not suggested:
            return factor_names[0]

        # Normalize string for matching (remove spaces, underscores, lowercase)
        def normalize(s):
            return s.lower().replace(' ', '').replace('_', '').replace('-', '')

        suggested_norm = normalize(suggested)

        # Try exact match
        for factor in factor_names:
            if normalize(factor) == suggested_norm:
                return factor

        # Try partial match
        for factor in factor_names:
            factor_norm = normalize(factor)
            if suggested_norm in factor_norm or factor_norm in suggested_norm:
                return factor

        # If no match, return first factor
        return factor_names[0] 