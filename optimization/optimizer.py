import logging
import numpy as np
import re
import copy
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

from ..llm_apis import BaseLLM, get_llm
from ..evaluation import BaseEvaluator
from .prompt_object import PromptStructure
from ..evaluation.unified_scoring import evaluate_with_unified_scoring

class Optimizer:
    """
    Responsible for using aPSF algorithm to optimize each factor in fusion prompt structure.
    Implements factor-level optimization through implicit factor positioning and local rewriting.
    """
    def __init__(self,
                 prompt_struct: PromptStructure,
                 eval_data: List[Dict[str, Any]],
                 evaluator: BaseEvaluator,
                 dataset_config: Dict[str, Any],
                 architect_llm_id: str = "architect",
                 worker_llm_id: str = "worker",
                 method_name: str = "aPSF",
                 enable_feedback: bool = False):  # New parameter
        
        self.prompt_struct = prompt_struct
        self.eval_data = eval_data
        self.evaluator = evaluator
        self.architect_llm = get_llm(architect_llm_id)
        self.worker_llm = get_llm(worker_llm_id)
        
        # Save dataset_config as instance variable
        self.dataset_config = dataset_config
        
        # Save method name for display
        self.method_name = method_name.upper() if method_name else "aPSF"
        
        # Get key information from dataset configuration
        self.input_key = dataset_config.get("input_key", "prompt")
        self.metric_key = dataset_config.get("metric")
        if not self.metric_key:
            raise ValueError("Dataset config must contain a 'metric' key.")

        self.metric_name = self.metric_key

        self.num_factors = len(self.prompt_struct.factors)
        self.candidates_per_step = dataset_config.get("candidates_per_step", 4)
        self.ucb1_exploration = dataset_config.get("ucb1_exploration_constant", np.sqrt(2))
        self.dap_patience_M = dataset_config.get("dap_patience_M", 4)
        self.dap_improvement_delta = dataset_config.get("dap_improvement_delta", 0.005)

        self.candidate_counts = [0] * self.num_factors
        self.candidate_values = [0.0] * self.num_factors
        self.total_steps = 0
        self.stagnation_counters = [0] * self.num_factors
        self.last_best_scores = [-1.0] * self.num_factors
        
        self.global_best_score = -1.0
        self.global_best_factor = None
        self.global_best_step = 0
        self.global_best_prompt_structure = None
        self.global_best_tokens = 0  # Cumulative token consumption when reaching optimum
        self.all_scores_history = []
        
        self.initial_score = self._evaluate_initial_structure()
        
        self.current_optimization_step = 0

        # New feedback related attributes
        self.enable_feedback = enable_feedback
        self.wrong_examples = []  # Store wrong questions
        self.feedback_history = []  # Store feedback history

        # New factor selection statistics
        self.factor_selection_history = []  # Details of factor selection per step
        self.factor_impact_stats = {}  # Impact statistics for each factor

        # Optimization stability statistics
        self.stability_stats = {
            'total_attempts': 0,  # Total number of attempts
            'successful_updates': 0,  # Number of successful updates
            'failed_attempts': 0,  # Number of failed attempts
            'per_factor_stats': {},  # Statistics for each factor
        }
        
        # Regression rate statistics
        self.regression_stats = {
            'total_accepted_updates': 0,  # Total number of accepted updates
            'total_regressions': 0,  # Total number of regressions
            'per_factor_regressions': {},  # Regression statistics for each factor
        }
        
        if self.enable_feedback:
            logging.info(" Feedback mechanism enabled")
    
    def _evaluate_initial_structure(self) -> float:
        """Evaluate the performance of initial fusion prompt structure as baseline"""
        logging.info(" Evaluating initial fusion prompt structure...")
        prompt_for_eval = self.prompt_struct.compose()
        predictions = self._generate_predictions(prompt_for_eval)
        
        # Use real-time determined accuracy instead of GSM8KEvaluator
        if hasattr(self, '_current_accuracy'):
            initial_score = self._current_accuracy
        else:
            # Fallback to original logic
            eval_results = self.evaluator.evaluate(predictions, self.eval_data)
            initial_score = eval_results.get(self.metric_name, 0.0)
            
        logging.info(f" Initial fusion structure score (validation): {initial_score:.4f}")
        logging.info(f" Initial fusion prompt: {self.prompt_struct.fusion_prompt}")

        self.global_best_score = initial_score
        # Fix: Use deep copy to save best structure, avoid contamination by subsequent modifications
        self.global_best_prompt_structure = self._deep_copy_structure(self.prompt_struct)
        # Record token consumption during initialization phase
        worker_stats = self.worker_llm.get_token_stats()
        architect_stats = self.architect_llm.get_token_stats()
        self.global_best_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']

        return initial_score

    def _generate_predictions(self, prompt_template: str, current_structure: PromptStructure = None) -> List[str]:
        """Generate predictions and perform unified scoring"""
        predictions = []
        correct_count = 0  # Add counter
        detailed_results = []  # Add this line
        
        # Use passed structure or default structure to display current prompt
        display_structure = current_structure if current_structure else self.prompt_struct
        
        for i, item in enumerate(self.eval_data):
            input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
            question = item.get(input_key, '')
            # Combine instruction with question directly
            formatted_prompt = f"{prompt_template}\n\n{question}"
            
            print(f"\n{'='*80}")
            print(f" aPSF validation sample {i+1}/{len(self.eval_data)}")
            print(f"{'='*80}")
            print(f" Question:\n   {question}")
            print(f"\n Current fusion prompt:\n   \"{prompt_template}\"")
            
            prediction = self.worker_llm.generate(formatted_prompt)
            predictions.append(prediction)
            print(f"\n aPSF response:\n   {prediction}")
            
            # Immediately process answer extraction and matching for each sample
            try:
                from ..evaluation.unified_scoring import UnifiedScorer
                scorer = UnifiedScorer(self.worker_llm, "apsf_validation")
                extracted_answer, target_answer, is_correct = scorer.extract_and_score(
                    prediction, item, self.evaluator
                )
                
                # Add: Save detailed results for feedback
                detailed_results.append({
                    'question': item.get('input', item.get('prompt', item.get('question', ''))),
                    'predicted_answer': prediction,
                    'extracted_answer': extracted_answer,
                    'correct_answer': target_answer,
                    'correct': is_correct,
                    'reasoning': prediction  # Complete reasoning process
                })
                
                # Display information (CompetitionMath uses LLM direct judgment, does not display extracted answers)
                evaluator_name = self.evaluator.__class__.__name__
                if evaluator_name == "CompetitionMathEvaluator":
                    # CompetitionMath: Display target answer and LLM judgment result
                    print(f"\n Target answer: '{target_answer}'")
                    verdict = " Correct" if is_correct else " Incorrect"
                    print(f" LLM judgment result: {verdict}")
                else:
                    # Other datasets: Display extracted answers
                    print(f"\n Intelligent extracted answer: '{extracted_answer}'")
                    print(f" Target answer: '{target_answer}'")
                    verdict = " Correct" if is_correct else " Incorrect"
                    print(f" Answer match: {verdict}")
                
                # Accumulate correct count
                if is_correct:
                    correct_count += 1
                    
            except Exception as e:
                print(f" Answer extraction failed: {e}")
                # Use simple answer extraction as alternative
                extracted_answer = self._extract_answer_from_prediction(prediction, item)
                target_answer = self._get_target_answer(item)
                is_correct = self._is_answer_correct(extracted_answer, target_answer, item)
                
                # Add: Save detailed results for feedback (exceptional cases)
                detailed_results.append({
                    'question': item.get('input', item.get('prompt', item.get('question', ''))),
                    'predicted_answer': prediction,
                    'extracted_answer': extracted_answer,
                    'correct_answer': target_answer,
                    'correct': is_correct,
                    'reasoning': prediction
                })
                
                # Display information (distinguish CompetitionMath)
                evaluator_name = self.evaluator.__class__.__name__
                if evaluator_name == "CompetitionMathEvaluator":
                    print(f"\n Target answer: '{target_answer}'")
                    verdict = " Correct" if is_correct else " Incorrect"
                    print(f" LLM judgment result: {verdict}")
                else:
                    print(f"  Standardized comparison: '{extracted_answer}' vs '{target_answer}'")
                    print(f"  Extracted answer: '{extracted_answer}'")
                    print(f"  Target answer: '{target_answer}'")
                    verdict = " Correct" if is_correct else " Incorrect"
                    print(f"  Answer match: {verdict}")

                # Exception path should also accumulate correct count, otherwise accuracy will be underestimated
                if is_correct:
                    correct_count += 1
        
        # Save detailed evaluation results for feedback mechanism
        self._last_evaluation_results = detailed_results
        
        # Set current accuracy to avoid repeated evaluation
        self._current_accuracy = correct_count / len(predictions) if predictions else 0.0
        
        return predictions

    def _generate_fusion_factor_candidates(self, factor_name: str, num_candidates: int = 4) -> List[str]:
        """
        Generate candidate replacement phrases for specific factors in fusion prompt.
        
        Args:
            factor_name: Name of factor to optimize
            num_candidates: Number of candidates to generate
            
        Returns:
            List[str]: List of candidate replacement phrases
        """
        meta_prompt = self.prompt_struct.create_optimization_meta_prompt(factor_name, num_candidates)
        
        print(f"\n Generating fusion candidates for factor '{factor_name}'...")
        print(f"{'─'*60}")
        print(" Candidate generation prompt sent to ARCHITECT:")
        print(meta_prompt)
        print(f"{'─'*60}")
        
        response = self.architect_llm.generate(meta_prompt)
        
        print(f" ARCHITECT response:")
        print(response)
        print(f"{'─'*60}")
        
        # Parse candidate phrases
        candidates = self._parse_fusion_candidates(response)
        
        # Ensure sufficient candidates
        if len(candidates) < num_candidates:
            candidates.extend(self._generate_default_fusion_candidates(factor_name, num_candidates - len(candidates)))
        
        return candidates[:num_candidates]

    def _parse_fusion_candidates(self, response: str) -> List[str]:
        """Parse fusion candidate response"""
        candidates = []
        
        # Find candidates marked with --- ALTERNATIVE ---
        alternative_pattern = r'---\s*ALTERNATIVE\s*---(.*?)(?=---\s*ALTERNATIVE\s*---|$)'
        matches = re.findall(alternative_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            candidate = match.strip()
            if candidate:
                # Clean candidate text
                candidate = re.sub(r'^\d+[\.\)]\s*', '', candidate)  # Remove numbering
                candidate = re.sub(r'^["\'\[\(]|["\'\]\)]$', '', candidate)  # Remove quotes and brackets
                candidate = candidate.strip()
                if candidate:  # Remove length restriction, allow candidates of any length
                    candidates.append(candidate)
        
        print(f" Parsed {len(candidates)} fusion candidates")
        
        return candidates

    def _generate_default_fusion_candidates(self, factor_name: str, num_needed: int) -> List[str]:
        """Generate default candidates for fusion factors"""
        factor_name_lower = factor_name.lower()
        defaults = []
        
        if any(keyword in factor_name_lower for keyword in ['tone', 'style']):
            defaults = [
                "in a professional manner",
                "with a helpful approach", 
                "in a clear and concise way",
                "using a structured method"
            ]
        elif any(keyword in factor_name_lower for keyword in ['format', 'method']):
            defaults = [
                "step by step",
                "systematically and thoroughly",
                "with detailed explanations",
                "in a logical sequence"
            ]
        elif any(keyword in factor_name_lower for keyword in ['perspective']):
            defaults = [
                "as if teaching a student",
                "from an expert's viewpoint",
                "with careful guidance",
                "as a helpful assistant"
            ]
        else:
            defaults = [
                "carefully and methodically",
                "with attention to detail",
                "using best practices",
                "thoroughly and accurately"
            ]
        
        return defaults[:num_needed]

    def _evaluate_fusion_candidate(self, factor_name: str, candidate_phrase: str) -> float:
        """
        Evaluate the performance of fusion factor candidates.
        Use LLM for intelligent factor replacement while maintaining fluency.
        """
        print(f"\n Testing fusion candidate: '{candidate_phrase}' for factor '{factor_name}'")

        # Create temporary structure for testing
        temp_structure = PromptStructure(
            task_description=self.prompt_struct.task_description,
            factors=self.prompt_struct.factors.copy(),
            fusion_prompt=self.prompt_struct.fusion_prompt,
            factor_mappings=self.prompt_struct.factor_mappings.copy() if hasattr(self.prompt_struct, 'factor_mappings') else {}
        )

        # Use LLM for intelligent factor replacement, maintaining fluency
        try:
            new_fusion_prompt = self._llm_smart_factor_replacement(
                current_prompt=temp_structure.fusion_prompt,
                factor_name=factor_name,
                old_factor_content=temp_structure.factors[factor_name],
                new_factor_content=candidate_phrase
            )

            if new_fusion_prompt and new_fusion_prompt != temp_structure.fusion_prompt:
                temp_structure.fusion_prompt = new_fusion_prompt
                temp_structure.factors[factor_name] = candidate_phrase
                print(f" LLM intelligent replacement completed")
                print(f" New fusion prompt: {new_fusion_prompt}")
            else:
                print(f" LLM replacement no changes, using direct update")
                temp_structure.factors[factor_name] = candidate_phrase
                temp_structure.fusion_prompt = self._regenerate_fusion_prompt_safe(temp_structure)

        except Exception as e:
            print(f" LLM intelligent replacement failed: {e}, using safe regeneration")
            temp_structure.factors[factor_name] = candidate_phrase
            temp_structure.fusion_prompt = self._regenerate_fusion_prompt_safe(temp_structure)

        # Evaluate the new fusion prompt
        test_prompt = temp_structure.compose()
        predictions = self._generate_predictions(test_prompt, temp_structure)

        # Use real-time accuracy
        score = getattr(self, '_current_accuracy', 0.0)

        # Feedback mechanism: If feedback is enabled, identify wrong questions and improve
        if self.enable_feedback and hasattr(self, '_last_evaluation_results'):
            self._collect_wrong_examples(self._last_evaluation_results)
            if self.wrong_examples:
                score = self._apply_feedback_improvement(temp_structure, score)

        print(f" Candidate score: {score:.4f}")

        return score

    def _llm_smart_factor_replacement(self, current_prompt: str, factor_name: str,
                                    old_factor_content: str, new_factor_content: str) -> str:
        """Precise string replacement of factor content without changing other factors"""

        # Direct string replacement
        if old_factor_content in current_prompt:
            new_prompt = current_prompt.replace(old_factor_content, new_factor_content, 1)
            return new_prompt

        # If exact match fails, return None, let caller handle
        print(f"  Warning: Unable to find factor content '{old_factor_content[:50]}...' in prompt")
        return None

    def _collect_wrong_examples(self, evaluation_results: List[Dict[str, Any]]):
        """Collect error samples and perform open-ended analysis"""
        self.wrong_examples.clear()
        
        for result in evaluation_results:
            if not result.get('correct', False):
                # Perform open-ended error analysis
                error_analysis = self._analyze_error_type(result)
                
                wrong_example = {
                    'question': result.get('question', ''),
                    'correct_answer': result.get('correct_answer', ''),
                    'predicted_answer': result.get('predicted_answer', ''),
                    'reasoning': result.get('reasoning', ''),
                    'error_analysis': error_analysis,  # Store complete open-ended analysis
                    'error_type': error_analysis.get('error_type', 'Unknown Error')  # Maintain compatibility
                }
                self.wrong_examples.append(wrong_example)
                
        if self.wrong_examples:
            print(f" Collected and analyzed {len(self.wrong_examples)} error samples")
    
    def _analyze_error_type(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Use architect model for completely open-ended error analysis"""
        question = result.get('question', '')
        correct_answer = str(result.get('correct_answer', ''))
        predicted_answer = str(result.get('predicted_answer', ''))
        reasoning = result.get('reasoning', '')
        
        # Completely open-ended error analysis prompt
        open_error_analysis_prompt = f"""
You are a professional AI error analysis expert. Please conduct a deep analysis of the following AI response error.

**Question:**
{question}

**Correct Answer:**
{correct_answer}

**AI Predicted Answer:**
{predicted_answer}

**AI Reasoning Process:**
{reasoning}

Please conduct a comprehensive error analysis without being limited to any predefined categories. Please freely:

1. **Identify Error Essence**: What is the root cause of this error?
2. **Define Error Type**: What type of error do you think this belongs to? Please name it with concise terminology
3. **Analyze Error Mechanism**: Why did this error occur? What caused it?
4. **Assess Error Impact**: What impact does this error have on overall reasoning?
5. **Propose Specific Improvements**: How should the prompt be improved to address this specific error?

**Response Format:**
Error Essence: [Deep analysis of the root cause of the error]
Error Type: [Your self-defined error type name]
Error Mechanism: [Detailed explanation of how the error was generated]
Error Impact: [Analysis of impact on overall reasoning]
Improvement Suggestion: [Specific prompt improvement suggestions]
"""

        try:
            print(f" Performing open-ended intelligent error analysis...")
            response = self.architect_llm.generate(open_error_analysis_prompt)
            
            # Parse open-ended analysis results
            analysis_result = self._parse_open_analysis_response(response)
            
            print(f" Open-ended analysis results:")
            print(f"   Error type: {analysis_result['error_type']}")
            print(f"   Error essence: {analysis_result['error_essence']}")
            
            return analysis_result
            
        except Exception as e:
            print(f" Open-ended error analysis failed: {e}")
            return self._fallback_open_analysis(result)

    def _parse_open_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse open-ended analysis response"""
        result = {
            'error_type': '',
            'error_essence': '',
            'error_mechanism': '',
            'error_impact': '',
            'improvement_suggestion': '',
            'full_analysis': response
        }
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('Error Essence:'):
                result['error_essence'] = line.split(':', 1)[1].strip()
            elif line.startswith('Error Type:'):
                result['error_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('Error Mechanism:'):
                result['error_mechanism'] = line.split(':', 1)[1].strip()
            elif line.startswith('Error Impact:'):
                result['error_impact'] = line.split(':', 1)[1].strip()
            elif line.startswith('Improvement Suggestion:'):
                result['improvement_suggestion'] = line.split(':', 1)[1].strip()
        
        # If no error type parsed, try to extract from text
        if not result['error_type']:
            result['error_type'] = self._extract_error_type_from_text(response)
        
        return result

    def _extract_error_type_from_text(self, text: str) -> str:
        """Intelligently extract error type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['calculation', 'math']):
            return "Math calculation related error"
        elif any(word in text_lower for word in ['logic', 'reasoning']):
            return "Logic reasoning related error"
        elif any(word in text_lower for word in ['understanding', 'comprehension']):
            return "Understanding related error"
        elif any(word in text_lower for word in ['format', 'structure']):
            return "Format structure related error"
        else:
            return "Comprehensive error"

    def _fallback_open_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Alternative open-ended analysis"""
        return {
            'error_type': 'Unanalyzed Error',
            'error_essence': 'Analysis failed, manual check required',
            'error_mechanism': 'Unknown',
            'error_impact': 'May affect overall performance',
            'improvement_suggestion': 'Suggest checking prompt generality',
            'full_analysis': 'Analysis process encountered exception'
        }

    def _apply_feedback_improvement(self, temp_structure: PromptStructure, original_score: float) -> float:
        """Apply feedback improvement based on open-ended error analysis - select factor optimization based on error type"""
        if not self.wrong_examples:
            return original_score
        
        print(f" Applying intelligent feedback improvement, based on {len(self.wrong_examples)} error questions")
        
        # Count error type distribution
        error_types = []
        for example in self.wrong_examples:
            error_analysis = example.get('error_analysis', {})
            if isinstance(error_analysis, dict):
                error_types.append(error_analysis.get('error_type', 'Unknown Error'))
        
        from collections import Counter
        error_counts = Counter(error_types)
        
        print(f" Open-ended error type distribution:")
        for error_type, count in error_counts.items():
            print(f"   {error_type}: {count} times")
        
        # Determine which factor needs optimization based on error type
        primary_error_type = error_counts.most_common(1)[0][0] if error_counts else None
        target_factor = self._map_error_to_factor(primary_error_type)
        
        if target_factor and target_factor in temp_structure.factors:
            print(f" Selected optimization factor '{target_factor}' based on error type '{primary_error_type}'")
            
            # Generate improvement suggestions for this factor
            improvement_suggestions = self._generate_improvement_suggestions()
            
            # Improve specific factor
            improved_score = self._improve_specific_factor(
                temp_structure, target_factor, improvement_suggestions, original_score
            )
            
            if improved_score > original_score:
                print(f" Factor-level feedback improvement successful! {original_score:.4f} → {improved_score:.4f}")
                self._log_feedback_success(improvement_suggestions, improved_score - original_score)
                return improved_score
            else:
                print(f" Factor-level feedback improvement ineffective, keeping original score: {original_score:.4f}")
        else:
            print(f" Unable to map error type to factor, skipping feedback improvement")
        
        return original_score
    
    def _map_error_to_factor(self, error_type: str) -> str:
        """Let LLM intelligently decide which factor to modify

        Let architect LLM directly select the most relevant factor based on error analysis.
        This is smarter, more accurate, and more general than keyword matching.
        """
        if not error_type:
            return None
        
        factor_names = list(self.prompt_struct.factors.keys())
        
        if not factor_names:
            return None
        
        # Collect detailed analysis of all error samples
        error_details = []
        for i, example in enumerate(self.wrong_examples[:3]):  # Show only first 3 as representatives
            error_analysis = example.get('error_analysis', {})
            error_details.append(f"""
Error Sample {i+1}:
- Question: {example.get('question', '')[:200]}...
- Error Type: {error_analysis.get('error_type', 'Unknown')}
- Error Essence: {error_analysis.get('error_essence', 'Unknown')}
- Improvement Suggestion: {error_analysis.get('improvement_suggestion', 'Unknown')}
""")
        
        # Build factor selection prompt
        factor_selection_prompt = f"""You are an expert in prompt optimization. Based on error analysis, you need to select which factor should be improved.

**Available Factors:**
{chr(10).join(f"{i+1}. {name}: {content}" for i, (name, content) in enumerate(self.prompt_struct.factors.items()))}

**Error Analysis Summary:**
- Primary Error Type: {error_type}
- Total Wrong Examples: {len(self.wrong_examples)}

**Representative Error Samples:**
{''.join(error_details)}

**Task:**
Which factor is MOST relevant to fix these errors? Consider:
1. Which factor's improvement would most directly address the error type?
2. Which factor's content is most related to the error mechanism?
3. Which factor has the highest potential to prevent similar errors?

**Important:** 
- Output ONLY the factor name, exactly as it appears in the list above
- Do NOT add any explanation or extra text
- Choose the single most relevant factor

Your answer (factor name only):"""

        try:
            print(f" Let LLM intelligently select target factor...")
            selected_factor = self.architect_llm.generate(factor_selection_prompt).strip()

            # Clean possible extra text
            for factor_name in factor_names:
                if factor_name in selected_factor:
                    print(f" LLM selected factor: '{factor_name}'")
                    return factor_name

            # If LLM returns invalid factor name, try fuzzy matching
            selected_lower = selected_factor.lower()
            for factor_name in factor_names:
                if factor_name.lower() in selected_lower or selected_lower in factor_name.lower():
                    print(f" LLM selected factor (fuzzy match): '{factor_name}'")
                    return factor_name

            # If still not found, return first factor
            print(f" LLM returned '{selected_factor}' not in factor list, using first factor")
            return factor_names[0]

        except Exception as e:
            print(f" LLM factor selection failed: {e}, using first factor")
            return factor_names[0]
    
    def _improve_specific_factor(self, temp_structure: PromptStructure, 
                                 target_factor: str, 
                                 improvement_suggestions: List[str],
                                 original_score: float) -> float:
        """Improve specific factor based on error analysis

        Args:
            temp_structure: Temporary prompt structure
            target_factor: Target factor name to improve
            improvement_suggestions: List of improvement suggestions
            original_score: Original score

        Returns:
            Improved score
        """
        old_factor_content = temp_structure.factors[target_factor]
        
        # Generate improvement prompt for this factor
        factor_improvement_prompt = f"""You are a prompt engineering expert. Based on error analysis, we need to improve a specific factor in the prompt.

**Target Factor:** {target_factor}
**Current Factor Content:** {old_factor_content}

**Identified Issues:**
{chr(10).join(f"- {suggestion}" for suggestion in improvement_suggestions)}

Please provide an improved version of this factor that:
1. Addresses the identified issues specifically
2. Maintains the original intent and style
3. Adds necessary guidance to prevent these errors
4. Keeps it concise and clear

Please output ONLY the improved factor content, without any explanation:"""

        try:
            # Use LLM to improve this factor
            new_factor_content = self.architect_llm.generate(factor_improvement_prompt).strip()

            if new_factor_content and new_factor_content != old_factor_content:
                print(f" Factor improvement content:")
                print(f"  Original: {old_factor_content}")
                print(f"  Improved: {new_factor_content}")

                # Use LLM intelligent replacement to put this factor into fusion prompt
                new_fusion_prompt = self._llm_smart_factor_replacement(
                    current_prompt=temp_structure.fusion_prompt,
                    factor_name=target_factor,
                    old_factor_content=old_factor_content,
                    new_factor_content=new_factor_content
                )

                if new_fusion_prompt and new_fusion_prompt != temp_structure.fusion_prompt:
                    # Apply improvement
                    temp_structure.factors[target_factor] = new_factor_content
                    temp_structure.fusion_prompt = new_fusion_prompt

                    # Re-evaluate
                    test_prompt = temp_structure.compose()
                    predictions = self._generate_predictions(test_prompt, temp_structure)

                    improved_score = getattr(self, '_current_accuracy', 0.0)
                    return improved_score
                else:
                    print(f" LLM intelligent replacement failed, keeping original factor")
            else:
                print(f" LLM did not generate valid improvement content")

        except Exception as e:
            print(f" Factor improvement process error: {e}")
        
        return original_score
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on open-ended error analysis"""
        print(f" Generating improvement suggestions based on {len(self.wrong_examples)} error samples")

        # Collect all error analyses
        error_analyses = []
        error_types = []
        improvement_suggestions = []

        for example in self.wrong_examples:
            error_analysis = example.get('error_analysis', {})
            if isinstance(error_analysis, dict):
                error_types.append(error_analysis.get('error_type', 'Unknown Error'))
                improvement_suggestions.append(error_analysis.get('improvement_suggestion', ''))
                error_analyses.append(error_analysis.get('full_analysis', ''))
        
        # Use architect model to comprehensively analyze all errors
        comprehensive_analysis_prompt = f"""
Based on the analysis of the following {len(self.wrong_examples)} error samples, please provide comprehensive prompt improvement suggestions:

**Error Type Distribution:**
{chr(10).join(f"- {error_type}" for error_type in error_types)}

**Specific Improvement Suggestions:**
{chr(10).join(f"- {suggestion}" for suggestion in improvement_suggestions if suggestion)}

Please generate 3-5 most important prompt improvement suggestions based on the above information, with the following requirements:
1. Highly targeted to solve the identified main problems
2. Specific and actionable, can be directly applied to the prompt
3. Clear priority, sorted by importance

**Response Format:**
1. [Most important improvement suggestion]
2. [Second most important improvement suggestion]
3. [Third most important improvement suggestion]
...
"""

        try:
            print(f" Generating comprehensive improvement suggestions...")
            response = self.architect_llm.generate(comprehensive_analysis_prompt)

            # Parse improvement suggestions
            suggestions = self._parse_improvement_suggestions(response)

            print(f" Generated improvement suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")

            return suggestions

        except Exception as e:
            print(f" Failed to generate comprehensive improvement suggestions: {e}")
            # Fallback: Use individual suggestions directly
            return [suggestion for suggestion in improvement_suggestions if suggestion][:3]

    def _parse_improvement_suggestions(self, response: str) -> List[str]:
        """Parse improvement suggestions response"""
        suggestions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match suggestions starting with numbers
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean numbers and symbols
                cleaned_suggestion = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '·']:
                    if cleaned_suggestion.startswith(prefix):
                        cleaned_suggestion = cleaned_suggestion[len(prefix):].strip()
                        break

                if cleaned_suggestion:
                    suggestions.append(cleaned_suggestion)
        
        return suggestions[:5]  # Return at most 5 suggestions

    def _apply_improvements_to_prompt(self, temp_structure: PromptStructure, suggestions: List[str]) -> str:
        """Apply improvement suggestions to prompt"""
        if not suggestions:
            return temp_structure.fusion_prompt
        
        # Use architect LLM to improve prompt
        improvement_prompt = f"""
The current prompt has the following issues:
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}

Please improve the following prompt based on these issues:
{temp_structure.fusion_prompt}

Please provide the improved prompt, maintaining the original structure but adding necessary improvements:
"""
        
        try:
            improved_prompt = self.architect_llm.generate(improvement_prompt)
            print(f" Applied improvement suggestions to generate new prompt")
            return improved_prompt.strip()
        except Exception as e:
            print(f" Error when applying improvement suggestions: {e}")
            return temp_structure.fusion_prompt
    
    def _log_feedback_success(self, suggestions: List[str], improvement: float):
        """Record successful feedback improvement"""
        feedback_record = {
            'suggestions': suggestions,
            'improvement': improvement,
            'step': self.current_optimization_step,
            'wrong_examples_count': len(self.wrong_examples)
        }
        self.feedback_history.append(feedback_record)
        
        print(f" Feedback record saved, cumulative improvement: {improvement:.4f}")

    def step(self):
        """
        Execute one round of aPSF optimization: Single-factor improvement based on error analysis

        Process:
        Step 1: Error-driven factor selection - Use step-level error analysis to derive improvement factor
        Step 2: Candidate generation and semantic filtering - Generate and filter improvement solutions
        Step 3: train-50 full evaluation and conservative acceptance - Use +1/50 threshold
        """
        total_optimization_steps = self.dataset_config.get("total_optimization_steps", 10)
        
        if self.current_optimization_step >= total_optimization_steps:
            print("Maximum optimization steps reached")
            return
        
        self.current_optimization_step += 1
        print(f"\n{'='*80}")
        print(f"aPSF optimization step {self.current_optimization_step}/{total_optimization_steps}")
        print(f"{'='*80}")
        
        current_score = self.global_best_score
        factor_names = self.prompt_struct.get_factor_names()
        
        if not factor_names:
            print("No factors to optimize")
            return
        
        print(f"\nStep 1: Error-driven factor selection")
        print(f"{'─'*80}")

        print(f"Running current prompt on train-50 and collecting errors...")
        current_prompt = self.prompt_struct.compose()
        predictions = self._generate_predictions(current_prompt)

        # Get extracted answers and complete reasoning process for error analysis
        # _generate_predictions saves detailed results to self._last_evaluation_results
        extracted_answers = []
        full_reasoning = []  # Complete reasoning process
        if hasattr(self, '_last_evaluation_results') and self._last_evaluation_results:
            # Use extracted answers (for correctness judgment)
            extracted_answers = [result['extracted_answer'] for result in self._last_evaluation_results]
            # Complete reasoning process (for error analysis)
            full_reasoning = [result.get('reasoning', result['predicted_answer']) for result in self._last_evaluation_results]
        else:
            # Fallback: Use complete output (not recommended, but ensures compatibility)
            extracted_answers = predictions
            full_reasoning = predictions

        print(f"\nError diagnosis based on step-level analysis...")
        print(f"Available factors: {', '.join(factor_names)}")

        # Pass current prompt and factor information to let LLM see complete context
        # Add factor selection history information to avoid over-focusing on single factor
        factor_selection_history_summary = self._get_recent_factor_selection_summary(factor_names)
        
        error_analysis = self.evaluator.collect_errors_with_step_analysis(
            extracted_answers, self.eval_data,
            llm=self.architect_llm,
            factor_names=factor_names,
            current_prompt=current_prompt,
            factors_dict=self.prompt_struct.factors,
            factor_selection_history=factor_selection_history_summary,
            full_reasoning=full_reasoning,  # Pass complete reasoning process for error analysis
            evaluation_results=self._last_evaluation_results  # Pass evaluation results to maintain consistent correctness judgment
        )
        
        # Collect confidence statistics
        factor_confidences = {}  # {factor_name: [confidence_list]}
        for err in error_analysis.get('errors', []):
            if 'step_analysis' in err:
                analysis = err['step_analysis']
                suggested = analysis.get('suggested_factor')
                confidence = analysis.get('confidence', 0.0)
                if suggested:
                    if suggested not in factor_confidences:
                        factor_confidences[suggested] = []
                    factor_confidences[suggested].append(confidence)
        
        # Calculate average confidence
        factor_avg_confidence = {}
        for factor, confs in factor_confidences.items():
            factor_avg_confidence[factor] = sum(confs) / len(confs) if confs else 0.0
        
        print(f"\n{'='*80}")
        print(f"Error analysis detailed results")
        print(f"{'='*80}")
        print(f"Total errors: {error_analysis['total_errors']}/50")
        print(f"Factor improvement effectiveness: {error_analysis['factor_effectiveness']}")
        print(f"Factor improvement priorities: {error_analysis['factor_priorities']}")
        print(f"\nFactor selection probability distribution:")
        for factor in factor_names:
            prob = error_analysis['factor_priorities'].get(factor, 0.0)
            avg_conf = factor_avg_confidence.get(factor, 0.0)
            count = error_analysis['factor_effectiveness'].get(factor, 0)
            print(f"  - {factor}: Selection probability={prob:.2%}, Average confidence={avg_conf:.2f}, Selection count={count}")
        
        if error_analysis['total_errors'] > 0 and error_analysis['errors']:
            print(f"\nError sample examples (first 3):")
            for i, err in enumerate(error_analysis['errors'][:3]):
                print(f"  Sample {i+1}: ID={err['sample_id']}, Answer={err['target_answer']}")
                if 'step_analysis' in err:
                    analysis = err['step_analysis']
                    print(f"    - Error description: {analysis.get('error_description', 'N/A')}")
                    print(f"    - Root cause: {analysis.get('root_cause', 'N/A')}")
                    print(f"    - Suggested factor: {analysis.get('suggested_factor', 'N/A')}")
                    print(f"    - Confidence: {analysis.get('confidence', 0.0):.2f}")
        
        print(f"{'='*80}\n")
        
        if error_analysis['total_errors'] == 0:
            print("No errors, optimization complete!")
            return

        factor_priorities = error_analysis.get('factor_priorities', {})

        if not factor_priorities:
            print("Unable to derive factor priorities from error analysis, using round-robin selection")
            selected_factor = factor_names[self.current_optimization_step % len(factor_names)]
        else:
            selected_factor = max(factor_priorities.items(), key=lambda x: x[1])[0]

        if selected_factor not in factor_names:
            print(f"Recommended factor '{selected_factor}' not in factor list, using first factor")
            selected_factor = factor_names[0]
        
        # Save factor selection details for this step
        step_factor_info = {
            'step': self.current_optimization_step,
            'selected_factor': selected_factor,
            'factor_priorities': factor_priorities,
            'factor_effectiveness': error_analysis['factor_effectiveness'],
            'factor_avg_confidence': factor_avg_confidence,
            'total_errors': error_analysis['total_errors']
        }
        self.factor_selection_history.append(step_factor_info)
        
        print(f"\n{'='*80}")
        print(f"Factor selection decision")
        print(f"{'='*80}")
        print(f"Selected improvement factor: {selected_factor}")
        print(f"Factor priority score: {factor_priorities.get(selected_factor, 0):.2%}")
        print(f"Average confidence: {factor_avg_confidence.get(selected_factor, 0.0):.2f}")
        print(f"Improvement reason: Among {error_analysis.get('total_errors')} errors, this factor is most effective for improvement")
        print(f"{'='*80}\n")
        
        print(f"\nStep 2: Intelligent optimization based on error analysis")
        print(f"{'─'*80}")

        # New method: Let LLM directly optimize complete prompt based on error analysis
        num_candidates = self.dataset_config.get("candidates_per_step", 4)
        print(f"Let optimization LLM optimize complete prompt based on error analysis, generating {num_candidates} candidates...")

        candidates_with_mappings = self._generate_error_driven_complete_prompts(
            selected_factor, error_analysis, num_candidates
        )

        print(f"\nGenerated factor description candidates:")
        for i, item in enumerate(candidates_with_mappings, 1):
            factor_desc = item['factor_description'][:80]
            print(f"  Candidate {i}: {factor_desc}{'...' if len(item['factor_description']) > 80 else ''}")

        # Semantic filtering logic
        if self.dataset_config.get("semantic_filtering_enabled", False):
            print(f"\nPerforming semantic filtering...")
            top_k = self.dataset_config.get("top_k_after_filtering", 2)

            # Extract candidate factor description list
            candidate_descs = [item['factor_description'] for item in candidates_with_mappings]

            # Call semantic filtering function
            filtered_descs = self._semantic_filter_candidates(
                candidate_descs,
                selected_factor,
                top_k
            )

            # Keep only selected candidates
            candidates_with_mappings = [
                item for item in candidates_with_mappings
                if item['factor_description'] in filtered_descs
            ][:top_k]

            print(f"Semantic filtering complete, retained {len(candidates_with_mappings)} out of {len(candidate_descs)} candidates")
        
        print(f"\nStep 3: train-50 full evaluation and conservative acceptance")
        print(f"{'─'*80}")

        candidate_scores = []
        candidate_prompts = []  # Store constructed complete prompts
        for i, cand_item in enumerate(candidates_with_mappings):
            print(f"\nEvaluating candidate {i+1}/{len(candidates_with_mappings)}")

            # Construct complete prompt from factor description for evaluation
            old_factor_content = self.prompt_struct.factors[selected_factor]
            new_factor_content = cand_item['factor_description']

            # First try LLM intelligent replacement to construct complete prompt
            complete_prompt = self._llm_smart_factor_replacement(
                current_prompt=self.prompt_struct.fusion_prompt,
                factor_name=selected_factor,
                old_factor_content=old_factor_content,
                new_factor_content=new_factor_content
            )

            if not complete_prompt:
                # Alternative: Direct string replacement
                complete_prompt = self.prompt_struct.fusion_prompt.replace(
                    old_factor_content, new_factor_content
                )

            candidate_prompts.append(complete_prompt)
            score = self._evaluate_complete_prompt_candidate(complete_prompt)
            candidate_scores.append(score)
            print(f"  Candidate {i+1} score: {score:.4f}")

            # Fix issue 2: Record each evaluation to history
            self.all_scores_history.append({
                'step': self.current_optimization_step,
                'factor': selected_factor,
                'candidate_idx': i,
                'score': score,
                'accepted': False  # Update later
            })
        
        best_idx = np.argmax(candidate_scores)
        best_score = candidate_scores[best_idx]
        best_candidate_item = candidates_with_mappings[best_idx]
        best_complete_prompt = candidate_prompts[best_idx]  # Use constructed complete prompt

        print(f"\n{'='*80}")
        print(f"Best candidate evaluation results")
        print(f"{'='*80}")
        print(f"Best complete prompt: {best_complete_prompt[:100]}{'...' if len(best_complete_prompt) > 100 else ''}")
        print(f"Best factor description: {best_candidate_item['factor_description'][:80]}...")
        print(f"Best candidate score: {best_score:.4f}")
        print(f"Current best score: {current_score:.4f}")
        print(f"Score difference: {best_score - current_score:.4f}")

        acceptance_threshold = self.dataset_config.get("acceptance_threshold", 1)
        acceptance_threshold_score = current_score + (acceptance_threshold / len(self.eval_data))

        print(f"\nAcceptance rule judgment (threshold: +{acceptance_threshold}/50 = +{acceptance_threshold/len(self.eval_data):.4f})")

        # Record attempt
        self.stability_stats['total_attempts'] += 1

        if best_score >= acceptance_threshold_score:
            print(f"\n  Accept improvement: {current_score:.4f} → {best_score:.4f} (+{best_score - current_score:.4f})")
            print(f"New complete prompt:")
            print(f"  {best_complete_prompt}")
            print(f"\nUpdated '{selected_factor}' factor description:")
            print(f"  {best_candidate_item['factor_description']}")
            
            # Fix issue 2: Mark best candidate as accepted
            if self.all_scores_history:
                for record in self.all_scores_history:
                    if (record['step'] == self.current_optimization_step and 
                        record['factor'] == selected_factor and 
                        record['candidate_idx'] == best_idx):
                        record['accepted'] = True
                        break
            
            # Record successful update
            self.stability_stats['successful_updates'] += 1
            
            # Update regression statistics
            self.regression_stats['total_accepted_updates'] += 1
            
            # Check if it's a regression (score decrease)
            if best_score < current_score:
                self.regression_stats['total_regressions'] += 1
                if selected_factor not in self.regression_stats['per_factor_regressions']:
                    self.regression_stats['per_factor_regressions'][selected_factor] = 0
                self.regression_stats['per_factor_regressions'][selected_factor] += 1

            # Update statistics for each factor
            if selected_factor not in self.stability_stats['per_factor_stats']:
                self.stability_stats['per_factor_stats'][selected_factor] = {
                    'attempts': 0,
                    'successes': 0,
                    'failures': 0
                }
            self.stability_stats['per_factor_stats'][selected_factor]['attempts'] += 1
            self.stability_stats['per_factor_stats'][selected_factor]['successes'] += 1

            # Update prompt structure: Directly use complete prompt constructed during evaluation
            self.prompt_struct.fusion_prompt = best_complete_prompt
            self.prompt_struct.factors[selected_factor] = best_candidate_item['factor_description']

            self.global_best_score = best_score
            self.global_best_factor = selected_factor
            self.global_best_step = self.current_optimization_step
            self.global_best_prompt_structure = self._deep_copy_structure(self.prompt_struct)
            # Record cumulative token consumption when reaching optimum
            worker_stats = self.worker_llm.get_token_stats()
            architect_stats = self.architect_llm.get_token_stats()
            self.global_best_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']

            self.stagnation_counters = [0] * len(factor_names)

            # Fix issue 1: Update UCB and DAP statistics
            factor_idx = factor_names.index(selected_factor)
            self._update_ucb_scores(factor_idx, candidate_scores)
            self._update_dap_statistics(factor_idx, best_score)
        else:
            print(f"\n  Reject: Insufficient improvement ({best_score:.4f} < {acceptance_threshold_score:.4f})")

            # Record failed attempt
            self.stability_stats['failed_attempts'] += 1

            # Update statistics for each factor
            if selected_factor not in self.stability_stats['per_factor_stats']:
                self.stability_stats['per_factor_stats'][selected_factor] = {
                    'attempts': 0,
                    'successes': 0,
                    'failures': 0
                }
            self.stability_stats['per_factor_stats'][selected_factor]['attempts'] += 1
            self.stability_stats['per_factor_stats'][selected_factor]['failures'] += 1

            factor_idx = factor_names.index(selected_factor) if selected_factor in factor_names else 0
            self.stagnation_counters[factor_idx] += 1

            # Fix issue 1: Update statistics even on rejection (record this attempt)
            self._update_ucb_scores(factor_idx, candidate_scores)
            self._update_dap_statistics(factor_idx, current_score)
        
        print(f"{'='*80}\n")
        
        self._update_structure_stats()
        self._print_apsf_step_summary(selected_factor, best_score, best_score >= acceptance_threshold_score)
    
    def _get_recent_factor_selection_summary(self, factor_names: List[str], window_size: int = 5) -> str:
        """
        Get summary of recent N-step factor selection history, used to prompt LLM to consider other factors

        Args:
            factor_names: List of all factor names
            window_size: Observation window size (recent N steps)

        Returns:
            Text summary of factor selection history
        """
        if not self.factor_selection_history:
            return ""
        
        # Get selection records for recent N steps
        recent_history = self.factor_selection_history[-window_size:]
        
        # Count selections for each factor in recent N steps
        recent_selections = {}
        for step_info in recent_history:
            selected = step_info['selected_factor']
            recent_selections[selected] = recent_selections.get(selected, 0) + 1

        # Find underexplored or rarely selected factors
        underexplored_factors = []
        for factor in factor_names:
            count = recent_selections.get(factor, 0)
            if count == 0:
                underexplored_factors.append(f"{factor} (Not selected)")
            elif count <= 1:
                underexplored_factors.append(f"{factor} (Selected {count} times)")

        # Find over-selected factors
        overexplored_factors = []
        for factor, count in recent_selections.items():
            if count >= window_size * 0.5:  # If proportion exceeds 50%
                overexplored_factors.append(f"{factor} (Selected {count} times)")

        # Build summary
        summary_parts = []
        summary_parts.append(f"\n**Recent {len(recent_history)} factor selection history:**")
        summary_parts.append(f"- Frequently selected: {', '.join(overexplored_factors) if overexplored_factors else 'None'}")
        summary_parts.append(f"- Less explored: {', '.join(underexplored_factors) if underexplored_factors else 'None'}")

        # Add recommendation if over-selection exists
        if overexplored_factors and underexplored_factors:
            summary_parts.append(f"\n**Recommendation:** If {', '.join(underexplored_factors)} are also relevant to solving the current errors, consider giving them opportunities to ensure balanced factor exploration and avoid over-focusing on a single factor.")        
        return "\n".join(summary_parts)
    
    def _generate_error_driven_complete_prompts(self, target_factor: str, error_analysis: Dict, num_candidates: int = 2) -> List[Dict]:
        """
        Generate optimized complete prompts based on error analysis

        Lets the LLM see:
        1. Current complete prompt
        2. Error analysis results
        3. Factor to optimize

        Output:
        1. Optimized complete prompt
        2. Updated factor description

        Args:
            target_factor: Factor name to optimize
            error_analysis: Error analysis results
            num_candidates: Number of candidates to generate

        Returns:
            List[Dict]: [{'complete_prompt': str, 'factor_description': str, 'reasoning': str}, ...]
        """
        current_prompt = self.prompt_struct.fusion_prompt
        current_factor_desc = self.prompt_struct.factors.get(target_factor, "")
        all_factors = self.prompt_struct.factors

        # Get dataset type info
        dataset_name = self.dataset_config.get('name', 'unknown')
        task_type = self.prompt_struct.task_description

        # Calculate accuracy info
        total_samples = 50  # Validation set size
        total_errors = error_analysis.get('total_errors', 0)
        correct_count = total_samples - total_errors
        accuracy = correct_count / total_samples * 100

        # Build error summary
        error_summary = self._build_error_summary(error_analysis)

        # Get factor selection history hint
        factor_names = list(all_factors.keys())
        factor_history_hint = self._get_recent_factor_selection_summary(factor_names)

        # Build other factors info
        other_factors_info = "\n".join([
            f"- {name}: {desc}"
            for name, desc in all_factors.items()
            if name != target_factor
        ])

        optimization_prompt = f"""You are optimizing a prompt for a {dataset_name} dataset ({task_type}).

Current Performance:
- Accuracy: {accuracy:.1f}% ({correct_count}/{total_samples} correct)
- The current prompt already works for {correct_count} samples - DO NOT break what's working!

Context:
- Current Complete Prompt: {current_prompt}
- Target Factor Segment: "{current_factor_desc}" (This is the part to replace)
- Role/Goal of this factor: {target_factor}

Error Analysis (for reference only):
{error_summary}

Task: Generate {num_candidates} improved versions of the "{target_factor}" segment.

CRITICAL CONSTRAINTS:
1. Output ONLY the new text segment.
2. The new segment must be grammatically compatible with the surrounding text.
3. PRESERVE what makes the current prompt work for correct samples.
4. Keep improvements CONCISE and GENERAL-PURPOSE for this dataset type.
5. Do NOT overfit to the specific error examples - improve the general approach.
6. Consider the nature of {dataset_name} tasks when making improvements.
7. Do NOT include markdown blocks, just raw JSON.

Output format: A valid JSON array of strings, e.g., ["description 1", "description 2"].

Generate candidates:"""


        try:
            response = self.architect_llm.generate(optimization_prompt)
            candidates = self._parse_error_driven_response(response, num_candidates)

            if not candidates:
                print(f"    LLM returned incorrect format, using fallback")
                candidates = self._generate_fallback_complete_prompts(target_factor, num_candidates)

            return candidates

        except Exception as e:
            print(f"   Error-driven optimization failed: {e}")
            return self._generate_fallback_complete_prompts(target_factor, num_candidates)

    def _build_error_summary(self, error_analysis: Dict) -> str:
        """Build error analysis summary"""
        total_errors = error_analysis.get('total_errors', 0)
        factor_effectiveness = error_analysis.get('factor_effectiveness', {})

        summary = f"**Total Errors:** {total_errors} out of 50 samples\n\n"
        summary += "**Factor Improvement Suggestions:**\n"

        for factor, count in factor_effectiveness.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            summary += f"- {factor}: {count} times ({percentage:.1f}%)\n"

        # Add detailed error sample information
        errors = error_analysis.get('errors', [])
        if errors:
            summary += "\n**Detailed Error Examples:**\n"
            for i, err in enumerate(errors[:3], 1):  # Show first 3
                summary += f"\n{'='*60}\n"
                summary += f"Example {i}:\n\n"

                # Question content
                question = err.get('question', 'N/A')
                summary += f"**Question:**\n{question[:500]}...\n\n"

                # Correct answer
                target = err.get('target_answer', 'N/A')
                summary += f"**Correct Answer:** {target}\n\n"

                # Correct solution steps
                ground_truth = err.get('ground_truth_steps', '')
                if ground_truth:
                    summary += f"**Correct Solution Steps:**\n{ground_truth[:600]}...\n\n"

                # Model's full reasoning process (key change: show more content)
                model_out = err.get('model_output', 'N/A')
                summary += f"**Model's Full Reasoning Process (Incorrect):**\n{str(model_out)[:800]}...\n\n"

                # Extracted answer (if any)
                extracted = err.get('extracted_answer', '')
                if extracted and extracted != model_out:
                    summary += f"**Model's Extracted Answer:** {extracted}\n\n"

                # Error analysis
                if 'step_analysis' in err:
                    analysis = err['step_analysis']
                    summary += f"**Analysis:**\n"
                    summary += f"- Suggested Factor: {analysis.get('suggested_factor', 'N/A')}\n"
                    summary += f"- Root Cause: {analysis.get('root_cause', 'N/A')}\n"
                    summary += f"- Error Description: {analysis.get('error_description', 'N/A')}\n"
                    summary += f"- Confidence: {analysis.get('confidence', 0.0):.2f}\n"

        return summary

    def _parse_error_driven_response(self, response: str, num_expected: int) -> List[Dict]:
        """Parse error-driven optimization response"""
        import json
        import re

        try:
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group())

                valid_candidates = []
                for cand in candidates:
                    # Handle two formats:
                    # 1. String array ["desc1", "desc2"]
                    # 2. Object array [{"factor_description": "desc1"}, ...]
                    if isinstance(cand, str):
                        # String format, use directly as factor_description
                        desc = cand.strip()
                        if desc:
                            # Clean Markdown formatting symbols
                            desc = re.sub(r'^\*+\s*', '', desc)
                            desc = re.sub(r'\s*\*+$', '', desc)
                            valid_candidates.append({'factor_description': desc})
                    elif isinstance(cand, dict) and 'factor_description' in cand and cand['factor_description']:
                        # Object format
                        desc = cand['factor_description']
                        desc = re.sub(r'^\*+\s*', '', desc)
                        desc = re.sub(r'\s*\*+$', '', desc)
                        desc = desc.strip()
                        cand['factor_description'] = desc
                        valid_candidates.append(cand)

                return valid_candidates[:num_expected]
        except Exception as e:
            print(f"    JSON parsing failed: {e}")

        return []

    def _generate_fallback_complete_prompts(self, target_factor: str, num_candidates: int) -> List[Dict]:
        """Fallback: generate simple factor description variants"""
        current_factor_desc = self.prompt_struct.factors.get(target_factor, "")

        # Simple variant strategy: only generate factor descriptions
        variants = [
            {
                'factor_description': f"{current_factor_desc} with careful attention to detail",
            },
            {
                'factor_description': f"{current_factor_desc} systematically",
            }
        ]

        return variants[:num_candidates]
    
    def _evaluate_complete_prompt_candidate(self, complete_prompt: str) -> float:
        """Evaluate performance of complete prompt candidate"""
        predictions = self._generate_predictions(complete_prompt)

        # Use real-time accuracy determination
        if hasattr(self, '_last_evaluation_results') and self._last_evaluation_results:
            correct_count = sum(1 for result in self._last_evaluation_results if result.get('correct', False))
            accuracy = correct_count / len(self._last_evaluation_results)
            return accuracy
        else:
            # Fallback: use evaluator
            eval_results = self.evaluator.evaluate(predictions, self.eval_data)
            return eval_results.get(self.metric_name, 0.0)

    def _semantic_filter_candidates(self, candidates: List[str], factor_name: str, top_k: int) -> List[str]:
        """
        Use LLM for semantic filtering and ranking of candidates

        Evaluate candidates in dataset context, keep top-K most promising ones
        """
        if len(candidates) <= top_k:
            return candidates

        filter_prompt = f"""You are a prompt optimization expert. I have {len(candidates)} candidate improvements for the '{factor_name}' factor.

Current factor: {self.prompt_struct.factors.get(factor_name, '')}

Candidate improvements:
"""

        for i, cand in enumerate(candidates, 1):
            filter_prompt += f"\n{i}. {cand}"

        filter_prompt += f"""

Based on the task context and the goal to improve model reasoning, please rank these candidates by their potential effectiveness.

Return only the top {top_k} candidates as a numbered list (1, 2, 3...) in order of preference, without explanation."""

        try:
            print(f"Running Architect LLM semantic filtering...")
            response = self.architect_llm.generate(filter_prompt)

            selected_indices = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    try:
                        idx = int(line[0]) - 1
                        if 0 <= idx < len(candidates):
                            selected_indices.append(idx)
                    except:
                        pass

            if selected_indices:
                filtered = [candidates[i] for i in selected_indices[:top_k]]
                print(f"Semantic filtering complete, kept {len(filtered)} candidates")
                return filtered
        except Exception as e:
            print(f"Semantic filtering failed: {e}, using first {top_k} candidates")

        return candidates[:top_k]

    def _print_apsf_step_summary(self, factor_name: str, best_score: float, accepted: bool):
        """Print aPSF optimization step summary"""
        print(f"\n{'='*80}")
        print(f"aPSF Step {self.current_optimization_step} Summary")
        print(f"{'='*80}")
        print(f"Improved factor: {factor_name}")
        print(f"Best candidate score: {best_score:.4f}")
        print(f"Accept status: {'Accepted' if accepted else 'Rolled back'}")
        print(f"Current fusion prompt: {self.prompt_struct.fusion_prompt}")
        print(f"Global best score: {self.global_best_score:.4f} (step {self.global_best_step})")
        print(f"{'='*80}\n")
    
    def print_optimization_summary(self):
        """Print complete optimization process statistics summary, including stability statistics"""
        print(f"\n{'='*80}")
        print(f" Optimization Process Statistics Summary")
        print(f"{'='*80}")
        print(f" Total optimization steps: {self.current_optimization_step}")
        print(f" Initial score: {self.initial_score:.4f}")
        print(f" Final score: {self.global_best_score:.4f}")
        print(f" Total improvement: {self.global_best_score - self.initial_score:.4f} "
              f"({(self.global_best_score - self.initial_score) / self.initial_score * 100:.2f}%)")
        print(f" Best step: {self.global_best_step}")
        print(f"{'='*80}\n")

        # Print stability statistics
        self.print_stability_statistics()

        # Print factor selection statistics
        if self.factor_selection_history:
            print(f"\n{'='*80}")
            print(f" Factor Selection Statistics")
            print(f"{'='*80}")
            factor_counts = {}
            for step_info in self.factor_selection_history:
                factor = step_info['selected_factor']
                factor_counts[factor] = factor_counts.get(factor, 0) + 1

            for factor, count in sorted(factor_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(self.factor_selection_history) * 100
                print(f"   {factor}: {count} times ({percentage:.1f}%)")
            print(f"{'='*80}\n")

    def _apply_best_fusion_candidate(self, factor_name: str, best_candidate: str):
        """Apply best candidate to fusion structure"""
        print(f"Applying best candidate to factor '{factor_name}'")

        # Use LLM smart replacement, consistent with _evaluate_fusion_candidate
        try:
            new_fusion_prompt = self._llm_smart_factor_replacement(
                current_prompt=self.prompt_struct.fusion_prompt,
                factor_name=factor_name,
                old_factor_content=self.prompt_struct.factors[factor_name],
                new_factor_content=best_candidate
            )

            if new_fusion_prompt and new_fusion_prompt != self.prompt_struct.fusion_prompt:
                self.prompt_struct.fusion_prompt = new_fusion_prompt
                self.prompt_struct.factors[factor_name] = best_candidate
                print(f" LLM smart replacement applied to main structure")
                print(f" Updated fusion prompt: {self.prompt_struct.fusion_prompt}")
            else:
                print(f" LLM smart replacement no change, using direct update")
                self.prompt_struct.update_factor(factor_name, best_candidate)
                print(f" Updated fusion prompt: {self.prompt_struct.fusion_prompt}")

        except Exception as e:
            print(f" LLM smart replacement failed: {e}, using direct update")
            self.prompt_struct.update_factor(factor_name, best_candidate)
            print(f" Updated fusion prompt: {self.prompt_struct.fusion_prompt}")

        # Update structure statistics
        self._update_structure_stats()

    def _update_ucb_scores(self, factor_idx: int, scores: List[float]):
        """Update UCB scores"""
        best_score_for_step = max(scores) if scores else 0.0

        self.candidate_counts[factor_idx] += 1
        self.candidate_values[factor_idx] = (
            (self.candidate_values[factor_idx] * (self.candidate_counts[factor_idx] - 1)) +
            best_score_for_step
        ) / self.candidate_counts[factor_idx]
        self.total_steps += 1

    def _update_dap_statistics(self, factor_idx: int, best_score: float):
        """Update DAP-UCB statistics"""
        if self.last_best_scores[factor_idx] != -1 and \
           best_score < self.last_best_scores[factor_idx] + self.dap_improvement_delta:
            self.stagnation_counters[factor_idx] += 1
        else:
            self.stagnation_counters[factor_idx] = 0
            self.last_best_scores[factor_idx] = best_score

    def _update_structure_stats(self):
        """Update statistics in PromptStructure"""
        factor_names = self.prompt_struct.get_factor_names()

        # Ensure structure has factor_stats attribute
        if not hasattr(self.prompt_struct, 'factor_stats'):
            self.prompt_struct.factor_stats = {}

        # Update statistics for each factor
        for i, factor_name in enumerate(factor_names):
            self.prompt_struct.factor_stats[factor_name] = {
                "selections": self.candidate_counts[i],
                "score": self.candidate_values[i],
                "best_score": self.last_best_scores[i],
                "max_improvement": max(0.0, self.last_best_scores[i] - self.initial_score) if self.last_best_scores[i] > 0 else 0.0,
                "patience_counter": self.stagnation_counters[i],
                "is_frozen": False  # Can implement freezing logic as needed
            }

    def get_success_rate(self) -> float:
        """
        Calculate optimization success rate

        Returns:
            float: Success rate (0-1), returns 0 if no attempts
        """
        if self.stability_stats['total_attempts'] == 0:
            return 0.0

        return self.stability_stats['successful_updates'] / self.stability_stats['total_attempts']

    def print_stability_statistics(self):
        """Print optimization stability statistics"""
        total = self.stability_stats['total_attempts']
        success = self.stability_stats['successful_updates']
        failed = self.stability_stats['failed_attempts']
        success_rate = self.get_success_rate()

        print(f"\n{'='*80}")
        print(f" Optimization Stability Statistics")
        print(f"{'='*80}")
        print(f" Total attempts: {total}")
        print(f" Successful updates: {success}")
        print(f" Failed attempts: {failed}")
        print(f" Success rate: {success_rate:.2%} ({success}/{total})")

        if self.stability_stats['per_factor_stats']:
            print(f"\n Stability per factor:")
            for factor, stats in self.stability_stats['per_factor_stats'].items():
                factor_rate = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
                print(f"   {factor}: {stats['successes']}/{stats['attempts']} "
                      f"success rate {factor_rate:.2%}")

        print(f"{'='*80}\n")

    def print_regression_statistics(self):
        """Print improvement success rate statistics (called after test set evaluation)"""
        success_rate = self.get_success_rate()

        print(f"\n{'='*80}")
        print(f" aPSF Improvement Success Rate Statistics")
        print(f"{'='*80}")
        print(f" Total improvement attempts: {self.stability_stats['total_attempts']}")
        print(f" Successful improvements: {self.stability_stats['successful_updates']}")
        print(f" Failed improvements: {self.stability_stats['failed_attempts']}")
        print(f" Improvement success rate: {success_rate:.2%} ({self.stability_stats['successful_updates']}/{self.stability_stats['total_attempts']})")

        # Show comparison of initial and best scores
        improvement = self.global_best_score - self.initial_score
        improvement_pct = (improvement / self.initial_score * 100) if self.initial_score > 0 else 0
        print(f"\n Score improvement:")
        print(f"   Initial validation score: {self.initial_score:.4f}")
        print(f"   Best validation score: {self.global_best_score:.4f}")
        print(f"   Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")

        # Display improvement contribution per factor
        if self.stability_stats['per_factor_stats']:
            print(f"\n Factor Improvement Contribution:")
            for factor, stats in self.stability_stats['per_factor_stats'].items():
                if stats['attempts'] > 0:
                    factor_rate = stats['successes'] / stats['attempts']
                    print(f"   {factor}: {stats['successes']} successful improvements / {stats['attempts']} attempts "
                          f"(success rate {factor_rate:.2%})")
        
        print(f"{'='*80}\n")

    def _print_fusion_step_summary(self, factor_name: str, best_candidate: str, best_score: float):
        """Print fusion optimization step summary"""
        print(f"\n{'='*80}")
        print(f" Fusion Optimization Step {self.current_optimization_step} Summary")
        print(f"{'='*80}")
        print(f" Optimized factor: {factor_name}")
        print(f" Best candidate: {best_candidate}")
        print(f" Best score: {best_score:.4f}")
        print(f" Current fusion prompt: {self.prompt_struct.fusion_prompt}")
        print(f" Global best score: {self.global_best_score:.4f} (step {self.global_best_step})")
        print(f"{'='*80}")

    # Keeping original methods including answer extraction
    def _extract_gsm8k_smart_answer(self, prediction: str) -> str:
        """Use LLM to intelligently extract GSM8K answer value, not relying on official format"""
        if not prediction:
            return ""

        print(f" Using LLM for intelligent answer extraction...")

        # Use LLM intelligent extraction directly, skip official format check
        return self._llm_extract_final_answer(prediction)

    def _llm_extract_final_answer(self, prediction: str) -> str:
        """Use LLM to extract the final numerical answer from response"""
        try:
            # Simplified extraction prompt, more direct
            extraction_prompt = f"""You are a mathematics expert. Please extract the final numerical answer from the following mathematical solution.

Mathematical Solution:
{prediction}

Please return only the final numerical answer, without any text, units, or symbols.

Examples:
Input: "Therefore, Natalia sold a total of 72 clips in April and May."
Output: 72

Input: "The answer is $125.50"
Output: 125.50

Now please extract the final numerical answer from the above solution:"""

            print(f" Calling LLM to extract answer...")

            # Use architect_llm to extract answer
            llm_response = self.architect_llm.generate(extraction_prompt)

            # Clean LLM response, extract pure number
            cleaned_response = llm_response.strip()
            print(f" LLM raw response: '{cleaned_response}'")

            # Improved number extraction, support thousands separator
            import re
            # Support number format with thousands separator
            number_match = re.search(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)', cleaned_response)
            if number_match:
                extracted_number = number_match.group(1).replace(',', '')  # Remove thousands separator

                # Verify if valid number
                try:
                    float(extracted_number)
                    print(f" LLM successfully extracted answer: '{extracted_number}'")
                    return extracted_number
                except (ValueError, TypeError):
                    pass

            # If LLM extraction fails, fallback to simple text extraction
            print(f"  LLM extraction failed, using simple text extraction...")

            # Check if LLM response is boolean but text contains numbers
            if cleaned_response.lower() in ['true', 'false'] and any(char.isdigit() for char in prediction):
                print(f"  Detected boolean but text contains numbers, retrying extraction...")
                return self._simple_extract_number(prediction)

            return self._simple_extract_number(prediction)

        except Exception as e:
            print(f" LLM extraction error: {e}")
            # Fallback to simple extraction on error
            return self._simple_extract_number(prediction)

    def _simple_extract_number(self, text: str) -> str:
        """Simple number extraction method, find the last number in text"""
        if not text:
            return ""

        print(f" Simple extraction: searching for numbers from text end...")

        # Extract all numbers, return the last one (support thousands separator)
        import re
        all_numbers = re.findall(r'([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)

        if all_numbers:
            # Remove thousands separator
            cleaned_numbers = [num.replace(',', '') for num in all_numbers]
            last_number = cleaned_numbers[-1]
            print(f" Found numbers: {cleaned_numbers}, using last one: '{last_number}'")
            return last_number

        print(" No numbers found")
        return ""

    # These methods are no longer needed as unified scoring is used
    def _extract_answer_from_prediction(self, prediction: str, item: Dict[str, Any]) -> str:
        """Extract answer from prediction text, using different strategies based on task type"""

        # Math reasoning tasks like GSM8K and MultiArith - use intelligent LLM extraction directly
        evaluator_type = str(type(self.evaluator)).lower()
        if ('gsm8k' in evaluator_type or 'multiarith' in evaluator_type
            or 'gsmhard' in evaluator_type or 'gsm_hard' in evaluator_type):  # Added GSM-hard detection
            return self._extract_gsm8k_smart_answer(prediction)

        # AQuA and other math problems with options
        elif 'aqua' in evaluator_type:
            return self.evaluator._match_number_to_option(prediction, item)

        # BBH and other multiple choice tasks - extract option letters
        elif 'bbh' in evaluator_type or self._is_multiple_choice_task(item):
            return self._extract_choice_answer(prediction)

        # Generic extraction for other tasks
        else:
            return self._extract_generic_answer(prediction)

    def _extract_gsm8k_strict_answer(self, prediction: str) -> str:
        """GSM8K answer extraction - prefer official format with fallback"""
        if not prediction:
            return ""

        # First try strict GSM8K official format: #### [number]
        pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
        match = re.search(pattern, prediction)

        if match:
            return match.group(1).strip()

        # If no official format, try extracting number from the final answer
        return self._extract_final_number_from_text(prediction)

    def _extract_final_number_from_text(self, text: str) -> str:
        """Extract final answer number from the end of text - kept as backup method"""
        if not text:
            return ""

        print(f" Starting text extraction, original: {text[-100:]}")  # Show last 100 chars

        # Split by sentences, search from end
        sentences = re.split(r'[.!?]\s+', text.strip())

        for i, sentence in enumerate(reversed(sentences)):
            print(f" Checking sentence {i+1}: '{sentence}'")

            # Optimized regex patterns with more answer formats
            patterns = [
                r'total\s+of\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'answer\s+is\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'altogether\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'equals?\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'=\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'sold\s+(?:a\s+total\s+of\s+)?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                # More flexible pattern for "Therefore, ... total of X clips"
                r'therefore.*?(?:total\s+of\s+|altogether\s+|is\s+)([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                r'so.*?(?:total\s+of\s+|altogether\s+|has\s+)([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
                # Common ending format
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:clips?|dollars?|items?|units?|students?)?\s*(?:in\s+total|altogether)?\.?\s*$'
            ]
            
            for j, pattern in enumerate(patterns):
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    number_str = match.group(1).replace(',', '')  # Remove thousands separator
                    try:
                        # Verify if valid number
                        float(number_str)
                        print(f" Pattern {j+1} matched: '{number_str}'")
                        return number_str
                    except (ValueError, TypeError):
                        continue

        # If no clear answer pattern found, extract the last number in text
        print(" Trying to extract all numbers...")
        all_numbers = self._extract_all_numbers(text)
        if all_numbers:
            last_number = all_numbers[-1].replace(',', '')
            print(f" Using last number: '{last_number}' (from {all_numbers})")
            return last_number

        print(" Failed to extract any number")
        return ""

    def _extract_choice_answer(self, prediction: str) -> str:
        """Extract option letter from multiple choice answer (A-Z) - improved version"""
        import re

        if not prediction:
            return ""

        print(f" Choice extraction - original answer: '{prediction[:200]}...'")

        # Match patterns by priority, prefer more explicit expressions (support A-Z)
        patterns = [
            # Highest priority: explicit answer statements
            r'(?:answer\s+is\s+|correct\s+answer\s+is\s+|choose\s*)[^\w]*\(?([A-Z])\)?',
            r'(?:option\s*|choice\s*)\(?([A-Z])\)?',
            r'therefore.*?answer.*?\(?([A-Z])\)?',

            # High priority: standard format
            r'\*\*\s*option\s*\(?([A-Z])\)?\s*\*\*',  # **Option (B)**
            r'\*\*\s*\(?([A-Z])\)?\s*\*\*',           # **B**
            r'answer.*?\(?([A-Z])\)?',                 # Answer: (B) or Answer B

            # Medium priority: parentheses format
            r'\(([A-Z])\)',                           # (A), (B), (C) etc
            r'([A-Z])\)',                             # A), B), C) etc

            # Lower priority: other formats
            r'option\s*([A-Z])(?:\s|$|\.)',           # option A
            r'choose\s*([A-Z])(?:\s|$|\.)',           # choose A
            r'^([A-Z])(?:\s|$|\.|,)',                 # letter at line start
            r'([A-Z])(?:\s+no|yes)(?:\s|$|\.|,)',     # A no, B yes etc

            # Lowest priority: standalone letter (prone to mismatch)
            r'\b([A-Z])\b'
        ]

        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, prediction, re.IGNORECASE | re.MULTILINE)
            if matches:
                # If multiple matches, prefer the last one (usually the final answer)
                extracted = matches[-1].upper()
                print(f" Pattern {i+1} matched: '{extracted}' (found {len(matches)} matches)")
                return extracted

        print(f" No valid option found")
        return ""

    def _extract_generic_answer(self, prediction: str) -> str:
        """Generic answer extraction method"""
        # Simply return the last line or last few words of prediction text
        lines = prediction.strip().split('\n')
        return lines[-1].strip() if lines else ""

    def _is_multiple_choice_task(self, item: Dict[str, Any]) -> bool:
        """Check if task is multiple choice (supports A-Z options)"""
        question = item.get('question', '') or item.get('prompt', '')
        # Check if contains option markers
        return bool(re.search(r'\([A-Za-z]\)', question) or 'Options:' in question)

    def _is_answer_correct(self, extracted_answer: str, target_answer: str, item: Dict[str, Any]) -> bool:
        """Check if extracted answer is correct"""

        # Math reasoning tasks - numerical comparison (including GSM8K, GSM-hard and MultiArith)
        evaluator_type = str(type(self.evaluator)).lower()
        if ('gsm8k' in evaluator_type or 'multiarith' in evaluator_type
            or 'gsmhard' in evaluator_type or 'gsm_hard' in evaluator_type):  # Added GSM-hard detection
            if not extracted_answer or not target_answer:
                return False  # Strictly require extracted answer
            try:
                return abs(float(extracted_answer) - float(target_answer)) < 1e-6
            except (ValueError, TypeError):
                return False

        # Multiple choice tasks - compare after normalization
        else:
            # Normalize answer: remove parentheses, spaces, convert to uppercase
            normalized_extracted = self._normalize_choice_answer(extracted_answer)
            normalized_target = self._normalize_choice_answer(target_answer)

            print(f" Normalized comparison: '{normalized_extracted}' vs '{normalized_target}'")

            return normalized_extracted == normalized_target

 

    def _normalize_choice_answer(self, answer: str) -> str:
        """Normalize multiple choice answer: remove parentheses, spaces, keep only core letter"""
        if not answer:
            return ""

        import re

        # First try matching multiple choice format (A), (B), ..., (Z)
        choice_match = re.search(r'\(([A-Z])\)', answer.upper())
        if choice_match:
            return choice_match.group(1)

        # Try matching "Answer: X" format
        answer_match = re.search(r'(?:Answer|answer):\s*([A-Z])', answer.upper())
        if answer_match:
            return answer_match.group(1)

        # Remove all non-alphanumeric characters, convert to uppercase
        normalized = re.sub(r'[^\w]', '', str(answer).strip()).upper()

        # Extended support for A-Z options
        if len(normalized) == 1 and normalized.isalpha():
            return normalized

        # Search from end to avoid extracting wrong letter (support full A-Z range)
        for char in reversed(normalized):
            if char.isalpha():
                return char

        return normalized

    def _get_target_answer(self, item: Dict[str, Any]) -> str:
        """Get target answer for display"""
        # Try multiple possible answer keys
        answer_keys = ['answer', 'target', 'correct', 'label']

        for key in answer_keys:
            if key in item:
                answer = item[key]
                # Handle GSM8K format answer
                if isinstance(answer, str) and '####' in answer:
                    return answer.split('####')[-1].strip()
                return str(answer).strip()

        return ""

    def _extract_all_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text (supports thousands separator)"""
        if not text:
            return []

        # Improved regex: support number format with thousands separator
        pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.?\d*'
        matches = re.findall(pattern, text)

        # Remove commas and return pure number strings
        cleaned_numbers = []
        for match in matches:
            # Remove thousands separator comma
            cleaned_num = match.replace(',', '')
            if cleaned_num:  # Ensure not empty
                cleaned_numbers.append(cleaned_num)

        return cleaned_numbers

    def _evaluate_candidate(self, factor_name: str, candidate_content: str) -> float:
        """Evaluate performance of single candidate factor content (backward compatible interface)"""
        return self._evaluate_fusion_candidate(factor_name, candidate_content)

    def collect_validation_errors(self, eval_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collect error samples during validation phase for reflection optimization"""
        logging.info(" Collecting validation error samples...")
        
        best_prompt = self.global_best_prompt_structure.compose()
        wrong_examples = []
        
        for i, item in enumerate(tqdm(eval_data, desc="Collecting error samples", leave=False)):
            input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
            question = item.get(input_key, '')
            # Combine instruction with question directly
            formatted_prompt = f"{best_prompt}\n\n{question}"

            prediction = self.worker_llm.generate(formatted_prompt)

            try:
                from ..evaluation.unified_scoring import UnifiedScorer
                scorer = UnifiedScorer(self.worker_llm, "validation_error_collection")
                _, _, is_correct = scorer.extract_and_score(prediction, item, self.evaluator)

                if not is_correct:
                    wrong_examples.append({
                        'question': question,
                        'prediction': prediction,
                        'expected': item.get('answer', item.get('target', 'Unknown')),
                        'item_data': item
                    })
            except Exception as e:
                logging.warning(f" Sample {i+1} error determination failed: {e}")

        logging.info(f" Collected {len(wrong_examples)} error samples")
        return wrong_examples

    def reflection_optimization(self, eval_data: List[Dict[str, Any]]) -> bool:
        """Reflection optimization based on error samples"""
        logging.info(" Starting reflection optimization phase...")

        # 1. Collect error samples from current best prompt
        wrong_examples = self.collect_validation_errors(eval_data)

        if len(wrong_examples) == 0:
            logging.info(" No error samples, skipping reflection optimization")
            return False

        # Check if error rate exceeds threshold
        error_rate = len(wrong_examples) / len(eval_data)
        error_threshold = self.dataset_config.get("reflection_error_threshold", 0.1)

        if error_rate < error_threshold:
            logging.info(f" Error rate {error_rate:.3f} below threshold {error_threshold:.3f}, skipping reflection optimization")
            return False

        logging.info(f" Error rate {error_rate:.3f} exceeds threshold, starting reflection optimization")

        # 2. Analyze error patterns
        error_analysis = self._analyze_error_patterns(wrong_examples)

        # 3. Generate improvement suggestions based on error analysis
        improvement_suggestions = self._generate_reflection_improvement_suggestions(error_analysis, wrong_examples)

        # 4. Generate reflection-optimized prompt
        reflection_prompt = self._generate_reflection_prompt(improvement_suggestions)

        # 5. Test reflection prompt on validation set
        reflection_score = self._evaluate_reflection_prompt(reflection_prompt, eval_data)

        # 6. Compare scores, decide whether to adopt reflection prompt
        current_best_score = self.global_best_score

        logging.info(f" Reflection optimization result comparison:")
        logging.info(f"  Original best score: {current_best_score:.4f}")
        logging.info(f"  Reflection optimization score: {reflection_score:.4f}")

        if reflection_score > current_best_score:
            improvement = reflection_score - current_best_score
            logging.info(f" Reflection optimization successful! Improved by {improvement:.4f}")
            # Update best prompt structure
            self.global_best_score = reflection_score
            self.global_best_prompt_structure.fusion_prompt = reflection_prompt
            return True
        else:
            logging.info(" Reflection optimization did not improve performance, keeping original prompt")
            return False

    def _analyze_error_patterns(self, wrong_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error sample patterns"""
        logging.info(" Analyzing error patterns...")

        # Limit number of error samples to analyze
        max_errors_to_analyze = self.dataset_config.get("reflection_max_errors", 10)
        sample_errors = wrong_examples[:max_errors_to_analyze]

        # Build error analysis prompt
        error_analysis_prompt = f"""
Please analyze the following error samples and identify common error patterns and problem types:

Error samples ({len(sample_errors)} total):
"""
        
        # Analyze selected error samples
        for i, example in enumerate(sample_errors):
            error_analysis_prompt += f"""
Sample {i+1}:
Question: {example['question']}
Model prediction: {example['prediction']}
Correct answer: {example['expected']}
---
"""

        error_analysis_prompt += f"""
Please summarize:
1. What are the main types of errors?
2. What are the root causes of these errors?
3. What aspects need improvement?

Total errors: {len(wrong_examples)}
Analyzed samples: {len(sample_errors)}

Please provide a concise analysis in English.
"""

        try:
            analysis_result = self.architect_llm.generate(error_analysis_prompt)
            logging.info(f" Error analysis result: {analysis_result}")

            return {
                'analysis_result': analysis_result,
                'total_errors': len(wrong_examples),
                'analyzed_errors': len(sample_errors),
                'sample_errors': sample_errors
            }
        except Exception as e:
            logging.error(f" Error analysis failed: {e}")
            return {'analysis_result': 'Analysis failed', 'total_errors': len(wrong_examples)}

    def _generate_reflection_improvement_suggestions(self, error_analysis: Dict[str, Any], wrong_examples: List[Dict[str, Any]]) -> str:
        """Generate reflection optimization improvement suggestions based on error analysis"""
        logging.info(" Generating reflection optimization improvement suggestions...")
        
        current_prompt = self.global_best_prompt_structure.compose()
        
        suggestion_prompt = f"""
Current optimal prompt:
{current_prompt}

Error analysis results:
{error_analysis.get('analysis_result', 'No analysis result')}

Total errors: {error_analysis.get('total_errors', 0)}

Please provide specific improvement suggestions based on the error analysis:
1. Which parts need to be modified?
2. How should they be modified to address the identified error patterns?
3. What are the specific improvement measures?

Please provide clear, actionable suggestions in English.
"""
        
        try:
            suggestions = self.architect_llm.generate(suggestion_prompt)
            logging.info(f" Improvement suggestions: {suggestions}")
            return suggestions
        except Exception as e:
            logging.error(f" Failed to generate improvement suggestions: {e}")
            return "Unable to generate improvement suggestions"

    def _generate_reflection_prompt(self, improvement_suggestions: str) -> str:
        """Generate reflection-optimized prompt based on improvement suggestions"""
        logging.info(" Generating reflection-optimized prompt...")
        
        current_prompt = self.global_best_prompt_structure.compose()
        
        reflection_prompt_template = f"""
Please optimize the current prompt based on the following improvement suggestions:

Current prompt:
{current_prompt}

Improvement suggestions:
{improvement_suggestions}

Please generate an improved prompt with the following requirements:
1. Maintain the core structure and style of the original prompt
2. Address the identified error patterns through optimization
3. Ensure the prompt is clear, specific, and easy to understand
4. Output only the final optimized prompt in English, without any explanations
5. The prompt should be suitable for mathematical reasoning tasks

Optimized prompt:
"""
        
        try:
            reflection_prompt = self.architect_llm.generate(reflection_prompt_template)
            # Clean generated prompt, remove possible prefix
            reflection_prompt = reflection_prompt.strip()
            if reflection_prompt.startswith("Optimized prompt:"):
                reflection_prompt = reflection_prompt.replace("Optimized prompt:", "").strip()

            logging.info(f" Reflection-optimized prompt: {reflection_prompt}")
            return reflection_prompt
        except Exception as e:
            logging.error(f" Failed to generate reflection prompt: {e}")
            return current_prompt  # Return original prompt on failure

    def _evaluate_reflection_prompt(self, reflection_prompt: str, eval_data: List[Dict[str, Any]]) -> float:
        """Evaluate performance of reflection-optimized prompt"""
        logging.info(" Evaluating reflection-optimized prompt...")
        
        total = len(eval_data)
        correct = 0

        for i, item in enumerate(tqdm(eval_data, desc="Evaluating reflection prompt", leave=False)):
            input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
            question = item.get(input_key, '')
            # Combine instruction with question directly
            formatted_prompt = f"{reflection_prompt}\n\n{question}"

            prediction = self.worker_llm.generate(formatted_prompt)

            # Show detailed info for first 3 samples for debugging
            if i < 3:
                print(f"\n Reflection validation sample {i+1}:")
                print(f" Prompt used: {formatted_prompt}")
                print(f" Question: {question}")
                print(f" Answer: {prediction}")

            try:
                # Use exact same code as standard validation
                from ..evaluation.unified_scoring import UnifiedScorer
                scorer = UnifiedScorer(self.worker_llm, "apsf_validation")  # Same as standard validation
                extracted_answer, target_answer, is_correct = scorer.extract_and_score(
                    prediction, item, self.evaluator
                )

                # Show detailed info for first 3 samples
                if i < 3:
                    normalized_extracted = scorer._normalize_choice_answer_apsf_style(extracted_answer)
                    normalized_target = scorer._normalize_choice_answer_apsf_style(target_answer)
                    print(f" Normalized comparison: '{normalized_extracted}' vs '{normalized_target}'")
                    print(f" Intelligently extracted answer: '{extracted_answer}'")
                    print(f" Target answer: '{target_answer}'")
                    verdict = " Correct" if is_correct else " Wrong"
                    print(f" Answer match: {verdict}")

                if is_correct:
                    correct += 1
            except Exception as e:
                logging.warning(f" Reflection evaluation failed: {e}")
                # Add fallback
                try:
                    extracted_answer = self._extract_answer_from_prediction(prediction, item)
                    target_answer = self._get_target_answer(item)
                    is_correct = self._is_answer_correct(extracted_answer, target_answer, item)
                    if is_correct:
                        correct += 1
                except Exception as e2:
                    logging.error(f" Fallback evaluation also failed: {e2}")

        reflection_score = (correct / total) if total else 0.0
        logging.info(f" Reflection prompt validation score: {reflection_score:.4f}")
        return reflection_score

    def evaluate_on_test_set(self, test_data: List[Dict[str, Any]]) -> float:
        """Evaluate best fusion prompt structure on test set (single pass), show question and match per sample then summarize"""

        best_prompt = self.global_best_prompt_structure.compose()
        total = len(test_data)
        correct = 0
        predictions = []

        for i, item in enumerate(tqdm(test_data, desc="Testing Final Fusion Structure", leave=False)):
            input_key = 'prompt' if 'prompt' in item else ('input' if 'input' in item else 'question')
            question = item.get(input_key, '')
            # Combine instruction with question directly
            formatted_prompt = f"{best_prompt}\n\n{question}"

            prediction = self.worker_llm.generate(formatted_prompt)
            predictions.append(prediction)

            # Per-sample display: detailed question, prompt, answer and matching process
            try:
                from ..evaluation.unified_scoring import UnifiedScorer
                scorer = UnifiedScorer(self.worker_llm, "apsf_test")
                extracted_answer, target_answer, is_correct = scorer.extract_and_score(prediction, item, self.evaluator)

                print(f"\n{'='*80}")
                print(f" aPSF Test Sample {i+1}/{total}")
                print(f"{'='*80}")
                print(f" Question:\n   {question}")
                print(f"\n Current aPSF fusion prompt:\n   {best_prompt}")
                print(f"\n aPSF response:\n   {prediction}")
                print(f"\n Intelligently extracted answer: '{extracted_answer}'")
                print(f" Target answer: '{target_answer}'")
                verdict = " Correct" if is_correct else " Wrong"
                print(f" Answer match: {verdict}")

                if is_correct:
                    correct += 1
            except Exception as e:
                print(f" Test sample {i+1} match determination failed: {e}")

        # Summarize after single pass
        test_score = (correct / total) if total else 0.0

        # Check if MMLU task and output categorized results
        if hasattr(self.evaluator, 'subject_mapping'):
            print("\n" + "="*80)
            print(" Using MMLU evaluator to generate subject-categorized results")
            print("="*80)
            mmlu_results = self.evaluator.evaluate(predictions, test_data)
            test_score = mmlu_results.get("Average", test_score)
            print(f"\n MMLU average accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        else:
            print(f"\n aPSF test set unified scoring results:")
            print(f"  Correct: {correct}/{total}")
            print(f"  Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")

        # Print improvement success rate statistics after test evaluation
        self.print_regression_statistics()

        logging.info(f" Final test score: {test_score:.4f}")
        return test_score

    def _print_training_summary(self):
        """Print fusion training set optimization summary"""
        logging.info("="*70)
        logging.info(" Fusion aPSF TRAINING OPTIMIZATION SUMMARY")
        logging.info("="*70)
        
        logging.info(f" INITIAL VALIDATION SCORE: {self.initial_score:.4f}")
        logging.info(f" HIGHEST VALIDATION SCORE: {self.global_best_score:.4f}")
        improvement = self.global_best_score - self.initial_score
        improvement_pct = (improvement / self.initial_score * 100) if self.initial_score > 0 else 0
        logging.info(f" IMPROVEMENT: +{improvement:.4f} ({improvement_pct:+.1f}%)")
        logging.info(f" ACHIEVED BY FACTOR: {self.global_best_factor}")
        logging.info(f" AT OPTIMIZATION STEP: {self.global_best_step}")
        logging.info(f" FINAL FUSION PROMPT: {self.global_best_prompt_structure.fusion_prompt}")
        logging.info("-" * 70)
        
        factor_names = self.prompt_struct.get_factor_names()
        factor_importance = []
        
        for i, name in enumerate(factor_names):
            avg_score = self.candidate_values[i]
            selections = self.candidate_counts[i]
            best_score = self.last_best_scores[i]
            
            factor_importance.append({
                'name': name,
                'avg_score': avg_score,
                'best_score': best_score,
                'selections': selections
            })
        
        factor_importance.sort(key=lambda x: x['avg_score'], reverse=True)
        
        logging.info(" FACTOR IMPORTANCE RANKING:")
        for i, factor in enumerate(factor_importance):
            is_best = "Best" if factor['name'] == self.global_best_factor else "  "
            logging.info(f"  {is_best}{i+1}. {factor['name']}: avg={factor['avg_score']:.4f}, best={factor['best_score']:.4f}, selections={factor['selections']}")
        
        logging.info("-" * 70)
        logging.info(f" TOTAL EVALUATIONS: {len(self.all_scores_history)}")

        # Token statistics
        logging.info("-" * 70)
        logging.info(" TOKEN USAGE STATISTICS:")
        worker_stats = self.worker_llm.get_token_stats()
        architect_stats = self.architect_llm.get_token_stats()
        total_tokens = worker_stats['total_tokens'] + architect_stats['total_tokens']
        total_calls = worker_stats['api_calls'] + architect_stats['api_calls']
        logging.info(f"   Worker LLM: {worker_stats['total_tokens']:,} tokens ({worker_stats['api_calls']} calls)")
        logging.info(f"   Architect LLM: {architect_stats['total_tokens']:,} tokens ({architect_stats['api_calls']} calls)")
        logging.info(f"   TOTAL (all steps): {total_tokens:,} tokens ({total_calls} API calls)")
        logging.info(f"   BEST ACHIEVED AT STEP: {self.global_best_step}")
        logging.info(f"   TOKENS AT BEST STEP: {self.global_best_tokens:,} tokens")

        logging.info("="*70)

    def print_final_summary(self, test_score: float):
        """Print fusion optimization final summary"""
        logging.info("="*70)
        logging.info(" Fusion aPSF FINAL EXPERIMENT SUMMARY")
        logging.info("="*70)
        
        logging.info(" SCORE SUMMARY:")
        logging.info(f"   Initial Validation Score:  {self.initial_score:.4f}")
        logging.info(f"   Best Validation Score:     {self.global_best_score:.4f}")
        logging.info(f"   Final Test Score:          {test_score:.4f}")
        
        val_improvement = self.global_best_score - self.initial_score
        val_improvement_pct = (val_improvement / self.initial_score * 100) if self.initial_score > 0 else 0
        
        generalization_gap = self.global_best_score - test_score
        logging.info("-" * 70)
        logging.info(" PERFORMANCE ANALYSIS:")
        logging.info(f"   Validation Improvement:    +{val_improvement:.4f} ({val_improvement_pct:+.1f}%)")
        logging.info(f"   Generalization Gap:        {generalization_gap:.4f}")
        if generalization_gap < 0.1:
            logging.info("   EXCELLENT generalization (gap < 0.1)")
        elif generalization_gap < 0.2:
            logging.info("   GOOD generalization (gap < 0.2)")
        else:
            logging.info("    Some overfitting detected (gap >= 0.2)")
        
        logging.info("-" * 70)
        logging.info(" FUSION PROMPT EVOLUTION:")
        logging.info(f"   BEST FUSION PROMPT: {self.global_best_prompt_structure.fusion_prompt}")
        logging.info(f"   OPTIMIZED FACTOR: {self.global_best_factor}")
        logging.info(f"   ACHIEVED AT STEP: {self.global_best_step}")
        logging.info(f"   TOTAL EVALUATIONS: {len(self.all_scores_history)}")
        logging.info("="*70)

    def get_optimized_prompt(self) -> str:
        """Get optimized fusion prompt"""
        if self.global_best_prompt_structure:
            return self.global_best_prompt_structure.compose()
        return self.prompt_struct.compose()

    def get_best_structure(self) -> PromptStructure:
        """Get best fusion prompt structure"""
        return self.global_best_prompt_structure

    def get_regression_rate(self) -> float:
        """Calculate regression rate

        Returns:
            float: Regression rate in range [0, 1]
        """
        total_updates = self.regression_stats.get('total_accepted_updates', 0)
        total_regressions = self.regression_stats.get('total_regressions', 0)

        if total_updates == 0:
            return 0.0

        return total_regressions / total_updates

    def _get_factor_analysis(self) -> dict:
        """Get detailed fusion factor analysis data"""
        factor_names = self.prompt_struct.get_factor_names()
        factor_analysis = {
            "factor_statistics": [],
            "factor_ranking": [],
            "optimization_summary": {
                "most_improved_factor": None,
                "least_improved_factor": None,
                "most_explored_factor": None,
                "convergence_info": {}
            }
        }
        
        factor_stats = []
        for i, name in enumerate(factor_names):
            avg_score = self.candidate_values[i]
            selections = self.candidate_counts[i]
            best_score = self.last_best_scores[i]
            
            factor_scores = [
                record['score'] for record in self.all_scores_history 
                if 'factor' in record and record['factor'] == name
            ]
            
            factor_stat = {
                'name': name,
                'average_score': avg_score,
                'best_score': best_score,
                'selections_count': selections,
                'score_variance': np.var(factor_scores) if factor_scores else 0,
                'score_range': {
                    'min': min(factor_scores) if factor_scores else 0,
                    'max': max(factor_scores) if factor_scores else 0
                },
                'all_scores': factor_scores,
                'is_global_best': name == self.global_best_factor
            }
            factor_stats.append(factor_stat)
        
        if not factor_stats:
             return factor_analysis

        factor_ranking = sorted(factor_stats, key=lambda x: x['average_score'], reverse=True)
        
        most_explored = max(factor_stats, key=lambda x: x['selections_count'])
        least_explored = min(factor_stats, key=lambda x: x['selections_count'])
        
        factor_analysis["factor_statistics"] = factor_stats
        factor_analysis["factor_ranking"] = factor_ranking
        factor_analysis["optimization_summary"] = {
            "most_improved_factor": self.global_best_factor,
            "most_explored_factor": most_explored['name'],
            "least_explored_factor": least_explored['name'],
            "exploration_balance": {
                "max_selections": most_explored['selections_count'],
                "min_selections": least_explored['selections_count'],
                "selection_variance": np.var([f['selections_count'] for f in factor_stats])
            }
        }
        
        return factor_analysis

    def _safe_regenerate_prompt(self, temp_structure: PromptStructure) -> str:
        """Helper method to safely regenerate prompt"""
        factor_descriptions = list(temp_structure.factors.values())

        if len(factor_descriptions) == 1:
            prompt = factor_descriptions[0]
        else:
            prompt = '. '.join(factor_descriptions)

        # No need to add input placeholder - questions handled separately

        return prompt

    def _regenerate_fusion_prompt_safe(self, temp_structure: PromptStructure) -> str:
        """Safely regenerate fusion prompt - use PromptStructure built-in method"""
        try:
            # Use PromptStructure's built-in method to regenerate fusion prompt
            temp_structure.fusion_prompt = temp_structure._generate_fusion_prompt()
            return temp_structure.fusion_prompt
        except Exception as e:
            print(f"Built-in method regeneration failed: {e}, using simple concatenation")
            return self._safe_regenerate_prompt(temp_structure)

    def _deep_copy_structure(self, structure: PromptStructure) -> PromptStructure:
        """
        Deep copy PromptStructure object to avoid reference pollution.

        Ensures saved global best structure won't be modified by subsequent optimization steps.

        Args:
            structure: PromptStructure object to copy

        Returns:
            New PromptStructure object completely independent from original via deep copy

        Example:
            Step 3: Found best score 0.80, save global_best = deepcopy(prompt_struct)
            Step 4: Continue optimization, modify prompt_struct to score 0.75
            Result: global_best still keeps Step 3 state, unaffected
        """
        try:
            # Use copy.deepcopy for deep copy
            return copy.deepcopy(structure)
        except Exception as e:
            print(f"Deep copy failed: {e}, trying manual construction")
            # If deepcopy fails, manually construct new object
            try:
                new_structure = PromptStructure(
                    task_description=structure.task_description,
                    factors=copy.deepcopy(structure.factors),
                    fusion_prompt=structure.fusion_prompt,
                    factor_mappings=copy.deepcopy(getattr(structure, 'factor_mappings', {}))
                )
                # Copy additional attributes
                if hasattr(structure, 'factor_stats'):
                    new_structure.factor_stats = copy.deepcopy(structure.factor_stats)
                if hasattr(structure, 'task_type'):
                    new_structure.task_type = structure.task_type
                if hasattr(structure, 'output_format'):
                    new_structure.output_format = structure.output_format
                if hasattr(structure, 'validation_info'):
                    new_structure.validation_info = copy.deepcopy(structure.validation_info)
                return new_structure
            except Exception as e2:
                print(f"Manual construction also failed: {e2}, returning original object (may cause reference issues)")
                return structure
    
    def save_factor_analysis(self, dataset_name: str, method_name: str = None):
        """
        Save factor analysis details to JSON file

        Args:
            dataset_name: Dataset name
            method_name: Method name (defaults to self.method_name)
        """
        if not method_name:
            method_name = self.method_name

        # Calculate overall impact statistics for each factor
        factor_impact_summary = {}
        factor_names = list(self.prompt_struct.factors.keys())

        for factor in factor_names:
            total_selections = 0
            total_priority = 0.0
            total_confidence = 0.0
            improvements = []

            for step_info in self.factor_selection_history:
                if step_info['selected_factor'] == factor:
                    total_selections += 1
                    total_priority += step_info['factor_priorities'].get(factor, 0.0)
                    total_confidence += step_info['factor_avg_confidence'].get(factor, 0.0)

            factor_impact_summary[factor] = {
                'total_selections': total_selections,
                'avg_priority': total_priority / total_selections if total_selections > 0 else 0.0,
                'avg_confidence': total_confidence / total_selections if total_selections > 0 else 0.0,
                'selection_rate': total_selections / len(self.factor_selection_history) if self.factor_selection_history else 0.0
            }

        # Build complete analysis report
        analysis_report = {
            'experiment_info': {
                'dataset': dataset_name,
                'method': method_name,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_optimization_steps': len(self.factor_selection_history),
                'initial_score': self.initial_score,
                'final_best_score': self.global_best_score,
                'improvement': self.global_best_score - self.initial_score
            },
            'factor_impact_summary': factor_impact_summary,
            'step_by_step_details': self.factor_selection_history,
            'all_scores_history': self.all_scores_history
        }

        # Save to file
        results_dir = "results/factor_analysis"
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{method_name}_{dataset_name}_factor_analysis_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Factor Analysis Report Saved")
        print(f"{'='*80}")
        print(f"File path: {filepath}")
        print(f"\nFactor Impact Summary:")
        for factor, stats in factor_impact_summary.items():
            print(f"  {factor}:")
            print(f"    - Selection count: {stats['total_selections']}")
            print(f"    - Selection rate: {stats['selection_rate']:.2%}")
            print(f"    - Avg priority: {stats['avg_priority']:.2%}")
            print(f"    - Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"{'='*80}\n")

        return filepath
    
