"""
aPSF Ablation Experiments Implementation
Contains various ablation experiment variants:
1. NoStructure - Removes the Architect stage
2. NoTone/NoFormat/NoPerspective - Removes a specific factor
3. NoDAP - Does not use dynamic patience mechanism
4. RandSelect - Randomly selects factors
5. Feedback - Uses error sample feedback
6. SmallArchitect - Uses smaller models
"""

import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional
from ..optimization import Architect, Optimizer
from ..optimization.prompt_object import PromptStructure
from ..llm_apis import get_llm
from ..evaluation import BaseEvaluator

def run_apsf_nostructure(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    NoStructure ablation experiment - Removes Architect stage, optimizes content only
    """
    logging.info("Running NoStructure ablation experiment - Removing Architect stage")
    
    # Create a simple prompt structure without Architect discovery
    simple_structure = PromptStructure(
        task_description=task_desc,
        factors={
            "main_content": "Let's think step by step."
        },
        fusion_prompt="Let's think step by step."
    )

    # Use standard optimizer but skip structure discovery
    from ..optimization import Optimizer
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    optimizer = Optimizer(
        simple_structure, eval_data, evaluator, config,
        method_name="aPSF-NoStructure"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"NoStructure optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": simple_structure.to_dict()
    }


def run_apsf_nofactor(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    exclude_factor: str = "tone",  # Options: tone, format, perspective
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    NoTone/NoFormat/NoPerspective ablation experiment - Removes a specific factor
    """
    logging.info(f"Running No{exclude_factor.capitalize()} ablation experiment - Removing {exclude_factor} factor")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # Remove the specified factor
    if exclude_factor.lower() in ["tone"]:
        # Remove tone-related factors
        factors_to_remove = ["tone", "style"]
    elif exclude_factor.lower() in ["format"]:
        # Remove format-related factors
        factors_to_remove = ["format", "method"]
    elif exclude_factor.lower() in ["perspective"]:
        # Remove perspective-related factors
        factors_to_remove = ["perspective", "viewpoint"]
    else:
        factors_to_remove = [exclude_factor]
    
    # Remove factors from structure
    for factor_name in list(prompt_struct.factors.keys()):
        if any(remove_key in factor_name.lower() for remove_key in factors_to_remove):
            logging.info(f"Removing factor: {factor_name}")
            del prompt_struct.factors[factor_name]

    # Regenerate fusion prompt
    prompt_struct.fusion_prompt = _regenerate_fusion_prompt(prompt_struct)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Use optimizer
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name=f"aPSF-No{exclude_factor.capitalize()}"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"No{exclude_factor.capitalize()} optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }


def run_apsf_nodap(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    NoDAP ablation experiment - Does not use dynamic patience mechanism
    """
    logging.info("Running NoDAP ablation experiment - Not using dynamic patience mechanism")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # Modify config to disable DAP
    modified_config = dataset_config.copy()
    modified_config['dap_patience_M'] = float('inf')  # Infinite patience, effectively disabling DAP
    modified_config['use_ucb'] = False  # Use random selection only
    
    # If step parameter is specified, override config
    if step is not None:
        modified_config['total_optimization_steps'] = step
    
    # Use optimizer
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, modified_config,
        method_name="aPSF-NoDAP"
    )
    
    # Run optimization
    total_steps = modified_config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"NoDAP optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }


def run_apsf_randselect(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    RandSelect ablation experiment - Randomly selects factors instead of UCB
    """
    logging.info("Running RandSelect ablation experiment - Random factor selection")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Create custom optimizer using random selection
    class RandomSelectOptimizer(Optimizer):
        def _select_factor_dap_ucb(self, factor_names: List[str]):
            """Override DAP-UCB selection, use random selection instead"""
            factor_idx = random.choice(list(range(self.num_factors)))
            selected_factor = factor_names[factor_idx]
            logging.info(f"Randomly selected factor: {selected_factor}")
            return selected_factor
    
    # Use random selection optimizer
    optimizer = RandomSelectOptimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name="aPSF-RandSelect"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"RandSelect optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }


def run_apsf_feedback(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Feedback ablation experiment - Uses error sample feedback mechanism
    """
    logging.info("Running Feedback ablation experiment - Using error sample feedback")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Use optimizer with feedback
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name="aPSF-Feedback",
        enable_feedback=True  # Enable feedback mechanism
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"Feedback optimization step {step + 1}/{total_steps}")
        optimizer.step()

        # Apply feedback every 3 steps
        if (step + 1) % 3 == 0 and optimizer.wrong_examples:
            logging.info(f"Applying feedback mechanism, current error samples: {len(optimizer.wrong_examples)}")
            optimizer._apply_feedback_refinement()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict(),
        "feedback_history": getattr(optimizer, 'feedback_history', [])
    }


def run_apsf_smallarchitect(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    SmallArchitect ablation experiment - Uses smaller model for Architect
    """
    logging.info("Running SmallArchitect ablation experiment - Using smaller Architect model")
    
    # Use smaller Architect model
    class SmallArchitect(Architect):
        def __init__(self):
            super().__init__()
            # Use smaller model, like GPT-3.5 or smaller
            self.llm = get_llm("small_architect")  # Need to define in config
    
    # Use small model Architect for structure discovery
    small_architect = SmallArchitect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = small_architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Use standard optimizer but with small model for architect
    optimizer = Optimizer(
        prompt_struct, eval_data, evaluator, config,
        architect_llm_id="small_architect",  # Use small model
        method_name="aPSF-SmallArchitect"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"SmallArchitect optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }


# Helper functions
def _construct_examples(samples: List[Dict[str, Any]]) -> str:
    """Construct example strings"""
    example_strings = []
    for i, sample in enumerate(samples, 1):
        input_text = sample.get('input', sample.get('prompt', sample.get('question', '')))
        output_text = sample.get('target', sample.get('answer', sample.get('output', '')))
        example_strings.append(
            f"--- Example {i} ---\nInput: {input_text}\nExpected Output: {output_text}"
        )
    return "\n\n".join(example_strings)


def _regenerate_fusion_prompt(prompt_struct: PromptStructure) -> str:
    """Regenerate fusion prompt"""
    factors_text = " ".join(prompt_struct.factors.values())
    return f"Please solve the following problem {factors_text}: {{input}}"


# ============ Added: Multi-armed bandit algorithm comparison experiments ============

def run_apsf_thompson(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Thompson Sampling ablation experiment - Uses Thompson Sampling for factor selection
    Probability matching strategy based on Bayesian posterior
    """
    logging.info("Running Thompson Sampling ablation experiment - Bayesian exploration strategy")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Create Thompson Sampling optimizer
    class ThompsonSamplingOptimizer(Optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Beta distribution parameters (success count alpha, failure count beta)
            self.alpha = np.ones(self.num_factors)  # Success count (initialized to 1 to avoid division by zero)
            self.beta = np.ones(self.num_factors)   # Failure count (initialized to 1 to avoid division by zero)
            # Added: Track historical best scores for each factor
            self.factor_best_scores = [-1.0] * self.num_factors
            logging.info(f"Initializing Thompson Sampling: {self.num_factors} factors")
        
        def _select_factor_dap_ucb(self, factor_names: List[str]):
            """Select factor using Thompson Sampling"""
            # Sample from Beta distribution for each factor
            samples = [np.random.beta(self.alpha[i], self.beta[i])
                      for i in range(self.num_factors)]
            selected_idx = int(np.argmax(samples))
            selected_factor = factor_names[selected_idx]

            # Save selected factor index for subsequent statistics update
            self._last_selected_factor_idx = selected_idx
            self._last_selected_factor_name = selected_factor

            logging.info(f"Thompson sampling values: {[f'{s:.3f}' for s in samples]}")
            logging.info(f"Alpha (success): {[f'{a:.1f}' for a in self.alpha]}")
            logging.info(f"Beta (failure): {[f'{b:.1f}' for b in self.beta]}")
            logging.info(f"Selected: {selected_factor}")
            
            return selected_factor
        
        def step(self):
            """Execute one optimization step, update Thompson statistics"""
            # Fix: Record historical best score before optimizing this factor
            factor_idx = getattr(self, '_last_selected_factor_idx', None)
            if factor_idx is not None:
                score_before = self.factor_best_scores[factor_idx]
            else:
                score_before = -1.0

            # Execute standard optimization step
            result = super().step()

            # Fix: Update Thompson statistics for this factor after optimization
            if factor_idx is not None:
                factor_name = self._last_selected_factor_name
                # Get this factor's best score for this round from parent class statistics
                current_score = self.last_best_scores[factor_idx]

                # Correct judgment: current best vs historical best for this factor
                if current_score > score_before:
                    self.alpha[factor_idx] += 1
                    self.factor_best_scores[factor_idx] = current_score
                    logging.info(f"Factor '{factor_name}' improved! {score_before:.4f} → {current_score:.4f}, alpha+1")
                else:
                    self.beta[factor_idx] += 1
                    logging.info(f"Factor '{factor_name}' no improvement (current: {current_score:.4f}, best: {score_before:.4f}), beta+1")

            return result
    
    # Use Thompson Sampling optimizer
    optimizer = ThompsonSamplingOptimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name="aPSF-Thompson"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"Thompson optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict(),
        "thompson_alpha": optimizer.alpha.tolist(),
        "thompson_beta": optimizer.beta.tolist()
    }


def run_apsf_roundrobin(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Round-robin ablation experiment - Round-robin factor selection
    Simplest uniform exploration strategy, optimizes each factor in sequence
    """
    logging.info("Running Round-robin ablation experiment - Round-robin selection strategy")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Create Round-robin optimizer
    class RoundRobinOptimizer(Optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.current_factor_idx = 0
            logging.info(f"Initializing Round-robin: {self.num_factors} factors")
        
        def _select_factor_dap_ucb(self, factor_names: List[str]):
            """Select factor in round-robin fashion"""
            factor_idx = self.current_factor_idx
            selected_factor = factor_names[factor_idx]
            self.current_factor_idx = (self.current_factor_idx + 1) % self.num_factors

            logging.info(f"Round-robin selection: {selected_factor} (round-robin order)")
            logging.info(f"Next will select: {factor_names[self.current_factor_idx]}")
            
            return selected_factor
    
    # Use Round-robin optimizer
    optimizer = RoundRobinOptimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name="aPSF-RoundRobin"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"Round-robin optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }


def run_apsf_greedy(
    task_desc: str,
    eval_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    evaluator: BaseEvaluator,
    dataset_config: Dict[str, Any],
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Greedy-best ablation experiment - Always selects factor with highest historical average score
    Pure exploitation strategy, no exploration, prone to local optima
    """
    logging.info("Running Greedy-best ablation experiment - Greedy selection strategy")
    
    # Use Architect to discover structure
    architect = Architect()
    example_data = _construct_examples(eval_data[:5])
    prompt_struct = architect.discover_structure(task_desc, example_data)
    
    # If step parameter is specified, override config
    config = dataset_config.copy()
    if step is not None:
        config['total_optimization_steps'] = step
    
    # Create Greedy optimizer
    class GreedyOptimizer(Optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Initial phase: Try each factor at least once
            self.min_tries_per_factor = 1
            self.initialization_phase = True
            logging.info(f"Initializing Greedy: {self.num_factors} factors")
        
        def _select_factor_dap_ucb(self, factor_names: List[str]):
            """Greedy selection: Always pick factor with highest historical average score"""
            # Initialization phase: Ensure each factor is tried at least once
            if self.initialization_phase:
                for i in range(self.num_factors):
                    if self.candidate_counts[i] < self.min_tries_per_factor:
                        logging.info(f"Initial exploration: {factor_names[i]} (not tried yet)")
                        return factor_names[i]
                self.initialization_phase = False
                logging.info("Initial exploration complete, entering greedy mode")

            # Greedy phase: Select factor with highest average score
            avg_scores = []
            for i in range(self.num_factors):
                if self.candidate_counts[i] > 0:
                    avg_scores.append(self.candidate_values[i])  # Use average directly
                else:
                    avg_scores.append(0.0)

            selected_idx = int(np.argmax(avg_scores))
            selected_factor = factor_names[selected_idx]

            logging.info(f"Greedy selection: {selected_factor} (highest average score)")
            logging.info(f"Factor average scores: {[f'{s:.4f}' for s in avg_scores]}")
            logging.info(f"Factor attempt counts: {self.candidate_counts}")
            
            return selected_factor
    
    # Use Greedy optimizer
    optimizer = GreedyOptimizer(
        prompt_struct, eval_data, evaluator, config,
        method_name="aPSF-Greedy"
    )
    
    # Run optimization
    total_steps = config.get("total_optimization_steps", 10)
    for step in range(total_steps):
        logging.info(f"Greedy optimization step {step + 1}/{total_steps}")
        optimizer.step()
    
    # Test set evaluation
    test_score = optimizer.evaluate_on_test_set(test_data)
    
    return {
        "final_score": test_score,
        "optimized_prompt": optimizer.get_optimized_prompt(),
        "best_structure": prompt_struct.to_dict()
    }
