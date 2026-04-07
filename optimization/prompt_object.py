from typing import List, Dict, Optional, Any
import re

class PromptStructure:
    """
    Fusion structure prompt: Fuse factors into natural language prompts and support implicit positioning optimization at the factor level.
    """
    def __init__(self, task_description: str, factors: Optional[Dict[str, str]] = None, 
                 fusion_prompt: Optional[str] = None, factor_mappings: Optional[Dict[str, Dict]] = None):
        """
        Initialize fusion prompt structure.

        Args:
            task_description (str): Brief description of the task.
            factors (Optional[Dict[str, str]]): Factor dictionary, keys are factor names, values are factor semantic content.
            fusion_prompt (Optional[str]): Complete natural language prompt after fusion.
            factor_mappings (Optional[Dict[str, Dict]]): Position mapping of factors in the fusion prompt.
        """
        self.task_description = task_description
        self.factors = factors if factors else {}
        self.fusion_prompt = fusion_prompt if fusion_prompt else ""
        
        # Factor mapping: Record the position of each factor's corresponding text segment in the fusion prompt
        # Format: {factor_name: {"start": int, "end": int, "phrase": str}}
        self.factor_mappings = factor_mappings if factor_mappings else {}

        # Initialize statistical data for optimization algorithms
        self.factor_stats = {
            name: self._create_initial_stats() for name in self.factors
        }
        
        # If fusion prompt is not provided, generate from factors
        if not self.fusion_prompt and self.factors:
            self.fusion_prompt = self._generate_fusion_prompt()

    def _create_initial_stats(self) -> Dict:
        """Create initial statistics dictionary for a single factor."""
        return {
            "selections": 0,          # (UCB1) Number of times selected
            "score": 0.0,             # (UCB1) Cumulative score
            "best_score": -1.0,       # (DAP-UCB) Historical best score
            "max_improvement": 0.0,   # (DAP-UCB) Maximum improvement Δ_k
            "patience_counter": 0,    # (DAP-UCB) Patience counter
            "is_frozen": False        # (DAP-UCB) Whether frozen
        }

    def _generate_fusion_prompt(self) -> str:
        """
        Fuse factors into natural language prompts.
        Improved version: Use LLM intelligent fusion to ensure natural fluency.
        """
        if not self.factors:
            return ""
        
        factor_contents = list(self.factors.values())
        
        # Single factor case: return directly
        if len(factor_contents) == 1:
            content = factor_contents[0].rstrip('.,;')
            return content + "."

        # Multiple factors case: use LLM intelligent fusion
        return self._llm_intelligent_fusion(factor_contents)
    
    def _llm_intelligent_fusion(self, factor_contents: List[str]) -> str:
        """Use LLM to intelligently fuse multiple factors"""
        try:
            from ..llm_apis import get_llm
            llm = get_llm("architect")  # Use architect LLM
            
            fusion_meta_prompt = f"""You are a language expert. Your task is to merge the following factors into one complete, natural sentence while preserving each factor's original meaning.

Factors to merge:
{chr(10).join(f"- {content.rstrip('.,;')}" for content in factor_contents)}

Requirements:
- Keep the original meaning of each factor intact
- Use appropriate connectors to link factors smoothly
- Output one complete, coherent sentence
- No additional commentary or explanation

Output the merged sentence:"""
            
            response = llm.generate(fusion_meta_prompt).strip()
            
            # Clean response
            response = response.rstrip('.,;')

            # Ensure ends with period
            if not response.endswith('.'):
                response += "."
                
            return response
            
        except Exception as e:
            print(f" LLM fusion failed, fallback to simple connection: {e}")
            # Fallback to improved simple connection
            return self._fallback_fusion(factor_contents)
    
    def _fallback_fusion(self, factor_contents: List[str]) -> str:
        """Fallback fusion strategy: improved simple connection"""
        # Clean factor content
        cleaned_factors = [content.rstrip('.,;') for content in factor_contents]
        
        # Intelligent connection (no semicolons)
        if len(cleaned_factors) == 2:
            fusion_prompt = f"{cleaned_factors[0]} and {cleaned_factors[1]}"
        else:
            # Multiple factors: connect the last one with and
            fusion_prompt = ", ".join(cleaned_factors[:-1]) + f", and {cleaned_factors[-1]}"

        # Add natural beginning
        if not fusion_prompt.lower().startswith(("let's", "to", "please", "first")):
            fusion_prompt = f"Let's {fusion_prompt.lower()}"

        # No need for input context - instruction should be standalone
        
        return fusion_prompt + "."

    def _convert_gerund_to_imperative(self, content: str) -> str:
        """Convert gerund form to imperative"""
        # Simple conversion rules
        if content.startswith("Providing"):
            return content.replace("Providing", "Provide")
        elif content.startswith("Outputting"):
            return content.replace("Outputting", "Output")
        elif content.startswith("Recognizing"):
            return content.replace("Recognizing", "Recognize")
        # More conversion rules can be added
        return content

    def _convert_to_tone_phrase(self, content: str) -> str:
        """Convert tone factors to natural language phrases"""
        tone_mapping = {
            "friendly": "in a friendly manner",
            "formal": "with a formal approach", 
            "casual": "in a casual style",
            "professional": "professionally",
            "helpful": "helpfully"
        }
        
        # Try to match known tone words
        content_lower = content.lower()
        for tone, phrase in tone_mapping.items():
            if tone in content_lower:
                return phrase

        # If no match, try to extract tone from content
        if "friendly" in content_lower or "warm" in content_lower:
            return "in a friendly manner"
        elif "formal" in content_lower or "professional" in content_lower:
            return "with a formal approach"
        else:
            return f"with a {content.strip()} approach"

    def _convert_to_format_phrase(self, content: str) -> str:
        """Convert format factors to natural language phrases"""
        format_mapping = {
            "step-by-step": "step by step",
            "systematic": "systematically",
            "structured": "in a structured way",
            "detailed": "with detailed explanations",
            "concise": "concisely"
        }
        
        content_lower = content.lower()
        for format_key, phrase in format_mapping.items():
            if format_key in content_lower:
                return phrase
        
        # Default conversion
        if "step" in content_lower:
            return "step by step"
        elif "detail" in content_lower:
            return "with detailed explanations"
        else:
            return f"using a {content.strip()} format"

    def _convert_to_perspective_phrase(self, content: str) -> str:
        """Convert perspective factors to natural language phrases"""
        content_lower = content.lower()
        if "second" in content_lower or "you" in content_lower:
            return "as if guiding someone through the process"
        elif "first" in content_lower or "i" in content_lower:
            return "from a first-person perspective"
        else:
            return f"from a {content.strip()} perspective"

    def _convert_to_instruction_phrase(self, content: str) -> str:
        """Convert other factors to instruction phrases"""
        # Remove common instruction words, keep core content
        content = re.sub(r'^(please|make sure|ensure|remember to)\s+', '', content.lower())
        return f"ensuring you {content.strip()}"

    def _intelligently_combine_phrases(self, phrases: List[str]) -> str:
        """Intelligently combine all factor phrases"""
        if not phrases:
            return "carefully and systematically"
        
        if len(phrases) == 1:
            return phrases[0]
        elif len(phrases) == 2:
            return f"{phrases[0]} and {phrases[1]}"
        else:
            # For multiple phrases, separate with commas, connect last one with and
            return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    def _map_factors_to_fusion(self, fusion_prompt: str):
        """
        Establish mapping relationship from factors to text segments in fusion prompt.
        Now completely relies on mapping information provided by LLM during structure discovery.
        """
        # Modified: No longer use fixed template matching, completely rely on Architect provided mapping
        # This method is now mainly used for validation and post-processing of mapping information
        print(f" Using LLM provided factor mapping, total {len(self.factor_mappings)} mappings")
        
        # Validate mapping validity
        for factor_name, mapping in self.factor_mappings.items():
            phrase = mapping.get("phrase", "")
            if phrase and phrase in fusion_prompt:
                print(f" Factor '{factor_name}' successfully mapped to: '{phrase}'")
            else:
                print(f" Factor '{factor_name}' mapping may have issues: '{phrase}'")

        # No longer need fixed phrase matching logic

    def update_factor(self, factor_name: str, new_content: str):
        """
        Update specified factor and regenerate fusion prompt.
        Now completely uses LLM intelligent replacement method, no longer depends on position mapping.
        """
        if factor_name not in self.factors:
            raise ValueError(f"Factor '{factor_name}' not found.")

        # Update factor content
        old_content = self.factors[factor_name]
        self.factors[factor_name] = new_content

        # Completely use LLM intelligent regeneration, no longer depend on complex position mapping
        print(f" Factor '{factor_name}' updated: '{old_content}' → '{new_content}'")
        print(f" Using LLM intelligent regeneration of fusion prompt...")

        # Regenerate entire fusion prompt to ensure natural language fluency
        self.fusion_prompt = self._generate_fusion_prompt()

        # Clean up outdated mapping information to avoid inconsistency
        if hasattr(self, 'factor_mappings'):
            self.factor_mappings.clear()
            print(f" Outdated factor mappings cleared")

    def add_factor(self, name: str, content: str):
        """Add new factor and regenerate fusion prompt."""
        if name not in self.factors:
            self.factors[name] = content
            self.factor_stats[name] = self._create_initial_stats()
            # Regenerate fusion prompt
            self.fusion_prompt = self._generate_fusion_prompt()

    def render(self, factor_order: Optional[List[str]] = None) -> str:
        """
        Return fused natural language prompt.

        Args:
            factor_order: Ignored in fusion prompt since factors are already fused

        Returns:
            str: Complete fused prompt.
        """
        return self.fusion_prompt

    def get_factor_names(self) -> List[str]:
        """Return list of all factor names."""
        return list(self.factors.keys())

    def compose(self, specific_factors: Optional[Dict[str, str]] = None) -> str:
        """
        Return complete executable prompt.

        In aPSF, fusion_prompt is generated by Architect or updated through _llm_smart_factor_replacement,
        no need to refusion. This method just returns the current fusion prompt.

        Args:
            specific_factors: Deprecated, parameter kept only for backward compatibility

        Returns:
            str: Complete fusion prompt (pure, fluent instruction)
        """
        # Directly return current fusion prompt
        # During optimization, fusion_prompt is updated directly through _llm_smart_factor_replacement
        return self.fusion_prompt

    def create_optimization_meta_prompt(self, factor_name: str, num_candidates: int) -> str:
        """
        Create optimization prompt for LLM autonomously identified factors
        """
        if factor_name not in self.factors:
            raise ValueError(f"Factor '{factor_name}' not found.")
        
        current_content = self.factors[factor_name]
        
        meta_prompt = f"""You are an expert prompt engineer. Your task is to generate improved alternatives for a specific factor in a fusion prompt.

=== Current Situation ===
Task: {self.task_description}

Complete Fusion Prompt:
{self.fusion_prompt}

Target Factor: {factor_name}
Factor Description: {current_content}

=== Your Mission ===
Generate {num_candidates} alternative phrases for the "{factor_name}" factor to improve the overall prompt's effectiveness.

=== Guidelines ===
1. **Stay Focused**: Each alternative should improve the aspect controlled by "{factor_name}"
2. **Maintain Natural Flow**: Alternatives should fit naturally into the fusion prompt
3. **Be Specific**: Generate concrete, actionable phrases rather than generic ones
4. **Keep Concise**: Keep the segment short and to the point - avoid verbose instructions
5. **Direct Impact**: Focus on changes that would meaningfully affect model performance

=== Response Format ===
Provide ONLY the alternative phrases in this exact format:

--- ALTERNATIVE ---
[replacement phrase 1]
--- ALTERNATIVE ---
[replacement phrase 2]
--- ALTERNATIVE ---
[replacement phrase 3]
--- ALTERNATIVE ---
[replacement phrase 4]

Generate alternatives that specifically improve the "{factor_name}" aspect:"""

        return meta_prompt.strip()

    def __str__(self):
        return f"PromptStructure(task='{self.task_description}', factors={len(self.factors)}, fusion='{self.fusion_prompt[:50]}...')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert PromptStructure to JSON serializable dictionary"""
        result = {
            "task_description": self.task_description,
            "factors": self.factors,
            "fusion_prompt": self.fusion_prompt,
            "factor_mappings": self.factor_mappings,
            "factor_stats": self.factor_stats
        }

        # Handle newly added attributes (if exist)
        if hasattr(self, 'task_type'):
            result["task_type"] = self.task_type.value if self.task_type else None
        
        if hasattr(self, 'output_format'):
            if self.output_format:
                result["output_format"] = {
                    "format_type": self.output_format.format_type,
                    "constraints": self.output_format.constraints,
                    "validation_rules": self.output_format.validation_rules,
                    "examples": self.output_format.examples
                }
            else:
                result["output_format"] = None
        
        if hasattr(self, 'validation_info'):
            result["validation_info"] = self.validation_info
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptStructure':
        """Rebuild PromptStructure object from dictionary"""
        instance = cls(
            task_description=data["task_description"],
            factors=data.get("factors", {}),
            fusion_prompt=data.get("fusion_prompt", ""),
            factor_mappings=data.get("factor_mappings", {})
        )

        # Restore factor statistics
        if "factor_stats" in data:
            instance.factor_stats = data["factor_stats"]

        # Restore newly added attributes
        if "task_type" in data and data["task_type"]:
            from .architect import TaskType
            instance.task_type = TaskType(data["task_type"])
        
        if "output_format" in data and data["output_format"]:
            from .architect import OutputFormat
            format_data = data["output_format"]
            instance.output_format = OutputFormat(
                format_type=format_data["format_type"],
                constraints=format_data["constraints"],
                validation_rules=format_data["validation_rules"],
                examples=format_data["examples"]
            )
        
        if "validation_info" in data:
            instance.validation_info = data["validation_info"]
            
        return instance