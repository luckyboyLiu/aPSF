# Improved Architect Implementation
import re
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
from ..llm_apis import get_llm, BaseLLM
from .prompt_object import PromptStructure

class TaskType(Enum):
    """Task type enumeration"""
    MULTIPLE_CHOICE = "multiple_choice"      # Multiple choice questions
    NUMERICAL_CALCULATION = "numerical"     # Numerical calculation
    TEXT_CLASSIFICATION = "classification"  # Text classification
    REASONING = "reasoning"                  # Reasoning tasks
    GENERATION = "generation"                # Generation tasks
    OTHER = "other"

@dataclass
class OutputFormat:
    """Output format configuration"""
    format_type: str                # Output format type
    constraints: List[str]          # Format constraints
    validation_rules: List[str]     # Validation rules
    examples: List[str]             # Format examples

@dataclass
class ArchitectConfig:
    """Architect configuration class"""
    verbose: bool = True
    log_level: str = "INFO"
    max_retries: int = 3
    fallback_enabled: bool = True
    strict_output_validation: bool = True  # Strict output validation

class TaskAnalyzer:
    """Task analyzer - Uses LLM to analyze task type and determine output format"""
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize task analyzer.

        Args:
            llm: Language model instance for task analysis.
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.mc_option_pattern = re.compile(r'^\s*(\([a-zA-Z]\)|[a-zA-Z][\.\)])\s+')
        
        self.output_formats = {
            TaskType.MULTIPLE_CHOICE: OutputFormat(
                format_type="Multiple Choice",
                constraints=[
                    "Answer must be a single letter (A, B, C, D, E, F, G, etc.)",
                    "No additional explanation after the letter"
                ],
                validation_rules=[
                    "Must match pattern: ^[A-Z]$",
                    "Case insensitive matching allowed"
                ],
                examples=["A", "B", "C", "D", "E", "F", "G"]
            ),
            TaskType.NUMERICAL_CALCULATION: OutputFormat(
                format_type="Numerical Calculation",
                constraints=[
                    "Answer must be a number",
                    "Use decimal notation when necessary"
                ],
                validation_rules=[
                    "Must be parseable as float",
                    "No units or currency symbols"
                ],
                examples=["42", "3.14", "100", "0.5"]
            ),
            TaskType.TEXT_CLASSIFICATION: OutputFormat(
                format_type="Text Classification",
                constraints=[
                    "Answer must be a predefined category",
                    "Use exact category names"
                ],
                validation_rules=[
                    "Must match one of the predefined categories",
                    "Case insensitive matching"
                ],
                examples=["Positive", "Negative", "Neutral"]
            ),
            TaskType.REASONING: OutputFormat(
                format_type="Reasoning Task",
                constraints=[
                    "Must show reasoning process",
                    "Conclude with explicit final answer"
                ],
                validation_rules=[
                    "Must contain reasoning steps",
                    "Final answer must be clearly marked"
                ],
                examples=["...reasoning process... Therefore, the final answer is: Yes.", 
                         "...reasoning process... Therefore, the final answer is: Option C is correct."]
            ),
            TaskType.GENERATION: OutputFormat(
                format_type="Generation Task",
                constraints=[
                    "Generate coherent and relevant text",
                    "Follow specified length requirements"
                ],
                validation_rules=[
                    "Must be grammatically correct",
                    "Must be relevant to the prompt"
                ],
                examples=["This is a summary about artificial intelligence...", 
                         "Once upon a time there was a mountain..."]
            )
        }
    
    def _create_classification_prompt(self, task_description: str, example_data: str) -> str:
        """Create a specialized prompt for task classification"""
        
        task_types_info = "\n".join([f'- `{e.value}`: {e.name}' for e in TaskType])
        
        return f"""You are a task classification expert. Your job is to analyze the given task information and select the most appropriate category from the list below.

=== Task Information ===
Task Description: {task_description}
Example Data:
{example_data}

=== Available Task Categories ===
{task_types_info}

Please strictly follow the requirements and only output the best matching task category ID (e.g., `multiple_choice`), without any other text or explanation."""

    def analyze_task(self, task_description: str, example_data: str) -> Tuple[TaskType, OutputFormat]:
        """
        Use LLM to analyze task type and return corresponding output format.
        This method first performs quick structural analysis to identify multiple choice questions,
        if that fails, it calls LLM for deeper classification.
        """
        # 1. Quick path: Identify multiple choice through structural analysis, which is both fast and accurate.
        lines = example_data.split('\n')
        mc_option_count = sum(1 for line in lines if self.mc_option_pattern.match(line))
        
        if mc_option_count >= 2:
            self.logger.info("Task type detected through structural analysis: Multiple Choice.")
            return TaskType.MULTIPLE_CHOICE, self.output_formats[TaskType.MULTIPLE_CHOICE]

        # 2. Deep analysis: If not obviously multiple choice, call LLM for intelligent classification.
        self.logger.info("Structural analysis did not match, calling LLM for in-depth task type analysis...")
        prompt = self._create_classification_prompt(task_description, example_data)
        
        try:
            response = self.llm.generate(prompt).strip().lower()
            
            # Map LLM response directly to TaskType enum
            detected_type = TaskType(response)
            self.logger.info(f"Task type identified by LLM: {detected_type.value}")
            
        except (ValueError, AttributeError) as e:
            # If LLM response is invalid or empty, log warning and fallback to 'other'
            self.logger.warning(f"Failed to parse task type from LLM response '{response}': {e}. Falling back to default type 'other'.")
            detected_type = TaskType.OTHER

        output_format = self.output_formats.get(detected_type, self._get_default_format())
        
        return detected_type, output_format
    
    def _get_default_format(self) -> OutputFormat:
        """Get default output format"""
        return OutputFormat(
            format_type="Generic Task",
            constraints=["Output must be direct and clear"],
            validation_rules=["Answer must be complete and accurate"],
            examples=["Specific answer"]
        )

class Architect:
    """
    Uses powerful LLM to automatically discover optimal fusion prompt structure for tasks.
    Improved version: Generate complete prompt first, then auto-decompose into factors.
    """
    
    def __init__(self, 
                 architect_llm_id: str = "architect",
                 config: Optional[ArchitectConfig] = None):
        """
        Initialize Architect.

        Args:
            architect_llm_id: ID of Architect LLM defined in config.py
            config: Architect configuration
        """
        self.architect_llm: BaseLLM = get_llm(architect_llm_id)
        self.config = config or ArchitectConfig()
        self.task_analyzer = TaskAnalyzer(llm=self.architect_llm)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger
    
    def discover_structure(self, task_description: str, example_data: str, initial_prompt: str = None) -> PromptStructure:
        """
        Discover fusion prompt structure - using improved method.

        Args:
            task_description: Task description
            example_data: Example data
            initial_prompt: Optional initial prompt (e.g., "Let's think step by step")
                          If provided, it will analyze this prompt and extract factors, then optimize based on it

        Returns:
            PromptStructure: Discovered prompt structure
        """
        self._validate_inputs(task_description, example_data)

        # Analyze task type and output format
        task_type, output_format = self.task_analyzer.analyze_task(task_description, example_data)

        # If initial prompt is provided, use different discovery strategy
        if initial_prompt:
            if self.config.verbose:
                print("\n" + "="*60)
                print(f"  Starting from initial prompt: {initial_prompt}")
                print(" Strategy: Analyze initial prompt and extract factors for optimization")
                print("="*60)

            # Analyze initial prompt and extract factors
            complete_prompt, factors, factor_mappings = self._analyze_initial_prompt_and_extract_factors(
                initial_prompt, task_description, example_data, task_type, output_format
            )
        else:
            # Original logic: Generate from scratch
            if self.config.verbose:
                print("\n" + "="*60)
                print(" Using improved structure discovery: Complete prompt + Auto factor decomposition")
                print("="*60)

            complete_prompt, factors, factor_mappings = self._generate_complete_prompt_with_auto_factors(
                task_description, example_data, task_type, output_format
            )

        # Validate and adjust factor count
        factors = self._validate_factor_count(factors)

        # Create improved structure object
        structure = PromptStructure(
            task_description=task_description,
            factors=factors,
            fusion_prompt=complete_prompt,
            factor_mappings=factor_mappings
        )

        # Add task type and output format information
        structure.task_type = task_type
        structure.output_format = output_format

        # Print detailed structure discovery results
        if self.config.verbose:
            self._print_improved_structure_info(structure)

        return structure

    def _analyze_initial_prompt_and_extract_factors(self, initial_prompt: str, task_description: str,
                                                    example_data: str, task_type: TaskType,
                                                    output_format: OutputFormat):
        """
        Analyze given initial prompt and extract optimization factors

        Args:
            initial_prompt: Initial prompt provided by user (e.g., "Let's think step by step")
            task_description: Task description
            example_data: Example data
            task_type: Task type
            output_format: Output format

        Returns:
            Tuple[str, Dict[str, str], Dict[str, dict]]: (complete prompt, factor dictionary, factor mappings)
        """

        # Create meta prompt for analyzing initial prompt
        from ..config import OPTIMIZATION_PARAMS
        min_factors = OPTIMIZATION_PARAMS.get("min_factors", 1)
        max_factors = OPTIMIZATION_PARAMS.get("max_factors", 6)
        
        analysis_meta_prompt = f"""Based on the initial prompt and task examples, generate a complete, usable instruction and decompose it into factors.

Initial Prompt: {initial_prompt}
Task Description: {task_description}
Task Type: {task_type.value}
Expected Output: {output_format.format_type}; constraints: {", ".join(output_format.constraints)}

Example Data:
{example_data}

Return ONLY the four sections below, plain text, no extra commentary:

Complexity Analysis:
<one sentence on why you chose N factors (must be within {min_factors}-{max_factors})>

Complete Instruction Template:
<concise, natural instruction; no numbering/bold/quotes; must respect the expected output constraints>

Factor Decomposition:
Factor1_<Name>: <one-line role>
Factor2_<Name>: <one-line role>
... (use {min_factors}-{max_factors} factors)

Factor Boundary Mapping:
Factor1_<Name>: "<verbatim substring copied from the Complete Instruction Template>"
Factor2_<Name>: "<verbatim substring copied from the Complete Instruction Template>"
... (names must exactly match Factor Decomposition; mappings must be verbatim substrings; avoid mapping the same sentence to multiple factors)

Do not output anything beyond these four sections."""

        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f"  ARCHITECT Analyzing Initial Prompt")
            print(f"{'='*80}")
            print(analysis_meta_prompt)
            print(f"{'─'*80}")

        # Call LLM for analysis
        response = self._call_llm_with_retry(analysis_meta_prompt)

        if self.config.verbose:
            print(f"\n  ARCHITECT Analysis Results:")
            print(response)
            print(f"{'─'*80}")

        # Parse response
        complete_prompt, factors, factor_mappings = self._parse_auto_factor_response(response)

        if self.config.verbose:
            print(f"\n  Extraction Results:")
            print(f"   Complete prompt: {complete_prompt}")
            print(f"   Number of factors: {len(factors)}")
            print(f"   Factors: {list(factors.keys())}")

        return complete_prompt, factors, factor_mappings

    def _generate_complete_prompt_with_auto_factors(self, task_description: str, example_data: str, 
                                                  task_type: TaskType, output_format: OutputFormat):
        """Core method: Let LLM generate complete prompt and auto-decompose into factors"""
        
        # Create improved meta prompt
        meta_prompt = self._create_auto_factor_meta_prompt(task_description, example_data, task_type, output_format)
        
        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f" ARCHITECT Improved Meta Prompt")
            print(f"{'='*80}")
            print(meta_prompt)
            print(f"{'─'*80}")
        
        # Call LLM
        response = self._call_llm_with_retry(meta_prompt)
        
        if self.config.verbose:
            print(f"\n ARCHITECT Response:")
            print(response)
            print(f"{'─'*80}")
        
        # Parse response
        complete_prompt, factors, factor_mappings = self._parse_auto_factor_response(response)
        
        return complete_prompt, factors, factor_mappings

    def _create_auto_factor_meta_prompt(self, task_description: str, example_data: str,
                                      task_type: TaskType, output_format: OutputFormat) -> str:
        """Create meta prompt for auto factor decomposition"""

        from ..config import OPTIMIZATION_PARAMS
        min_factors = OPTIMIZATION_PARAMS.get("min_factors", 1)
        max_factors = OPTIMIZATION_PARAMS.get("max_factors", 6)

        return f"""Based on the task description and examples, generate a complete, usable instruction and decompose it into factors.

Task Description: {task_description}
Task Type: {task_type.value}
Expected Output: {output_format.format_type}; constraints: {", ".join(output_format.constraints)}

Example Data:
{example_data}

Return ONLY the four sections below, plain text, no extra commentary:

Complexity Analysis:
<one sentence on why you chose N factors (must be within {min_factors}-{max_factors})>

Complete Instruction Template:
<concise, natural instruction; no numbering/bold/quotes; must respect the expected output constraints>

Factor Decomposition:
Factor1_<Name>: <one-line role>
Factor2_<Name>: <one-line role>
... (use {min_factors}-{max_factors} factors)

Factor Boundary Mapping:
Factor1_<Name>: "<verbatim substring copied from the Complete Instruction Template>"
Factor2_<Name>: "<verbatim substring copied from the Complete Instruction Template>"
... (names must exactly match Factor Decomposition; mappings must be verbatim substrings; avoid mapping the same sentence to multiple factors)

Do not output anything beyond these four sections."""

    def _parse_auto_factor_response(self, response: str) -> Tuple[str, Dict[str, str], Dict[str, dict]]:
        """Parse auto factor decomposition response"""

        # Extract complete instruction template
        complete_prompt = self._extract_complete_instruction(response)

        # Extract factor names and semantic descriptions
        factor_semantics = self._extract_auto_factors(response)

        # Build factor mappings (extracts actual text segments)
        factor_mappings = self._build_factor_mappings(complete_prompt, factor_semantics, response)

        # Build factors dict with actual text segments from mappings
        factors = {}
        for factor_name in factor_semantics.keys():
            # Try direct matching or fuzzy matching of factor names
            matched_mapping = None
            if factor_name in factor_mappings:
                matched_mapping = factor_mappings[factor_name]
            else:
                # Fuzzy matching: Check if factor_name contains mapping keys, or vice versa
                for mapping_key in factor_mappings.keys():
                    # Extract core part of factor name (remove Factor1-, ** prefixes, etc.)
                    factor_core = factor_name.replace('**', '').replace('Factor', '').strip()
                    if '‑' in factor_core:
                        factor_core = factor_core.split('‑', 1)[-1].strip()
                    elif '-' in factor_core:
                        factor_core = factor_core.split('-', 1)[-1].strip()
                    # Remove leading numbers
                    import re
                    factor_core = re.sub(r'^\d+\s*', '', factor_core).strip()

                    # Check if core name matches
                    if factor_core.lower() == mapping_key.lower() or mapping_key.lower() in factor_core.lower():
                        matched_mapping = factor_mappings[mapping_key]
                        break

            if matched_mapping:
                # Use actual text segment from mapping
                if "full_text" in matched_mapping:
                    candidate_text = matched_mapping["full_text"]
                elif "phrase" in matched_mapping:
                    candidate_text = matched_mapping["phrase"]
                else:
                    candidate_text = None

                # Verify that text actually exists in complete_prompt
                if candidate_text and candidate_text in complete_prompt:
                    factors[factor_name] = candidate_text
                else:
                    # Text not in prompt, try to find similar segments in prompt
                    print(f"  Warning: Factor '{factor_name}' mapping text not in prompt, attempting fuzzy search")
                    found_text = self._find_factor_text_in_prompt(factor_semantics[factor_name], complete_prompt, factor_name)
                    factors[factor_name] = found_text if found_text else factor_semantics[factor_name]
            else:
                # No mapping found, try to find in prompt
                print(f"  Warning: Factor '{factor_name}' has no mapping, attempting to find in prompt")
                found_text = self._find_factor_text_in_prompt(factor_semantics[factor_name], complete_prompt, factor_name)
                factors[factor_name] = found_text if found_text else factor_semantics[factor_name]

        # Final validation: Ensure all factor content is in prompt
        for factor_name, factor_text in factors.items():
            if factor_text not in complete_prompt:
                print(f"  Error: Factor '{factor_name}' content '{factor_text[:50]}...' not in prompt!")

        return complete_prompt, factors, factor_mappings

    def _find_factor_text_in_prompt(self, semantic_desc: str, prompt: str, factor_name: str) -> str:
        """Try to find text segments in prompt that are most relevant to semantic description"""
        import re

        # Extract keywords from semantic description
        keywords = re.findall(r'\b\w{4,}\b', semantic_desc.lower())

        # Find segments in prompt that contain keywords
        # Split prompt by semicolon or comma
        segments = re.split(r'[;,]', prompt)

        best_match = None
        best_score = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            segment_lower = segment.lower()
            # Calculate number of matching keywords
            score = sum(1 for kw in keywords if kw in segment_lower)
            if score > best_score:
                best_score = score
                best_match = segment

        return best_match

    def _extract_complete_instruction(self, response: str) -> str:
        """Extract complete instruction template"""
        import re
        
        patterns = [
            r'\*\*Complete Instruction Template:\*\*(.*?)(?:\*\*Factor Decomposition:\*\*|$)',
            r'Complete.*?Instruction[^:]*:(.*?)(?:Factor|$)',
            r'\*\*Complete\s+Instruction\s+Template:\*\*(.*?)(?:\*\*Factor\s+Decomposition:\*\*|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                instruction = match.group(1).strip()
                # Clean format
                instruction = re.sub(r'^\[|\]$', '', instruction)
                instruction = re.sub(r'\n+', ' ', instruction).strip()
                
                # If contains **tag:**, truncate at first tag
                # Completely remove second paragraph (tag + content)
                markdown_heading = re.search(r'\*\*[^*]+:\*\*', instruction)
                if markdown_heading:
                    # Keep only content before the tag
                    instruction = instruction[:markdown_heading.start()].strip()
                
                # No need for {input} placeholder - question will be provided separately
                
                return instruction
        
        # Fallback: Return a simple instruction without input placeholder
        return "Please analyze the problem step by step and provide your answer."

    def _extract_auto_factors(self, response: str) -> Dict[str, str]:
        """Extract auto-decomposed factors"""
        import re

        factors = {}

        # Find factor decomposition section
        patterns = [
            r'\*\*Factor Decomposition:\*\*(.*?)(?:\*\*Factor Boundary Mapping:\*\*|$)',
            r'Factor.*?Decomposition[^:]*:(.*?)(?:Boundary|Mapping|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                factors_section = match.group(1).strip()
                lines = factors_section.split('\n')

                for line in lines:
                    line = line.strip()
                    if 'Factor' in line and ':' in line:
                        # Parse: **Factor1‑Name:** Description or Factor1-Name: Description
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            name_part = parts[0].strip()
                            description = parts[1].strip()

                            # Clean factor name: remove **, -, ‑, number prefixes, etc.
                            # For example "**Factor1‑Comprehension" -> "Comprehension"
                            factor_name = name_part
                            factor_name = factor_name.replace('**', '').replace('*', '').strip()
                            # Handle Factor1‑Name or Factor1-Name format
                            if '‑' in factor_name:  # Unicode minus sign
                                factor_name = factor_name.split('‑', 1)[-1].strip()
                            elif '-' in factor_name:  # Regular minus sign
                                factor_name = factor_name.split('-', 1)[-1].strip()
                            # Remove square brackets
                            factor_name = re.sub(r'^\[|\]$', '', factor_name).strip()

                            if factor_name:
                                factors[factor_name] = description
                break

        # If parsing fails, provide default factors
        if not factors:
            factors = {
                "Problem Understanding": "Analyze and understand the given problem",
                "Solution Process": "Apply systematic approach to solve the problem",
                "Answer Formation": "Format the answer according to requirements"
            }

        return factors

    def _build_factor_mappings(self, complete_prompt: str, factors: Dict[str, str], response: str) -> Dict[str, dict]:
        """Build factor mapping relationships"""
        factor_mappings = {}
        
        # Try to extract explicit mappings from response
        mappings_from_response = self._extract_explicit_mappings(response)
        
        if mappings_from_response:
            factor_mappings.update(mappings_from_response)
        else:
            # Auto-build mappings (divide complete prompt equally among factors)
            factor_mappings = self._auto_build_factor_mappings(complete_prompt, factors)
        
        return factor_mappings

    def _extract_explicit_mappings(self, response: str) -> Dict[str, dict]:
        """Extract explicit factor mappings from response - extract factor boundary markers"""
        import re
        mappings = {}

        # Look for boundary mapping section - supports multiple formats
        patterns = [
            r'\*\*Factor Boundary Mapping:?\*\*\s*(.*?)(?:Generate now|$)',
            r'Factor Boundary Mapping:?\s*(.*?)(?:Generate now|$)',
        ]

        mapping_section = None
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                mapping_section = match.group(1).strip()
                break

        if mapping_section:
            lines = mapping_section.split('\n')

            for line in lines:
                line = line.strip()
                if not line or line.startswith('---'):
                    continue

                # Match format: - **FactorName:** "text" or [FactorName]: "text"
                # First find the position of the first quote
                quote_start = line.find('"')
                if quote_start == -1:
                    quote_start = line.find("'")

                if quote_start > 0 and ':' in line[:quote_start]:
                    # Factor name is before the colon before the quotes
                    colon_pos = line[:quote_start].rfind(':')
                    name_part = line[:colon_pos]
                    content_part = line[quote_start:]

                    # Clean factor name
                    factor_name = name_part.strip()
                    factor_name = factor_name.replace('[', '').replace(']', '')
                    factor_name = factor_name.replace('**', '').replace('*', '')
                    factor_name = factor_name.replace('- ', '').replace('→', '')
                    factor_name = factor_name.strip()

                    if not factor_name:
                        continue

                    # Extract complete content within quotes
                    content = content_part.strip()
                    if content.startswith('"') or content.startswith("'"):
                        quote_char = content[0]
                        last_quote = content.rfind(quote_char, 1)
                        if last_quote > 0:
                            full_text = content[1:last_quote].strip()
                            mappings[factor_name] = {
                                "phrase": full_text,
                                "full_text": full_text
                            }

        return mappings

    def _auto_build_factor_mappings(self, complete_prompt: str, factors: Dict[str, str]) -> Dict[str, dict]:
        """Auto-build factor mapping relationships"""
        factor_mappings = {}
        
        # Divide prompt into segments for each factor
        factor_names = list(factors.keys())
        prompt_length = len(complete_prompt)
        segment_length = prompt_length // len(factor_names)
        
        for i, factor_name in enumerate(factor_names):
            start_pos = i * segment_length
            end_pos = (i + 1) * segment_length if i < len(factor_names) - 1 else prompt_length
            
            # Try to split on word boundaries
            if end_pos < prompt_length:
                while end_pos > start_pos and complete_prompt[end_pos] != ' ':
                    end_pos -= 1
            
            phrase = complete_prompt[start_pos:end_pos].strip()
            
            factor_mappings[factor_name] = {
                "start": start_pos,
                "end": end_pos,
                "phrase": phrase
            }
        
        return factor_mappings

    def _validate_factor_count(self, factors: Dict[str, str]) -> Dict[str, str]:
        """Validate and adjust factor count"""
        from ..config import OPTIMIZATION_PARAMS
        
        min_factors = OPTIMIZATION_PARAMS.get("min_factors", 2)
        max_factors = OPTIMIZATION_PARAMS.get("max_factors", 4)
        
        factor_count = len(factors)
        
        if factor_count < min_factors:
            self.logger.warning(f"Factor count too low ({factor_count} < {min_factors}), adding default factors")
            # Add default factors
            while len(factors) < min_factors:
                factors[f"Additional Factor {len(factors) + 1}"] = "Additional processing step"
        
        elif factor_count > max_factors:
            self.logger.warning(f"Factor count too high ({factor_count} > {max_factors}), keeping first {max_factors}")
            # Keep first max_factors
            factor_items = list(factors.items())[:max_factors]
            factors = dict(factor_items)
        
        return factors

    def _print_improved_structure_info(self, structure: PromptStructure):
        """Print improved structure information"""
        print(f"\n{'='*80}")
        print(f" aPSF Structure Discovery Results")
        print(f"{'='*80}")
        print(f" Complete Instruction Prompt (clean, fluent):")
        print(f"   {structure.fusion_prompt}")
        print(f"\n Factor Decomposition (Total: {len(structure.factors)}):")
        
        # Display each factor and its corresponding actual text segment
        for i, (factor_name, factor_text) in enumerate(structure.factors.items(), 1):
            print(f"   {i}. [{factor_name}]")
            print(f"      Text Segment: \"{factor_text}\"")
        
        print(f"{'='*80}\n")

    def _validate_inputs(self, task_description: str, example_data: str) -> None:
        """Validate input parameters"""
        # Allow empty task description, don't validate
        if not example_data or not example_data.strip():
            raise ValueError("Example data cannot be empty")
    
    def _call_llm_with_retry(self, prompt: str) -> str:
        """LLM call with retry mechanism"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.architect_llm.generate(prompt)
                if response and response.strip():
                    return response
                self.logger.warning(f"LLM returned empty response, attempt {attempt + 1}/{self.config.max_retries}")
            except Exception as e:
                self.logger.error(f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        raise RuntimeError("LLM call retry limit reached")

    def _print_task_analysis(self, task_type: TaskType, output_format: OutputFormat) -> None:
        """Print task analysis results"""
        print("\n" + "="*60)
        print(" Task Type Analysis")
        print("="*60)
        print(f" Detected Task Type: {task_type.value}")
        print(f" Expected Output Format: {output_format.format_type}")
        print(" Format Constraints:")
        for i, constraint in enumerate(output_format.constraints, 1):
            print(f"   {i}. {constraint}")
        print(" Validation Rules:")
        for i, rule in enumerate(output_format.validation_rules, 1):
            print(f"   {i}. {rule}")
        print("="*60)
