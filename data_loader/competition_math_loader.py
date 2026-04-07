import os
import re
import pyarrow.parquet as pq
from .base_loader import BaseLoader
import random

class CompetitionMathLoader(BaseLoader):
    """Load CompetitionMath dataset (Parquet format).

    Dataset contains fields:
    - problem: Mathematical problem text
    - level: Difficulty level (Level 1-5)
    - type: Problem type (e.g., Algebra, Geometry, etc.)
    - solution: Solution steps and process
    """
    
    def _load_data(self):
        """Load CompetitionMath data from Parquet file"""
        
        if not self.path:
            raise ValueError("Loader requires data path to be specified")
        
        # If path is directory, find parquet files
        if os.path.isdir(self.path):
            parquet_files = [f for f in os.listdir(self.path) if f.endswith('.parquet')]
            if not parquet_files:
                raise FileNotFoundError(f"No Parquet files found in {self.path}")
            parquet_file = os.path.join(self.path, parquet_files[0])
        else:
            parquet_file = self.path
        
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"Data file not found: {parquet_file}")
        
        try:
            # Read Parquet file
            table = pq.read_table(parquet_file)
            
            # Convert to list of dictionaries
            all_data = []
            for i in range(table.num_rows):
                row = table.take([i])
                item = {}
                for col in table.column_names:
                    item[col] = row[col][0].as_py()
                all_data.append(item)
            
            # Process data format
            processed_data = self._process_competition_math_data(all_data)
            
            # Keep only Level 5 data
            level5_data = [item for item in processed_data if item.get('level', '') == 'Level 5']
            print(f" Original data: {len(processed_data)} samples total")
            print(f" Filtered Level 5 data: {len(level5_data)} samples")
            
            # Shuffle data but don't split, let run_experiments.py split dynamically based on config
            # Use unified random seed for reproducibility
            from ..config import DATA_SPLIT_CONFIG
            random.seed(DATA_SPLIT_CONFIG['random_seed'])
            random.shuffle(level5_data)
            
            # Put all data in both train and test, let caller sample as needed
            self.data = {
                "train": level5_data,
                "test": level5_data
            }
            
            print(f" Successfully loaded CompetitionMath dataset (Level 5): {len(level5_data)} samples total")
            
            # Count Level 5 data type distribution
            type_dist = {}
            for item in level5_data:
                type_ = item.get('type', 'Unknown')
                type_dist[type_] = type_dist.get(type_, 0) + 1
            
            print(f"\n Level 5 type distribution:")
            for type_, count in sorted(type_dist.items(), key=lambda x: -x[1]):
                print(f"   {type_}: {count} samples")
            
        except Exception as e:
            print(f" Error reading CompetitionMath data: {e}")
            raise
    
    def _process_competition_math_data(self, raw_data):
        """Process CompetitionMath data format"""
        processed_data = []
        
        for idx, item in enumerate(raw_data):
            try:
                problem = str(item.get("problem", "")).strip()
                level = str(item.get("level", "")).strip()
                type_ = str(item.get("type", "")).strip()
                solution = str(item.get("solution", "")).strip()
                
                # Validate data integrity
                if not problem:
                    print(f"  Item {idx} has empty problem field, skipping")
                    continue
                
                # Extract final answer from problem (usually in \boxed{})
                answer = self._extract_answer_from_problem(problem, solution)
                
                # Build standardized input-output format
                processed_item = {
                    # CompetitionMath original fields
                    "problem": problem,
                    "level": level,
                    "type": type_,
                    "solution": solution,

                    # Standardized fields (compatible with different evaluators)
                    "prompt": problem,
                    "input": problem,
                    "question": problem,
                    "target": answer,
                    "answer": answer,
                    "output": answer,

                    # Additional information fields
                    "question_type": "mathematical_reasoning",
                    "domain": "mathematical_reasoning",
                    "explanation": solution,
                    "difficulty_level": level,
                    "subject": type_,
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"  Error processing item {idx}: {e}")
                continue
        
        return processed_data
    
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
    
    def _extract_answer_from_problem(self, problem: str, solution: str) -> str:
        """Extract answer from problem or solution"""
        
        # Method 1: Extract last numerical value or expression from end of solution
        if solution:
            # Try to extract content from \boxed{} (handle nested braces)
            boxed_match = self._extract_boxed_answer(solution)
            if boxed_match:
                return boxed_match
            
            # Try to extract from end (last few lines may contain answer)
            lines = solution.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('\\'):
                    # Clean LaTeX commands from line
                    cleaned = re.sub(r'\\[a-zA-Z]+\{', '', line)
                    cleaned = re.sub(r'\}', '', cleaned)
                    cleaned = cleaned.strip()
                    if cleaned and len(cleaned) < 200:
                        return cleaned
        
        # Method 2: Extract \boxed{} from problem
        boxed_match = self._extract_boxed_answer(problem)
        if boxed_match:
            return boxed_match
        
        # Method 3: If no explicit answer found, try to extract key data from last sentence
        if solution:
            # Extract last line
            lines = [l.strip() for l in solution.split('\n') if l.strip()]
            if lines:
                last_line = lines[-1]
                # Remove LaTeX formatting
                cleaned = re.sub(r'\\[a-zA-Z]+\{', '', last_line)
                cleaned = re.sub(r'\}', '', cleaned)
                if len(cleaned) < 200:
                    return cleaned
        
        # Final fallback: return "Unable to extract"
        return "Unable to extract"
    
    def get_formatted_sample(self, sample):
        """Get formatted sample for display and debugging"""
        problem = sample.get("problem", "")
        level = sample.get("level", "")
        type_ = sample.get("type", "")
        solution = sample.get("solution", "")
        answer = sample.get("answer", "")
        
        formatted = f"【{level} | {type_}】\n\n"
        formatted += f"Problem:\n{problem}\n\n"
        formatted += f"Answer: {answer}\n\n"
        formatted += f"Solution steps:\n{solution}"
        
        return formatted
