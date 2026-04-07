import pandas as pd
import ast
import os
import random
from .base_loader import BaseLoader

class GSM8KLoader(BaseLoader):
    """Load local GSM8K dataset (supports TSV and Parquet formats)."""""
    
    def _load_data(self):
        """Load GSM8K data from local files"""
        
        # Check for parquet file
        parquet_file = os.path.join(self.path, "train-00000-of-00001.parquet")
        
        if os.path.exists(parquet_file):
            # Use new parquet data
            print(f" Using new parquet data: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            all_data = self._process_parquet_data(df)
            
            # Set random seed to ensure consistent data split
            from ..config import DATA_SPLIT_CONFIG
            random.seed(DATA_SPLIT_CONFIG['random_seed'])
            random.shuffle(all_data)
            
            # Split train and test sets
            split_idx = int(len(all_data) * 0.8)
            train_data = all_data[:split_idx]
            test_data = all_data[split_idx:]
            
        else:
            # Fallback to original TSV format
            print(" Using TSV format data")
            train_file = os.path.join(self.path, "gsm_train.tsv")
            test_file = os.path.join(self.path, "gsm_test.tsv")
            
            # Load training set
            if os.path.exists(train_file):
                # Read TSV file, tab-separated, no header
                train_df = pd.read_csv(train_file, sep='\t', header=None, names=['question', 'answer', 'solution'])
                train_data = self._process_tsv_data(train_df)
            else:
                train_data = []
                print(f"Warning: {train_file} not found")
            
            # Load test set (if exists)
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file, sep='\t', header=None, names=['question', 'answer', 'solution'])
                test_data = self._process_tsv_data(test_df)
            else:
                # If no test set, use last 20% of train data as test
                split_idx = int(len(train_data) * 0.8)
                test_data = train_data[split_idx:]
                train_data = train_data[:split_idx]
                print(f"Warning: {test_file} not found, using last 20% of train data as test")
        
        self.data = {
            "train": train_data,
            "test": test_data
        }
        
        print(f" Loaded GSM8K: {len(train_data)} train, {len(test_data)} test samples")
    
    def _process_tsv_data(self, df):
        """Process TSV data format - based on actual data format"""
        processed_data = []
        
        for idx, row in df.iterrows():
            question = row['question'].strip()  # Question
            answer = str(row['answer']).strip()  # Answer (converted to string)
            solution = row['solution']           # Solution process
            
            # Handle bytes format solution (e.g.: b'Natalia sold...')
            if isinstance(solution, str) and solution.startswith("b'") and solution.endswith("'"):
                try:
                    # Remove b' and ', then handle escape characters
                    solution_content = solution[2:-1]  # Remove b' and '
                    # Handle common escape characters
                    solution_content = solution_content.replace('\\n', '\n')
                    solution_content = solution_content.replace('\\t', '\t')
                    solution_content = solution_content.replace('\\\\', '\\')
                    solution = solution_content
                except Exception as e:
                    print(f"Warning: Failed to parse solution at row {idx}: {e}")
                    solution = str(solution)
            
            processed_item = {
                # GSM8K standard fields
                "question": question,
                "answer": answer,
                "solution": solution,
                
                # Compatibility fields (to adapt to different evaluators)
                "prompt": question,
                "target": answer,
                "input": question,
                "output": answer,
                
                # Numerical answer (for mathematical evaluation)
                "numerical_answer": self._extract_numerical_answer(answer)
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _process_parquet_data(self, df):
        """Process Parquet data format"""
        processed_data = []
        
        for idx, row in df.iterrows():
            question = row['question'].strip()
            answer_solution = row['answer'].strip()
            
            # Extract final numerical answer from solution
            numerical_answer = self._extract_final_answer(answer_solution)
            
            processed_item = {
                "question": question,
                "answer": str(numerical_answer) if numerical_answer is not None else answer_solution,
                "solution": answer_solution,
                "prompt": question,
                "target": str(numerical_answer) if numerical_answer is not None else answer_solution,
                "input": question,
                "output": str(numerical_answer) if numerical_answer is not None else answer_solution,
                "numerical_answer": numerical_answer
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _extract_final_answer(self, solution):
        """Extract final numerical answer from GSM8K solution"""
        import re
        
        # GSM8K format: "step by step <<calculation=result>> final answer"
        # Extract all <<...=number>> results, take the last one
        calculations = re.findall(r'<<.*?=(\d+(?:\.\d+)?)>>', solution)
        
        if calculations:
            try:
                return float(calculations[-1])
            except:
                pass
        
        # If not found, try to extract the last number
        numbers = re.findall(r'(\d+(?:\.\d+)?)', solution)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        
        return None
    
    def _extract_numerical_answer(self, answer_str):
        """Extract numerical value from answer string"""
        try:
            # Try to convert directly to number
            return float(answer_str)
        except:
            # If failed, try to extract numbers from string
            import re
            numbers = re.findall(r'-?\d+\.?\d*', str(answer_str))
            if numbers:
                try:
                    return float(numbers[-1])  # Take the last number
                except:
                    pass
            return None