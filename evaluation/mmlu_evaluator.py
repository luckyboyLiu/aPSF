import re
from typing import List, Dict, Any
from .accuracy_evaluator import AccuracyEvaluator

class MMLUEvaluator(AccuracyEvaluator):
    """
    MMLU specialized evaluator - outputs results by subject category
    """
    
    def __init__(self, extractor_llm_id: str = "worker"):
        super().__init__(extractor_llm_id)
        
        # MMLU subject category mapping
        self.subject_mapping = {
            # Humanities
            "formal_logic": "Humanities",
            "high_school_european_history": "Humanities", 
            "high_school_us_history": "Humanities",
            "high_school_world_history": "Humanities",
            "international_law": "Humanities",
            "jurisprudence": "Humanities",
            "logical_fallacies": "Humanities",
            "moral_disputes": "Humanities",
            "moral_scenarios": "Humanities",
            "philosophy": "Humanities",
            "prehistory": "Humanities",
            "professional_law": "Humanities",
            "world_religions": "Humanities",
            
            # Social Science
            "econometrics": "Social Science",
            "high_school_geography": "Social Science",
            "high_school_government_and_politics": "Social Science", 
            "high_school_macroeconomics": "Social Science",
            "high_school_microeconomics": "Social Science",
            "high_school_psychology": "Social Science",
            "human_sexuality": "Social Science",
            "professional_psychology": "Social Science",
            "public_relations": "Social Science",
            "security_studies": "Social Science",
            "sociology": "Social Science",
            "us_foreign_policy": "Social Science",
            
            # STEM
            "abstract_algebra": "STEM",
            "anatomy": "STEM", 
            "astronomy": "STEM",
            "college_biology": "STEM",
            "college_chemistry": "STEM",
            "college_computer_science": "STEM",
            "college_mathematics": "STEM",
            "college_physics": "STEM",
            "computer_security": "STEM",
            "conceptual_physics": "STEM",
            "electrical_engineering": "STEM",
            "elementary_mathematics": "STEM",
            "high_school_biology": "STEM",
            "high_school_chemistry": "STEM",
            "high_school_computer_science": "STEM",
            "high_school_mathematics": "STEM",
            "high_school_physics": "STEM",
            "high_school_statistics": "STEM",
            "machine_learning": "STEM",
            "medical_genetics": "STEM",
            "virology": "STEM",
            
            # Other
            "business_ethics": "Other",
            "clinical_knowledge": "Other",
            "college_medicine": "Other",
            "global_facts": "Other",
            "human_aging": "Other",
            "management": "Other",
            "marketing": "Other",
            "miscellaneous": "Other",
            "nutrition": "Other",
            "professional_accounting": "Other",
            "professional_medicine": "Other",
        }
    
    def evaluate(self, predictions: List[str], references: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate in MMLU standard format: calculate accuracy for each subject category
        """
        print(f"\n Starting MMLU subject category evaluation - {len(predictions)} samples")

        # Organize data by subject category
        subject_results = {}
        category_results = {"Humanities": [], "Social Science": [], "STEM": [], "Other": []}
        
        # Evaluate each sample
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            subject = ref.get('subject', 'unknown')
            category = self.subject_mapping.get(subject, 'Other')

            # Evaluate single sample
            result = self.evaluate_single_prediction(pred, ref, show_detail=False)
            is_correct = result["is_correct"]

            # Record subject results
            if subject not in subject_results:
                subject_results[subject] = []
            subject_results[subject].append(is_correct)

            # Record category results
            category_results[category].append(is_correct)
            
            # Show details for first few samples
            if i < 5:
                status = "Yes" if is_correct else "No"
                print(f"\nSample {i+1} [{subject}] [{category}] {status}")
                print(f"   Original response: {pred[:100]}{'...' if len(pred) > 100 else ''}")
                print(f"   Extracted answer: '{result['extracted_answer']}'")
                print(f"   Target answer: '{result['target_answer']}'")
                print("-" * 50)
        
        # Calculate subject accuracies
        subject_accuracies = {}
        for subject, results in subject_results.items():
            if results:  # Ensure there are results
                accuracy = sum(results) / len(results)
                subject_accuracies[subject] = accuracy
            else:
                subject_accuracies[subject] = 0.0

        # Calculate category accuracies
        category_accuracies = {}
        for category, results in category_results.items():
            if results:  # Ensure there are results
                accuracy = sum(results) / len(results)
                category_accuracies[category] = accuracy
            else:
                category_accuracies[category] = 0.0

        # Calculate overall accuracy
        all_results = []
        for results in category_results.values():
            all_results.extend(results)
        overall_accuracy = sum(all_results) / len(all_results) if all_results else 0.0

        # Construct standard MMLU format results
        mmlu_results = {
            "Humanities": category_accuracies["Humanities"],
            "Social Science": category_accuracies["Social Science"],
            "STEM": category_accuracies["STEM"],
            "Other": category_accuracies["Other"],
            "Average": overall_accuracy
        }

        # Add detailed subject results
        mmlu_results["subject_details"] = subject_accuracies

        # Print results in standard format
        self._print_mmlu_results(mmlu_results)
        
        return mmlu_results
    
    def _print_mmlu_results(self, results: Dict[str, float]):
        """Print results in MMLU standard format"""
        print("\n" + "="*80)
        print(" MMLU Evaluation Results (by Subject Category)")
        print("="*80)
        
        # Main category results table
        print(f"{'Model':<20} {'Humanities':<12} {'Social Science':<15} {'STEM':<8} {'Other':<8} {'Average':<8}")
        print("-" * 80)
        
        humanities_pct = results["Humanities"] * 100
        social_pct = results["Social Science"] * 100
        stem_pct = results["STEM"] * 100
        other_pct = results["Other"] * 100
        avg_pct = results["Average"] * 100
        
        print(f"{'Your Model':<20} {humanities_pct:<12.1f} {social_pct:<15.1f} {stem_pct:<8.1f} {other_pct:<8.1f} {avg_pct:<8.1f}")
        
        # Detailed category accuracies
        print(f"\n{'Category':<20} {'Accuracy':<10} {'Percentage':<12}")
        print("-" * 50)
        
        for category in ["Humanities", "Social Science", "STEM", "Other"]:
            accuracy = results[category]
            percentage = accuracy * 100
            print(f"{category:<20} {accuracy:<10.4f} {percentage:<12.2f}%")
        
        print("-" * 50)
        print(f"{'Average':<20} {results['Average']:<10.4f} {results['Average']*100:<12.2f}%")
        
        # Detailed subject results
        if "subject_details" in results:
            print("\n" + "="*80)
            print(" Detailed Subject Results")
            print("="*80)
            
            subject_details = results["subject_details"]
            sorted_subjects = sorted(subject_details.items(), key=lambda x: x[1], reverse=True)
            
            print(f"{'Subject':<35} {'Category':<15} {'Accuracy':<10} {'Percentage':<12}")
            print("-" * 80)
            
            for subject, accuracy in sorted_subjects:
                category = self.subject_mapping.get(subject, 'Other')
                percentage = accuracy * 100
                print(f"{subject:<35} {category:<15} {accuracy:<10.4f} {percentage:<12.2f}%")
        
        print("="*80) 