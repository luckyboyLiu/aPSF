# --- Evaluation Configuration ---
# Whether to use LLM for semantic answer comparison (more accurate but slower)
USE_LLM_ANSWER_COMPARISON = True

# LLM model for answer extraction (options: "worker" or "architect")
# "worker": Use worker model for extraction (faster)
# "architect": Use architect model for extraction (potentially better)
ANSWER_EXTRACTOR_LLM = "architect"  # Default: use architect

# --- LLM API Configurations ---
# It's recommended to use environment variables for API keys for security.
# For example: os.getenv("OPENAI_API_KEY")
import os

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "google": os.getenv("GOOGLE_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "siliconflow": os.getenv("SILICONFLOW_API_KEY", ""),
    "groq": os.getenv("GROQ_API_KEY", ""),
    "dashscope": os.getenv("DASHSCOPE_API_KEY", ""),
    "ollama": "any-key-works",
    "agentify": os.getenv("AGENTIFY_API_KEY", ""),
}

# --- Custom API Endpoints ---
API_BASE_URLS = {
    "qwen_vllm": "http://localhost:8000/v1", # vLLM OpenAI compatible endpoint
    "siliconflow": "https://api.siliconflow.cn/v1",  # SiliconFlow API
    "openrouter": "https://openrouter.ai/api/v1",  # Add OpenRouter API endpoint
    "local_llm": "http://localhost:9600/v1", # Local vLLM endpoint
    "qwen3_vllm": "http://localhost:8000/v1", # vLLM OpenAI compatible endpoint
    "oss_vllm": "http://localhost:8002/v1",  # Local vLLM (gpt-oss-120b) endpoint
    "groq": "https://api.groq.com/openai/v1", # Groq API endpoint
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1", # Alibaba DashScope API endpoint
    # Add ollama OpenAI compatible endpoint
    "ollama": "http://127.0.0.1:11434/v1",  # ollama OpenAI compatible API endpoint
    "agentify": "https://api.agentify.top/v1",  # Agentify API endpoint
}

# --- Model Definitions for Experiments ---
# Configure the architect (structure discovery & optimization) and worker (task execution) LLMs.
# Supports any OpenAI-compatible endpoint (vLLM, Ollama, cloud APIs, etc.)
MODELS = {
    "architect": {
        "provider": "openai",
        "api_base_id": "local_llm",       # Change to your endpoint key in API_BASE_URLS
        "model_name": "your-architect-model",  # e.g., "Qwen/Qwen3-8B" or local path
        "api_key": "openai",              # vLLM accepts any string; for cloud APIs use the key name above
        "temperature": 0.7,
        "max_tokens": 8192,
        "top_p": 1.0,
    },
    "worker": {
        "provider": "openai",
        "api_base_id": "local_llm",       # Change to your endpoint key in API_BASE_URLS
        "model_name": "your-worker-model",  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
        "api_key": "openai",
        "temperature": 0.0,
        "max_tokens": 8192,
        "top_p": 1.0,
    },
    "generalization_test_model": {
        "provider": "llama_local",
        "model_name": "meta-llama/Llama-3-8B-Instruct",
        "temperature": 0.0,
    },
}

# --- Dataset Configurations ---
# Dataset paths (assuming a 'data' folder in project root)
DATA_PATHS = {
    "gsm8k": "data/gsm_data",
    "bbh": "data/BIG-Bench-Hard-data",
    "bbh_hard": "data/BIG-Bench-Hard-data",
    "bbh_colored_objects": "data/BIG-Bench-Hard-data",
    "bbh_web_of_lies": "data/BIG-Bench-Hard-data",
    "bbh_movie_recommendation": "data/BIG-Bench-Hard-data",
    "aqua": "data/AQuA-data",
    "multiarith": "data/MultiArith-data",
    "humaneval": "data/human_eval",
    "xcopa": "data/xcopa",
    "gsm_hard": "data/GSM-hard",
    "billsum": "data/billsum",
    "spider": "data/spider",
    "mmlu": "data/MMLU-data",
    "bbh_all": "data/BIG-Bench-Hard-data",
    "aime2025": "data/AIME2025",
    "competition_math": "data/competition_math",
    "gpqa": "data/gpqa",
    "gpqa_chemistry": "data/gpqa",
    "gpqa_physics": "data/gpqa",
    "gpqa_biology": "data/gpqa",
    "squad_v2": "data/SQuAD",
}
# --- Data Splitting Configuration ---
DATA_SPLIT_CONFIG = {
    "random_seed": 42,  # Unified random seed for reproducibility (can change to 43, 44 for testing)
    "split_strategy": "no_overlap",  # Ensure no overlap between train and test sets
    "shuffle_before_split": True,  # Shuffle data before splitting
}
# Dataset specific configurations
DATASET_CONFIG = {
    "gsm8k": {
        "loader": "GSM8KLoader",
        "evaluator": "GSM8KEvaluator",  # Changed to dedicated GSM8K evaluator
        "metric": "accuracy",
        "val_split": "train",
        "test_split": "test",
        "val_size": 50,  # aPSF needs 50 samples for train-50 evaluation
        "test_size": 1495,
    },
    "bbh": {
        "loader": "BBHLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "val_split": "train",
        "test_split": "test",
        "val_size": 200,
    },
    "bbh_hard": {
        "loader": "BBHLoader",
        "evaluator": "AccuracyEvaluator", 
        "metric": "average_accuracy",
        "input_key": "prompt",
        "val_split": "train",
        "test_split": "test",
        "val_size": 200,
    },
    "bbh_colored_objects": {
        "loader": "BBHSingleTaskLoader",
        "evaluator": "AccuracyEvaluator", 
        "metric": "accuracy",
        "input_key": "input",  # BBH uses input field
        "val_split": "validation",  # Modified: use validation to distinguish
        "test_split": "test",       # Modified: use test to distinguish
        "val_size": 50,  # aPSF only uses 50 samples for template discovery
        "test_size": 150,  # Modified: ensure val_size + test_size = 200, no overlap
        "task_file": "reasoning_about_colored_objects.json",  # Specify exact file
    },
    "bbh_web_of_lies": {
        "loader": "BBHSingleTaskLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "input_key": "input",  
        "val_split": "validation",  
        "test_split": "test",       
        "val_size": 50,  
        "test_size": 200,  
        "task_file": "web_of_lies.json",  
    },
    "bbh_movie_recommendation": {
        "loader": "BBHSingleTaskLoader",
        "evaluator": "AccuracyEvaluator", 
        "metric": "accuracy",
        "input_key": "input",  # BBH uses input field
        "val_split": "validation",  # Modified: use validation to distinguish
        "test_split": "test",       # Modified: use test to distinguish
        "val_size": 50,  # aPSF only uses 50 samples for template discovery
        "test_size": 200,  # Modified: ensure no overlap, total 250 samples
        "task_file": "movie_recommendation.json",  # Specify exact file
    },
    "aqua": {  # AQuA dataset config
        "loader": "AQuALoader",
        "evaluator": "AQuAEvaluator",  # Use dedicated AQuA evaluator
        "metric": "accuracy",
        "input_key": "input",  # Input is the question
        "target_key": "correct",  # Target is the correct answer
        "val_split": "train",
        "test_split": "test", 
        "val_size": 50,  # Take 50 for validation in aPSF
        "test_size": 200,
    },
    "multiarith": {  # MultiArith dataset config
        "loader": "MultiArithLoader",
        "evaluator": "MultiArithEvaluator",
        "metric": "accuracy",
        "input_key": "question",  # Input field
        "target_key": "answer",   # Target field
        "val_split": "train",
        "test_split": "test", 
        "val_size": 50,  # Take 50 for validation in aPSF
        "test_size": 480,  # Take 400 samples from test set
    },
    
    "gsm_hard": {
        "loader": "GSMHardLoader",
        "evaluator": "GSMHardEvaluator",
        "metric": "accuracy",
        "val_split": "train",
        "input_key": "input",
        "target_key": "target",
        "test_split": "test",
        "val_size": 50,
        "test_size": 1056,
    },
    "humaneval": {
        "loader": "GSM8KLoader",  # HumanEval loader not available; placeholder
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "input_key": "prompt",           # Add input field
        "target_key": "canonical_solution",  # Add target field
        "val_split": "validation",       # Modified: avoid overlap with test
        "test_split": "test",
        "val_size": 100,                # Take 100 from test set for validation
        "test_size": 64,                # Added: HumanEval usually has 164 samples, avoid overlap
    },
    "xcopa": {
        "loader": "XCOPALoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 200,
    },
    "billsum": {
        "loader": "BillSumLoader",
        "evaluator": "RougeEvaluator",
        "metric": "rouge-L",
        "val_split": "train",
        "test_split": "test",
        "val_size": 200,
    },
    "spider": {
        "loader": "SpiderLoader",
        "evaluator": "ExecutionEvaluator",
        "metric": "execution_accuracy",
        "val_split": "train",
        "test_split": "dev", # Spider validation set is usually called 'dev'
        "val_size": 200,
    },
    "mmlu": {
        "loader": "MMLULoader",
        "evaluator": "MMLUEvaluator",
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "dev",
        "test_split": "test",
        "val_size": 50,          # 50 samples per subject for validation
        "test_size": None,       # Dynamically calculated: total per subject minus 50
        "split_by_subject": True, # Split data by subject, 50 validation per subject
    },
    "mmlu_all": {
        "loader": "MMLUAllSubjectsLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 50,          # 50 samples per subject for validation
        "test_size": None,       # Remaining samples per subject for test
    },
    "bbh_all": {
        "loader": "BBHMultiTaskLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "val_split": "train",
        "test_split": "test",
        "val_size": 50,  # Validation set size
        "test_size": 150,  # Test set size
    },
    "squad_v2": {
        "loader": "SquadV2Loader",
        "evaluator": "SquadV2Evaluator",
        "metric": "f1",  # Main metric is F1
        "input_key": "question",
        "target_key": "answer",
        "val_split": "test",  # Take validation from test set
        "test_split": "test",  # Remaining as test set
        "val_size": 50,  # Validation set size, using seed 42
        "test_size": 500,  # All remaining samples for test set
    }
}

# All MMLU subjects list (defined before DATASET_CONFIG.update() for use in update)
MMLU_ALL_SUBJECTS = [
    # Mathematics (9)
    "abstract_algebra", "college_mathematics", "elementary_mathematics",
    "high_school_mathematics", "high_school_statistics",
    "formal_logic", "logical_fallacies", "machine_learning", "econometrics",
    # Science (14)
    "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_physics", "conceptual_physics", "high_school_biology",
    "high_school_chemistry", "high_school_physics", "electrical_engineering",
    "clinical_knowledge", "medical_genetics", "virology", "nutrition",
    # Computer Science (3)
    "college_computer_science", "high_school_computer_science", "computer_security",
    # Social Science (11)
    "high_school_psychology", "professional_psychology", "sociology",
    "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_us_history", "high_school_world_history", "high_school_european_history",
    "high_school_geography", "prehistory", "human_aging",
    # Humanities (8)
    "philosophy", "world_religions", "moral_disputes", "moral_scenarios",
    "high_school_government_and_politics", "jurisprudence", "international_law",
    "human_sexuality",
    # Business & Law (7)
    "business_ethics", "professional_law", "professional_accounting",
    "management", "marketing", "public_relations", "us_foreign_policy",
    # Medicine (2)
    "college_medicine", "professional_medicine",
    # Other (3)
    "miscellaneous", "global_facts", "security_studies"
]

# MMLU subject categories - grouped by type
MMLU_CATEGORIES = {
    "Mathematics": [
        "abstract_algebra", "college_mathematics", "elementary_mathematics",
        "high_school_mathematics", "high_school_statistics",
        "formal_logic", "logical_fallacies", "machine_learning", "econometrics"
    ],
    "Science": [
        "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_physics", "conceptual_physics", "high_school_biology",
        "high_school_chemistry", "high_school_physics", "electrical_engineering",
        "clinical_knowledge", "medical_genetics", "virology", "nutrition"
    ],
    "Computer_Science": [
        "college_computer_science", "high_school_computer_science", "computer_security"
    ],
    "Social_Science": [
        "high_school_psychology", "professional_psychology", "sociology",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_us_history", "high_school_world_history", "high_school_european_history",
        "high_school_geography", "prehistory", "human_aging"
    ],
    "Humanities": [
        "philosophy", "world_religions", "moral_disputes", "moral_scenarios",
        "high_school_government_and_politics", "jurisprudence", "international_law",
        "human_sexuality"
    ],
    "Business_Law": [
        "business_ethics", "professional_law", "professional_accounting",
        "management", "marketing", "public_relations", "us_foreign_policy"
    ],
    "Medicine": [
        "college_medicine", "professional_medicine"
    ],
    "Other": [
        "miscellaneous", "global_facts", "security_studies"
    ]
}

# All BBH task list
BBH_ALL_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", 
    "disambiguation_qa", "dyck_languages", "formal_fallacies", 
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting"
]

# Update bbh_all config in existing DATASET_CONFIG
DATASET_CONFIG.update({
    "bbh_all": {
        "loader": "BBHAllTasksLoader",  # Use correct loader
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 50,  # Take 50 per task for validation
        "test_size": 200,  # Take 200 per task for test
        "tasks": BBH_ALL_TASKS,  # All task list
    },
    "mmlu_all": {
        "loader": "MMLUAllSubjectsLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 50,  # 50 samples per subject for validation
        "test_size": None,  # Remaining samples per subject for test
        "subjects": MMLU_ALL_SUBJECTS,  # All subject list (updated after variable definition)
    },
    "aime2025": {
        "loader": "AIME2025Loader",
        "evaluator": "AIME2025Evaluator",  # Use dedicated AIME evaluator
        "metric": "accuracy",
        "input_key": "question",  # Input field
        "target_key": "answer",   # Target field
        "val_split": "train",     # AIME uses train split as validation for prompt optimization
        "test_split": "test",     # Test set for final evaluation
        "val_size": 6,           # First 60% as validation (30*0.6=18)
        "test_size": 24,          # Last 40% as test (30*0.4=12)
    },
    "competition_math": {
        "loader": "CompetitionMathLoader",
        "evaluator": "CompetitionMathEvaluator",  # Use dedicated CompetitionMath evaluator
        "metric": "accuracy",
        "input_key": "problem",   # Input field
        "target_key": "answer",   # Target field
        "val_split": "train",     # Use train split as validation
        "test_split": "test",     # Test set for final evaluation
        "val_size": 50,           # Validation set size
        "test_size": 500,         # Test set size
    },
    
    # GPQA (Google-Proof Q&A) - Expert-designed challenging science QA
    "gpqa": {
        "loader": "GPQALoader",
        "evaluator": "AccuracyEvaluator",  # Use accuracy evaluator (multiple choice)
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 30,           # Diamond subset is small, 30 questions for validation
        "test_size": 168,         # Remaining 168 as test set (198-30)
    },
    
    # GPQA - Chemistry questions only (93 total)
    "gpqa_chemistry": {
        "loader": "GPQALoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 25,           # 25 chemistry questions for validation
        "test_size": 68,          # Remaining 68 as test set (93-25)
        "domain_filter": "Chemistry",  # Only load chemistry questions
    },
    
    # GPQA - Physics questions only (86 total)
    "gpqa_physics": {
        "loader": "GPQALoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 25,           # 25 physics questions for validation
        "test_size": 61,          # Remaining 61 as test set (86-25)
        "domain_filter": "Physics",    # Only load physics questions
    },
    
    # GPQA - Biology questions only (19 total, fewer samples)
    "gpqa_biology": {
        "loader": "GPQALoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 5,            # 5 biology questions for validation (few samples)
        "test_size": 14,          # Remaining 14 as test set (19-5)
        "domain_filter": "Biology",    # Only load biology questions
    }
})

# Add config for each individual BBH task
for task_name in BBH_ALL_TASKS:
    DATASET_CONFIG[f"bbh_{task_name}"] = {
        "loader": "BBHSingleTaskLoader",
        "evaluator": "AccuracyEvaluator", 
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test", 
        "val_size": 50,
        "test_size": 200,
        "task_file": f"{task_name}.json",
    }

# Add configuration for each MMLU subject
for subject in MMLU_ALL_SUBJECTS:
    DATASET_CONFIG[f"mmlu_{subject}"] = {
        "loader": "MMLULoader",
        "evaluator": "MMLUEvaluator",
        "metric": "accuracy",
        "input_key": "input",
        "val_split": "validation",
        "test_split": "test",
        "val_size": 50,
        "test_size": 100,
        "subject": subject,
    }

# --- aPSF Optimization Hyperparameters ---
OPTIMIZATION_PARAMS = {
    "total_optimization_steps": 10,
    "candidates_per_step": 4,
    "optimizer_algorithm": "apsf",

    "semantic_filtering_enabled": False,
    "top_k_after_filtering": 2,
    "acceptance_threshold": 1,

    "verbose_output": True,
    "show_all_qa_pairs": True,

    "min_factors": 2,
    "max_factors": 6,
    "auto_factor_mode": True,
    "factor_selection_strategy": "error_driven",

    "enable_reflection": False,
    "reflection_error_threshold": 0.0,
    "reflection_max_errors": 10,

    # Initial prompt configuration
    # If set to None, aPSF will generate prompt from scratch
    # If set to a string (e.g., "Let's think step by step"), aPSF will optimize from that prompt
    "initial_prompt": None,  # Default: None, generate from scratch

    # Common initial prompt presets (can be selected at runtime)
    "initial_prompt_presets": {
        "cot": "Let's think step by step.",
        "analytical": "Let's analyze this problem carefully and solve it systematically.",
        "expert": "As an expert, let me approach this problem methodically.",
        "empty": "",  # Empty prompt, equivalent to None
    }
}
# --- Experiment and Logging Configurations ---
RESULTS_DIR = "results"
LOG_FILE = "experiment_logs.log"

