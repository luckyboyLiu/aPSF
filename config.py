# --- LLM API Configurations ---
# It's recommended to use environment variables for API keys for security.
# For example: os.getenv("OPENAI_API_KEY")
API_KEYS = {
    "openai": "YOUR_OPENAI_API_KEY",  # 对于本地vLLM可以留空或任意字符串
    "google": "YOUR_GOOGLE_API_KEY",  # 替换为您的 Google API 密钥
    "openrouter": "YOUR_OPENROUTER_API_KEY", # 可选，用于访问多种模型
}

# --- 自定义 API 端点 ---
API_BASE_URLS = {
    "qwen_vllm": "http://localhost:8000/v1" # vLLM OpenAI 兼容端点
}

# --- Model Definitions for Experiments ---
# 论文中使用的模型
MODELS = {
    # Architect/Optimizer: 使用您通过 vLLM 部署的 Qwen-2.5
    "architect": {
        "provider": "openai", # 使用我们修改过的 OpenAI 兼容 provider
        "api_base_id": "qwen_vllm", # 引用上面的 vLLM 端点
        "model_name": "/workspace/lhy/Qwen2.5-VL-7B-Instruct", # 在vLLM中加载的模型名称/路径
        "temperature": 0.7,
        "top_p": 1.0,
    },
    # Worker/Scorer: 使用本地的 Llama-2 模型
    "worker": {
        "provider": "llama_local", # 使用我们实现的本地加载器
        "model_name": "/workspace/lhy/Llama-2-7b-chat-hf", # 本地模型路径
        "temperature": 0.0,
        # 为本地模型添加生成参数 (可选, 但建议)
        "max_new_tokens": 512, # 限制生成长度
    },
    # 用于跨模型泛化测试的模型
    "generalization_test_model": {
        "provider": "llama_local", # 本地 Llama 模型的示例
        "model_name": "meta-llama/Llama-3-8B-Instruct",
        "temperature": 0.0,
    },
}

# --- Dataset Configurations ---
# 数据集路径 (假设在项目根目录下有一个 'data' 文件夹)
DATA_PATHS = {
    "gsm8k": "data/gsm8k",
    "bbh": "data/bbh",
    "humaneval": "data/humaneval",
    "xcopa": "data/xcopa",
    "billsum": "data/billsum",
    "spider": "data/spider",
}

# 数据集具体配置
DATASET_CONFIG = {
    "gsm8k": {
        "loader": "GSM8KLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "accuracy",
        "val_split": "train",
        "test_split": "test",
        "val_size": 200,
    },
    "bbh": {
        "loader": "BBHLoader",
        "evaluator": "AccuracyEvaluator",
        "metric": "average_accuracy",
        "val_split": "train",
        "test_split": "test",
        "val_size": 200,
    },
    "humaneval": {
        "loader": "HumanEvalLoader",
        "evaluator": "PassAtKEvaluator",
        "metric": "pass@1",
        "val_split": "test", # HumanEval 通常使用 test 集的一部分做验证
        "test_split": "test",
        "val_size": 100, # 从 test 集中取 100 个做验证
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
        "test_split": "dev", # Spider 的验证集通常称为 'dev'
        "val_size": 200,
    },
}


# --- aPSF Optimization Hyperparameters ---
OPTIMIZATION_PARAMS = {
    "total_optimization_steps": 6,
    "candidates_per_step": 4, # 论文中的 N
    "optimizer_algorithm": "dap_ucb",
    "ucb1_exploration_constant": 2.0, # 论文中的 c=sqrt(2)
    
    # DAP-UCB 专用参数
    "dap_patience_M": 4, # 停滞容忍轮数
    "dap_improvement_delta": 0.005, # 最小改进阈值 (0.5%)
}

# --- Experiment and Logging Configurations ---
RESULTS_DIR = "results"
LOG_FILE = "experiment_logs.log"