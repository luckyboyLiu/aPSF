import os

# 定义项目根目录
project_root = "/workspace/lhy/PSF/psf/"

# 定义目录结构
dirs = [
    "llm_apis",
    "data_loader",
    "evaluation",
    "optimization",
    "baselines",
    "results"  # 存放实验结果
]

# 定义文件结构
files = {
    ".": [
        "config.py",
        "main.py",
        "run_experiments.py",
        "README.md"
    ],
    "llm_apis": [
        "__init__.py",
        "base_api.py",
        "gpt_api.py",
        "google_api.py",  # 用于 text-bison@001
        "llama_api.py"
    ],
    "data_loader": [
        "__init__.py",
        "base_loader.py",
        "gsm8k_loader.py",
        "bbh_loader.py",
        "humaneval_loader.py",
        "xcopa_loader.py",
        "billsum_loader.py",
        "spider_loader.py"
    ],
    "evaluation": [
        "__init__.py",
        "base_evaluator.py",
        "accuracy_evaluator.py",
        "pass_at_k_evaluator.py",
        "rouge_evaluator.py",
        "execution_evaluator.py"
    ],
    "optimization": [
        "__init__.py",
        "prompt_object.py",
        "architect.py",
        "optimizer.py"
    ],
    "baselines": [
        "__init__.py",
        "opro_baseline.py",
        "protegi_baseline.py",
        "dspy_baseline.py"
    ]
}

def create_structure():
    """创建项目目录和文件"""
    # 创建目录
    for d in dirs:
        dir_path = os.path.join(project_root, d)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # 创建文件
    for d, file_list in files.items():
        for f in file_list:
            file_path = os.path.join(project_root, d, f)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as fp:
                    pass  # 创建空文件
                print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_structure()
    print("\nProject structure for aPSF created successfully.")
