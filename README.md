# aPSF: Auto Prompt Structure Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Automatic prompt optimization through structured factor-level refinement**

## Overview

aPSF (Auto Prompt Structure Fusion) is an automatic prompt optimization framework that discovers and refines prompt structures through error-driven factor selection and multi-armed bandit (DAP-UCB) algorithms. Given a task description and a small set of examples, aPSF automatically:

1. **Discovers** the latent factor structure of a prompt (e.g., tone, format, perspective) via an Architect LLM.
2. **Selects** the most impactful factor to refine at each step using a DAP-UCB bandit policy.
3. **Optimizes** the selected factor with error-driven feedback, producing improved prompt candidates.
4. **Evaluates** candidates with a unified scoring pipeline and accepts improvements.

### Supported Benchmarks

| Category | Datasets |
|----------|----------|
| Math | `gsm8k`, `multiarith`, `gsm_hard`, `aime2025`, `competition_math` |
| Logic | `aqua`, `bbh_all` (27 tasks), `bbh_<task_name>` |
| Knowledge | `mmlu`, `mmlu_<subject>` (57 subjects), `gpqa`, `gpqa_<domain>` |
| Code | `humaneval` |

## Installation

### Requirements

- Python >= 3.9 (recommended 3.10 / 3.11)

```bash
pip install -r requirements.txt
```

The core dependencies include `openai`, `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `tqdm`, and `tabulate`. See `requirements.txt` for the full list.

## API Configuration

Set your API keys as environment variables (recommended) or edit `config.py` directly:

```bash
export OPENAI_API_KEY="sk-..."
export SILICONFLOW_API_KEY="..."
export GROQ_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

aPSF uses two LLM roles configured in `config.py` under `MODELS`:

| Role | Purpose | Example |
|------|---------|---------|
| `architect` | Structure discovery & factor optimization | GPT-4o, Qwen3-8B, gpt-oss-120b |
| `worker` | Task execution (answer generation) | Llama-3.1-8B, Qwen2.5-7B |

### Local LLM Support

aPSF is compatible with any OpenAI-compatible endpoint. For local deployment:

```bash
# Ollama
ollama run qwen2.5:7b
# Set api_base_id to "ollama" in config.py

# vLLM
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
# Set api_base_id to "qwen_vllm" or "local_llm" in config.py
```

## Data Preparation

Dataset paths are defined in `config.py` under `DATA_PATHS`. Organize your data as follows:

```
data/
├── gsm_data/              # GSM8K
├── BIG-Bench-Hard-data/   # BBH (27 tasks)
├── AQuA-data/             # AQuA
├── MultiArith-data/       # MultiArith
├── MMLU-data/             # MMLU (57 subjects)
├── GSM-hard/              # GSM-Hard
├── AIME2025/              # AIME 2025
├── competition_math/      # Competition Math
├── gpqa/                  # GPQA
└── human_eval/            # HumanEval
```

Update the paths in `DATA_PATHS` to point to your local data directory.

## Usage

### Quick Test

Verify your setup with the built-in sanity check:

```bash
python main.py
```

This tests the Architect's structure discovery and the Worker LLM's generation capability.

### Running Experiments

```bash
python run_experiments.py --dataset <DATASET> --method <METHOD> [OPTIONS]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name (e.g., `gsm8k`, `bbh_all`, `mmlu`, `gpqa`) |
| `--method` | Optimization method (see below) |

**Optional arguments:**

| Argument | Description |
|----------|-------------|
| `--feedback` | Enable reflection-based optimization using error feedback |
| `--resume` | Resume from the last checkpoint |
| `--step N` | Override the number of optimization steps |
| `--initial-prompt TEXT` | Start optimization from a given prompt (presets: `cot`, `analytical`, `expert`) |

### Examples

```bash
# aPSF on GSM8K
python run_experiments.py --dataset gsm8k --method apsf

# With Chain-of-Thought initial prompt
python run_experiments.py --dataset gsm8k --method apsf --initial-prompt cot

# Enable reflection optimization
python run_experiments.py --dataset gsm8k --method apsf --feedback

# Full BBH benchmark (27 tasks) with checkpoint resume
python run_experiments.py --dataset bbh_all --method apsf --resume

# Single BBH task
python run_experiments.py --dataset bbh_web_of_lies --method apsf

# MMLU single subject
python run_experiments.py --dataset mmlu_abstract_algebra --method apsf

# GPQA
python run_experiments.py --dataset gpqa --method apsf
```

### Available Methods

| Method | Description |
|--------|-------------|
| `apsf` | **aPSF** (full framework) |
| `opro` | OPRO baseline |
| `protegi` | ProTeGi baseline |
| `dspy` | DSPy baseline |
| `ape` | APE baseline |
| `grips` | GrIPS baseline |
| `empty_cot` | Zero-shot Chain-of-Thought |

**Ablation variants:**

| Method | Description |
|--------|-------------|
| `apsf_nostructure` | Without structure optimization |
| `apsf_notone` | Without tone factor |
| `apsf_noformat` | Without format factor |
| `apsf_noperspective` | Without perspective factor |
| `apsf_nodap` | Without DAP-UCB |
| `apsf_randselect` | Random factor selection |
| `apsf_thompson` | Thompson Sampling |
| `apsf_greedy` | Greedy selection |
| `apsf_roundrobin` | Round Robin selection |

## Results & Logs

| Output | Location |
|--------|----------|
| Logs | `experiment_logs.log` |
| Results | `results/` (JSON with timestamps) |
| Checkpoints | `checkpoints/` (supports `--resume`) |

## Project Structure

```
aPSF/
├── run_experiments.py          # Main experiment entry point & CLI
├── main.py                     # Quick test script
├── config.py                   # Model, API, dataset, and hyperparameter configuration
├── checkpoint_manager.py       # Checkpoint save/load management
├── checkpoint_utils.py         # Checkpoint utility functions
├── requirements.txt            # Python dependencies
│
├── optimization/               # Core aPSF algorithm
│   ├── architect.py            # Prompt structure discovery (Architect LLM)
│   ├── optimizer.py            # Factor-level optimization with DAP-UCB
│   └── prompt_object.py        # PromptStructure representation
│
├── evaluation/                 # Scoring & evaluation
│   ├── unified_scoring.py      # Unified scoring pipeline
│   ├── base_evaluator.py       # Base evaluator class
│   ├── accuracy_evaluator.py   # General accuracy evaluator
│   ├── gsm8k_evaluator.py      # GSM8K evaluator
│   ├── gsm_hard_evaluator.py   # GSM-Hard evaluator
│   ├── multiarith_evaluator.py # MultiArith evaluator
│   ├── aime2025_evaluator.py   # AIME 2025 evaluator
│   ├── competition_math_evaluator.py  # Competition Math evaluator
│   ├── aqua_evaluator.py       # AQuA evaluator
│   ├── mmlu_evaluator.py       # MMLU evaluator
│   └── execution_evaluator.py  # Code execution evaluator
│
├── data_loader/                # Dataset loaders
│   ├── base_loader.py          # Base loader class
│   ├── gsm8k_loader.py         # GSM8K
│   ├── gsm_hard_loader.py      # GSM-Hard
│   ├── multiarith_loader.py    # MultiArith
│   ├── aime2025_loader.py      # AIME 2025
│   ├── competition_math_loader.py  # Competition Math
│   ├── aqua_loader.py          # AQuA
│   ├── bbh_loader.py           # BBH (single task)
│   ├── bbh_single_task_loader.py   # BBH single task loader
│   ├── bbh_multi_task_loader.py    # BBH multi-task loader
│   ├── bbh_all_tasks_loader.py     # BBH all 27 tasks
│   ├── mmlu_loader.py          # MMLU (single subject)
│   ├── mmlu_all_subjects_loader.py # MMLU all 57 subjects
│   └── gpqa_loader.py          # GPQA
│
├── llm_apis/                   # LLM API wrappers
│   ├── base_api.py             # Base API class
│   ├── gpt_api.py              # OpenAI-compatible API (GPT, vLLM, Ollama, etc.)
│   ├── google_api.py           # Google Gemini API
│   └── llama_api.py            # Local Llama model API
│
├── baselines/                  # Baseline & ablation experiments
│   ├── apsf_ablation.py        # aPSF ablation studies
│   └── apsf_comparative.py     # Comparative experiments
│
├── checkpoints/                # Checkpoint files (generated at runtime)
└── results/                    # Experiment result JSONs (generated at runtime)
```

## Configuration Reference

Key hyperparameters in `config.py` under `OPTIMIZATION_PARAMS`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_optimization_steps` | 10 | Number of optimization iterations |
| `candidates_per_step` | 4 | Candidate prompts generated per step |
| `factor_selection_strategy` | `error_driven` | Factor selection strategy |
| `min_factors` / `max_factors` | 2 / 6 | Factor count range |
| `enable_reflection` | `False` | Enable reflection-based refinement |
| `initial_prompt` | `None` | Starting prompt (`None` = generate from scratch) |

## FAQ

**Q: How do I use a different LLM as the architect or worker?**
A: Edit the `MODELS` section in `config.py`. Set `provider`, `api_base_id`, `model_name`, and `api_key` for each role. Any OpenAI-compatible endpoint works.

**Q: GPU acceleration?**
A: Install `torch` with CUDA support and `accelerate`. Local models via `llama_api.py` will automatically use GPU when available.

**Q: Missing dependencies?**
A: Run `pip install -r requirements.txt`. For tokenization issues, ensure `sentencepiece` is installed.

**Q: How to add a new dataset?**
A: 1) Create a loader in `data_loader/` inheriting from `BaseLoader`. 2) Create an evaluator in `evaluation/` if needed. 3) Add the dataset config to `DATASET_CONFIG` in `config.py`.

## License

MIT License

## Citation

```bibtex
@article{apsf2025,
  title={aPSF: Auto Prompt Structure Fusion},
  year={2025}
}
```
