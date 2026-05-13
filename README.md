<h2 align="center">
  <a href="https://arxiv.org/abs/2604.06699">Adaptive Prompt Structure Factorization: A Framework for Self-Discovering and Optimizing Compositional Prompt Programs</a>
</h2>

<h5 align="center">
  If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!
</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.06699-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2604.06699)
[![ACL 2026](https://img.shields.io/badge/ACL%202026-Main%20Conference-blue.svg)](https://arxiv.org/abs/2604.06699)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</h5>

## News

- **[2026.04.06]** 🎉 Our paper is accepted to ACL 2026 Main Conference!

## Why Do We Need aPSF?

Prompt optimization has become an important way to improve the performance of large language models, but existing methods often treat prompts as a single monolithic text block. This makes optimization difficult: when a prompt fails, it is hard to determine whether the issue comes from the reasoning strategy, output format, role description, task decomposition, or another hidden component.

Moreover, manually designing and tuning prompt programs requires substantial human effort. As tasks become more complex, prompt programs often need multiple interacting instructions, and searching over the whole prompt space directly can be inefficient, unstable, and difficult to interpret.

Our **Adaptive Prompt Structure Factorization (aPSF)** addresses these challenges by automatically discovering the latent structure of a prompt and optimizing it at the factor level. Instead of rewriting the entire prompt at each step, aPSF identifies which component matters most and performs targeted refinement.

## Adaptive Prompt Structure Factorization (aPSF)

## Structure Factorization

aPSF first decomposes a prompt into a set of interpretable structural factors, such as task framing, reasoning style, answer format, constraint specification, and perspective. This factorized representation makes the prompt program easier to analyze, debug, and optimize.

## Error-Guided Factor Selection

Given model predictions and task feedback, aPSF identifies the factor that is most likely responsible for current errors. This allows the optimizer to focus on the most impactful part of the prompt rather than repeatedly modifying unrelated instructions.

## Interventional Single-Factor Optimization

After selecting a factor, aPSF performs targeted intervention on that single factor while keeping the remaining prompt structure stable. This improves controllability and helps isolate which prompt changes are truly beneficial.

## Unified Candidate Evaluation

aPSF evaluates optimized prompt candidates through a unified scoring pipeline and accepts updates that improve task performance. This creates an adaptive optimization loop that progressively discovers and improves compositional prompt programs.

## Can We Trust aPSF? Yes!

- aPSF automatically discovers structured prompt factors from task descriptions and examples.
- aPSF performs targeted factor-level optimization instead of rewriting the whole prompt blindly.
- aPSF supports diverse reasoning and knowledge-intensive benchmarks, including math, logic, and knowledge tasks.
- aPSF is compatible with both hosted APIs and OpenAI-compatible local LLM endpoints.

## Supported Benchmarks

| Category | Datasets |
| --- | --- |
| Math | `gsm8k`, `multiarith`, `gsm_hard`, `aime2025`, `competition_math` |
| Logic | `aqua`, `bbh_all` (27 tasks), `bbh_<task_name>` |
| Knowledge | `mmlu`, `mmlu_<subject>` (57 subjects), `gpqa`, `gpqa_<domain>` |
| Code | `human_eval` |

## Directory Specification

### Core Code

- `main.py` provides a quick sanity check for aPSF.
- `run_experiments.py` is the main entry point for running experiments.
- `config.py` contains model, API, dataset, and experiment configurations.
- `data_loader/` contains dataset loaders.
- `evaluation/` contains task-specific evaluation logic.

### aPSF Components

- The **Architect** LLM discovers prompt structure and optimizes selected factors.
- The **Worker** LLM executes the current prompt program and generates task answers.
- The optimization loop performs factor discovery, error-guided factor selection, candidate generation, evaluation, and checkpointing.

## Environment Setup

### Requirements

- Python >= 3.9, recommended 3.10 or 3.11

Install dependencies:

```bash
pip install -r requirements.txt
```

The core dependencies include `openai`, `torch`, `transformers`, `accelerate`, `datasets`, `numpy`, `tqdm`, and `tabulate`. Please check `requirements.txt` for the full dependency list.

## API Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export SILICONFLOW_API_KEY="..."
export GROQ_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

You can also edit `config.py` directly.

aPSF uses two LLM roles configured in `config.py` under `MODELS`:

| Role | Purpose | Example Models |
| --- | --- | --- |
| `architect` | Structure discovery and factor optimization | Qwen3-8B, gpt-oss-120b |
| `worker` | Task execution and answer generation | Llama-3.1-8B, Qwen2.5-7B |

### Local LLM Support

aPSF is compatible with any OpenAI-compatible endpoint. For local deployment, you may use Ollama or vLLM.

```bash
# Ollama
ollama run qwen2.5:7b
# Then set api_base_id to "ollama" in config.py

# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000
# Then set api_base_id to "qwen_vllm" or "local_llm" in config.py
```

## Data Preparation

Dataset paths are defined in `config.py` under `DATA_PATHS`. Please organize your data as follows:

```text
data/
├── gsm_data/              # GSM8K
├── BIG-Bench-Hard-data/   # BBH, 27 tasks
├── AQuA-data/             # AQuA
├── MultiArith-data/       # MultiArith
├── MMLU-data/             # MMLU, 57 subjects
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

This checks whether the Architect can perform structure discovery and whether the Worker LLM can generate answers successfully.

### Running Experiments

```bash
python run_experiments.py --dataset <DATASET> --method <METHOD> [OPTIONS]
```

Required arguments:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset name, such as `gsm8k`, `bbh_all`, `mmlu`, or `gpqa` |
| `--method` | Method to run, such as `apsf` or an ablation variant |

Optional arguments:

| Argument | Description |
| --- | --- |
| `--feedback` | Enable reflection-based optimization using error feedback |
| `--resume` | Resume from the last checkpoint |
| `--step N` | Override the number of optimization steps |
| `--initial-prompt TEXT` | Start optimization from a given prompt; presets include `cot`, `analytical`, and `expert` |

### Examples

```bash
# aPSF on GSM8K
python run_experiments.py --dataset gsm8k --method apsf

# Start with a Chain-of-Thought initial prompt
python run_experiments.py --dataset gsm8k --method apsf --initial-prompt cot

# Enable reflection-based optimization
python run_experiments.py --dataset gsm8k --method apsf --feedback

# Full BBH benchmark, 27 tasks, with checkpoint resume
python run_experiments.py --dataset bbh_all --method apsf --resume

# Single BBH task
python run_experiments.py --dataset bbh_web_of_lies --method apsf

# Single MMLU subject
python run_experiments.py --dataset mmlu_abstract_algebra --method apsf

# GPQA
python run_experiments.py --dataset gpqa --method apsf
```

## FAQ

**Q: How do I use a different LLM as the architect or worker?**

A: Edit the `MODELS` section in `config.py`. Set `provider`, `api_base_id`, `model_name`, and `api_key` for each role. Any OpenAI-compatible endpoint can be used.

**Q: Does aPSF support local models?**

A: Yes. You can use local models through OpenAI-compatible endpoints, such as Ollama or vLLM.

**Q: Can I resume an interrupted experiment?**

A: Yes. Use the `--resume` flag when running `run_experiments.py`.

**Q: How do I enable feedback-based optimization?**

A: Add the `--feedback` flag. This enables reflection-based optimization using error feedback.

**Q: How do I add a new dataset?**

A: Create a loader in `data_loader/` that inherits from `BaseLoader`, create an evaluator in `evaluation/` if needed, and add the dataset configuration to `DATASET_CONFIG` in `config.py`.

## Acknowledgement

We are grateful for the following awesome projects and resources:

- [OpenAI](https://openai.com/)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [vLLM](https://github.com/vllm-project/vllm)
- [Ollama](https://ollama.com/)

## Citation

If you find this project helpful, please consider citing our work:

```bibtex
@misc{liu2026adaptivepromptstructurefactorization,
  title={Adaptive Prompt Structure Factorization: A Framework for Self-Discovering and Optimizing Compositional Prompt Programs},
  author={Haoyue Liu and Zhichao Wang and Yongxin Guo and Haoran Shou and Xiaoying Tang},
  year={2026},
  eprint={2604.06699},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.06699}
}
```
