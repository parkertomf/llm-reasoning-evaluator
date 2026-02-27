![Build Status](https://github.com/parkertomf/llm-reasoning-evaluator/actions/workflows/lint.yml/badge.svg)

# Automated Evaluation of LLM Reasoning (Work in Progress)

## Overview
This LLM benchmarking pipeline evaluates the reasoning performance of large language models.

The current scope is a baseline with one model and one dataset: the [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model and the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of "basic mathematical problems that require multi-step reasoning."

## Baseline Setup
- Dataset: GSM8K (test split, 1319 problems)
- Model: Qwen2.5-1.5B-Instruct
- Decoding: Greedy (for reproducibility of results)
- Max new tokens: 16
- Batch size: 32

## Running the Evaluation
### Requirements
- Python
    - Tested on Python 3.11.
- Nvidia GPU recommended for faster runtime.
- If you encounter out-of-memory errors, reduce the batch size.
    - 32 was tested on an 8GB VRAM GPU

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
Note that the first run will download ~3GB, by default located at `C:\Users\<your-username>\.cache\huggingface`.
```bash
python eval.py
```
**Hugging Face Authentication (Optional)**

If you encounter rate limit warnings when downloading on the first run, create and it is a problem for you, then:
1. Create/login to Hugging Face: https://huggingface.co
2. Go to: https://huggingface.co/settings/tokens
3. Create a **Read** token.
4. Follow the instructions [here](https://huggingface.co/docs/huggingface_hub/guides/cli
) to install `huggingface_hub` to your CLI
5. Run:
    ```bash
    hf auth login
    ```

## Baseline Results
| **Metric** | **Value** |
| :--- | ---: |
| **Problems Tested** | 1319 (full set) |
| **Correct** | 118 |
| **Incorrect** | 1167 |
| **Extraction Failures** | 34 |
| **Accuracy** | 8.9% |
| **Extraction Success Rate** | 97.4% |
| **Accuracy on Extraction Success** | 9.2% |

## Implementation Notes
### Batch Size
Results vary slightly (<0.5%) across batch sizes likely due to batch size affecting PyTorch kernel selection and therefore the possibility of different token selection in some cases.

Execution time by batch size is a U-curve with a Goldilocks zone of efficiency.

On my machine, experimentation suggests that the most accurate and most time-efficient batch size is in the range of 16-64, so I chose to stick with 32. Although accuracy was lower at both ends, extraction success rate remained relatively stable, suggesting that extraction success is not affected by batching, even though the model's ability to do the actual math is. An important caveat is that since the signal size of the accuracy variation is small (8.5%-8.9% range) range, it is not certain.

### Response Extraction
The prompt requests that the model respond only with the numerical answer to the question, so extraction just verifies that.

## Next Steps
- Add prompting variants
- Structured prediction logging
- Comparative evaluation script
- Add another model and dataset
