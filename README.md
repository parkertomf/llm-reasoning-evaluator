![Build Status](https://github.com/parkertomf/llm-reasoning-evaluator/actions/workflows/lint.yml/badge.svg)

# llm-reasoning-evaluator v0 (WIP)
Automated Evaluation of LLM Reasoning Using Structured Rubrics

# Results


| **Model** | Qwen/Qwen2.5-1.5B-Instruct |
| :--- | ---: |
| **Dataset** | GSM8K |
| **Problems Tested** | 1319 (full set) |
| **Correct** |  |
| **Incorrect** |  |
| **Extraction Failures** |  |
| **Accuracy** |  |
| **Extraction Success Rate** |  |
| **Accuracy on Extraction Success** |  |

Results vary slightly (<0.3%) across batch sizes likely due to batch size affecting PyTorch kernel selection and therefore the possibility of different token selection in some cases.

For example:
 64:
Model: Qwen/Qwen2.5-1.5B-Instruct
Problems Tested: 1319
Correct: 116
Incorrect: 1170
Extraction Failures: 33
Accuracy: 8.8%
Extraction Success Rate: 97.5%
Accuracy on Extraction Success: 9.0%

128
Problems Tested: 1319
Correct: 113
Incorrect: 1174
Extraction Failures: 32
Accuracy: 8.6%
Extraction Success Rate: 97.6%
Accuracy on Extraction Success: 8.8%


256
Model: Qwen/Qwen2.5-1.5B-Instruct
Problems Tested: 1319
Correct: 112
Incorrect: 1176
Extraction Failures: 31
Accuracy: 8.5%
Extraction Success Rate: 97.6%
Accuracy on Extraction Success: 8.7% 







Model name
Eval rule v0
How to run
Baseline result

# How to Run
### Requirements:
- Nvidia GPU because CUDA is used. Note that this code was tested on nividia geforce rtx 4070 laptop gpu, so you may also have issues on a less powerful nvidia gpu, such as running out of memory with with higher eval counts, if you have one with less 8gb vram in particular.
- Though maybe the code will fallback to CPU if a computer doesn't have an nvidia gpu? I don't know


Model used
Prompt format
Extraction rule v0
Evaluation rule v0
How to run
Baseline result

What you should include in the README:
Run 200–500 with the same settings (deterministic) and record the result in README.
Model name (exact string)
Note that first run will download ~3GB
Any auth requirements (if gated)
GPU requirement (optional but helpful)
Record all three metrics + model + exact decoding params in README.

Model name

Decoding params

Extraction rule v0

Metrics (N, accuracy, format compliance)

How to run