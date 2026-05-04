![Build Status](https://github.com/parkertomf/llm-reasoning-evaluator/actions/workflows/tests.yml/badge.svg)
![Build Status](https://github.com/parkertomf/llm-reasoning-evaluator/actions/workflows/lint.yml/badge.svg)

# Automated Evaluation of LLM Reasoning (Work in Progress)

## Overview
This LLM benchmarking pipeline evaluates the reasoning performance of large language models.

The current scope is one model and one dataset:
- The [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model
- The [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset of "basic mathematical problems that require multi-step reasoning."

Prompt strategy options include:
- **Baseline**: Prompts the model to give the answer in a specific extractible way, but provides no other instructions.
- **Answer Only**: Prompts the model to respond only with the numerical answer.
- **Chain of Thought**: Baseline plus instructing the model to think step-by-step.
- **One Shot**: Baseline plus providing a user/assistant history of one question and (chain of thought) answer from the training datasplit from GSM8K.

See the [Results](#results) section below for examples of exact phrasing of prompt strategies.

## Baseline Setup
- Dataset: GSM8K (test split, 1319 problems)
- Model: Qwen2.5-1.5B-Instruct
- Decoding: Greedy (for reproducibility of results)
- Max new tokens: 1024
- Batch size: 32

## Running the Evaluation
### Requirements
- Python
    - Tested on Python 3.11.
- Nvidia GPU recommended for faster runtime.
- If you encounter out-of-memory errors, reduce the batch size.
    - 32 was tested on an 8GB VRAM GPU

### Setup

Install the requirements in your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
Command Line Arguments
| Argument | Short | Description |
| :--- | :---: | :---: | 
| `--prompt-strategy` | `-ps` |  How the model is prompted before each question. Options: `baseline`, `answer-only`, `cot`, `one-shot` (default: `baseline`). |
| `--question-count` | `-qc` | How many questions with which to prompt the model (default: `1319`—the total number of test questions in GSM8K). |
| `--max-new-tokens` | `-mnt` | Max tokens for a model's response: low values may run faster; high values may increase performance (default: `1024`). Note that `answer-only` requires only `8` for performance. |
| `--batch-size` | `-bs` | Batch size for each inference loop (default: `32`). |
| `--verbose` | `-v` | Prints the summary data rather than only saving it to the summary file.|

```bash
python eval.py
```

Note that the first run will download ~3GB, by default located at `C:\Users\<your-username>\.cache\huggingface`.

**Hugging Face Authentication (Optional)**

If you encounter rate limit warnings when downloading on the first run, and it is a problem for you, then:
1. Create/login to Hugging Face: https://huggingface.co
2. Go to: https://huggingface.co/settings/tokens
3. Create a **Read** token.
4. Follow the instructions [here](https://huggingface.co/docs/huggingface_hub/guides/cli
) to install `huggingface_hub` to your CLI
5. Run:
    ```bash
    hf auth login
    ```

## Results
### Getting Your Own Results

Results from local runs are stored in the `output` directory. For each run, both a `<timestamp>_<question-count>_<prompt-strategy>_results.jsonl` and `<timestamp>_<question-count>_<prompt-strategy>_summary.json` are produced. The former contains details for each question/answer in the run, and the latter contains summary information for the run overall.

### Sample Results

Here are real examples of the `json` produced for the summary file for each prompting strategy, as well as the first line (first question from GSM8K) of the `jsonl` (note that the both have been prettified for the purposes of readability here to be more than one line):

<details>
<summary><b>Baseline</b></summary>

**Summary**
```json
{
  "dataset": "gsm8k",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_strategy": "baseline",
  "question_count": 1319,
  "correct": 100,
  "incorrect": 1123,
  "extraction_failures": 96,
  "accuracy": "7.6%",
  "extraction_success_rate": "92.7%",
  "accuracy_on_extraction_success": "8.2%"
}
```

**Result Line**
```json
{
  "formatted_prompt": "<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>\n<|im_start|>assistant\n",
  "model_response": "#### 8",
  "extracted_model_answer": "8",
  "correct_answer": "18",
  "answer_status": "incorrect"
}
```
</details>

<details>
<summary><b>Answer Only</b></summary>

**Summary**
```json
{
  "dataset": "gsm8k",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_strategy": "answer-only",
  "question_count": 1319,
  "correct": 118,
  "incorrect": 1167,
  "extraction_failures": 34,
  "accuracy": "8.9%",
  "extraction_success_rate": "97.4%",
  "accuracy_on_extraction_success": "9.2%"
}
```

**Result Line**
```json
{
  "formatted_prompt": "<|im_start|>system\nFor each question, respond only with your numerical answer.<|im_end|>\n<|im_start|>user\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>\n<|im_start|>assistant\n",
  "model_response": "48",
  "extracted_model_answer": "48",
  "correct_answer": "18",
  "answer_status": "incorrect"
}
```
</details>

<details>
<summary><b>Chain of Thought</b></summary>

**Summary**
```json
{
  "dataset": "gsm8k",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_strategy": "cot",
  "question_count": 1319,
  "correct": 488,
  "incorrect": 197,
  "extraction_failures": 634,
  "accuracy": "37.0%",
  "extraction_success_rate": "51.9%",
  "accuracy_on_extraction_success": "71.2%"
}
```

**Result Line**
```json
{
  "formatted_prompt": "<|im_start|>system\nExplain your answer step by step.\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>\n<|im_start|>assistant\n",
  "model_response": "To determine how much Janet makes from selling the eggs at the farmers' market every day, we need to follow these steps:\n\n1. **Calculate total eggs laid per day**: \n   - Janet's ducks lay 16 eggs per day.\n\n2. **Determine the number of eggs eaten for breakfast**:\n   - Janet eats 3 eggs for breakfast every morning.\n   \n3. **Calculate remaining eggs after breakfast**:\n   - Remaining eggs = Total eggs - Eggs eaten for breakfast\n   - Remaining eggs = 16 - 3 = 13\n\n4. **Determine the number of eggs used to bake muffins**:\n   - Janet uses 4 eggs to bake muffins for her friends every day.\n\n5. **Calculate remaining eggs after baking muffins**:\n   - Remaining eggs = Remaining eggs after breakfast - Eggs used for muffins\n   - Remaining eggs = 13 - 4 = 9\n\n6. **Calculate earnings from selling the remaining eggs**:\n   - Janet sells each egg for $2.\n   - Earnings = Remaining eggs * Price per egg\n   - Earnings = 9 * $2 = $18\n\nTherefore, Janet makes $18 every day at the farmers' market.\n\nFinal Answer: $#### 18",
  "extracted_model_answer": "18",
  "correct_answer": "18",
  "answer_status": "correct"
}
```
</details>

<details>
<summary><b>One Shot</b></summary>

**Summary**
```json
{
  "dataset": "gsm8k",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_strategy": "one-shot",
  "question_count": 1319,
  "correct": 684,
  "incorrect": 512,
  "extraction_failures": 123,
  "accuracy": "51.9%",
  "extraction_success_rate": "90.7%",
  "accuracy_on_extraction_success": "57.2%"
}
```

**Result Line**
```json
{
  "formatted_prompt": "<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\nNatalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72<|im_end|>\n<|im_start|>user\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>\n<|im_start|>assistant\n",
  "model_response": "Janet's ducks lay a total of 16*3 = <<16*3=48>>48 eggs for breakfast.\nShe also uses up 4*3 = <<4*3=12>>12 eggs to bake muffins for her friends.\nSo Janet has 48-12-3 = <<48-12-3=33>>33 eggs left over.\n\nTherefore, Janet makes 33*2 = <<33*2=66>>66 dollars every day at the farmers' market.\n\n#### 66",
  "extracted_model_answer": "66",
  "correct_answer": "18",
  "answer_status": "incorrect"
}
```
</details>

### Summary Comparison Between Prompting Strategies

| Strategy | Accuracy | Extraction Success Rate  | Accuracy on Extraction Success |
| :--- | :---: | :---: | :---: |
| **One Shot** | 51.9% | 90.7% | 57.2% |
| **Chain of Thought** | 37.0% | 51.9% | 71.2% |
| **Answer Only** | 8.9% | 97.4% | 9.2% |
| **Baseline** | 7.6% | 92.7% | 8.2% |

Note that these results come from running the evaluations with no command line arguments aside from the prompt strategy, thus, for all of the above, the following were true:
- Batch size: `32`
- Question Count: `1319` (all of the test questions in GSM8K)
- Max New Tokens: `1024` (although answer only performs just as well with as few as 8, and execution time remains around 30 seconds)

### Analysis
Chain of thought had by far the best accuracy on extraction success, but also by far the worst extraction success rate.

One shot overall had the best accuracy due to its combined great extraction rate and accuracy on extraction success. I wonder if one shot combined with chain of thought would produce great results, but one has to be careful with overloading smaller models with too many instructions, in my experience.

It is striking that merely suggesting thinking step by step (chain of thought) causes such a drastic drop in extraction success rate compared to baseline (51.9% vs 92.7%). This goes to show the above point that it is easy to overload smaller models on instructions—add the ""Explain your answer step by step," and the "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'" is often forgotten.

Baseline is the worst, even worse than answer only. This is surprising since using its response to think through an answer typically makes a model perform better—that is the whole idea behind chain of thought, and answer only makes it answer without any of that thinking. Yet even ignoring extraction as a factor, answer only had better results than baseline: a 9.2% vs 8.2% success rate.

## Implementation Notes
### Batch Size
Results vary slightly (<0.5% for answer only) across batch sizes likely due to batch size affecting PyTorch kernel selection and therefore the possibility of different token selection in some cases.

Execution time by batch size is a U-curve with a Goldilocks zone of efficiency.

On my machine, experimentation suggests that the most accurate and most time-efficient batch size is in the range of 16-64, so I chose to stick with 32. Although accuracy was lower at both ends, extraction success rate remained relatively stable, suggesting that extraction success is not affected by batching, even though the model's ability to do the actual math is. An important caveat is that since the signal size of the accuracy variation is small (8.5%-8.9% range) range, it is not certain.

### Response Extraction
With the exception of the answer only prompting method, GSM8K evaluation follows the standard answer extraction method used in the official repository, reimplemented here.
https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

### Prompt Format
For each prompt strategy, I experimented with various formats until I was satisfied. These are author-designed prompt variants, not benchmark-optimized prompts.

All prompts, with the exception of answer only (which requests the numerical answer in isolation), request that the numerical response be prefixed with "#### ", and they all give an example of that format. Further analysis could be done without that example; in my experience, extraction success suffers without it, unsurprisingly.

See the [Sample Results](#sample-results) section above for examples of exact phrasing of prompt strategies.

## Next Steps
- Error clustering
- Identify failure patterns
- Try one refinement loop
- Analyze tradeoffs, add to ReadME
- Potentially add further prompting variants (two shot, few shot...)
- Potentially add other models or datasets
