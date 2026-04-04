import argparse
from datetime import datetime
from typing import get_args

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.models import ModelWrapper
from evaluator.runner import run_eval
from evaluator.types import VALID_PROMPTING_STRATEGIES
from evaluator.utils import get_project_root, log_summary


def main():
    args = get_args()

    # Dataset and model are hardcoded for now. Input capacity may be added with the addition of more options in the future.
    dataset = Gsm8kDatasetWrapper(args.question_count, args.prompt_strategy)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model_wrapper = ModelWrapper(model_name)

    # Set up paths for detailed question/answer results and for overall summary statistics.
    base_path = get_project_root() / "output"
    output_file_base_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_qc{args.question_count}_{args.prompt_strategy}"
    results_file_path = base_path / f"{output_file_base_name}_results.jsonl"
    summary_file_path = base_path / f"{output_file_base_name}_summary.json"

    run_eval(dataset, model_wrapper, args.batch_size, args.max_new_tokens, results_file_path)

    log_summary(
        results_file_path=results_file_path,
        summary_file_path=summary_file_path,
        dataset_name=dataset.name,
        model_name=model_name,
        prompt_strategy=args.prompt_strategy,
        verbose=args.verbose
    )


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM reasoning ability.")
    parser.add_argument("-ps", "--prompt-strategy", default="baseline",
                        choices=VALID_PROMPTING_STRATEGIES,
                        help="How the model is prompted before each question")
    parser.add_argument("-bs", "--batch-size", type=int, default=32,
                        help="Batch size for each inference loop")
    parser.add_argument("-qc", "--question-count", type=int, default=1319,
                        help="How many questions with which to prompt the model")
    parser.add_argument(
        "-mnt", "--max-new-tokens", type=int, default=8, help=
        "Max tokens for a model's response: low values run faster; high values may increase performance"
    )
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction,
                             help="Print result summary.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
