from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import get_args

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.models import ModelWrapper
from evaluator.runner import run_eval
from evaluator.types import VALID_PROMPTING_STRATEGIES
from evaluator.utils import get_project_root
512


def main():
    args = get_args()

    # Dataset and model are hardcoded for now. Input capacity may be added with the addition of more options in the future.
    dataset = Gsm8kDatasetWrapper(args.question_count, args.prompt_strategy)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model_wrapper = ModelWrapper(model_name)

    # TODO: Add filepath
    log_file_path = get_project_root() / "results" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_qc{args.question_count}_{args.prompt_strategy}.jsonl"

    decoded_responses = run_eval(dataset, model_wrapper, args.batch_size, args.max_new_tokens, log_file_path)

    # correct, incorrect, extract_fails = sort_output(decoded_responses, dataset)
    # print_statistics(correct, incorrect, extract_fails, model_name, dataset.name, args.prompt_strategy)


def get_args():
    parser = ArgumentParser(description='Evaluate LLM reasoning ability.')
    parser.add_argument('-ps', '--prompt-strategy', default='baseline',
                        choices=VALID_PROMPTING_STRATEGIES,
                        help='How the model is prompted before each question')
    parser.add_argument('-bs', '--batch-size', type=int, default=32,
                        help='Batch size for each inference loop')
    parser.add_argument('-qc', '--question-count', type=int, default=1319,
                        help='How many questions with which to prompt the model')
    parser.add_argument(
        '-mnt',
        '--max-new-tokens',
        type=int,
        default=16,  # Note / TODO?: 8 performed the same for answer-only as 16
        help=
        'Max tokens for a model\'s response: low values run faster, high values increase performance'
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
