from evaluator.models import ModelWrapper
from evaluator.datasets import GSM8KDataset
from evaluator.utils import sort_output, print_statistics
from evaluator.runner import run_eval
from argparse import ArgumentParser

# (baseline: nothing, forced_float: asked to supply the float answer in isolation, cot: chain-of-thought)

def main():
    prompt_strategy, batch_size, question_count = get_args()

    # These are hardcoded for now. Input capacity will be added with the addition of more options in the future.
    dataset = GSM8KDataset(question_count)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_wrapper = ModelWrapper(model_name)

    decoded_responses = run_eval(dataset, model_wrapper, batch_size)

    correct, incorrect, extract_fails = sort_output(decoded_responses, dataset)
    print_statistics(correct, incorrect, extract_fails, model_name, dataset.name)

def get_args():
    parser = ArgumentParser(
        description='Evaluate LLM reasoning ability.')
    parser.add_argument('-ps',
                        '--prompt-strategy',
                        default='isolate',
                        choices=['baseline', 'isolate', 'cot'],
                        help='How the model is prompted before each question')
    parser.add_argument('-bs',
                        '--batch-size',
                        type=int,
                        default=32,
                    help='Batch size for each inference loop')
    parser.add_argument('-qc',
                        '--question_count',
                        type=int,
                        default=1319,
                        help='How many questions with which to prompt the model.')
    args = parser.parse_args()
    return args.prompt_strategy, args.batch_size, args.question_count

if __name__ == "__main__":
    main()
