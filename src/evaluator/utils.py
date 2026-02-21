from typing import Tuple, List
from evaluator.datasets import GSM8KDataset


def sort_output(
    eval_count: int,
    model_responses: List[str],
    dataset: GSM8KDataset,
) -> Tuple[int, int, int]:
    correct, incorrect, extract_fails = 0, 0, 0
    for i in range(eval_count):
        model_res = model_responses[i]
        if not dataset.is_valid_answer(model_res):
            extract_fails += 1
        elif model_res == dataset.correct_answers[i]:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect, extract_fails


def print_statistics(
    correct: int,
    incorrect: int,
    extract_fails: int,
    model_name: str,
    eval_count: int,
) -> None:
    print(f"\nModel: {model_name}\n"
          f"Problems Tested: {eval_count}\n"
          f"Correct: {correct}\n"
          f"Incorrect: {incorrect}\n"
          f"Extraction Failures: {extract_fails}\n"
          f"Accuracy: {(correct / eval_count * 100):.1f}%\n"
          f"Accuracy on Extraction Success: {(correct / (eval_count - extract_fails) * 100):.1f}%\n"
          f"Format Compliance: {((eval_count - extract_fails) / eval_count * 100):.1f}%\n")
