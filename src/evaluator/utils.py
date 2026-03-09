from typing import Tuple, List
from evaluator.datasets import Gsm8kDataset


def sort_output(
    decoded_responses: List[str],
    dataset: Gsm8kDataset,
) -> Tuple[int, int, int]:
    """Get the counts of correct answers, incorrect answers, and extraction failures from a model's response."""
    correct, incorrect, extract_fails = 0, 0, 0
    for i in range(len(decoded_responses)):
        model_res = decoded_responses[i]
        extracted = dataset.extract_answer(model_res)
        if not dataset.is_valid_answer(extracted):
            extract_fails += 1
        elif extracted == dataset.correct_answers[i]:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect, extract_fails


def print_statistics(
    correct: int,
    incorrect: int,
    extract_fails: int,
    model_name: str,
    dataset_name: str,
    prompt_strategy: str,
) -> None:
    question_count = correct + incorrect + extract_fails
    extracted = question_count - extract_fails
    accuracy_on_extracted = f"{correct / extracted * 100:.1f}%" if extracted else "N/A"
    print(f"\nModel: {model_name}\n"
          f"Dataset: {dataset_name}\n"
          f"Problems Tested: {question_count}\n"
          f"Prompting Strategy: {prompt_strategy}\n"
          f"Correct: {correct}\n"
          f"Incorrect: {incorrect}\n"
          f"Extraction Failures: {extract_fails}\n"
          f"Accuracy: {(correct / question_count * 100):.1f}%\n"
          f"Extraction Success Rate: {(extracted / question_count * 100):.1f}%\n"
          f"Accuracy on Extraction Success: {accuracy_on_extracted}\n")
