from dataclasses import asdict
from json import dumps
from pathlib import Path

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import ResultRecord

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

# def sort_output(
#     decoded_responses: List[str],
#     dataset: Gsm8kDatasetWrapper,
# ) -> Tuple[int, int, int]:
#     """Get the counts of correct answers, incorrect answers, and extraction failures from a model's response."""
#     correct, incorrect, extract_fails = 0, 0, 0
#     for i in range(len(decoded_responses)):
#         model_res = decoded_responses[i]
#         extracted = dataset.extract_answer(model_res)
#         if not dataset.is_valid_answer(extracted):
#             extract_fails += 1
#         elif extracted == dataset.correct_answers[i]:
#             correct += 1
#         else:
#             incorrect += 1
#     return correct, incorrect, extract_fails


def log_results(
    log_file_path: Path,
    start: int,
    formatted_prompts: list[str],
    model_responses: list[str],
    dataset: Gsm8kDatasetWrapper,
) -> None:
    """Logs question / answer data to an output .jsonl in the results directory."""
    # TODO: Maybe a doesnt work if file doesnt exist already?
    with log_file_path.open('a') as f:
        for i, model_res in enumerate(model_responses):
            model_answer = dataset.extract_answer(model_res)
            correct_answer = dataset.correct_answers[i + start]
            answer_status = dataset.get_answer_status(model_answer, correct_answer)
            result_record = ResultRecord(formatted_prompts[i], model_responses[i], model_answer, correct_answer, answer_status)
            f.write(dumps(asdict(result_record)))

# def print_statistics(
#     correct: int,
#     incorrect: int,
#     extract_fails: int,
#     model_name: str,
#     dataset_name: str,
#     prompt_strategy: str,
# ) -> None:
#     question_count = correct + incorrect + extract_fails
#     extracted = question_count - extract_fails
#     accuracy_on_extracted = f"{correct / extracted * 100:.1f}%" if extracted else "N/A"
#     print(f"\nModel: {model_name}\n"
#           f"Dataset: {dataset_name}\n"
#           f"Problems Tested: {question_count}\n"
#           f"Prompting Strategy: {prompt_strategy}\n"
#           f"Correct: {correct}\n"
#           f"Incorrect: {incorrect}\n"
#           f"Extraction Failures: {extract_fails}\n"
#           f"Accuracy: {(correct / question_count * 100):.1f}%\n"
#           f"Extraction Success Rate: {(extracted / question_count * 100):.1f}%\n"
#           f"Accuracy on Extraction Success: {accuracy_on_extracted}\n")
