import json
import dataclasses
from pathlib import Path

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus, ResultRecord, ResultSummary


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def log_results(
    results_file_path: Path,
    start: int,
    formatted_prompts: list[str],
    model_responses: list[str],
    dataset: Gsm8kDatasetWrapper,
) -> None:
    """Log question / answer data to an output .jsonl in the results directory."""
    with results_file_path.open('a') as f:
        for i, model_res in enumerate(model_responses):
            extracted_model_answer = dataset.extract_answer(model_res)
            correct_answer = dataset.correct_answers[i + start]
            answer_status = dataset.get_answer_status(extracted_model_answer, correct_answer)
            result_record = ResultRecord(
                formatted_prompt=formatted_prompts[i],
                model_response=model_responses[i],
                extracted_model_answer=extracted_model_answer,
                correct_answer=correct_answer,
                answer_status=answer_status,
            )
            f.write(f"{json.dumps(dataclasses.asdict(result_record))}\n")


def log_summary(
    results_file_path: Path,
    summary_file_path: Path,
    dataset_name: str,
    model_name: str,
    prompt_strategy: str,
    verbose: bool,
) -> None:
    """Log overall data for a run."""
    with results_file_path.open('r') as res:
        correct, incorrect, extract_fails = 0, 0, 0
        for line in res:
            match json.loads(line)["answer_status"]:
                case AnswerStatus.CORRECT:
                    correct += 1
                case AnswerStatus.INCORRECT:
                    incorrect += 1
                case AnswerStatus.INVALID:
                    extract_fails += 1

    with summary_file_path.open('w') as summ:
        question_count = correct + incorrect + extract_fails
        extracted = question_count - extract_fails
        result_summary = ResultSummary(
            dataset=dataset_name,
            model=model_name,
            prompt_strategy=prompt_strategy,
            question_count=question_count,
            correct=correct,
            incorrect=incorrect,
            extraction_failures=extract_fails,
            accuracy=f"{(correct / question_count * 100):.1f}%",
            extraction_success_rate=f"{(extracted / question_count * 100):.1f}%",
            accuracy_on_extraction_success=f"{correct / extracted * 100:.1f}%"
            if extracted else "N/A",
        )
        summ.write(json.dumps(dataclasses.asdict(result_summary)))

    if verbose:
        print(f"\nDataset: {dataset_name}\n"
              f"Model: {model_name}\n"
              f"Prompting Strategy: {prompt_strategy}\n"
              f"Question Count: {question_count}\n"
              f"Correct: {correct}\n"
              f"Incorrect: {incorrect}\n"
              f"Extraction Failures: {extract_fails}\n"
              f"Accuracy: {result_summary.accuracy}\n"
              f"Extraction Success Rate: {result_summary.extraction_success_rate}\n"
              f"Accuracy on Extraction Success: {result_summary.accuracy_on_extraction_success}\n")
