import dataclasses
import json

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus, ResultRecord
from evaluator.utils import log_results, log_summary
from tests.conftest import MOCK_DATASET

MOCK_FORMATTED_PROMPTS = [
    f"<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\n{MOCK_DATASET['test'][0]['question']}<|im_end|>\n<|im_start|>assistant\n",
    f"<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\n{MOCK_DATASET['test'][1]['question']}<|im_end|>\n<|im_start|>assistant\n"
]
MOCK_MODEL_RESPONSES = ["The answer is 42", "According to Vegeta, it is over #### 9000"]
EXPECTED_EXTRACTED_ANSWERS = [AnswerStatus.INVALID, "9000"]
EXPECTED_ANSWER_STATUSES = [AnswerStatus.INVALID, AnswerStatus.CORRECT]
EXPECTED_CORRECT_ANSWERS = ["42", "9000"]
MOCK_DATASET_NAME = "test_dataset"
MOCK_MODEL_NAME = "test_model"
MOCK_PROMPT_STRATEGY = "baseline"


class TestUtils:

    def test_log_results(self, patch_load_dataset, tmp_path):
        results_file_path = tmp_path / "results.jsonl"
        dataset = Gsm8kDatasetWrapper(question_count=2, prompt_strategy="baseline")
        log_results(
            results_file_path=results_file_path,
            start=0,
            formatted_prompts=MOCK_FORMATTED_PROMPTS,
            model_responses=MOCK_MODEL_RESPONSES,
            dataset=dataset,
        )
        with results_file_path.open('r') as res:
            lines = res.readlines()
            assert len(lines) == len(MOCK_FORMATTED_PROMPTS)
            for i, line in enumerate(lines):
                result_record = json.loads(line)
                assert result_record["formatted_prompt"] == MOCK_FORMATTED_PROMPTS[i]
                assert result_record["model_response"] == MOCK_MODEL_RESPONSES[i]
                assert result_record["extracted_model_answer"] == EXPECTED_EXTRACTED_ANSWERS[i]
                assert result_record["correct_answer"] == EXPECTED_CORRECT_ANSWERS[i]
                assert result_record["answer_status"] == EXPECTED_ANSWER_STATUSES[i]

    def test_log_summary(self, tmp_path):
        # Set up results file for log_summary to parse
        results_file_path = tmp_path / "results.jsonl"
        with results_file_path.open('a') as res:
            for i, _ in enumerate(MOCK_MODEL_RESPONSES):
                result_record = ResultRecord(
                    formatted_prompt=MOCK_FORMATTED_PROMPTS[i],
                    model_response=MOCK_MODEL_RESPONSES[i],
                    extracted_model_answer=EXPECTED_EXTRACTED_ANSWERS[i],
                    correct_answer=EXPECTED_CORRECT_ANSWERS[i],
                    answer_status=EXPECTED_ANSWER_STATUSES[i],
                )
                res.write(f"{json.dumps(dataclasses.asdict(result_record))}\n")

        summary_file_path = tmp_path / "summary.json"
        log_summary(
            results_file_path=results_file_path,
            summary_file_path=summary_file_path,
            dataset_name=MOCK_DATASET_NAME,
            model_name=MOCK_MODEL_NAME,
            prompt_strategy=MOCK_PROMPT_STRATEGY,
            verbose=False,
        )

        with summary_file_path.open('r') as summ:
            summary = json.loads(summ.read())
            assert summary["dataset"] == MOCK_DATASET_NAME
            assert summary["model"] == MOCK_MODEL_NAME
            assert summary["prompt_strategy"] == MOCK_PROMPT_STRATEGY
            assert summary["question_count"] == 2
            assert summary["correct"] == 1
            assert summary["incorrect"] == 0
            assert summary["extraction_failures"] == 1
            assert summary["accuracy"] == "50.0%"
            assert summary["extraction_success_rate"] == "50.0%"
            assert summary["accuracy_on_extraction_success"] == "100.0%"

    def test_log_summary_0_percent_extraction_rate(self, tmp_path):
        """Covering this case ensures a divide by 0 is avoided."""
        # Set up results file for log_summary to parse
        results_file_path = tmp_path / "results.jsonl"
        with results_file_path.open('a') as res:
            result_record = ResultRecord(
                formatted_prompt=MOCK_FORMATTED_PROMPTS[0],
                model_response=MOCK_MODEL_RESPONSES[0],
                extracted_model_answer=EXPECTED_EXTRACTED_ANSWERS[0],
                correct_answer=EXPECTED_CORRECT_ANSWERS[0],
                answer_status=EXPECTED_ANSWER_STATUSES[0],
            )
            res.write(f"{json.dumps(dataclasses.asdict(result_record))}\n")

        summary_file_path = tmp_path / "summary.json"
        log_summary(
            results_file_path=results_file_path,
            summary_file_path=summary_file_path,
            dataset_name=MOCK_DATASET_NAME,
            model_name=MOCK_MODEL_NAME,
            prompt_strategy=MOCK_PROMPT_STRATEGY,
            verbose=False,
        )

        with summary_file_path.open('r') as summ:
            summary = json.loads(summ.read())
            assert summary["question_count"] == 1
            assert summary["extraction_failures"] == 1
            assert summary["accuracy_on_extraction_success"] == "N/A"
