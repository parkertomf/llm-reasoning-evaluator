import json

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus
from evaluator.utils import log_results
from tests.conftest import MOCK_DATASET

MOCK_FORMATTED_PROMPTS = [
    f"<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\n{MOCK_DATASET['test'][0]['question']}<|im_end|>\n<|im_start|>assistant\n",
    f"<|im_start|>system\nFor each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'<|im_end|>\n<|im_start|>user\n{MOCK_DATASET['test'][1]['question']}<|im_end|>\n<|im_start|>assistant\n"
]

MOCK_MODEL_RESPONSES = ["The answer is 42", "According to Vegeta, it is over #### 9000"]
EXPECTED_EXTRACTED_ANSWERS = [AnswerStatus.INVALID, "9000"]
EXPECTED_ANSWER_STATUSES = [AnswerStatus.INVALID, AnswerStatus.CORRECT]
EXPECTED_CORRECT_ANSWERS = ["42", "9000"]


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
            for i, line in enumerate(res):
                result_record = json.loads(line)
                assert result_record["formatted_prompt"] == MOCK_FORMATTED_PROMPTS[i]
                assert result_record["model_response"] == MOCK_MODEL_RESPONSES[i]
                assert result_record["extracted_model_answer"] == EXPECTED_EXTRACTED_ANSWERS[i]
                assert result_record["correct_answer"] == EXPECTED_CORRECT_ANSWERS[i]
                assert result_record["answer_status"] == EXPECTED_ANSWER_STATUSES[i]
