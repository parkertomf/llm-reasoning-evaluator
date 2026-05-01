from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus

from tests.conftest import MOCK_DATASET

class TestDatasets:

    def test_init_one_shot(self, patch_load_dataset):
        """Test initialization for the one shot prompt strategy."""
        dataset = Gsm8kDatasetWrapper(question_count=1, prompt_strategy="one-shot")

        # Verify the training examples are added to the messages correctly, additionally ensuring correct mocking
        assert dataset.base_messages[1]["content"] == MOCK_DATASET["train"][0]["question"]
        assert dataset.base_messages[2]["content"] == MOCK_DATASET["train"][0]["answer"]

    def test_extract_answer(self, patch_load_dataset):
        """Test the extract_answer method for for all prompt strategies other than answer-only, which have shared extraction logic."""
        dataset = Gsm8kDatasetWrapper(question_count=1, prompt_strategy="baseline")

        # Standard correct format
        assert dataset.extract_answer("The answer is #### 42") == "42"

        # Correct using commas
        assert dataset.extract_answer("The answer is #### 9,000") == "9000"

        # Correct with signage
        assert dataset.extract_answer("The answer is #### -42") == "-42"

        # Correct float
        assert dataset.extract_answer("The answer is #### 13.37") == "13.37"

        # Undesired formatting, but still extractable
        assert dataset.extract_answer(
            "The answer is #### 42 (the answer to the Great Question)") == "42"

        # Incorrect (missing "#### ")
        assert dataset.extract_answer("42") == AnswerStatus.INVALID

        # Incorrect (missing number)
        assert dataset.extract_answer(
            "The answer is #### the meaning of Life, the Universe, and Everything."
        ) == AnswerStatus.INVALID

    def test_extract_answer_answer_only(self, patch_load_dataset):
        """Test the extract_answer method for the answer-only prompt strategy, which has unique extraction logic."""
        dataset = Gsm8kDatasetWrapper(question_count=1, prompt_strategy="answer-only")

        # Standard correct format
        assert dataset.extract_answer("42") == "42"

        # Correct format using commas
        assert dataset.extract_answer("9,000") == "9000"

        # Correct with signage
        assert dataset.extract_answer("-42") == "-42"

        # Correct float
        assert dataset.extract_answer("13.37") == "13.37"

        # Incorrect (passes a non-float string)
        assert dataset.extract_answer("The answer is 42.") == AnswerStatus.INVALID
