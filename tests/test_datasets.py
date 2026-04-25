import pytest
from unittest.mock import patch

from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus

MOCK_DATASET = {
    "test": [{
        "question":
        "What is the Answer to the Great Question of Life, The Universe, and Everything?",
        "answer": "#### 42"
    }],
    "train": [{
        "question": "What does the scouter say about his power level?",
        "answer": "According to Vegeta, it's over 9000, so final answer: #### 9,001"
    }]
}


class TestDatasets:

    @pytest.fixture(autouse=True)
    def patch_load_dataset(self):
        with patch("evaluator.datasets.load_dataset", return_value=MOCK_DATASET):
            yield

    def test_extract_answer_general(self):
        """Test the extract_answer method for for all prompt strategies other than answer-only, which have shared extraction logic."""
        dataset = Gsm8kDatasetWrapper(question_count=1, prompt_strategy="baseline")

        # Standard correct format
        assert dataset.extract_answer("The answer is #### 42") == "42"

        # Correct using commas
        assert dataset.extract_answer("The answer is #### 9,001") == "9001"

        # Correct with signage
        assert dataset.extract_answer("The answer is #### -42") == "-42"

        # Correct float
        assert dataset.extract_answer("The answer is #### 13.37") == "13.37"

        # Undesired formatting, but still extractable
        assert dataset.extract_answer(
            "The answer is #### 42 (the answer to the Great Question).") == "42"

        # Incorrect (missing "#### ")
        assert dataset.extract_answer("42") == AnswerStatus.INVALID

        # Incorrect (missing number)
        assert dataset.extract_answer(
            "The answer is #### the meaning of Life, the Universe, and Everything."
        ) == AnswerStatus.INVALID

    def test_extract_answer_answer_only(self):
        """Test the extract_answer method for the answer-only prompt strategy, which has unique extraction logic."""
        dataset = Gsm8kDatasetWrapper(question_count=1, prompt_strategy="answer-only")

        # Standard correct format
        assert dataset.extract_answer("42") == "42"

        # Correct format using commas
        assert dataset.extract_answer("9,001") == "9001"

        # Correct with signage
        assert dataset.extract_answer("-42") == "-42"

        # Correct float
        assert dataset.extract_answer("13.37") == "13.37"

        # Incorrect (passes a non-float string)
        assert dataset.extract_answer("The answer is 42.") == AnswerStatus.INVALID
