from evaluator.datasets import Gsm8kDatasetWrapper
from evaluator.types import AnswerStatus


class TestDatasets:

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
        assert dataset.extract_answer("The answer is #### 42 (the meaning of life).") == "42"

        # Incorrect (missing "#### ")
        assert dataset.extract_answer("42") == AnswerStatus.INVALID

        # Incorrect (missing number)
        assert dataset.extract_answer(
            "The answer is #### the meaning of life.") == AnswerStatus.INVALID

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
