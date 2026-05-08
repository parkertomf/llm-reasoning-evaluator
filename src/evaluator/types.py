from dataclasses import dataclass
from enum import Enum
from typing import Literal, get_args

StrategyType = Literal["baseline", "answer-only", "cot", "one-shot"]
VALID_PROMPTING_STRATEGIES = get_args(StrategyType)


class AnswerStatus(str, Enum):
    # Declare as a subclass of str to allow json serialization.
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INVALID = "invalid"


@dataclass
class ResultRecord:
    """
    Class representing an individual record for a single question / answer.
    
    Used in the results output jsonl file.
    """
    index: int
    formatted_prompt: str
    model_response: str
    extracted_model_answer: str
    correct_answer: str
    answer_status: AnswerStatus


@dataclass
class ResultSummary:
    """
    Class representing the result summary for an entire run.
    
    Used in the summary output json file.
    """
    dataset: str
    model: str
    prompt_strategy: StrategyType
    question_count: int
    correct: int
    incorrect: int
    extraction_failures: int
    accuracy: str
    extraction_success_rate: str
    accuracy_on_extraction_success: str
