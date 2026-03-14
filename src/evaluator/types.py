from dataclasses import dataclass
from typing import Literal, get_args

AnswerStatus = Literal['correct', 'incorrect', 'invalid']
StrategyType = Literal['baseline', 'answer-only', 'cot', 'one-shot']
VALID_PROMPTING_STRATEGIES = get_args(StrategyType)


@dataclass
class ResultRecord:
    """Class representing an individual record in the result jsonl file."""
    formatted_prompt: str
    model_response: str
    model_answer: str
    correct_answer: str
    answer_status: str
