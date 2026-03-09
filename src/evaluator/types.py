from typing import Literal, get_args

StrategyType = Literal['baseline', 'answer-only', 'cot', 'one-shot']
VALID_PROMPTING_STRATEGIES = get_args(StrategyType)
