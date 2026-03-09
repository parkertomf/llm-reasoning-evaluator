from re import compile
from datasets import load_dataset
from typing import Literal
from evaluator.types import StrategyType

ANS_REGEX = compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

class Gsm8kDataset:
    """Loads, preprocesses, and handles GSM8K test examples."""

    def __init__(self, question_count: int, prompt_strategy: StrategyType):
        self.name = "gsm8k"
        self.prompt_strategy = prompt_strategy

        dataset = list(load_dataset(self.name, "main")["test"])  # main as opposed to Socratic
        subset = dataset[:question_count]
        self.questions = [qa["question"] for qa in subset]
        self.correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in subset]

        match self.prompt_strategy:
            case "baseline":
                # 0/10 extracted. Insane that this could make such a difference
                # self.base_prompt = "Conclude your response with your final answer in the format: '#### '.\nFor example: '#### 42'"

                # # 2/10 correct, 10/10 extracted
                self.base_prompt = "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'"

                # 1/10 correct, 3/10 extracted
                # self.base_prompt = "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '[rest of response]#### 42'"

                # Fixed extraction, 2/10 correct 5/10 extracted
                # "For each question, you MUST prefix the final answer with these characters: '#### '."
                
                # 0/10
                # Tried: "For each question, you MUST end your response with your final answer in the format: '#### [final answer]'" <—- 0 success
                # "End your response with a line containing exactly: #### [answer]"
            case "answer-only":
                self.base_prompt = "For each question, respond only with your numerical answer."
            case "cot":
                self.base_prompt = "Explain step by step. End your response with your final answer in the format: '#### '.\nFor example: '#### 42'" 

    def extract_answer(self, text: str) -> str:
        """
        Extract and return the numerical solution from an answer.

        For prompts other than "answer-only", follow the standard answer extraction method used in the official repository:
        https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
        """
        if self.prompt_strategy == "answer-only":
            return text
        else:
            match = ANS_REGEX.search(text)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS

    def is_valid_answer(self, ans: str) -> bool:
        """
        Return True if an answer is valid for the GSM8K dataset, i.e. it is a float.

        GSM8K only intends positive integer solutions, however:
            A) There are some exceptions that are likely mistakes, according to this paper. https://arxiv.org/html/2405.00332v1
            B) A model could also mistakenly think the answer is negative or a decimal, so that would still be a valid response, albeit an incorrect one.
        """
        try:
            float(ans)
            return True
        except (ValueError, TypeError):
            return False
