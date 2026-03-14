from re import compile
from datasets import load_dataset
from evaluator.types import AnswerStatus, StrategyType

ANS_REGEX = compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


class Gsm8kDatasetWrapper:
    """Loads, preprocesses, and handles GSM8K test examples."""

    def __init__(self, question_count: int, prompt_strategy: StrategyType):
        self.name = "gsm8k"
        self.prompt_strategy = prompt_strategy

        self.dataset = load_dataset(self.name, "main")  # main as opposed to Socratic

        # Tested subset
        subset = list(self.dataset["test"])[:question_count]
        self.questions = [qa["question"] for qa in subset]
        self.correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in subset]

        # TODO: Make formatted prompts here. whole own function maybe

        self.base_prompt = "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'"
        match self.prompt_strategy:
            case "answer-only":
                self.base_prompt = "For each question, respond only with your numerical answer."
            case "cot":
                self.base_prompt = f"Explain your answer step by step.\n{self.base_prompt}"
            case "one-shot":
                # TODO Possible that this is supposed to be a whole chat template like we're continuing the conversation. Investigate
                self.base_prompt = (f"\nHere is a full example question and appropriate response:\n"
                                    f"User: '{self.dataset['train'][0]['question']}'\n"
                                    f"Assistant: '{self.dataset['train'][0]['answer']}'\n"
                                    f"{self.base_prompt}")
        # print("Base prompt: " + self.base_prompt)
            
    def extract_answer(self, text: str) -> str:
        """
        Extract and return the numerical solution from an answer.

        For prompts other than "answer-only", follow the standard answer extraction method used in the official repository:
        https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
        
        For "answer-only", the expectation is that the model response is a number in isolation without the preceeding '#### ' or anything else.
        """
        if self.prompt_strategy == "answer-only":
            return text
            # TODO: Need the old verification back for this. Or something. dam
        else:
            match = ANS_REGEX.search(text)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS

    def get_answer_status(self, model_ans: str, correct_ans: str) -> AnswerStatus:
        if model_ans == INVALID_ANS:
            return "invalid"
        elif model_ans == correct_ans:
            return "correct"
        else:
            return "incorrect"