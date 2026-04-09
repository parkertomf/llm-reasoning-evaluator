import re

from datasets import load_dataset

from evaluator.types import AnswerStatus, StrategyType

ANS_REGEX = re.compile(r"#### (\-?[0-9\.\,]+)")
BASE_SYSTEM_CONTENT = "For each question, you MUST prefix the final answer with these characters: '#### '.\nFor example: '#### 42'"


class Gsm8kDatasetWrapper:
    """Loads, preprocesses, and handles GSM8K test examples."""

    def __init__(self, question_count: int, prompt_strategy: StrategyType):
        self.name = "gsm8k"
        self.prompt_strategy = prompt_strategy

        dataset = load_dataset(self.name, "main")  # main as opposed to Socratic

        # Tested subset
        subset = list(dataset["test"])[:question_count]
        self.questions = [qa["question"] for qa in subset]
        self.correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in subset]

        # Everything for a prompt aside from a question
        self.base_messages = [{"role": "system", "content": BASE_SYSTEM_CONTENT}]
        match self.prompt_strategy:
            case "answer-only":
                self.base_messages[0][
                    "content"] = "For each question, respond only with your numerical answer."
            case "cot":
                self.base_messages[0][
                    "content"] = f"Explain your answer step by step.\n{BASE_SYSTEM_CONTENT}"
            case "one-shot":
                self.base_messages.extend([{
                    "role": "user",
                    "content": dataset["train"][0]["question"]
                }, {
                    "role": "assistant",
                    "content": dataset["train"][0]["answer"]
                }])

    def get_messages(self, question: str) -> list[dict]:
        return self.base_messages + [{"role": "user", "content": question}]

    def extract_answer(self, text: str) -> str:
        """
        Extract and return the numerical solution from an answer.

        For prompts other than "answer-only", follow the standard answer extraction method used in the official repository:
        https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
        
        For "answer-only", the expectation is that the model response is a number in isolation without the preceeding '#### ' or anything else.
        """
        if self.prompt_strategy == "answer-only":
            # GSM8K only intends positive integer solutions, however:
            # A) there are some exceptions that are likely mistakes, according to this paper. https://arxiv.org/html/2405.00332v1
            # B) A model could also mistakenly think the answer is negative or a decimal, so that would still be a valid response, albeit an incorrect one.
            try:
                float(text)
                return text
            except (ValueError):
                return AnswerStatus.INVALID
        else:
            match = ANS_REGEX.search(text)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return AnswerStatus.INVALID

    def get_answer_status(self, extracted_model_ans: str, correct_ans: str) -> AnswerStatus:
        if extracted_model_ans == AnswerStatus.INVALID:
            return AnswerStatus.INVALID
        elif extracted_model_ans == correct_ans:
            return AnswerStatus.CORRECT
        else:
            return AnswerStatus.INCORRECT
