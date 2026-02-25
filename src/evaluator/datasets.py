from datasets import load_dataset


class GSM8KDataset:
    """Loads, preprocesses, and handles GSM8K test examples."""

    def __init__(self, eval_count: int):
        dataset = list(load_dataset("gsm8k", "main")["test"])  # main as opposed to Socratic
        subset = dataset[:eval_count]
        self.questions = [qa["question"] for qa in subset]
        self.correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in subset]
        self.base_prompt = "For each question, respond only with your numerical answer."

    def is_valid_answer(self, ans: str) -> bool:
        """
        Return True if an answer is valid for the GSM8K dataset, i.e. it is a float.

        GSM8K only intends positive integer solutions, however:
            A) there are some exceptions that are likely mistakes, according to this paper. https://arxiv.org/html/2405.00332v1
            B) A model could also mistakenly think the answer is negative or a decimal, so that would still be a valid response, albeit an incorrect one.
        """
        try:
            float(ans)
            return True
        except (ValueError):
            return False
