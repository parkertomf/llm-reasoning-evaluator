from datasets import load_dataset


class GSM8KDataset:

    def __init__(self, eval_count: int):
        dataset = list(load_dataset("gsm8k", "main")["test"])  # main as opposed to Socratic
        subset = dataset[:eval_count]
        self.questions = [qa["question"] for qa in subset]
        self.correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in subset]
        self.base_prompt = "For each question, respond only with your numerical answer."

    def is_valid_answer(self, ans: str) -> bool:
        return ans.isdigit()
