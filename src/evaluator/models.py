from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:

    def __init__(self, name: str):
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto",
                                                          device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
