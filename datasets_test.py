from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")  # As opposed to Socratic
for i in range(3):
    print(f"Question: {dataset['test'][i]['question']} \n Answer: {dataset['test'][i]['answer']} \n")

# Useful info here: https://github.com/openai/grade-school-math#calculation-annotations


# https://huggingface.co/datasets/openai/gsm8k


# Switch to using secret with token param in from_pretrained