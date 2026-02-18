from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Dataset and model setup
dataset = list(load_dataset("gsm8k", "main")["test"])  # main as opposed to Socratic
questions = [qa["question"] for qa in dataset[:5]]
correct_answers = [qa["answer"][qa["answer"].rindex("####") + 4 :] for qa in dataset[:5]]
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Prompt formatting for model
messages = [{"role": "system", "content": "For each question, respond only with your numerical answer."}]
formatted_messages = []
for q in questions:
    messages.append({"role": "user", "content": q})
    formatted_messages.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    messages.pop()
model_inputs = tokenizer(formatted_messages, padding=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    do_sample=False,  # Consistent output. Greedy token generation
    # The following 3 params are default values and are irrelevant parameters when used with do_sample=False
    # They are set simply to suppress an info log.
    temperature=1.0,
    top_k=50,
    top_p=1.0
)

# Remove the prompt tokens from the output so only the model’s new answer remains.
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

model_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for i, q in enumerate(questions):
    print(f"Question: {q}\nCorrect Response: {correct_answers[i]}\nModel Response: {model_responses[i]}\n\n")
