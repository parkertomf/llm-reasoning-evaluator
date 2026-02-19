from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

eval_count = 100

# Dataset and model setup
dataset = list(load_dataset("gsm8k", "main")["test"])  # main as opposed to Socratic
questions = [qa["question"] for qa in dataset[:eval_count]]
correct_answers = [qa["answer"][qa["answer"].rindex("####") + 5:] for qa in dataset[:eval_count]]
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Prompt formatting for model
messages = [{
    "role": "system",
    "content": "For each question, respond only with your numerical answer."
}]
formatted_messages = []
for q in questions:
    messages.append({"role": "user", "content": q})
    formatted_messages.append(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    messages.pop()
model_inputs = tokenizer(formatted_messages, padding=True, return_tensors="pt").to(model.device)

# Actually generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16,
    do_sample=False,  # Consistent output. Greedy token generation
    # The following 3 params are default values and are irrelevant parameters when used with do_sample=False
    # They are set simply to suppress an info log.
    temperature=1.0,
    top_k=50,
    top_p=1.0)

# Remove the prompt tokens from the output so only the model’s new answer remains.
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

model_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Handle output
correct, incorrect, extraction_failures = 0, 0, 0
for i in range(eval_count):
    model_res = model_responses[i]
    if not model_res.isdigit():
        extraction_failures += 1
    elif model_res == correct_answers[i]:
        correct += 1
    else:
        incorrect += 1

# for i, q in enumerate(questions):
#     print(
#         f"Question: {q}\nCorrect Response: {correct_answers[i]}\nModel Response: {model_responses[i]}\n\n"
#     )

print(
    f"Model: {model_name}\n"
    f"Problems Tested: {eval_count}\n"
    f"Correct: {correct}\n"
    f"Incorrect: {incorrect}\n"
    f"Extraction Failures: {extraction_failures}\n"
    f"Accuracy: {(correct / eval_count * 100):.1f}%\n"
    f"Accuracy on Extraction Success: {(correct / (eval_count - extraction_failures) * 100):.1f}%\n"
    f"Format Compliance: {((eval_count - extraction_failures) / eval_count * 100):.1f}%")
