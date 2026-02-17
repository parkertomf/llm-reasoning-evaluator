from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

dataset = load_dataset("gsm8k", "main")  # main as opposed to Socratic
q, a = dataset['test'][0]['question'], dataset['test'][0]['answer']
numerical_answer = a[a.rindex('####') + 4:]

messages = [
    {"role": "system", "content": "For each question, respond only with your numerical answer."},
    #{"role": "system", "content": "For each question, after responding, conclude your message with the following, where r is the numerical result: ####r."},
    {"role": "user", "content": q}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False,  # Consistent output. Greedy token generation
    # The following 3 params are default values and are set to suppress an info log.
    temperature=1.0,
    top_k=50,
    top_p=1.0
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Question: {q}\nGround Truth: {numerical_answer}\nResponse: {response}")

# questions = []
# ground_truths = []
# for i in range(5):
#     q, a = dataset['test'][i]['question'], dataset['test'][i]['answer']
#     questions.append(q)
#     ground_truths.append(a)