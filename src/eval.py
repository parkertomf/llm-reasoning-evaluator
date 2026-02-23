from evaluator.models import ModelWrapper
from evaluator.datasets import GSM8KDataset
from evaluator.utils import sort_output, print_statistics

def main():
    # These are hardcoded for now. Input capacity will be added with the addition of more options in the future.
    eval_count = 5
    dataset = GSM8KDataset(eval_count)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model_wrapper = ModelWrapper(model_name)

    # Prompt formatting for model
    messages = [{"role": "system", "content": dataset.base_prompt}]
    formatted_messages = []
    for q in dataset.questions:
        messages.append({"role": "user", "content": q})
        formatted_messages.append(
            model_wrapper.tokenizer.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True))
        messages.pop()
    model_inputs = model_wrapper.tokenizer(formatted_messages, padding=True,
                                           return_tensors="pt").to(model_wrapper.model.device)

    # Actually generate
    generated_ids = model_wrapper.model.generate(
        **model_inputs,
        max_new_tokens=16,
        do_sample=False,  # Consistent output. Greedy token generation
        # The following 3 params are default values and are irrelevant parameters when used with do_sample=False
        # They are set simply to suppress an info log.
        temperature=1.0,
        top_k=50,
        top_p=1.0)

    # Remove the prompt tokens from the output so only the model’s new answers remain.
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    model_responses = model_wrapper.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    correct, incorrect, extract_fails = sort_output(eval_count, model_responses, dataset)
    print_statistics(correct, incorrect, extract_fails, model_name, eval_count)


if __name__ == "__main__":
    main()
