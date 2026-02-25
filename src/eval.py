from evaluator.models import ModelWrapper
from evaluator.datasets import GSM8KDataset
from evaluator.utils import sort_output, print_statistics
from torch import inference_mode
from tqdm import tqdm


def main():
    # These are hardcoded for now. Input capacity will be added with the addition of more options in the future.
    eval_count = 500
    dataset = GSM8KDataset(eval_count)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model_wrapper = ModelWrapper(model_name)

    BATCH_SIZE = 256  # Handle questions in batches to avoid running out of memory.
    decoded_responses = []
    with inference_mode():
        for i in tqdm(range(0, len(dataset.questions), BATCH_SIZE), desc="Evaluating"):
            batch = dataset.questions[i:i + BATCH_SIZE]
            formatted_prompts = [
                model_wrapper.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": dataset.base_prompt},
                        {"role": "user", "content": q}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ) for q in batch
            ]

            # Run inference (tokenization and generation).
            model_inputs = model_wrapper.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(model_wrapper.model.device)
            response_tokens = model_wrapper.model.generate(
                **model_inputs,
                max_new_tokens=16,
                do_sample=False,  # Use greedy token generation for consistent output.
                # The following 3 params are default values and irrelevant when used with do_sample=False; they are set to suppress an info log.
                temperature=1.0,
                top_k=50,
                top_p=1.0)
            
            # Remove the prompt tokens from the output so only the model’s new answers remain.
            new_tokens = response_tokens[:, model_inputs.input_ids.shape[-1]:]

            decoded = model_wrapper.tokenizer.batch_decode(new_tokens.cpu(), skip_special_tokens=True)
            decoded_responses.extend(decoded)

    correct, incorrect, extract_fails = sort_output(decoded_responses, dataset)
    print_statistics(correct, incorrect, extract_fails, model_name)


if __name__ == "__main__":
    main()
