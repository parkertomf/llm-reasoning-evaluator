from typing import List
from torch import inference_mode
from tqdm import tqdm
from evaluator.models import ModelWrapper
from evaluator.datasets import Gsm8kDataset


def run_eval(
    dataset: Gsm8kDataset,
    model_wrapper: ModelWrapper,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    """Format dataset questions for the model, run the inference loop, and decode the response."""
    decoded_responses = []
    with inference_mode():
        for i in tqdm(range(0, len(dataset.questions), batch_size), desc="Evaluating"):
            batch = dataset.questions[i:i + batch_size]
            formatted_prompts = [
                model_wrapper.tokenizer.apply_chat_template([{
                    "role": "system",
                    "content": dataset.base_prompt
                }, {
                    "role": "user",
                    "content": q
                }], tokenize=False, add_generation_prompt=True) for q in batch
            ]

            # Run inference (tokenization and generation).
            model_inputs = model_wrapper.tokenizer(
                formatted_prompts, padding=True, return_tensors="pt").to(model_wrapper.model.device)
            response_tokens = model_wrapper.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy token generation for consistent output.
                # The following 3 params are default values and irrelevant when used with do_sample=False; they are set to suppress an info log.
                temperature=1.0,
                top_k=50,
                top_p=1.0)

            # Remove the prompt tokens from the output so only the model’s new answers remain.
            new_tokens = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, response_tokens)
            ]

            decoded = model_wrapper.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            decoded_responses.extend(decoded)
    return decoded_responses
