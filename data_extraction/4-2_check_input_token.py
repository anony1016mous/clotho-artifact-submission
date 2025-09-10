import os
import json
import argparse
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
import tiktoken

from clotho.exp_config import supported_datasets, task2template, task2input_key, task2answer_key_list

MODEL = "gpt-4o-mini"
def get_encoder_for_model(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def normalize_message_content(content):
    """Flatten message content to plain text (string or list-of-dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            txt = part.get("text") if isinstance(part, dict) else None
            if isinstance(txt, str):
                parts.append(txt)
        return "\n".join(parts)
    return str(content)

def count_prompt_tokens(messages, model_name=MODEL) -> int:
    enc = get_encoder_for_model(model_name)
    buf = []
    for m in messages:
        role = m.get("role", "user")
        content_text = normalize_message_content(m.get("content", ""))
        buf.append(f"{role}: {content_text}")
    text = "\n".join(buf) + "\n"
    return len(enc.encode(text))

def main():
    parser = argparse.ArgumentParser(description="Count tokens of prompts using tiktoken.")
    parser.add_argument("--dataset_type", "-t", type=str, required=True, help="e.g., spell_check")
    parser.add_argument("--dataset_name", "-d", type=str, required=True, help="e.g., misspell_injected_wikipedia (jsonl without extension)")
    parser.add_argument("--prompt_template", "-p", type=str, required=True, help="e.g., messages_template")
    parser.add_argument("--max_records", type=int, default=None, help="Only process first N items (debug)")
    args = parser.parse_args()

    DATASET_TYPE = args.dataset_type
    DATASET_NAME = args.dataset_name
    prompt_template_name = args.prompt_template

    assert DATASET_TYPE in supported_datasets, \
        f"Unsupported dataset type: {DATASET_TYPE}. Supported: {list(supported_datasets.keys())}"
    assert DATASET_NAME in supported_datasets[DATASET_TYPE], \
        f"Unsupported dataset name: {DATASET_NAME}. Supported: {supported_datasets[DATASET_TYPE]}"

    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(
            f"Unsupported prompt template: {prompt_template_name}. "
            f"Available: {list(task2template[DATASET_TYPE].keys())}"
        )

    # Load dataset
    dataset = []
    with open(f'./dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    if args.max_records:
        dataset = dataset[: args.max_records]

    print(f"Counting tokens for {len(dataset)} samples with model={MODEL}...")
    total_tokens = 0

    for input_index, data in enumerate(tqdm(dataset, desc="Tokenizing")):
        if task2input_key[DATASET_TYPE] is None:
            input_vars = data
        else:
            input_vars = data[task2input_key[DATASET_TYPE]]

        messages = prompt_template_func(input_vars)
        prompt_tokens = count_prompt_tokens(messages, MODEL)
        total_tokens += prompt_tokens

        print(f"[{input_index}] Prompt tokens = {prompt_tokens}")

    print("="*50)
    print(f"Total tokens across dataset: {total_tokens}")
    print(f"Average per sample: {total_tokens/len(dataset):.2f}")

if __name__ == "__main__":
    main()