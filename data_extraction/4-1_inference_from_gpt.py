import os, sys
import json
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI, APIError, RateLimitError

from clotho.exp_config import supported_datasets, task2template, task2input_key, task2answer_key_list

MODEL = "gpt-4o-mini"
NUM_INFERENCE_RUNS = 10

REPEAT = 10
TEMPERATURE = 0.8
MAX_TOKENS = 512
TOP_LOGPROBS = 5
RETRY_MAX = 6
RETRY_BASE_SLEEP = 1.5

def _openai_retry(fn, *args, **kwargs):
    """Simple exponential backoff wrapper."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            sleep = RETRY_BASE_SLEEP ** attempt
            time.sleep(sleep)
            if attempt == RETRY_MAX:
                raise
        except APIError as e:
            # Retry transient 5xx; re-raise others
            if getattr(e, "status", None) and int(e.status) >= 500:
                sleep = RETRY_BASE_SLEEP ** attempt
                time.sleep(sleep)
                if attempt == RETRY_MAX:
                    raise
            else:
                raise

def pack_choice(choice):
    text = choice.message.content or ""
    tokens = []
    if choice.logprobs and getattr(choice.logprobs, "content", None):
        for item in choice.logprobs.content:
            if hasattr(item, "token"):
                tokens.append({
                    "token": item.token,
                    "logprob": item.logprob,
                })
    return {
        "text": text,
        "tokens": tokens
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI gpt-4o-mini inference with obtaining logprobs.")
    parser.add_argument("--dataset_type", "-t", type=str, required=True, help="e.g., spell_check")
    parser.add_argument("--dataset_name", "-d", type=str, required=True, help="e.g., misspell_injected_wikipedia (jsonl)")
    parser.add_argument("--prompt_template", "-p", type=str, required=True, help="e.g., messages_template")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for decoding reproducibility")
    parser.add_argument("--n", type=int, default=NUM_INFERENCE_RUNS, help="Completions per input")
    parser.add_argument("--max_records", type=int, default=None, help="Debug: only process first N items")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    MODEL_NAME = 'gpt4o_mini'
    DATASET_TYPE = args.dataset_type
    DATASET_NAME = args.dataset_name # expected to be in jsonl format
    
    assert DATASET_TYPE in supported_datasets, f"Unsupported dataset type: {DATASET_TYPE}. Supported types: {list(supported_datasets.keys())}"
    assert DATASET_NAME in supported_datasets[DATASET_TYPE], f"Unsupported dataset name: {DATASET_NAME}. Supported names for {DATASET_TYPE}: {supported_datasets[DATASET_TYPE]}"
    
    prompt_template_name = args.prompt_template
    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")

    # Load dataset
    dataset = []
    with open(f'./dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    if args.max_records is not None:
        dataset = dataset[: args.max_records]
    
    output_result_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}.json'
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"[OpenAI] Running {MODEL} with {args.n} Inferences & logprobs")
    print(f"Writing to: {output_result_path}")

    all_records = []
    for input_index, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        if task2input_key[DATASET_TYPE] is None:
            input_vars = data
        else:
            input_vars = data[task2input_key[DATASET_TYPE]]
        
        answer_key_list = task2answer_key_list[DATASET_TYPE]
        GT = [data[answer_key] for answer_key in answer_key_list]
        
        messages = prompt_template_func(input_vars)
        
        inferences = _openai_retry(
            client.chat.completions.create,
            model=MODEL,
            messages=messages,
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=True,
            **({"seed": args.seed} if args.seed is not None else {})
        )

        inference_list= []
        for choice in inferences.choices:
            inference_list.append(pack_choice(choice))
        
        record = {
            "input_index": input_index,
            # "input_data": input_vars,
            # "prompt_messages": messages,
            "inferences": [pack_choice(choice) for choice in inferences.choices],
            "gt": GT,
        }
        all_records.append(record)
    
    with open(output_result_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)