import os

from clotho.metrics.logprobs import calc_confidence_scores, calc_average_entropy
from clotho.exp_config import supported_datasets, task2template, task2input_key, task2answer_key_list

import argparse
import torch
import numpy as np
import pandas as pd
import json
import glob
import re

from tqdm import tqdm
from collections import defaultdict

REPEAT = 10
TEMPERATURE = 0.8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference or extract last input hidden states.")
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help= "Model to use for inference (ex: llama)",
    )
    parser.add_argument(
        "--dataset_type",
        "-t",
        type=str,
        help="Type of dataset to use (ex: spell_check)."
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        help="Name of the dataset to use (ex: misspell_injected_wikipedia)."
    )
    parser.add_argument(
        "--prompt_template",
        "-p",
        type=str,
        help="Name of the prompt template to use (ex: messages_template)."
    )
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_TYPE = args.dataset_type
    DATASET_NAME = args.dataset_name # expected to be in jsonl format
    
    assert DATASET_TYPE in supported_datasets, f"Unsupported dataset type: {DATASET_TYPE}. Supported types: {list(supported_datasets.keys())}"
    assert DATASET_NAME in supported_datasets[DATASET_TYPE], f"Unsupported dataset name: {DATASET_NAME}. Supported names for {DATASET_TYPE}: {supported_datasets[DATASET_TYPE]}"
    
    prompt_template_name = args.prompt_template
    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")

    dataset = []
    with open(f'./dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    if MODEL_NAME == 'llama':
        import clotho.models.llama as model_wrapper
    elif MODEL_NAME == 'gemma':
        import clotho.models.gemma as model_wrapper
    elif MODEL_NAME == 'mistral':
        import clotho.models.mistral as model_wrapper
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}. Available models: {['llama', 'gemma', 'mistral']}")

    model_suffix = "" if MODEL_NAME == "llama" else f"_{MODEL_NAME}"
    inference_result_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}.json'
    
    with open(inference_result_path, 'r') as f:
        inference_results = json.load(f)
            
    print(f"Compute log probabilities + token entropy for each generation as confidence scores... ({DATASET_TYPE}, {DATASET_NAME}, {prompt_template_name})")
    
    result_conf = []
    result_ent = []
    for i, (data, test_result) in enumerate(zip(tqdm(dataset), inference_results)):
        if task2input_key[DATASET_TYPE] is None:
            test_input = data
        else:
            test_input = data[task2input_key[DATASET_TYPE]]

        answer_key_list = task2answer_key_list[DATASET_TYPE]
        GT = [data[answer_key] for answer_key in answer_key_list]
          
        inferences = test_result['inferences']
        
        prompt = prompt_template_func(test_input)
        
        for j, output in enumerate(inferences):
            confidence_score_results = calc_confidence_scores(model_wrapper, prompt, output)
            entropy_results = calc_average_entropy(model_wrapper, prompt, output)
            result_conf.append({
                'input_index': i,
                'inference_index': j,
                'average_log_probs': confidence_score_results['average_log_probs'],
                'output_log_props': confidence_score_results['output_log_probs'],
                'perplexity': confidence_score_results['perplexity'],
            })
            result_ent.append({
                'input_index': i,
                'inference_index': j,
                'per_token_entropies': entropy_results['per_token_entropies'],
                'average_entropy': entropy_results['average_entropy'],
                'gt': GT,
            })
            
    conf_df = pd.DataFrame(result_conf)
    conf_df_path = f'./results{model_suffix}/{DATASET_TYPE}/precalculated_metrics/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}_log_probs.pkl'
    
    if not os.path.exists(os.path.dirname(conf_df_path)):
        os.makedirs(os.path.dirname(conf_df_path))
    
    conf_df.to_pickle(conf_df_path)

    ent_df = pd.DataFrame(result_ent)
    ent_df_path = f'./results{model_suffix}/{DATASET_TYPE}/precalculated_metrics/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}_average_token_entropy.pkl'
    
    if not os.path.exists(os.path.dirname(ent_df_path)):
        os.makedirs(os.path.dirname(ent_df_path))

    ent_df.to_pickle(ent_df_path)
    
    print(f"Log probabilities and token entropy saved to {conf_df_path}")