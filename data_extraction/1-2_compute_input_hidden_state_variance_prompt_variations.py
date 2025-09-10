import os
from clotho.exp_config import supported_datasets, task2template, task2input_key, task2template_variation

import argparse

import torch
import numpy as np
import pandas as pd
import json
import importlib
import glob

from tqdm import tqdm
from collections import defaultdict

NUM_INFERENCE_RUNS = 10
target_layers = [11, 16, 21, 26]    # Observe only promising layers (to save storage space)

def sum_feature_variances(input_tensor):
    mean_vector = torch.mean(input_tensor, dim=0, keepdim=True)
    centered_data = input_tensor - mean_vector

    n_samples =  input_tensor.size(0)
    cov_matrix = torch.mm(centered_data.t(), centered_data) / (n_samples - 1)
    total_variance = torch.trace(cov_matrix)
    
    return total_variance.item()

def max_variance_direction(input_tensor):
    mean_vector = torch.mean(input_tensor, dim=0, keepdim=True)
    centered_data = input_tensor - mean_vector

    n_samples = input_tensor.size(0)
    cov_matrix = torch.mm(centered_data.t(), centered_data) / (n_samples - 1)

    # 대칭화 (symmetrize)
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.t())

    eigvals = torch.linalg.eigvalsh(cov_matrix)  # 대칭행렬용 고유값만 계산
    max_variance = torch.max(eigvals)
    return max_variance.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference or extract last input hidden states.")
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
    
    DATASET_TYPE = args.dataset_type
    DATASET_NAME = args.dataset_name # expected to be in jsonl format
    
    assert DATASET_TYPE in supported_datasets, f"Unsupported dataset type: {DATASET_TYPE}. Supported types: {list(supported_datasets.keys())}"
    assert DATASET_NAME in supported_datasets[DATASET_TYPE], f"Unsupported dataset name: {DATASET_NAME}. Supported names for {DATASET_TYPE}: {supported_datasets[DATASET_TYPE]}"
    
    prompt_template_name = args.prompt_template
    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")
    
    print(f"Compute input hidden states variances generated from prompt variations... ({DATASET_TYPE}, {DATASET_NAME}, {prompt_template_name})")
    
    input_hidden_states_path = f'./results/{DATASET_TYPE}/input_hidden_states/{prompt_template_name}/{DATASET_NAME}/'
    input_hidden_states_variance_path = f'./results/{DATASET_TYPE}/input_hidden_states/{prompt_template_name}/from_variations/{DATASET_NAME}'
    
    rows = []
    for l in target_layers:
        with open(f'{input_hidden_states_path}/layer_{l}/hidden_vectors.pt', 'rb') as f:
            original_hidden_states = torch.tensor(torch.load(f))
        
        hidden_states_variations = []
        for variation_file in glob.glob(f'{input_hidden_states_variance_path}/layer_{l}/*.pt'):
            with open(variation_file, 'rb') as f:
                hidden_states_variations.append(torch.tensor(torch.load(f)))
                
        for input_index in tqdm(list(range(len(hidden_states_variations[0]))), desc=f"Layer {l}"):
            input_states_variations = [original_hidden_states[input_index]] + [v[input_index] for v in hidden_states_variations]
            
            input_states_variance_feature_sum = sum_feature_variances(torch.stack(input_states_variations))
            input_states_variance_max_direction = max_variance_direction(torch.stack(input_states_variations))
            
            rows.append({
                'layer': l,
                'input_index': input_index,
                'input_states_variance_feature_sum': input_states_variance_feature_sum,
                'input_states_variance_max_direction': input_states_variance_max_direction
            })
    
    result_path = f'./results/{DATASET_TYPE}/precalculated_metrics/{prompt_template_name}/from_variation_{DATASET_NAME}_input_state_variances.pkl'
    
    df = pd.DataFrame(rows)
    df.to_pickle(result_path)
    print(f"Input hidden states variances saved to {result_path}")
    
