from clotho.exp_config import supported_datasets, task2template, task2labeler

import argparse
import torch
import numpy as np
import pandas as pd
import json
import os
import glob
import re

from tqdm import tqdm
from collections import defaultdict

REPEAT = 10
TEMPERATURE = 0.8

def get_test_results(inference_results_w_GT, labeler, prompt_template_name):
    labels = []

    for test_result in inference_results_w_GT:
        actual_list = test_result['inferences']
        expected = test_result['GT']
        
        labels_per_input = []
        for actual in actual_list:
            # TODO: Consider error types? (e.g., formatting error vs. semantic error)
            is_correct, error = labeler(actual, expected, prompt_template_name)
            labels_per_input.append((is_correct, error))

        labels.append(labels_per_input)
    
    return labels

def calculate_variance_repeated_generations(input_tensor):
    mean_vector = torch.mean(input_tensor, dim=0, keepdim=True)
    centered_data = input_tensor - mean_vector

    n_samples =  input_tensor.size(0)
    cov_matrix = torch.mm(centered_data.t(), centered_data) / (n_samples - 1)
    total_variance = torch.trace(cov_matrix)
    
    return total_variance.item()


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
    parser.add_argument(
        "--use_variation",
        "-v",
        action='store_true',
        help="Use template variations if available (default: False)."
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
    
    output_labeler = task2labeler[DATASET_TYPE]

    inference_result_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}.json'
    
    output_state_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/output_hidden_states/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}/'
    
    with open(inference_result_path, 'r') as f:
        inference_results = json.load(f)
            
    print(f"Compute variance among repeated generations as consistency scores... ({DATASET_TYPE}, {DATASET_NAME}, {prompt_template_name})")
    
    labels = get_test_results(inference_results, output_labeler, prompt_template_name)
    labels_aggregated = [sum([l[0] for l in input_labels]) for input_labels in labels]
    
    print(f'Any correct ratio: {sum([1 for input_labels in labels if any([l[0] for l in input_labels])]) / len(labels)}')
    print(f'All correct ratio: {sum([1 for input_labels in labels if all([l[0] for l in input_labels])]) / len(labels)}')
    
    if args.use_variation:
        print(f"Variances are computed with prompt template variations enabled.")
        variation_inference_result_path = f'./results/{DATASET_TYPE}/inference_results/{prompt_template_name}/from_variations/{DATASET_NAME}_T{TEMPERATURE}.json'
        variation_output_state_path = f'./results/{DATASET_TYPE}/output_hidden_states/{prompt_template_name}/from_variations/{DATASET_NAME}_T{TEMPERATURE}/'
        
        with open(variation_inference_result_path, 'r') as f:
            variation_inference_results = json.load(f)
            
        variation_labels = get_test_results(variation_inference_results, output_labeler, prompt_template_name)
        variation_labels_aggregated = [sum([l[0] for l in input_labels]) for input_labels in variation_labels]
        
        print(f'Any correct ratio (variations): {sum([1 for input_labels in variation_labels if any([l[0] for l in input_labels])]) / len(variation_labels)}')
        print(f'All correct ratio (variations): {sum([1 for input_labels in variation_labels if all([l[0] for l in input_labels])]) / len(variation_labels)}')
        
        rows = []
        for target_setting_dir in tqdm(glob.glob(os.path.join(variation_output_state_path, "token_*_layer_*"))):
            m = re.search(r'token_(.+)_layer_(\d+)', os.path.basename(target_setting_dir))
            token_index = m.group(1)
            layer_index = int(m.group(2))
            
            # For each token index and layer setting:
            for repeated_states_file in glob.glob(os.path.join(target_setting_dir, 'input_*.pt')):
                m = re.search(r'input_(\d+)', os.path.basename(repeated_states_file))
                input_index = int(m.group(1))
                
                hidden_states_repeated = torch.load(repeated_states_file)
                
                # append first output state of the original generation
                original_output_state_file = os.path.join(output_state_path, f'token_{token_index}_layer_{layer_index}/input_{input_index}.pt')
                if not os.path.exists(original_output_state_file):
                    print(f"Original output state file not found: {original_output_state_file}")
                    assert 'token_IE' not in original_output_state_file, f"Original output state file for the inference end token not found: {original_output_state_file}"
                    continue
                    
                original_hidden_states_repeated = torch.load(original_output_state_file) # [REPEAT, 1, 4096]
                original_hidden_state = original_hidden_states_repeated[0] # [1, 4096]

                # [REPEAT-1, 1, 4096] -> [REPEAT-1, 4096]
                hidden_states_repeated = hidden_states_repeated.squeeze(1).to(torch.float32)
                
                hidden_states_repeated = torch.cat((original_hidden_state, hidden_states_repeated), dim=0)

                variance = calculate_variance_repeated_generations(hidden_states_repeated)
                score = variation_labels_aggregated[input_index] + int(labels[input_index][0][0])
                original_score = labels_aggregated[input_index]
                
                rows.append({
                    'layer_index': layer_index,
                    'input_index': input_index,
                    'token_index': token_index,
                    'variance': variance,
                    'score': score,
                    'original_score': original_score
                })
                
                del hidden_states_repeated
                
        df = pd.DataFrame(rows)
        
        result_df_path = f'./results/{DATASET_TYPE}/precalculated_metrics/{prompt_template_name}/from_variation_{DATASET_NAME}_T{TEMPERATURE}_output_state_variances.pkl'
        
        if not os.path.exists(os.path.dirname(result_df_path)):
            os.makedirs(os.path.dirname(result_df_path))
        
        df.to_pickle(result_df_path)
        print(f"Output state variances (with prompt variations) saved to {result_df_path}")
        exit(0)
    
    # Variance among generations with the same prompt template (known to suffer from 'confident but wrong' answer issue)
    rows = []
    for target_setting_dir in tqdm(glob.glob(os.path.join(output_state_path, "token_*_layer_*"))):
        m = re.search(r'token_(.+)_layer_(\d+)', os.path.basename(target_setting_dir))
        token_index = m.group(1)
        layer_index = int(m.group(2))
        
        # For each token index and layer setting:
        for repeated_states_file in glob.glob(os.path.join(target_setting_dir, 'input_*.pt')):
            m = re.search(r'input_(\d+)', os.path.basename(repeated_states_file))
            input_index = int(m.group(1))

            hidden_states_repeated = torch.load(repeated_states_file)

            # [REPEAT, 1, 4096] -> [REPEAT, 4096]
            hidden_states_repeated = hidden_states_repeated.squeeze(1).to(torch.float32)

            variance = calculate_variance_repeated_generations(hidden_states_repeated)
            score = labels_aggregated[input_index]
            
            rows.append({
                'layer_index': layer_index,
                'input_index': input_index,
                'token_index': token_index,
                'variance': variance,
                'score': score
            })
            
            del hidden_states_repeated
            
    df = pd.DataFrame(rows)
    
    result_df_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/precalculated_metrics/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}_output_state_variances.pkl'
    
    if not os.path.exists(os.path.dirname(result_df_path)):
        os.makedirs(os.path.dirname(result_df_path))
    
    df.to_pickle(result_df_path)
    print(f"Output state variances saved to {result_df_path}")
