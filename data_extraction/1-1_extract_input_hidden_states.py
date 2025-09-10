import os

from clotho.exp_config import supported_datasets, task2template, task2input_key, task2template_variation

import argparse

import torch
import numpy as np
import json
import importlib

from tqdm import tqdm
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference or extract last input hidden states.")
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default='llama',
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
    if MODEL_NAME == 'llama': # num_layers=32
        from clotho.models import llama
        get_hidden_states = llama.get_hidden_states
        TARGET_LAYERS = [21, 23]
    elif MODEL_NAME == 'gemma': # num_layers=42
        from clotho.models import gemma
        get_hidden_states = gemma.get_hidden_states
        TARGET_LAYERS = [28]
    elif MODEL_NAME == 'mistral': # num_layers=32
        from clotho.models import mistral
        get_hidden_states = mistral.get_hidden_states
        TARGET_LAYERS = [16, 22]
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}. Available models: {['llama', 'gemma', 'mistral']}")

    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")
    
    dataset = []
    with open(f'./dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    if args.use_variation:
        template_variations = importlib.import_module(task2template_variation[DATASET_TYPE])
        try:
            prompt_variation_func, variations = template_variations.variation_map[prompt_template_name]
        except KeyError:
            raise ValueError(f"Unsupported prompt template variation: {prompt_template_name}. Available variations for {DATASET_TYPE}: {list(template_variations.variation_map.keys())}")

        result_dir = f'./results_{MODEL_NAME}/{DATASET_TYPE}/input_hidden_states/{prompt_template_name}/from_variations/{DATASET_NAME}/'
        
        os.makedirs(result_dir, exist_ok=True)
        
        hidden_vectors_by_layers = defaultdict(list)
        
        for data in tqdm(dataset):
            if task2input_key[DATASET_TYPE] is None:
                input_variables = data
            else:
                input_variables = data[task2input_key[DATASET_TYPE]]
            
            for variation in variations:
                template_func = lambda input_variables: prompt_variation_func(input_variables, variation)
                
                res = get_hidden_states(input_variables, template_func, produce_inference_results=False)
            
                for i, layer_hidden in enumerate(res['hidden_states']):
                    if i not in TARGET_LAYERS:
                        continue
                    
                    hidden_vectors_by_layers[(i, variation)].append(layer_hidden[0, -1].cpu().to(torch.float32).numpy())

        for (i, variation), hidden_vectors in hidden_vectors_by_layers.items():
            os.makedirs(os.path.join(result_dir, f'layer_{i}'), exist_ok=True)
            torch.save(np.array(hidden_vectors), os.path.join(result_dir, f'layer_{i}/hidden_vectors_{variation}.pt'))

        print('Finished extracting hidden states with variations.')
        exit(0)
    
    print(f"Extracting hidden states... ({DATASET_TYPE}, {DATASET_NAME}, {prompt_template_name})")
    
    if MODEL_NAME == 'llama':
        result_dir = f'./results/{DATASET_TYPE}/input_hidden_states/{prompt_template_name}/{DATASET_NAME}/'
    else:
        result_dir = f'./results_{MODEL_NAME}/{DATASET_TYPE}/input_hidden_states/{prompt_template_name}/{DATASET_NAME}/'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    hidden_vectors_by_layers = defaultdict(list)
        
    for data in tqdm(dataset):
        if task2input_key[DATASET_TYPE] is None:
            input_variables = data
        else:
            input_variables = data[task2input_key[DATASET_TYPE]]
        
        res = get_hidden_states(input_variables, prompt_template_func, produce_inference_results=False)
        
        # Only use the hidden states of the last token (TODO: try mean/max pooling as well)
        for i, layer_hidden in enumerate(res['hidden_states']):
            if i not in TARGET_LAYERS:
                continue
            
            hidden_vectors_by_layers[i].append(layer_hidden[0, -1].cpu().to(torch.float32).numpy())

    for i, hidden_vectors in hidden_vectors_by_layers.items():
        os.makedirs(os.path.join(result_dir, f'layer_{i}'), exist_ok=True)
        torch.save(np.array(hidden_vectors), os.path.join(result_dir, f'layer_{i}/hidden_vectors.pt'))
