import os
import torch

from clotho.exp_config import supported_datasets, task2template, task2input_key, task2answer_key_list, task2template_variation, task2template_keywords

import argparse

import gc
import numpy as np
import json
import importlib

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
        help="Name of the dataset to use (ex: misspell_injected_wordnet)."
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
    
    dataset = []
    with open(f'./dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    
    prompt_template_name = args.prompt_template
    if MODEL_NAME == 'llama':
        from clotho.models import llama
        get_hidden_states_during_generation = llama.get_hidden_states_during_generation
        TARGET_LAYERS = [16, 23]
    elif MODEL_NAME == 'gemma':
        from clotho.models import gemma
        get_hidden_states_during_generation = gemma.get_hidden_states_during_generation
        TARGET_LAYERS = [21, 28]
    elif MODEL_NAME == 'mistral':
        from clotho.models import mistral
        get_hidden_states_during_generation = mistral.get_hidden_states_during_generation
        TARGET_LAYERS = [16, 22]
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}. Available models: {['llama', 'gemma', 'mistral']}")
    
    try:
        prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    except KeyError:
        raise ValueError(f"Unsupported prompt template: {prompt_template_name}. Available templates for {DATASET_TYPE}: {list(task2template[DATASET_TYPE].keys())}")
    
    # Use prompt variations for inference (for even more diverse outputs)
    if args.use_variation:
        template_variations = importlib.import_module(task2template_variation[DATASET_TYPE])
        try:
            prompt_variation_func, variations = template_variations.variation_map[prompt_template_name]
        except KeyError:
            raise ValueError(f"Unsupported prompt template variation: {prompt_template_name}. Available variations for {DATASET_TYPE}: {list(template_variations.variation_map.keys())}")
        
        output_result_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/from_variations/{DATASET_NAME}_T{TEMPERATURE}.json'
            
        state_result_dir = f'./results_{MODEL_NAME}/{DATASET_TYPE}/output_hidden_states/{prompt_template_name}/from_variations/{DATASET_NAME}_T{TEMPERATURE}/'
        
        os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
        os.makedirs(state_result_dir, exist_ok=True)
        
        if os.path.exists(output_result_path):
            print(f'Loading existing results from {output_result_path}')
            with open(output_result_path, 'r') as f:
                inference_results_w_GT = json.load(f)
        else:
            inference_results_w_GT = []
        for input_index, data in enumerate(tqdm(dataset)):
            if input_index < len(inference_results_w_GT):
                print(f'Skipping input {input_index} as results already exist.')
                continue
            
            if task2input_key[DATASET_TYPE] is None:
                input_variables = data
            else:
                input_variables = data[task2input_key[DATASET_TYPE]]
            answer_key_list = task2answer_key_list[DATASET_TYPE]
            GT = [data[answer_key] for answer_key in answer_key_list]
            
            inferences = []
            # generated_tokens = []
            hidden_state_snapshots = []
            for variation in variations:
                template_func = lambda input_variables: prompt_variation_func(input_variables, variation)
                
                res = get_hidden_states_during_generation(input_variables, template_func, repeat=1, target_layers=TARGET_LAYERS, temperature=TEMPERATURE, template_keywords=None)
                inferences.append(res['inference_results'][0])  # Only one inference per variation
                # generated_tokens.append(res['generated_tokens'][0])  # Only one generated token per variation
                
                hidden_state_snapshots.append(res['hidden_state_snapshots'][0])
            
            repeated_states = defaultdict(list)
            for per_repeat_result in hidden_state_snapshots:
                for token_index, state_dict in per_repeat_result.items():
                    for layer_index, hidden_state in state_dict:
                        repeated_states[(token_index, layer_index)].append(hidden_state)
            
            for (token_index, layer_index), hidden_states in repeated_states.items():
                if len(hidden_states) < len(variations):
                    continue
                
                hidden_states_tensor = torch.stack(hidden_states, dim=0)
                os.makedirs(os.path.join(state_result_dir, f'token_{token_index}_layer_{layer_index}'), exist_ok=True)
                torch.save(hidden_states_tensor, os.path.join(state_result_dir, f'token_{token_index}_layer_{layer_index}/input_{input_index}.pt')) # repeated hidden states for each input, token index, and layer
                
                del hidden_states
                del hidden_states_tensor
            
            del res
            del repeated_states
            del hidden_state_snapshots
            gc.collect()
            # torch.cuda.empty_cache()
        
            inference_results_w_GT.append({
                'inferences': inferences,
                # 'generated_tokens': generated_tokens,
                'GT': GT
            })
            
            with open(output_result_path, 'w') as f:
                json.dump(inference_results_w_GT, f, indent=4)
            
        print('Finished generating outputs and recording hidden states with prompt variations.')
        exit(0)
    
    # default mode: repeated inferences with the original prompt template
    print(f"Generate {REPEAT} inferences for each input and record hidden states while generation... ({DATASET_TYPE}, {DATASET_NAME}, {prompt_template_name})")

    output_result_path = f'./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}.json'
    state_result_dir = f'./results_{MODEL_NAME}/{DATASET_TYPE}/output_hidden_states/{prompt_template_name}/{DATASET_NAME}_R{REPEAT}_T{TEMPERATURE}/'
            
    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    os.makedirs(state_result_dir, exist_ok=True)

    if os.path.exists(output_result_path):
        print(f'Loading existing results from {output_result_path}')
        with open(output_result_path, 'r') as f:
            inference_results_w_GT = json.load(f)
    else:
        inference_results_w_GT = []

    for input_index, data in enumerate(tqdm(dataset)):
        if input_index < len(inference_results_w_GT):
            print(f'Skipping input {input_index} as results already exist.')
            continue
        
        if task2input_key[DATASET_TYPE] is None:
            input_variables = data
        else:
            input_variables = data[task2input_key[DATASET_TYPE]]
        answer_key_list = task2answer_key_list[DATASET_TYPE]
        GT = [data[answer_key] for answer_key in answer_key_list]
        
        res = get_hidden_states_during_generation(input_variables, prompt_template_func, repeat=REPEAT, target_layers=TARGET_LAYERS, temperature=TEMPERATURE, template_keywords=None)
        
        repeated_states = defaultdict(list)
        for per_repeat_result in res["hidden_state_snapshots"]:
            for token_index, state_dict in per_repeat_result.items():
                for layer_index, hidden_state in state_dict:
                    repeated_states[(token_index, layer_index)].append(hidden_state)
        
        for (token_index, layer_index), hidden_states in repeated_states.items():
            if len(hidden_states) < REPEAT:
                continue  # Skip if not enough repetitions (some generations are shorter)
            
            hidden_states_tensor = torch.stack(hidden_states, dim=0)
            
            os.makedirs(os.path.join(state_result_dir, f'token_{token_index}_layer_{layer_index}'), exist_ok=True)
            torch.save(hidden_states_tensor, os.path.join(state_result_dir, f'token_{token_index}_layer_{layer_index}/input_{input_index}.pt')) # repeated hidden states for each input, token index, and layer
            # print("Saved tensor shape:", hidden_states_tensor.shape)
            del hidden_states
            del hidden_states_tensor
            
        inference_results_w_GT.append({
            'inferences': res['inference_results'],
            # 'generated_tokens': res['generated_tokens'],
            'GT': GT
        })

        del res
        del repeated_states
        gc.collect()
        # torch.cuda.empty_cache()
        
        if input_index % 10 == 0:
            print(f'Memory Status for Input {input_index}:', torch.cuda.memory_summary(device=None, abbreviated=True))
                                    
        with open(output_result_path, 'w') as f:
            json.dump(inference_results_w_GT, f, indent=4)
