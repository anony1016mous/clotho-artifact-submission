from clotho.exp_config import supported_datasets, task2template, task2input_key, task2labeler

from tqdm import tqdm

import torch
import numpy as np

def from_input(input_variables, task, prompt_template_name='messages_template', num_inference_runs=1, target_layers=[11, 16, 21, 26], produce_inference_results=True, model='llama'):
    assert task in supported_datasets, f"Unsupported task: {task}. Supported tasks: {list(supported_datasets.keys())}"
    assert prompt_template_name in task2template[task], f"Unsupported prompt template: {prompt_template_name}. Available templates for {task}: {list(task2template[task].keys())}"
    
    prompt_template_func = task2template[task][prompt_template_name]
    
    input_key = task2input_key[task]
    if input_key is None:
        input_variables = input_variables
    else:
        input_variables = input_variables[input_key]
    
    if model == 'llama':
        import clotho.models.llama as llama
        get_hidden_states = llama.get_hidden_states
        
    elif model == 'gemma':
        import clotho.models.gemma as gemma 
        get_hidden_states = gemma.get_hidden_states
        
    elif model == 'mistral':
        import clotho.models.mistral as mistral
        get_hidden_states = mistral.get_hidden_states
    
    res = get_hidden_states(input_variables, prompt_template_func, produce_inference_results=produce_inference_results, num_inference_runs=num_inference_runs)

    layer2hidden = {i: res['hidden_states'][i][0, -1].cpu().to(torch.float32).numpy() for i in target_layers}
    
    if produce_inference_results:
        return res['inference_results'], layer2hidden
    else:
        return None, layer2hidden


def evaluate(inference_outputs, expected_output, task, prompt_template_name):
    output_labeler = task2labeler[task]
    
    label_results = []
    for actual_output in inference_outputs:
        is_correct, error = output_labeler(actual_output, expected_output, prompt_template_name)
        label_results.append(is_correct)
        
    test_score = sum(label_results) / len(label_results)
    return test_score, label_results
    