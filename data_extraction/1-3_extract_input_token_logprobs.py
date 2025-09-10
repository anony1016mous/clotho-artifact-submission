from clotho.metrics.logprobs import calc_confidence_scores_input
from clotho.exp_config import task2template, task2input_key, supported_datasets

import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
import pandas as pd

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "gemma", "mistral"])
    args = parser.parse_args()

    model = args.model
    
    model_suffix = f'_{model}'
    tasks = list(supported_datasets.keys())
    
    if model == 'llama':
        import clotho.models.llama as llama_model_wrapper
        model_wrapper = llama_model_wrapper
        
    elif model == 'gemma':
        import clotho.models.gemma as gemma_model_wrapper
        model_wrapper = gemma_model_wrapper
    
    elif model == 'mistral':
        import clotho.models.mistral as mistral_model_wrapper
        model_wrapper = mistral_model_wrapper

    for task in tasks:
        dataset_names = supported_datasets[task]
        prompt_template_name = "messages_template"
        prompt_template_func = task2template[task][prompt_template_name]
        
        os.makedirs(f'../clotho/results{model_suffix}/{task}/precalculated_metrics/{prompt_template_name}', exist_ok=True)
        
        for dataset_name in dataset_names:
            print(f'* Processing {task} - {dataset_name}')
            
            result_df_path = f'../clotho/results{model_suffix}/{task}/precalculated_metrics/{prompt_template_name}/{dataset_name}_input_logprobs.pkl'
            
            input_key = task2input_key[task]
            dataset = []
            with open(f'../clotho/dataset/{task}/{dataset_name}.jsonl') as f:
                for line in f:
                    dataset.append(json.loads(line))
            
            rows = []
            for i, data in enumerate(tqdm(dataset)):
                if input_key is None:
                    input_variables = data
                else:
                    input_variables = data[input_key]
                
                input_prompt = prompt_template_func(input_variables)
                
                res_input = calc_confidence_scores_input(
                    model_wrapper=model_wrapper,
                    prompt=input_prompt,
                )
                
                rows.append({
                    'input_index': i,
                    'token_log_probs': res_input['input_log_probs'],
                    'average_log_probs': res_input['average_log_probs'],
                    'perplexity': res_input['perplexity']
                })
                
            result_df = pd.DataFrame(rows)
            result_df.to_pickle(result_df_path)
            print(f'* * Results saved to {result_df_path}')
