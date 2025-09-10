from clotho.preprocessing.feature_reduction import PCAFeatureReducer
from clotho.metrics.sa import SA
import clotho.dataset as clotho_dataset

from utils import load_input_hidden_states, get_test_results, target_testsuites, prompt_templates

from sklearn.decomposition import PCA

import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import os
import json
import numpy as np

from tqdm import tqdm

import argparse

def simulate_refset_based_SA(target_task, target_layer, pass_threshold=0.5):
    rows = []
    
    with open(f'./initial_tests{model_name_prefix}/{target_task}/inference_results.json', 'r') as f:
        inference_results = json.load(f)
    LIH_init = torch.load(f'./initial_tests{model_name_prefix}/{target_task}/hidden_vectors_layer_{target_layer}.pt', weights_only=False)

    hidden_states = load_input_hidden_states(target_task, target_layer, model=model)
    _, _, test_scores = get_test_results(model, target_task)

    test_scores = np.concatenate((test_scores, inference_results['test_scores']), axis=0)
    LIH = np.concatenate((hidden_states, LIH_init), axis=0)

    assert len(LIH) == len(test_scores)

    dim_reducer = PCAFeatureReducer(n_components=n_features)
    dim_reducer.fit(LIH)

    for seed in tqdm(list(range(10)), desc="10 random seeds"):
        for refset_size in refset_sizes:
            reference_set_indices = random.sample(range(hidden_states.shape[0]), refset_size - len(LIH_init))

            passing_reference_set_indices = [i for i in reference_set_indices if test_scores[i] > pass_threshold]
            passing_initial_set_indices = [i for i in range(LIH_init.shape[0]) if inference_results['test_scores'][i] > pass_threshold]
            
            scores = {}
            for mode in ['MDSA', 'MLSA', 'DSA']:
                sa = SA(mode=mode, dimension_reducer=dim_reducer)
                sa.register_reference_set(np.concatenate((hidden_states[passing_reference_set_indices], LIH_init[passing_initial_set_indices]), axis=0))

                scores[mode] = sa.calculate(LIH)
            
            for input_index, test_score in enumerate(test_scores):
                rows.append({
                    'task': target_task,
                    'layer': target_layer,
                    'input_index': input_index,
                    'test_score': test_score,
                    'reference_set_size': refset_size,
                    'label': 'ref' if input_index in reference_set_indices else 'test',
                    'seed': seed,
                    'pred_score_MDSA': scores['MDSA'][input_index],
                    'pred_score_MLSA': scores['MLSA'][input_index],
                    'pred_score_DSA': scores['DSA'][input_index],
                    'n_features': n_features
                })
                
    del hidden_states
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refset_sizes", type=int, nargs='+', default=[100, 200, 300, 400, 500])
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--model", type=str, default='llama')
    parser.add_argument("--pass_threshold", type=float, default=0.5)

    args = parser.parse_args()

    refset_sizes = args.refset_sizes
    n_features = args.n_features
    pass_threshold = args.pass_threshold
    model = args.model

    target_layer_map = {
        'llama': 21,
        'gemma': 28,
        'mistral': 22
    }
    
    model_name_prefix = f'_{model}'
    target_layer = target_layer_map[model]

    for task in tqdm(target_testsuites, desc="Simulating Reference Set Based SA"):
        print(f'Processing task {task}')
        result_df = simulate_refset_based_SA(task, target_layer, pass_threshold=pass_threshold)
        os.makedirs(f"../clotho/results{model_name_prefix}/{task}/precalculated_metrics", exist_ok=True)
        result_df.to_pickle(f"../clotho/results{model_name_prefix}/{task}/precalculated_metrics/LIH_unweighted_SA_refset_layer_{target_layer}_{n_features}_{pass_threshold}.pkl")