from clotho.exp_config import task2template, task2labeler
from scipy.stats import pearsonr, rankdata
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import glob
import json
import os

import numpy as np

import logging

prompt_templates = {
    'syntactic_bug_detection': 'messages_template',
    'spell_check': 'messages_template',
    'github_typo_check': 'messages_template',
    'json_repair': 'messages_template',
    'pos_detection': 'messages_template',
    'topic_classification': 'messages_template',
    'adding_odd_numbers': 'messages_template',
    'model_name_extraction': 'messages_template'
}

target_testsuites = {
    'syntactic_bug_detection': ['syntactic_bug_injected'],
    'spell_check': ['misspell_injected_wordnet'],
    'github_typo_check': ['github_typo_corpus_cleaned'],
    'json_repair': ['invalid_json_dataset_2166', 'invalid_json_dataset_4397'],
    'pos_detection': ['cleaned_and_sampled_pos_tags', 'cleaned_and_sampled_pos_tags_trainset'],
    'topic_classification': ['ag_news_test'],
    'adding_odd_numbers': ['integer_sequences_length_1_to_10_uniform'],
    'model_name_extraction': ['ml_arxiv_papers_no_conflicting_labels', 'synthetic_abstracts_gpt4o_3600']
}


def convert_to_normalized_rank_score(scores):
    ranks = rankdata(scores, method='average')
    ranks_norm = (ranks - 1) / len(ranks) - 1
    return ranks_norm


def calc_pearson_correlation(inverse_adequacy_scores, test_result_scores):
    assert len(inverse_adequacy_scores) == len(test_result_scores), "Length mismatch between inverse adequacy scores and test result scores ({} vs {})".format(
        len(inverse_adequacy_scores), len(test_result_scores))
    
    corr, pval = pearsonr(inverse_adequacy_scores, test_result_scores)
    
    return {
        'pearson_correlation': corr,
        'p_value': pval
    }


def label_test_outputs(inference_results_w_GT, labeler, prompt_template_name):
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


def get_test_results_all_tasks(num_inferences=10, temperature=0.8, return_integer_scores=False):
    inference_results = {}
    labels = {}
    test_scores = {}
    
    for task in tqdm(target_testsuites.keys(), desc="Processing datasets"):
        _inference_results, _labels, _test_scores = get_test_results(task, num_inferences=num_inferences, temperature=temperature, return_integer_scores=return_integer_scores)

        inference_results[task] = _inference_results
        labels[task] = _labels
        test_scores[task] = _test_scores

    return inference_results, labels, test_scores


def get_test_results(model, task, num_inferences=10, temperature=0.8):
    prompt_template_name = prompt_templates[task]
    dataset_names = target_testsuites[task]
    output_labeler = task2labeler[task]
    
    results_suffix = '' if model == 'llama' else f'_{model}'
    inference_results = []
    for dataset_name in dataset_names:
        inference_result_path = f'../clotho/results{results_suffix}/{task}/inference_results/{prompt_template_name}/{dataset_name}_R{num_inferences}_T{temperature}.json'
        
        with open(inference_result_path, 'r') as f:
            inference_results.extend(json.load(f))
    
        if model == 'gpt4o_mini':
            labels = label_test_outputs_closed(inference_results, output_labeler, prompt_template_name)
        else:
            labels = label_test_outputs(inference_results, output_labeler, prompt_template_name)
    
    test_scores = [sum([label[0] for label in labels[i]]) / len(labels[i]) for i in range(len(labels))]
    return inference_results, labels, test_scores
        

def load_input_hidden_states(task, target_layer, model='llama', from_variations=False):
    hidden_vectors_all = []
    results_suffix = '' if model == 'llama' else f'_{model}'

    for dataset_name in target_testsuites[task]:
        input_hidden_states_path = f'../clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/{dataset_name}/layer_{target_layer}/hidden_vectors.pt'

        if from_variations:
            input_hidden_states_path = f'../clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/from_variations/{dataset_name}/layer_{target_layer}/hidden_vectors.pt'

        if not os.path.exists(input_hidden_states_path):
            raise FileNotFoundError(f"Input hidden states path does not exist: {input_hidden_states_path}")
        
        with open(input_hidden_states_path, 'rb') as f:
            hidden_vectors_all.append(torch.tensor(torch.load(f, weights_only=False)))

    hidden_vectors = torch.cat(hidden_vectors_all, dim=0)
    return hidden_vectors


def load_input_hidden_states_target_dataset(task, target_layer, target_dataset_name, from_variations=False):
    input_hidden_states_path = f'../clotho/results/{task}/input_hidden_states/{prompt_templates[task]}/{target_dataset_name}/layer_{target_layer}/hidden_vectors.pt'

    if from_variations:
        input_hidden_states_path = f'../clotho/results/{task}/input_hidden_states/{prompt_templates[task]}/from_variations/{target_dataset_name}/layer_{target_layer}/hidden_vectors.pt'

    if not os.path.exists(input_hidden_states_path):
        raise FileNotFoundError(f"Input hidden states path does not exist: {input_hidden_states_path}")
        
        with open(input_hidden_states_path, 'rb') as f:
            hidden_vectors = torch.tensor(torch.load(f, weights_only=False))

    return hidden_vectors


def load_output_hidden_states(task, target_layer, token='IE', aggregate_scheme='first', num_inferences=10, temperature=0.8):
    hidden_vectors_all = []
    
    for dataset_name in target_testsuites[task]:
        hidden_vectors_per_dataset = {}
        
        output_hidden_states_path = f'../clotho/results/{task}/output_hidden_states/{prompt_templates[task]}/{dataset_name}_R{num_inferences}_T{temperature}'
        
        for inputwise_hidden_states_path in glob.glob(os.path.join(output_hidden_states_path, f'token_{token}_layer_{target_layer}', 'input_*.pt')):
            input_index = int(os.path.basename(inputwise_hidden_states_path).split('_')[-1].split('.')[0])
            
            with open(inputwise_hidden_states_path, 'rb') as f:
                repeated_states = torch.load(f)
                repeated_states = repeated_states.squeeze(1).to(torch.float32)
            
            if aggregate_scheme == 'mean':
                state = repeated_states.mean(dim=0, keepdim=True)
            elif aggregate_scheme == 'first':
                state = repeated_states[0:1, :]
            
            hidden_vectors_per_dataset[input_index] = state
            
        max_input_index = max(hidden_vectors_per_dataset.keys())
        assert len(hidden_vectors_per_dataset) == max_input_index + 1, f"Missing input indices in {dataset_name} for token {token} at layer {target_layer}: {len(hidden_vectors_per_dataset)} != {max_input_index + 1}"
            
        ordered_list = sorted([(hidden_index, hidden_vector) for hidden_index, hidden_vector in hidden_vectors_per_dataset.items()], key=lambda x: x[0])
        ordered_list = [hidden_vector for _, hidden_vector in ordered_list]

        hidden_vectors_all.extend(ordered_list)

    hidden_vectors = torch.cat(hidden_vectors_all, dim=0)
    return hidden_vectors


def convert_test_score_to_integer(test_scores, num_inferences=10):
    if len(test_scores) == 0:
        return []
    if not isinstance(test_scores[0], float):
        raise TypeError("Test scores to be converted must be floats.")

    return [int((score + 1e-3) * num_inferences) for score in test_scores]

    
def get_vis_components(vectors, random_state=42):
    pca = PCA(n_components=10, random_state=random_state)
    tsne = TSNE(n_components=2, random_state=random_state)
    vectors_pca = pca.fit_transform(vectors)
    vectors_tsne = tsne.fit_transform(vectors_pca)
    
    return vectors_tsne


def get_gmm_components_perplexity(component_weights):
    pi = component_weights
    return np.exp(-(pi[pi>0] * np.log(pi[pi>0])).sum())