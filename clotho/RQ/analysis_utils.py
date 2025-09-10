from clotho.exp_config import task2template, task2labeler
from scipy.stats import pearsonr
from tqdm import tqdm

import torch
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import rankdata

cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"], N=11) 

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

def label_test_outputs(inference_results_w_GT, labeler, prompt_template_name):
    labels = []

    for test_result in inference_results_w_GT:
        actual_list = test_result['inferences']
        expected = test_result['GT']
        
        labels_per_input = []
        for actual in actual_list:
            is_correct, error = labeler(actual, expected, prompt_template_name)
            labels_per_input.append((is_correct, error))

        labels.append(labels_per_input)
    
    return labels

def label_test_outputs_closed(inference_results_w_GT, labeler, prompt_template_name):
    labels = []

    for test_result in inference_results_w_GT:
        actual_list = test_result['inferences']
        expected = test_result['gt']
        
        labels_per_input = []
        for actual in actual_list:
            actual = actual['text']
            is_correct, error = labeler(actual, expected, prompt_template_name)
            labels_per_input.append((is_correct, error))

        labels.append(labels_per_input)
    
    return labels

def get_test_results_all_tasks(model='llama', num_inferences=10, temperature=0.8):
    inference_results = {}
    labels = {}
    test_scores = {}

    results_suffix = f'_{model}'

    for dataset_type in tqdm(target_testsuites.keys(), desc="Processing datasets"):
        prompt_template_name = prompt_templates[dataset_type]
        dataset_names = target_testsuites[dataset_type]
        output_labeler = task2labeler[dataset_type]
        
        inference_results[dataset_type] = []
        for dataset_name in dataset_names:
            inference_result_path = f'../clotho/results{results_suffix}/{dataset_type}/inference_results/{prompt_template_name}/{dataset_name}_R{num_inferences}_T{temperature}.json'

            with open(inference_result_path, 'r') as f:
                inference_results[dataset_type].extend(json.load(f))
            
        labels[dataset_type] = label_test_outputs(
            inference_results[dataset_type],
            output_labeler,
            prompt_template_name
        )
        
        test_scores[dataset_type] = [sum([label[0] for label in labels[dataset_type][i]]) / len(labels[dataset_type][i]) for i in range(len(labels[dataset_type]))]
        
    return inference_results, labels, test_scores


def get_test_results(model, task, num_inferences=10, temperature=0.8):
    prompt_template_name = prompt_templates[task]
    dataset_names = target_testsuites[task]
    output_labeler = task2labeler[task]
    
    results_suffix = f'_{model}'
    inference_results = []
    if model in ['gpt4o_mini', 'claude-3-5-haiku-20241022', 'gemini-2.5-flash-lite']:
        inference_result_path = f'../clotho/results{results_suffix}/{task}/inference_results/{prompt_template_name}/concat_dataset_R{num_inferences}_T{temperature}.json'
        with open(inference_result_path, 'r') as f:
            inference_results.extend(json.load(f))
    else:    
        for dataset_name in dataset_names:
            inference_result_path = f'../clotho/results{results_suffix}/{task}/inference_results/{prompt_template_name}/{dataset_name}_R{num_inferences}_T{temperature}.json'
            with open(inference_result_path, 'r') as f:
                inference_results.extend(json.load(f))
    
    if model in ['gpt4o_mini', 'claude-3-5-haiku-20241022', 'gemini-2.5-flash-lite']:
        labels = label_test_outputs_closed(inference_results, output_labeler, prompt_template_name)
    else:
        labels = label_test_outputs(inference_results, output_labeler, prompt_template_name)
    
    test_scores = [sum([label[0] for label in labels[i]]) / len(labels[i]) for i in range(len(labels))]
    return inference_results, labels, test_scores
        

def load_input_hidden_states(task, target_layer, model='llama', from_variations=False):
    hidden_vectors_all = []
    results_suffix = f'_{model}'

    for dataset_name in target_testsuites[task]:
        input_hidden_states_path = f'/root/workspace/clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/{dataset_name}/layer_{target_layer}/hidden_vectors.pt'

        if from_variations:
            input_hidden_states_path = f'../clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/from_variations/{dataset_name}/layer_{target_layer}/hidden_vectors.pt'

        if not os.path.exists(input_hidden_states_path):
            raise FileNotFoundError(f"Input hidden states path does not exist: {input_hidden_states_path}")
        
        with open(input_hidden_states_path, 'rb') as f:
            hidden_vectors_all.append(torch.tensor(torch.load(f)))
            
    hidden_vectors = torch.cat(hidden_vectors_all, dim=0)
    return hidden_vectors


def load_input_hidden_states_target_dataset(task, target_layer, target_dataset_name, model='llama', from_variations=False):
    results_suffix = f'_{model}'
    
    input_hidden_states_path = f'../clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/{target_dataset_name}/layer_{target_layer}/hidden_vectors.pt'

    if from_variations:
        input_hidden_states_path = f'../clotho/results{results_suffix}/{task}/input_hidden_states/{prompt_templates[task]}/from_variations/{target_dataset_name}/layer_{target_layer}/hidden_vectors.pt'

    if not os.path.exists(input_hidden_states_path):
        raise FileNotFoundError(f"Input hidden states path does not exist: {input_hidden_states_path}")
        
    with open(input_hidden_states_path, 'rb') as f:
        hidden_vectors = torch.tensor(torch.load(f, weights_only=False))

    return hidden_vectors


def load_initial_testset(task, target_layer):
    with open(f'../experiments/initial_tests/{task}/inference_results.json', 'r') as f:
        inference_results = json.load(f)
        LIH_init = torch.load(f'../experiments/initial_tests/{task}/hidden_vectors_layer_{target_layer}.pt')

    return inference_results['test_scores'], LIH_init


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

def convert_prob_to_descrete_labels(probabilities, binarize=False, binarize_choice='majority_pass'):
    if binarize:
        if binarize_choice == 'majority_pass':
            return (probabilities > 0.5).astype(int)
        elif binarize_choice == 'all_pass':
            return (probabilities == 1).astype(int)
        elif binarize_choice == 'any_pass':
            return (probabilities > 0).astype(int)
        else:
            raise ValueError("Unsupported binarization choice: {}".format(binarize_choice))
    else:
        weights = np.clip(np.round(probabilities, 1), 0.0, 1.0)
        return weights

def convert_to_normalized_rank_score(scores, invert=False):
    scores = np.asarray(scores)
    ranks = rankdata(scores, method='average') #smallest value gets rank 1, largest gets rank N
    ranks_norm = (ranks - 1) / (len(ranks) - 1)
    if invert:
        ranks_norm = 1.0 - ranks_norm
    return ranks_norm

def convert_to_rank_score(scores, invert=False):
    scores = np.asarray(scores)
    ranks = rankdata(scores, method='average') #smallest value gets rank 1, largest gets rank N
    if invert:
        ranks = len(ranks) + 1 - ranks
    return ranks

def get_vis_components(vectors, random_state=42):
    pca = PCA(n_components=10, random_state=random_state)
    tsne = TSNE(n_components=2, random_state=random_state)
    vectors_pca = pca.fit_transform(vectors)
    vectors_tsne = tsne.fit_transform(vectors_pca)
    
    return vectors_tsne
    
    
def plot_logprob_ranks(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='logprob_rank', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Predicted Logprobs (Normalized Ranks)')
    ax.legend([], [], frameon=False)
    
def plot_ranks(df_vis, ax=None, cname='logprob'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    ranks = rankdata(df_vis[cname].to_numpy(), method='average')
    ranks = (ranks - 1) / len(ranks) - 1
    
    df_vis[f'_ranks_{cname}'] = ranks

    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue=f'_ranks_{cname}', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Predicted Logprobs (Normalized Ranks)')
    ax.legend([], [], frameon=False)
    
    
def plot_test_scores(df_vis, ax=None, cname='test_score'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue=cname, palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title(f'{cname}')
    ax.legend([], [], frameon=False)
    
    
def plot_probability_density(df_vis, ax=None, l_q=25, u_q=95, cname='logprob'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    s = df_vis[cname].to_numpy()
    vmin, vmax = np.percentile(s, (l_q, u_q))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    df_vis['_normed'] = norm(df_vis[cname])

    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='_normed', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title(f'normalized {cname} ({l_q}%-{u_q}%)')
    ax.legend([], [], frameon=False)


def plot_reference_points(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='label', palette={'ref': 'blue', 'test': 'orange'}, style='label', alpha=0.7, ax=ax)
    
    texts = []
    points = []
    for _, row in df_vis[df_vis['label'] == 'ref'].iterrows():
        x, y = row['Component 1'], row['Component 2']
        texts.append(ax.text(x, y, f'({row["test_score"]:.2f})', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')))
        points.append(ax.plot(x, y, 'x', markersize=5, color='red')[0])
    ax.grid(alpha=0.3)
    ax.set_title('Reference Points')
    ax.legend([], [], frameon=False)
    
    _ = adjust_text(texts, add_objects=points, ax=ax, arrowprops=dict(arrowstyle='->', color='black'), lw=0.5, force_text=0.8, force_points=1.0, expand_texts=(5, 5), expand_points=(5, 5), only_move={'points': 'none', 'text': 'xy'})
    
    
def plot_score_distribution(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    order = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sns.boxplot(data=df_vis, x='test_score', y='logprob', ax=ax, palette='RdYlGn', showfliers=False, order=order)
    ax.set_title('Logprob Distribution by Test Score')
    ax.set_xlabel('Test Score')
    ax.set_ylabel('Predicted Logprob')
    ax.legend([], [], frameon=False)
    
def plot_score_correlation(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    vmin, vmax = np.percentile(df_vis['logprob'], (1, 100))
    
    sns.violinplot(data=df_vis, x='test_score', y='logprob', ax=ax, palette='RdYlGn', inner=None, bw=0.2)
    ax.set_title('Correlation between Test Score and Predicted Logprob')
    ax.set_xlabel('Test Score')
    ax.set_ylabel('Predicted Logprob')
    ax.set_ylim(vmin, vmax)
    ax.grid(alpha=0.3)
    

def plot_uncertainties(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df_vis['uncertainty_rank'] = rankdata(df_vis['uncertainty'], method='average')
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='uncertainty_rank', palette='coolwarm', alpha=0.7, ax=ax)
        
    ax.grid(alpha=0.3)
    ax.set_title('Uncertainty of Predictions')
    ax.legend([], [], frameon=False)
    
    
def clipped_minmax_normalize(values, lower_q=1, upper_q=99):
    values = np.asarray(values, dtype=float)
    
    lower = np.percentile(values, lower_q)
    upper = np.percentile(values, upper_q)
    
    clipped = np.clip(values, lower, upper)
    
    normed = (clipped - lower) / (upper - lower)
    
    return normed, lower, upper

def calculate_pearson_correlation(predicted_scores, observed_scores, lower_q=0, upper_q=100):
    if lower_q > 0 or upper_q < 100:
        predicted_logprobs_normed, lb, ub = clipped_minmax_normalize(predicted_scores, lower_q, upper_q)
    else:
        predicted_logprobs_normed = predicted_scores

    r, p_val = pearsonr(predicted_logprobs_normed, observed_scores)
    return r, p_val

def entries_to_df(entries):
    rows = []
    for e in entries:
        u = e.get("uncertainties", {}) or {}
        rows.append({
            "input_index": e.get("entry_index"),
            "cluster_assignment_entropy": u.get("cluster_assignment_entropy"),
            "predictive_entropy": u.get("predictive_entropy"),
            "semantic_entropy": u.get("semantic_entropy"),
            "num_semantic_clusters": u.get("num_semantic_clusters"),
        })
    return pd.DataFrame(rows)

def process_semantic_entropy_json(json_path: str, out_pkl_path: str=None):
    with open(json_path, "r") as f:
        entries = json.load(f)
    df = entries_to_df(entries)
    return df

def load_semantic_entropy_df(model, target_task):
    dataset_names = target_testsuites[target_task]
    _, _, test_scores = get_test_results(model, target_task)

    semantic_ent_df = []
    index_offset = 0
    for dataset_name in dataset_names:
        df = process_semantic_entropy_json(f"../metrics/results/semantic_entropy/results_{model}/{target_task}_{dataset_name}_R10_T0.8.json")
        df.input_index += index_offset
        index_offset += df.input_index.max() + 1
        semantic_ent_df.append(df)

    semantic_ent_df = pd.concat(semantic_ent_df, ignore_index=True)
    semantic_ent_df['sem_ent_rank'] = convert_to_rank_score(semantic_ent_df['semantic_entropy'], invert=True)
    semantic_ent_df['task'] = target_task
    semantic_ent_df['test_score'] = semantic_ent_df['input_index'].apply(lambda x: test_scores[x])
    semantic_ent_df['test_rank'] = convert_to_rank_score(semantic_ent_df['test_score'])
    return semantic_ent_df

def load_gmm_df(model_name, target_task, data='all', target_iteration = 50, seed=0):
    models_suffix = f'_{model_name}'
    if model_name == 'llama':
        LIH_target_layer = 21
    elif model_name == 'gemma':
        LIH_target_layer = 28
    elif model_name == 'mistral':
        LIH_target_layer = 22
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    lih_gmm_df = []
    df = pd.read_pickle(f'../experiments/results_GMM/LIH_refset_iter_10{models_suffix}/{target_task}/layer_{LIH_target_layer}_seed_{seed}/balanced_pca.pkl')
    lih_gmm_df.append(df)
    
    lih_gmm_df = pd.concat(lih_gmm_df, ignore_index=True)
    lih_gmm_df['task'] = target_task
    if data == 'all':
        lih_gmm_df = lih_gmm_df[(lih_gmm_df['iteration'] == target_iteration)].iloc[:-10]
    else:
        lih_gmm_df = lih_gmm_df[(lih_gmm_df['iteration'] == target_iteration) & (lih_gmm_df['label'] == data)]
    lih_gmm_df['logprob_nn_rank'] = convert_to_rank_score(lih_gmm_df['logprob'], invert=False)
    lih_gmm_df['test_rank'] = convert_to_rank_score(lih_gmm_df['test_score'])
    return lih_gmm_df

def load_loh_variance_df(model, target_task):
    _, _, test_scores = get_test_results(model, target_task)
    if model == 'gemma':
        target_loh_layer = 28
    elif model == 'mistral':
        target_loh_layer = 22
    elif model == 'llama':
        target_loh_layer = 23
    else:
        raise ValueError(f'Unknown model: {model}')
    
    prompt_template_name = prompt_templates[target_task]
    dataset_names = target_testsuites[target_task]
    models_suffix = f'_{model}'
    loh_variance_df = []
    input_index_offset = 0
    
    for dataset_name in dataset_names:
        df = pd.read_pickle(f'../results{models_suffix}/{target_task}/precalculated_metrics/{prompt_template_name}/{dataset_name}_R10_T0.8_output_state_variances.pkl')
        
        dataset_size = df['input_index'].max() + 1
        df['input_index'] += input_index_offset
        input_index_offset += dataset_size
        loh_variance_df.append(df)
    
    loh_variance_df = pd.concat(loh_variance_df, ignore_index=True)
    loh_variance_df = loh_variance_df[
        (loh_variance_df['layer_index'] == target_loh_layer) &
        (loh_variance_df['token_index'] == 'IE')
    ][['input_index', 'variance']]
    loh_variance_df['var_rank'] = convert_to_rank_score(loh_variance_df['variance'], invert=True)
    loh_variance_df['test_score'] = loh_variance_df['input_index'].apply(lambda x: test_scores[x])
    loh_variance_df['test_rank'] = convert_to_rank_score(loh_variance_df['test_score'])
    return  loh_variance_df

def load_output_logprobs_df(model, target_task):
    dataset_names = target_testsuites[target_task]
    prompt_template_name = prompt_templates[target_task]
    _, _, test_scores = get_test_results(model, target_task)
    models_suffix = f'_{model}'
    
    logprobs_df = []
    index_offset = 0
    for dataset_name in dataset_names:
        df = pd.read_pickle(f'../results{models_suffix}/{target_task}/precalculated_metrics/{prompt_template_name}/{dataset_name}_R10_T0.8_log_probs.pkl')
        df.input_index += index_offset
        index_offset += df.input_index.max() + 1
        logprobs_df.append(df)

    logprobs_df = pd.concat(logprobs_df, ignore_index=True)
    logprobs_df = logprobs_df[logprobs_df['inference_index'] == 0] # use only index @ 0
    
    logprobs_df['task'] = target_task
    logprobs_df['logprob_rank'] = convert_to_rank_score(logprobs_df['average_log_probs'], invert=False)
    logprobs_df['test_score'] = logprobs_df['input_index'].apply(lambda x: test_scores[x])
    logprobs_df['test_rank'] = convert_to_rank_score(logprobs_df['test_score'])
    return logprobs_df

def load_token_entropy_df(model_name, target_task):
    prompt_template_name = prompt_templates[target_task]
    dataset_names = target_testsuites[target_task]
    _, _, test_scores = get_test_results(model_name, target_task)
    
    models_suffix = f'_{model_name}'
    token_entropy_df = []
    input_index_offset = 0
    for dataset_name in dataset_names:
        df = pd.read_pickle(f'../results{models_suffix}/{target_task}/precalculated_metrics/{prompt_template_name}/{dataset_name}_R10_T0.8_average_token_entropy.pkl')
        dataset_size = df['input_index'].max() + 1
        df['input_index'] += input_index_offset
        input_index_offset += dataset_size
        token_entropy_df.append(df)
    
    token_entropy_df = pd.concat(token_entropy_df, ignore_index=True)
    token_entropy_df = token_entropy_df[token_entropy_df['inference_index'] == 0] # use only index @ 0
    
    token_entropy_df['task'] = target_task
    token_entropy_df['tok_ent_rank'] = convert_to_rank_score(token_entropy_df['average_entropy'], invert=True)
    token_entropy_df['test_score'] = token_entropy_df['input_index'].apply(lambda x: test_scores[x])
    token_entropy_df['test_rank'] = convert_to_rank_score(token_entropy_df['test_score'])
    return token_entropy_df