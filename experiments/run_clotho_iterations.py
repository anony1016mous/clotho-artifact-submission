import gc
import json
import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from collections import defaultdict
from utils import get_test_results, target_testsuites, prompt_templates, load_input_hidden_states, get_vis_components, get_gmm_components_perplexity, convert_to_normalized_rank_score
from plot_utils import *

import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, rankdata
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

import time
from datetime import datetime

from clotho.preprocessing.feature_reduction import PCAFeatureReducer
from clotho.metrics.reference_based.density_estimation import GMMScorer

from clotho.al.sampling import *    # reference set extension methods for Active Learning

import argparse
import logging

PASS_THRESH = 0.5

def main(target_task, target_layer, refset_extension_methods=['random', 'diversity'], scoring_model='GMM', num_add=10, random_seeds=[42], target_llm='llama'):
    target_llm_suffix = f'_{target_llm}'
    
    with open(f'./initial_tests{target_llm_suffix}/{target_task}/inference_results.json', 'r') as f:
        inference_results = json.load(f)
    LIH_init = torch.load(f'./initial_tests{target_llm_suffix}/{target_task}/hidden_vectors_layer_{target_layer}.pt', weights_only=False)
    LIH_target = load_input_hidden_states(target_task, target_layer, model=target_llm)
    _, _, test_scores = get_test_results(target_llm, target_task)
    
    LIH_vis = get_vis_components(np.concatenate((LIH_target, LIH_init), axis=0))
    
    logging.info(f"Loaded initial LIH with shape {LIH_init.shape} and target LIH with shape {LIH_target.shape}")
    logging.info(f"target refset_extension_methods: {refset_extension_methods}")

    for refset_extension_method in refset_extension_methods:
        logging.info(f"Starting experiment with task: {target_task}, LIH layer: {target_layer}, scoring_model: {scoring_model}, refset_extension_method: {refset_extension_method}, random_seeds: {random_seeds}")
        
        for random_seed in tqdm(random_seeds, position=0, desc="Experimenting with {} random seeds (task: {}, scoring_model: {}, target_layer: {}, refset_extension_method: {})".format(len(random_seeds), target_task, scoring_model, target_layer, refset_extension_method), leave=True):
            result_df, iteration_records = simulate_iterations(
                target_task=target_task,
                target_layer=target_layer,
                LIH_init=LIH_init,
                LIH_target=LIH_target,
                labels_init=inference_results['test_scores'],
                labels_target=test_scores,
                scoring_model=scoring_model,
                dim_reduction_method='pca',
                refset_extension_method=refset_extension_method,
                random_seed=random_seed,
                n_iterations=50,
                max_n_components=50,
                max_n_features=100,
                update_term=1,
                num_add=num_add,
                LIH_vis=LIH_vis,
                result_suffix=target_llm_suffix,
                target_llm=target_llm,
                plot_intermediate_results=True,
            )

            result_df.to_pickle(f'./results_{scoring_model}/LIH_refset_iter_{num_add}{target_llm_suffix}/{target_task}/layer_{target_layer}_seed_{random_seed}/{refset_extension_method}_pca.pkl')

            with open(f'./results_{scoring_model}/LIH_refset_iter_{num_add}{target_llm_suffix}/{target_task}/layer_{target_layer}_seed_{random_seed}/{refset_extension_method}_pca_iteration_records.json', 'w') as f:
                json.dump(iteration_records, f, indent=2)


def simulate_iterations(target_task, target_layer, LIH_init, LIH_target, labels_init, labels_target, scoring_model='GMM', dim_reduction_method='pca', refset_extension_method='diversity', random_seed=42, n_iterations=50, update_term=1, max_n_components=50, max_n_features=100, num_add=10, LIH_vis=None, result_suffix='_llama', target_llm=None, plot_intermediate_results=False):
    LIH = np.concatenate((LIH_target, LIH_init), axis=0)
    labels = np.concatenate((labels_target, labels_init), axis=0)
    labels_binary = (labels > PASS_THRESH).astype(int)
    
    if LIH_vis is None:
        LIH_vis = get_vis_components(LIH, random_state=random_seed)
    
    current_reference_set_indices = list(range(len(LIH_target), len(LIH)))
    current_n_features = 10
    current_n_components = 5

    components_perplexities = []
    
    scorer = None
    LIH_rep = {} # feature_number -> feature representation mapping
    
    df_all_iterations = pd.DataFrame()
    iteration_records_all = {}
    
    for i in tqdm(range(n_iterations), desc="Simulating iterations", position=1, leave=False):
        skip = False
        
        iteration_record = {
            'duration_feature_reduction': None,
            'duration_model_fitting': None,
            'duration_reference_set_extension': None,
            'duration_total': None,
            'lower_n_features': None,
            'current_n_features': None,
            'larger_n_features': None,
            'selected_n_features': None,
            'n_components': None,
            'n_reference_set': len(current_reference_set_indices)
        }

        num_effective_references = len([l for l in labels[current_reference_set_indices] if l > PASS_THRESH])
            
        start_time = time.time()
        if current_n_features not in LIH_rep:
            if dim_reduction_method == 'pca':
                dim_reducer = PCAFeatureReducer(n_features=current_n_features)
            else:
                raise NotImplementedError(f"Dimensionality reduction method '{dim_reduction_method}' is not implemented.")
            
            LIH_rep[current_n_features] = dim_reducer.fit_transform(LIH)
            
            iteration_record['duration_feature_reduction'] = time.time() - start_time

        is_fitted_to_current_reference_set = False
        if i > 0 and (update_term == 1 or (i+1) % update_term == 1):
            """
            Dimensionality Selection Checkpoint
            """
            logging.info('* * * Dimensionality Selection Checkpoint (Iteration {}) * * *'.format(i+1))
            
            CLUSTER_UPDATE_SIZE = 1
            MIN_CLUSTER_SIZE = 5
            
            # 1. Update n_components (based on the active components records)
            if len(components_perplexities) > 0:
                logging.info(f'Perplexities for the previous phase: {components_perplexities}')
                assert len(components_perplexities) == update_term
                
                perplexity_record = np.array(sorted(components_perplexities, reverse=True))
                saturated_n_comp_thresh = current_n_components - (CLUSTER_UPDATE_SIZE * 0.5)
                understuffed_n_comp_thresh = current_n_components - (CLUSTER_UPDATE_SIZE * 1.5)
                
                if len(perplexity_record[perplexity_record > saturated_n_comp_thresh]) >= max(len(components_perplexities) // 2, 1):
                    next_n_components = min(current_n_components + CLUSTER_UPDATE_SIZE, max_n_components)
                    if num_effective_references // 10 > next_n_components:
                        logging.info(f'Increasing n_components from {current_n_components} to {next_n_components} based on perplexity. (more than {saturated_n_comp_thresh})')
                        old_n_components = current_n_components
                        current_n_components = next_n_components

                elif len(perplexity_record[perplexity_record < understuffed_n_comp_thresh]) == len(components_perplexities):
                    next_n_components = max(current_n_components - CLUSTER_UPDATE_SIZE, MIN_CLUSTER_SIZE)
                    logging.info(f'Decreasing n_components from {current_n_components} to {next_n_components} based on perplexity. (less than {understuffed_n_comp_thresh})')
                    old_n_components = current_n_components
                    current_n_components = next_n_components

                components_perplexities = []
            
            # 2. Update n_features
            current_n_features = get_next_n_features(
                LIH,
                LIH_rep,
                current_reference_set_indices,
                labels,
                current_n_features,
                current_n_components,
                random_seed,
                max_n_features,
                target_llm,
                scoring_model=scoring_model,
            )

        if scoring_model == 'wGMM':
            reference_set_labels = labels[current_reference_set_indices]
            if len(reference_set_labels[reference_set_labels > 0]) < 10:
                logging.info(f"[Weighted GMM] simulate_iterations ({i+1}th iteration): Not enough nonzero-score reference points to fit Weighted GMM. (Current: {len(reference_set_labels[reference_set_labels > 0])}, Required: 10) Skipping this iteration.")
                skip = True
            
        elif scoring_model == 'GMM':
            reference_set_labels = (labels[current_reference_set_indices] > PASS_THRESH).astype(int)
            if len(reference_set_labels[reference_set_labels == 1]) < 10:
                logging.info(f"[Unweighted GMM] simulate_iterations ({i+1}th iteration): Not enough passing reference points to fit GMM. (Current: {len(reference_set_labels[reference_set_labels == 1])}, Required: 10) Skipping this iteration.")
                skip = True
        
        if not skip:
            start_time_model_fitting = time.time()
            scorer = construct_scorer(current_n_components, random_seed, target_llm=target_llm)
            scorer.fit(LIH_rep[current_n_features][current_reference_set_indices], reference_set_labels)
            iteration_record['duration_model_fitting'] = time.time() - start_time_model_fitting
                
            logging.info(f"Fitted Weighted GMM with n_features={current_n_features}, n_components={current_n_components}, data_size={len(current_reference_set_indices)}")
            logging.info(f"converged_: {scorer.model.converged_}, n_iter: {scorer.model.n_iter_}, lower_bounds_: {scorer.model.lower_bound_}, weights_: {scorer.model.weights_}, means_: {scorer.model.means_.shape}")
            logging.info(f"Sample Weights: {reference_set_labels}")

            assert scorer.check_is_fitted()
            assert scorer._n_features() == current_n_features
            assert scorer._n_components() == current_n_components
            
            iteration_record['n_features'] = current_n_features
            iteration_record['n_components'] = current_n_components

            perplexity = get_gmm_components_perplexity(scorer.model.weights_)
            logging.info(f'model.weights_: {scorer.model.weights_}')
            logging.info(f'Components perplexity: {perplexity:.2f}')
            components_perplexities.append(perplexity)

        
            start_time_model_scoring = time.time()
            
            pred_scores = scorer.score(LIH_rep[current_n_features])
            rank_normalized = convert_to_normalized_rank_score(pred_scores)

            iteration_record['duration_model_scoring'] = time.time() - start_time_model_scoring
            
            eval_scores = {
                'pearson': pearsonr(pred_scores, labels)[0],
                'pearson_norm_rank': pearsonr(rank_normalized, labels)[0],
                'roc_auc': roc_auc_score(labels_binary, pred_scores)
            }
            
            logging.info(f'''
* * * [Iteration {i + 1}] scoring_model={scoring_model}, refset_extension_method={refset_extension_method}, random_seed={random_seed}
Current reference set size: {len(current_reference_set_indices)}
Current n_features: {current_n_features}
Current n_components: {current_n_components}
Pearson correlation w/ observed scores: {eval_scores['pearson']:.4f}
Pearson correlation w/ observed scores (norm_rank): {eval_scores['pearson_norm_rank']:.4f}
ROC AUC score: {eval_scores['roc_auc']:.4f}''')

        start_time_reference_set_extension = time.time()
        uncertainties = None

        if refset_extension_method == 'random':
            new_reference_set_indices = extend_reference_set_random(current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add, random_seed=random_seed)

        elif refset_extension_method == 'diversity_euclidean':
            new_reference_set_indices = extend_reference_set_diversity(current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add, metric='euclidean')

        elif refset_extension_method == 'uncertainty':
            if scorer is None:
                logging.warning(f'Scorer is not fitted due to the insufficient effective reference points. Using random sampling instead.')
                new_reference_set_indices = extend_reference_set_random(current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add, random_seed=random_seed)
            else:
                new_reference_set_indices, uncertainties = extend_reference_set_uncertainty(scorer, current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add)
                
        elif refset_extension_method == 'balanced':
            if scorer is None:
                logging.warning(f'Scorer is not fitted due to the insufficient effective reference points. Using diversity sampling instead.')
                new_reference_set_indices = extend_reference_set_diversity(current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add, metric='euclidean')
            else:
                new_reference_set_indices_by_uncertainty, uncertainties = extend_reference_set_uncertainty(scorer, current_reference_set_indices, LIH_rep[current_n_features], num_add=num_add // 2)
                new_reference_set_indices_by_diversity = extend_reference_set_diversity(current_reference_set_indices + new_reference_set_indices_by_uncertainty, LIH_rep[current_n_features], num_add=num_add - (num_add // 2), metric='euclidean')
                assert set(new_reference_set_indices_by_uncertainty).isdisjoint(set(new_reference_set_indices_by_diversity)), "The two sets of new reference indices should be disjoint."
                new_reference_set_indices = new_reference_set_indices_by_uncertainty + new_reference_set_indices_by_diversity

        else:
            raise NotImplementedError(f"Reference set extension method '{refset_extension_method}' is not implemented.")

        iteration_record['duration_reference_set_extension'] = time.time() - start_time_reference_set_extension
        iteration_record['duration_total'] = time.time() - start_time

        if not skip:
            df_vis = pd.DataFrame(LIH_vis, columns=['Component 1', 'Component 2'])
            df_vis['input_index'] = list(range(len(LIH)))
            df_vis['test_score'] = labels
            df_vis['logprob'] = scorer.score(LIH_rep[current_n_features])
            df_vis['logprob_rank'] = scorer.rank(LIH_rep[current_n_features], normalize=True)
            df_vis['label'] = ['ref' if idx in current_reference_set_indices else 'test' for idx in range(len(LIH))]
            if uncertainties is not None:
                df_vis['uncertainty'] = uncertainties
            else:
                df_vis['uncertainty'] = np.nan

            if plot_intermediate_results and (i == 0 or (i+1) % 10 == 0):
                fig, axes = plt.subplots(1, 5, figsize=(35, 6), sharex=False, sharey=False)

                plot_probability_density(df_vis, ax=axes[0], cname='test_score')
                plot_logprob_ranks(df_vis, ax=axes[1])
                plot_reference_points_simple(df_vis, ax=axes[2])
                if uncertainties is not None:
                    plot_uncertainties(df_vis, ax=axes[3])
                plot_score_distribution(df_vis, ax=axes[4])

                fig.suptitle(f'{target_task}: iteration {i+1} ({refset_extension_method}, {dim_reduction_method}, n_feat={current_n_features}, n_comp={current_n_components}) => pearson={eval_scores["pearson_norm_rank"]:.4f}, roc-auc={eval_scores["roc_auc"]:.4f}', fontsize=12)
                plt.tight_layout()
                os.makedirs(f'./results_{scoring_model}/LIH_refset_iter_{num_add}{result_suffix}/{target_task}/layer_{target_layer}_seed_{random_seed}/n_{len(current_reference_set_indices)}', exist_ok=True)
                plt.savefig(f'./results_{scoring_model}/LIH_refset_iter_{num_add}{result_suffix}/{target_task}/layer_{target_layer}_seed_{random_seed}/n_{len(current_reference_set_indices)}/{refset_extension_method}_{dim_reduction_method}.png')
                plt.close(fig)
                gc.collect()
            
            df_vis['iteration'] = i + 1
            df_vis['n_components'] = current_n_components
            df_vis['n_features'] = current_n_features
            
            df_all_iterations = pd.concat([df_all_iterations, df_vis], ignore_index=True)
    
        assert len(new_reference_set_indices) == num_add, f"Expected {num_add} new reference set indices, got {len(new_reference_set_indices)}."
        
        iteration_records_all["iteration_" + str(i + 1)] = iteration_record
        current_reference_set_indices.extend(new_reference_set_indices)
        
    df_all_iterations['task'] = target_task
    df_all_iterations['layer'] = target_layer
    df_all_iterations['random_seed'] = random_seed
    df_all_iterations['scoring_model'] = scoring_model
    df_all_iterations['refset_extension_method'] = refset_extension_method
    df_all_iterations['dim_reduction_method'] = dim_reduction_method
    return df_all_iterations, iteration_records_all


def get_next_n_features(LIH, LIH_rep, current_reference_set_indices, labels, current_n_features, current_n_components, random_seed, max_n_features, target_llm, dim_reduction_method='pca', scoring_model='GMM'):
    next_n_features_higher = min(current_n_features + 10, max_n_features)
    next_n_features_lower = max(5, current_n_features - 10)
    
    if next_n_features_higher not in LIH_rep:
        if dim_reduction_method == 'pca':
            dim_reducer = PCAFeatureReducer(n_features=next_n_features_higher)
        else:
            raise NotImplementedError(f"Dimensionality reduction method '{dim_reduction_method}' is not implemented.")
        
        LIH_rep[next_n_features_higher] = dim_reducer.fit_transform(LIH)

    if next_n_features_lower not in LIH_rep:
        if dim_reduction_method == 'pca':
            dim_reducer = PCAFeatureReducer(n_features=next_n_features_lower)
        else:
            raise NotImplementedError(f"Dimensionality reduction method '{dim_reduction_method}' is not implemented.")
        
        LIH_rep[next_n_features_lower] = dim_reducer.fit_transform(LIH)

    if next_n_features_lower != current_n_features and next_n_features_lower in LIH_rep:
        rep_LD = LIH_rep[next_n_features_lower]
    else:
        rep_LD = None

    if next_n_features_higher != current_n_features and next_n_features_higher in LIH_rep:
        rep_HD = LIH_rep[next_n_features_higher]
    else:
        rep_HD = None
    
    if rep_LD is not None or rep_HD is not None:
        return select_feature_dimension_by_CV(
            labels[current_reference_set_indices],
            LIH_rep[current_n_features][current_reference_set_indices],
            rep_LD[current_reference_set_indices] if rep_LD is not None else None,
            rep_HD[current_reference_set_indices] if rep_HD is not None else None,
            current_n_components,
            random_seed,
            scoring_model=scoring_model,
            target_llm=target_llm
        )
        
    else:
        return current_n_features # No need of feature selection
    

def select_feature_dimension_by_CV(reference_scores, rep_original, rep_LD, rep_HD, n_components, random_state, scoring_model, target_llm):
    assert not (rep_LD is None and rep_HD is None), "At least one of rep_LD or rep_HD must be provided."
    
    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    score_correlations = {
        'LD': [],
        'original': [],
        'HD': [],
    }

    if scoring_model == 'wGMM':
        train_labels = (reference_scores > PASS_THRESH).astype(int)
    else:
        train_labels = reference_scores

    scorer_original = construct_scorer(n_components, random_seed=random_state, n_init=10, target_llm=target_llm)
    scorer_LD = construct_scorer(n_components, random_seed=random_state, n_init=10, target_llm=target_llm)
    scorer_HD = construct_scorer(n_components, random_seed=random_state, n_init=10, target_llm=target_llm)

    for train_idx, test_idx in kf.split(rep_original):
        if scoring_model == 'wGMM':
            num_effective_references = len([l for l in reference_scores[train_idx] if l > PASS_THRESH])
        else:
            num_effective_references = len([l for l in reference_scores[train_idx] if l > PASS_THRESH])
        if num_effective_references < n_components:
            logging.warning(f"Not enough effective references to perform cross-validation (n_components={n_components})")
            return rep_original.shape[1]

        scorer_original.fit(rep_original[train_idx], train_labels[train_idx])
        predicted_scores_original = convert_to_normalized_rank_score(scorer_original.score(rep_original[test_idx]))
        correlation_original = calculate_score_correlations_safe(predicted_scores_original, reference_scores[test_idx])
        score_correlations['original'].append(correlation_original)

        if rep_LD is not None:
            scorer_LD.fit(rep_LD[train_idx], train_labels[train_idx])
            predicted_scores_LD = convert_to_normalized_rank_score(scorer_LD.score(rep_LD[test_idx]))
            correlation_LD = calculate_score_correlations_safe(predicted_scores_LD, reference_scores[test_idx])
            score_correlations['LD'].append(correlation_LD)

        if rep_HD is not None: 
            scorer_HD.fit(rep_HD[train_idx], train_labels[train_idx])
            predicted_scores_HD = convert_to_normalized_rank_score(scorer_HD.score(rep_HD[test_idx]))
            correlation_HD = calculate_score_correlations_safe(predicted_scores_HD, reference_scores[test_idx])
            score_correlations['HD'].append(correlation_HD)
        
    max_mean_correlation = mean_correlation_original = np.mean(score_correlations['original'])
    best_CV = 'original'
    if len(score_correlations['LD']) > 0:
        mean_correlation_LD = np.mean(score_correlations['LD'])
        if mean_correlation_LD > max_mean_correlation and np.min(score_correlations['LD']) > 0.1:
            max_mean_correlation = mean_correlation_LD
            best_CV = 'LD'

    if len(score_correlations['HD']) > 0:
        mean_correlation_HD = np.mean(score_correlations['HD'])
        if mean_correlation_HD > max_mean_correlation:
            max_mean_correlation = mean_correlation_HD
            best_CV = 'HD'
            
    logging.info(str({
        "best_CV": best_CV,
        "score_correlations": score_correlations
    }))
    logging.info(f'Best CV: {best_CV}')
    if len(score_correlations['LD']) > 0:
        logging.info('Mean LD score correlations: {}'.format(np.mean(score_correlations['LD'])))
    if len(score_correlations['HD']) > 0:
        logging.info('Mean HD score correlations: {}'.format(np.mean(score_correlations['HD'])))
    logging.info('Mean Original score correlations: {}'.format(np.mean(score_correlations['original'])))

    if len(score_correlations['LD']) > 0:
        logging.info('Min LD score correlations: {}'.format(np.min(score_correlations['LD'])))
    if len(score_correlations['HD']) > 0:
        logging.info('Min HD score correlations: {}'.format(np.min(score_correlations['HD'])))
    logging.info('Min Original score correlations: {}'.format(np.min(score_correlations['original'])))


    if best_CV == 'HD':
        return rep_HD.shape[1]
    
    if best_CV == 'LD':
        return rep_LD.shape[1]

    return rep_original.shape[1]


def calculate_score_correlations_safe(predicted_scores, true_scores):
    predicted_scores = np.asarray(predicted_scores)
    true_scores = np.asarray(true_scores)

    if np.allclose(np.var(true_scores), 0.0):
        const_val = true_scores[0]
        mae = np.mean(np.abs(predicted_scores - const_val))  # predicted_scores range: [0, 1]
        return 1 - mae  
    else:
        return pearsonr(predicted_scores, true_scores)[0]


def construct_scorer(current_n_components, random_seed, n_init=30, target_llm='llama'):
    return GMMScorer(n_components=current_n_components, random_state=random_seed, verbose=False, max_iter=500, reg_covar=1e-3, tol=1e-3, n_init=n_init, use_custom_class=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Adequacy Modelling Experiment w/ Probability Density Model")
    parser.add_argument('--target_task', type=str, required=True, help='Target task for the experiment')
    parser.add_argument('--scoring_model', type=str, help='Scoring model for the experiment', default='GMM')
    parser.add_argument('--refset_extension_methods', type=str, nargs='+', default=['random', 'diversity'], help='Methods for reference set extension')
    parser.add_argument('--target_llm', type=str, default='llama', help='Target LLM for the experiment')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0], help='Random seeds for the experiment')

    args = parser.parse_args()
    
    target_layer_map = {
        'llama': 21,
        'gemma': 28,
        'mistral': 22
    }

    result_suffix = f"_{args.target_llm}"
    target_layer = target_layer_map[args.target_llm]

    refset_extension_methods_str = '_'.join(args.refset_extension_methods)
    os.makedirs(f"logs_{args.scoring_model}/{target_layer}{result_suffix}/{refset_extension_methods_str}/{args.target_task}", exist_ok=True)
    log_filename = f"logs_{args.scoring_model}/{target_layer}{result_suffix}/{refset_extension_methods_str}/{args.target_task}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    main(
        target_task=args.target_task,
        target_layer=target_layer,
        scoring_model=args.scoring_model,
        refset_extension_methods=args.refset_extension_methods,
        random_seeds=args.seeds,
        num_add=10,
        target_llm=args.target_llm
    )