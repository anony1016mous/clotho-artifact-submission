import numpy as np
import random
import torch

import logging

from scipy.spatial.distance import cdist


def extend_reference_set_uncertainty(scoring_model, refset_indices, target_vectors, num_add=10):
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))

    uncertainties = scoring_model.get_uncertainties(target_vectors)
    logging.info(f"Uncertainty stats: mean={np.mean(uncertainties):.4f}, std={np.std(uncertainties):.4f}, min={np.min(uncertainties):.4f}, max={np.max(uncertainties):.4f}")
    logging.info(f"Uncertainties: {uncertainties}")

    top_n_uncertain_indices = uncertainties[candidate_indices].argsort()[::-1][:num_add]
    top_n_uncertain_indices = [candidate_indices[i] for i in top_n_uncertain_indices]
    
    assert set(top_n_uncertain_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, top_n_uncertain_indices)
    
    return top_n_uncertain_indices, uncertainties


def extend_reference_set_random(refset_indices, target_vectors, num_add=10, random_seed=42):
    np.random.seed(random_seed)
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))
    
    selected_indices = np.random.choice(candidate_indices, size=num_add, replace=False).tolist()
    
    assert set(selected_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, selected_indices)
    
    return selected_indices


def extend_reference_set_diversity(refset_indices, target_vectors, num_add=10, metric='euclidean'):
    # deterministic version
    candidate_indices = [i for i in range(len(target_vectors)) if i not in refset_indices]
    if len(candidate_indices) < num_add:
        raise ValueError("Not enough selectable indices to add {} new indices.".format(num_add))
    
    selected_candidates = candidate_indices
    
    base = cdist(target_vectors[selected_candidates], target_vectors[refset_indices], metric=metric)  # (Nc, |R|)
    min_d = base.min(axis=1)                                                # (Nc,)
    
    picked_mask = np.zeros(len(selected_candidates), dtype=bool)
    chosen_local = []
    
    for _ in range(min(num_add, len(selected_candidates))):
        scores = np.where(picked_mask, -np.inf, min_d)
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break
        picked_mask[j] = True
        chosen_local.append(j)

        d_new = cdist(target_vectors[selected_candidates], target_vectors[selected_candidates][j:j+1], metric=metric).ravel()  # (Nc,)
        d_new = np.where(np.isnan(d_new), 1.0 if metric == 'cosine' else np.nanmax(d_new[np.isfinite(d_new)]), d_new)
        min_d = np.minimum(min_d, d_new)

    top_n_diverse_indices = [selected_candidates[i] for i in chosen_local]
    
    assert set(top_n_diverse_indices).isdisjoint(set(refset_indices)), "Selected indices must not overlap with reference set indices. Current reference set indices: {}, selected indices: {}".format(refset_indices, top_n_diverse_indices)
    
    return top_n_diverse_indices
