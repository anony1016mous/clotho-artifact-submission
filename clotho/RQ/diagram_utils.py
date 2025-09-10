import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from analysis_utils import load_loh_variance_df, load_token_entropy_df, load_gmm_df

import scienceplots
plt.style.use(["science", "grid", "nature"])
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"], N=11)

task_alias_map = {
    "syntactic_bug_detection": "SYN-BUG",
    "spell_check": "SPELL-CHECK",
    "github_typo_check": "GH-TYPO",
    "json_repair": "JSON-FIX",
    "pos_detection": "POS-TAG",
    "topic_classification": "TOPIC-CLS",
    "adding_odd_numbers": "ODD-ADD",
    "model_name_extraction": "MODEL-EX",
}

def load_and_merge_data(model, task, data='all'):
    variance_df = load_loh_variance_df(model, task)
    token_ent_df = load_token_entropy_df(model, task)
    lih_gmm_df = load_gmm_df(model, task, data)

    # LOH variance
    var_df = variance_df[['input_index', 'variance']].copy()
    # Token entropy
    token_ent_df = token_ent_df[['input_index', 'average_entropy']].copy()
    # LIH GMM
    lih_df = lih_gmm_df[['input_index', 'logprob', 'test_score']].copy()

    merged = var_df.merge(token_ent_df, on='input_index', how='inner').merge(lih_df, on='input_index', how='inner')
    merged['task'] = task
    merged['variance_rank'] = merged['variance'].rank(ascending=True) # lower variance -> stable -> rank1
    merged['logprob_rank'] = merged['logprob'].rank(ascending=False) # higher logprob -> stable -> rank1
    merged['average_entropy_rank'] = merged['average_entropy'].rank(ascending=True) # lower entropy -> stable -> rank1
    
    return merged[['task','input_index','variance', 'logprob', 'average_entropy', 'variance_rank', 'logprob_rank', 'average_entropy_rank', 'test_score']]

def _make_masks_for_score(score_df, score, top_n_threshold, bottom_n_threshold):
    if score == 'high':
        A_certain = score_df['logprob_rank'] <= top_n_threshold
        B_certain = score_df['variance_rank'] <= top_n_threshold
        C_certain = score_df['average_entropy_rank'] <= top_n_threshold
    else:
        A_certain = score_df['logprob_rank'] >= bottom_n_threshold
        B_certain = score_df['variance_rank'] >= bottom_n_threshold
        C_certain = score_df['average_entropy_rank'] >= bottom_n_threshold

    return A_certain, B_certain, C_certain


def _select_score_slice(df, score):
    if score == 'high':
        return df[df['test_score'].isin([1.0, 0.9, 0.8])].copy()
    else:
        return df[df['test_score'].isin([0.0, 0.1, 0.2])].copy()


def compute_case_counts_for_score_3way(df, score, top_n_threshold, bottom_n_threshold):
    """
      A = GMM certain (logprob-based)
      B = LOH certain (variance-based)
      C = ENT certain (average-entropy-based)
    """
    score_df = _select_score_slice(df, score)
    n_total = len(score_df)
    A, B, C = _make_masks_for_score(score_df, score, top_n_threshold, bottom_n_threshold)

    # Disjoint partitions
    ABC = (A & B & C)
    AB_only = (A & B & ~C)
    AC_only = (A & ~B & C)
    BC_only = (~A & B & C)
    A_only = (A & ~B & ~C)
    B_only = (~A & B & ~C)
    C_only = (~A & ~B & C)

    return {
        "A_only": int(A_only.sum()),
        "B_only": int(B_only.sum()),
        "AB_only": int(AB_only.sum()),
        "C_only": int(C_only.sum()),
        "AC_only": int(AC_only.sum()),
        "BC_only": int(BC_only.sum()),
        "ABC": int(ABC.sum()),
        "n_total": n_total,
    }
    
def create_venn3_grid_for_score(task_list, score, base_config, percentile=0.25, save_path=None):
    """
    A=BGMM(logprob), B=LOH(variance), C=ENT(average_entropy)
    """
    if len(task_list) == 4:
        nrows, ncols = 1, 4
    else:
        nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8), constrained_layout=True)    
    axes = axes.flatten()

    for i, task in enumerate(task_list):
        ax = axes[i]
        cfg = dict(base_config)
        cfg['task'] = task

        task_df = load_and_merge_data(cfg['model'], cfg['task'])
        if task_df.empty:
            ax.axis('off')
            ax.set_title(f'{task}\n(no data)')
            continue

        top_n = int(percentile * len(task_df))
        bottom_n = int((1 - percentile) * len(task_df))
        
        # Counts for Venn diagram
        counts = compute_case_counts_for_score_3way(task_df, score, top_n, bottom_n)
        n_total = counts['n_total']
        if n_total == 0:
            ax.axis('off')
            ax.set_title(f'{task}\n(no {score} slice)')
            continue

        venn_counts = (
            counts['A_only'],
            counts['B_only'],
            counts['AB_only'],
            counts['C_only'],
            counts['AC_only'],
            counts['BC_only'],
            counts['ABC'],
        )

        v = venn3(subsets=venn_counts, set_labels=('Clotho', 'LOHS-Var', 'Tok-Ent'), ax=ax)
        
        def pct(x):
            return f'{(100.0 * x / n_total):.1f}\%'
        
        label_map = {
            '100': counts['A_only'],
            '010': counts['B_only'],
            '110': counts['AB_only'],
            '001': counts['C_only'],
            '101': counts['AC_only'],
            '011': counts['BC_only'],
            '111': counts['ABC']
        }
        
        for region_id, val in label_map.items():
            label = v.get_label_by_id(region_id)
            if label:
                label.set_text(f'{val}\n({pct(val)})')
                label.set_fontsize(12)

        covered = sum([counts[k] for k in ['A_only', 'B_only', 'C_only', 'AB_only', 'AC_only', 'BC_only', 'ABC']])
        ax.set_title(
            f'{task_alias_map[task]}\n'
            f'covered={covered}/{n_total} ({pct(covered)})',
            pad = 20,
            fontsize=16
        )
    
    for j in range(len(task_list), nrows * ncols):
        axes[j].axis('off')

    if save_path:
        fig.tight_layout() 
        fig.savefig(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()