import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from scipy.stats import rankdata

from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"], N=11)

def plot_logprob_ranks(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='logprob_rank', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Predicted Logprobs (Normalized Ranks)')
    ax.legend([], [], frameon=False)


def plot_probability_density(df_vis, ax=None, cname='logprob', lq=0, uq=100):
    df_vis = df_vis.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    s = df_vis[cname].to_numpy()
    if lq > 0 or uq < 100:
        vmin, vmax = np.percentile(s, (lq, uq))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        df_vis['_normed'] = norm(df_vis[cname])
    else:
        df_vis['_normed'] = df_vis[cname]

    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='_normed', palette=cmap, alpha=0.7, ax=ax)
    ax.grid(alpha=0.3)
    ax.set_title('Probability Density: {}'.format(cname))
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
    
    
def plot_reference_points_simple(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df_vis_sorted = df_vis.sort_values(by='label', key=lambda x: x.eq('ref'))
    sns.scatterplot(data=df_vis_sorted, x='Component 1', y='Component 2', hue='label', palette={'ref': 'blue', 'test': 'orange'}, style='label', alpha=0.7, ax=ax)

    ax.grid(alpha=0.3)
    ax.set_title('Reference Points')
    ax.legend([], [], frameon=False)
    
    
def plot_score_distribution(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    order = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sns.boxplot(data=df_vis, x='test_score', y='logprob', ax=ax, palette='RdYlGn', showfliers=False, order=order)
    ax.set_title('Logprob Distribution by Test Score')
    ax.set_xlabel('Test Score')
    ax.set_ylabel('Predicted Logprob')
    ax.legend([], [], frameon=False)
    

def plot_uncertainties(df_vis, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    df_vis['uncertainty_rank'] = rankdata(df_vis['uncertainty'], method='average')
    sns.scatterplot(data=df_vis, x='Component 1', y='Component 2', hue='uncertainty_rank', palette='coolwarm', alpha=0.7, ax=ax)
        
    ax.grid(alpha=0.3)
    ax.set_title('Uncertainty of Predictions')
    ax.legend([], [], frameon=False)


def plot_uncertainty_logprob_overview(df_vis):
    fig, axes = plt.subplots(1, 4, figsize=(32, 6))

    sns.histplot(df_vis['uncertainty'].dropna(), bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Uncertainty Distribution")
    axes[0].set_xlabel("Uncertainty Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    sns.histplot(df_vis['logprob'], bins=30, kde=True, ax=axes[1], color='lightcoral')
    axes[1].set_title("Logprob Distribution")
    axes[1].set_xlabel("Logprob Value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    # 3. Logprob vs Uncertainty (Scatterplot with test_score colormap)
    cutoff = df_vis['logprob'].quantile(0.01)
    df_filtered = df_vis[df_vis['logprob'] >= cutoff]

    sc = axes[2].scatter(
        df_filtered['logprob'], df_filtered['uncertainty'],
        c=df_filtered['test_score'], cmap=cmap, alpha=0.7, s=1
    )
    axes[2].set_title(f"Logprob vs. Uncertainty (>{0.01*100:.1f}th percentile)")
    axes[2].set_xlabel("Logprob")
    axes[2].set_ylabel("Uncertainty")
    axes[2].grid(alpha=0.3)

    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label("Test Score")

    if df_vis['uncertainty'].notna().any():
        corr = df_vis[['logprob', 'uncertainty']].corr().iloc[0, 1]
        axes[2].annotate(f"Pearson r = {corr:.2f}",
                         xy=(0.05, 0.05), xycoords="axes fraction",
                         ha="left", va="top", fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    sc = axes[3].scatter(
        df_vis['logprob'], df_vis['uncertainty'],
        c=df_vis['test_score'], cmap=cmap, alpha=0.7, s=1
    )
    axes[3].set_title(f"Logprob vs. Uncertainty")
    axes[3].set_xlabel("Logprob")
    axes[3].set_ylabel("Uncertainty")
    axes[3].grid(alpha=0.3)

    cbar = plt.colorbar(sc, ax=axes[3])
    cbar.set_label("Test Score")

    plt.tight_layout()
    return fig, axes