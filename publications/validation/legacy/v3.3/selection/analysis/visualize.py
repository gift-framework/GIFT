"""Visualization: Pareto plots, rank histograms, null distributions."""

from __future__ import annotations
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_pareto(all_scores, frontier, gift_scores, observable_name, output_dir):
    if not HAS_MPL:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    comps = [s["comp"] for s in all_scores]
    errs = [max(s["err"], 1e-10) for s in all_scores]
    ax.scatter(comps, errs, alpha=0.3, s=10, c='gray', label='All formulas')
    f_comps = [s["comp"] for s in frontier]
    f_errs = [max(s["err"], 1e-10) for s in frontier]
    ax.plot(f_comps, f_errs, 'b-o', markersize=4, linewidth=1.5, label='Pareto frontier')
    ax.scatter([gift_scores["comp"]], [max(gift_scores["err"], 1e-10)],
               c='red', s=200, marker='*', zorder=5, label='GIFT formula')
    ax.set_xlabel('Complexity', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(f'Pareto Frontier: {observable_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, 'pareto_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_null_distribution(null_errors, gift_error, p_value, observable_name,
                           output_dir, null_type="random"):
    if not HAS_MPL or not null_errors:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(null_errors, bins=50, alpha=0.7, color='steelblue',
            label=f'Null distribution (n={len(null_errors)})')
    ax.axvline(gift_error, color='red', linewidth=2, linestyle='--',
               label=f'GIFT (p={p_value:.4f})')
    ax.set_xlabel('Error score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Null Model ({null_type}): {observable_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f'null_{null_type}_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_rank_histogram(all_scores, gift_rank, observable_name, output_dir,
                        metric="total"):
    if not HAS_MPL:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    scores = sorted([s[metric] for s in all_scores])
    n = len(scores)
    ax.bar(range(n), scores, color='steelblue', alpha=0.5, width=1.0)
    if 0 < gift_rank <= n:
        ax.axvline(gift_rank - 1, color='red', linewidth=2, linestyle='--',
                   label=f'GIFT rank: #{gift_rank}/{n}')
    ax.set_xlabel(f'Rank by {metric}', fontsize=12)
    ax.set_ylabel(f'{metric} score', fontsize=12)
    ax.set_title(f'Formula Ranking: {observable_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(output_dir, f'rank_{metric}_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path
