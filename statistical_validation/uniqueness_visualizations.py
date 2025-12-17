#!/usr/bin/env python3
"""
Visualization Module for GIFT Uniqueness Tests

Generates publication-quality plots showing the statistical uniqueness
of the GIFT framework's topological configuration.

Author: GIFT Framework Team
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set up publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# GIFT constants
GIFT_B2 = 21
GIFT_B3 = 77


class UniquenessVisualizer:
    """Generate visualizations for uniqueness test results."""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = Path(__file__).parent / "results" / "uniqueness_tests"
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self) -> Dict:
        """Load all available results."""
        results = {}

        # Load Sobol results
        sobol_path = self.results_dir / "sobol_results.csv"
        if sobol_path.exists():
            results['sobol'] = pd.read_csv(sobol_path)

        # Load LHS results
        lhs_path = self.results_dir / "lhs_results.csv"
        if lhs_path.exists():
            results['lhs'] = pd.read_csv(lhs_path)

        # Load grid search results
        grid_path = self.results_dir / "grid_search_results.csv"
        if grid_path.exists():
            results['grid'] = pd.read_csv(grid_path)

        # Load bootstrap results
        bootstrap_path = self.results_dir / "bootstrap_results.json"
        if bootstrap_path.exists():
            with open(bootstrap_path, 'r') as f:
                results['bootstrap'] = json.load(f)

        # Load LEE correction
        lee_path = self.results_dir / "lee_correction.json"
        if lee_path.exists():
            with open(lee_path, 'r') as f:
                results['lee'] = json.load(f)

        return results

    def plot_chi2_distribution(self, df: pd.DataFrame, title: str = "Chi-squared Distribution",
                               save_name: str = "chi2_distribution.png"):
        """Plot histogram of chi-squared values with GIFT highlighted."""
        fig, ax = plt.subplots(figsize=(12, 8))

        gift_row = df[df['is_gift'] == True]
        other_rows = df[df['is_gift'] == False]

        # Plot histogram of alternatives
        alt_chi2 = other_rows['chi2'].values
        alt_chi2_finite = alt_chi2[np.isfinite(alt_chi2)]

        # Use log scale for better visualization
        bins = np.logspace(np.log10(max(1, alt_chi2_finite.min())),
                          np.log10(alt_chi2_finite.max()), 50)

        ax.hist(alt_chi2_finite, bins=bins, alpha=0.7, color='steelblue',
                edgecolor='black', linewidth=0.5, label='Alternative configurations')

        # Mark GIFT configuration
        if len(gift_row) > 0:
            gift_chi2 = gift_row['chi2'].values[0]
            ax.axvline(gift_chi2, color='red', linewidth=3, linestyle='--',
                      label=f'GIFT (b2=21, b3=77): chi2={gift_chi2:.1f}')

            # Add annotation
            ax.annotate(f'GIFT\nchi2={gift_chi2:.1f}',
                       xy=(gift_chi2, ax.get_ylim()[1] * 0.8),
                       fontsize=12, color='red', ha='center')

        ax.set_xscale('log')
        ax.set_xlabel('Chi-squared')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_parameter_space_heatmap(self, df: pd.DataFrame,
                                     title: str = "Chi-squared Landscape",
                                     save_name: str = "parameter_space.png"):
        """Plot heatmap of chi-squared in (b2, b3) parameter space."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create pivot table
        pivot = df.pivot_table(values='chi2', index='b3', columns='b2', aggfunc='min')

        # Apply log transform for better visualization
        log_chi2 = np.log10(pivot.values + 1)

        # Create heatmap
        im = ax.imshow(log_chi2, aspect='auto', origin='lower',
                      cmap='viridis_r',
                      extent=[pivot.columns.min(), pivot.columns.max(),
                             pivot.index.min(), pivot.index.max()])

        # Mark GIFT configuration
        ax.scatter([GIFT_B2], [GIFT_B3], s=300, c='red', marker='*',
                  edgecolor='white', linewidth=2, zorder=5, label='GIFT (21, 77)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='log10(chi2 + 1)')

        ax.set_xlabel('b2 (Second Betti number)')
        ax.set_ylabel('b3 (Third Betti number)')
        ax.set_title(title)
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_contour_map(self, df: pd.DataFrame,
                         title: str = "Chi-squared Contours",
                         save_name: str = "contour_map.png"):
        """Plot contour map of chi-squared in parameter space."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Create grid
        b2_unique = sorted(df['b2'].unique())
        b3_unique = sorted(df['b3'].unique())

        # Create 2D array for contours
        Z = np.full((len(b3_unique), len(b2_unique)), np.nan)

        for _, row in df.iterrows():
            b2_idx = b2_unique.index(row['b2'])
            b3_idx = b3_unique.index(row['b3'])
            chi2_val = row['chi2']
            if np.isfinite(chi2_val):
                Z[b3_idx, b2_idx] = np.log10(chi2_val + 1)

        # Smooth the data
        Z_smooth = gaussian_filter(np.nan_to_num(Z, nan=np.nanmax(Z)), sigma=1)

        B2, B3 = np.meshgrid(b2_unique, b3_unique)

        # Plot contours
        levels = np.linspace(np.nanmin(Z_smooth), np.nanmax(Z_smooth), 20)
        cs = ax.contourf(B2, B3, Z_smooth, levels=levels, cmap='viridis_r')
        ax.contour(B2, B3, Z_smooth, levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)

        # Mark GIFT configuration
        ax.scatter([GIFT_B2], [GIFT_B3], s=400, c='red', marker='*',
                  edgecolor='white', linewidth=2, zorder=5)
        ax.annotate('GIFT\n(21, 77)', xy=(GIFT_B2, GIFT_B3), xytext=(GIFT_B2 + 5, GIFT_B3 + 10),
                   fontsize=12, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))

        cbar = plt.colorbar(cs, ax=ax, label='log10(chi2 + 1)')

        ax.set_xlabel('b2 (Second Betti number)')
        ax.set_ylabel('b3 (Third Betti number)')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_cumulative_distribution(self, df: pd.DataFrame,
                                     title: str = "Cumulative Distribution of Chi-squared",
                                     save_name: str = "cumulative_distribution.png"):
        """Plot cumulative distribution function of chi-squared."""
        fig, ax = plt.subplots(figsize=(12, 8))

        gift_row = df[df['is_gift'] == True]
        other_rows = df[df['is_gift'] == False]

        # Sort chi-squared values
        alt_chi2 = np.sort(other_rows['chi2'].values)
        alt_chi2_finite = alt_chi2[np.isfinite(alt_chi2)]
        cdf = np.arange(1, len(alt_chi2_finite) + 1) / len(alt_chi2_finite)

        # Plot CDF
        ax.plot(alt_chi2_finite, cdf, 'b-', linewidth=2, label='Alternative configurations')

        # Mark GIFT
        if len(gift_row) > 0:
            gift_chi2 = gift_row['chi2'].values[0]
            gift_percentile = np.mean(alt_chi2_finite <= gift_chi2)

            ax.axvline(gift_chi2, color='red', linewidth=2, linestyle='--',
                      label=f'GIFT chi2={gift_chi2:.1f}')
            ax.axhline(gift_percentile, color='red', linewidth=1, linestyle=':')

            ax.annotate(f'GIFT at {gift_percentile*100:.2f}th percentile',
                       xy=(gift_chi2, gift_percentile),
                       xytext=(gift_chi2 * 2, gift_percentile + 0.1),
                       fontsize=11, color='red',
                       arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_xscale('log')
        ax.set_xlabel('Chi-squared')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_bootstrap_distribution(self, bootstrap_results: Dict,
                                    title: str = "Bootstrap Distribution of Chi-squared Difference",
                                    save_name: str = "bootstrap_distribution.png"):
        """Plot bootstrap distribution of (min_alternative - GIFT)."""
        # Note: This requires storing the bootstrap samples, not just summary stats
        # For now, we'll create a simulated visualization based on the summary

        fig, ax = plt.subplots(figsize=(12, 8))

        mean = bootstrap_results['bootstrap_diff_mean']
        std = bootstrap_results['bootstrap_diff_std']
        ci_lower = bootstrap_results['ci_lower']
        ci_upper = bootstrap_results['ci_upper']

        # Generate approximate distribution
        x = np.linspace(mean - 4*std, mean + 4*std, 1000)
        y = stats.norm.pdf(x, mean, std)

        ax.fill_between(x, y, alpha=0.3, color='steelblue')
        ax.plot(x, y, 'b-', linewidth=2, label='Bootstrap distribution')

        # Mark confidence interval
        ax.axvline(ci_lower, color='orange', linewidth=2, linestyle='--',
                  label=f'95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')
        ax.axvline(ci_upper, color='orange', linewidth=2, linestyle='--')

        # Shade CI region
        ci_x = x[(x >= ci_lower) & (x <= ci_upper)]
        ci_y = stats.norm.pdf(ci_x, mean, std)
        ax.fill_between(ci_x, ci_y, alpha=0.4, color='orange')

        # Mark zero
        ax.axvline(0, color='red', linewidth=2, linestyle='-',
                  label='GIFT = Best alternative')

        ax.set_xlabel('Chi-squared difference (min alternative - GIFT)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_significance_summary(self, results: Dict,
                                  save_name: str = "significance_summary.png"):
        """Create summary plot showing significance across all tests."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Chi-squared comparison bar chart
        ax1 = axes[0, 0]
        methods = ['Sobol', 'LHS', 'Grid']
        gift_chi2s = []
        best_alt_chi2s = []

        for method, key in zip(methods, ['sobol', 'lhs', 'grid']):
            if key in results:
                df = results[key]
                gift_row = df[df['is_gift'] == True]
                other_rows = df[df['is_gift'] == False]
                if len(gift_row) > 0:
                    gift_chi2s.append(gift_row['chi2'].values[0])
                    best_alt_chi2s.append(other_rows['chi2'].min())
                else:
                    gift_chi2s.append(0)
                    best_alt_chi2s.append(0)
            else:
                gift_chi2s.append(0)
                best_alt_chi2s.append(0)

        x = np.arange(len(methods))
        width = 0.35

        ax1.bar(x - width/2, gift_chi2s, width, label='GIFT', color='steelblue')
        ax1.bar(x + width/2, best_alt_chi2s, width, label='Best Alternative', color='coral')
        ax1.set_ylabel('Chi-squared')
        ax1.set_title('GIFT vs Best Alternative by Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Percentile ranking
        ax2 = axes[0, 1]
        percentiles = []
        for key in ['sobol', 'lhs', 'grid']:
            if key in results:
                df = results[key]
                gift_row = df[df['is_gift'] == True]
                other_rows = df[df['is_gift'] == False]
                if len(gift_row) > 0:
                    gift_chi2 = gift_row['chi2'].values[0]
                    n_better = len(other_rows[other_rows['chi2'] < gift_chi2])
                    percentile = 100 * (1 - n_better / len(other_rows))
                    percentiles.append(percentile)
                else:
                    percentiles.append(0)
            else:
                percentiles.append(0)

        colors = ['green' if p > 99 else 'orange' if p > 90 else 'red' for p in percentiles]
        ax2.barh(methods, percentiles, color=colors)
        ax2.set_xlabel('Percentile Ranking')
        ax2.set_title('GIFT Uniqueness Percentile')
        ax2.axvline(99, color='green', linestyle='--', label='99th percentile')
        ax2.axvline(95, color='orange', linestyle='--', label='95th percentile')
        ax2.set_xlim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. Bootstrap confidence interval
        ax3 = axes[1, 0]
        if 'bootstrap' in results:
            boot = results['bootstrap']
            ax3.errorbar(['GIFT - Best Alt'],
                        [boot['bootstrap_diff_mean']],
                        yerr=[[boot['bootstrap_diff_mean'] - boot['ci_lower']],
                              [boot['ci_upper'] - boot['bootstrap_diff_mean']]],
                        fmt='o', markersize=10, capsize=10, capthick=2,
                        color='steelblue', ecolor='coral')
            ax3.axhline(0, color='red', linestyle='--', label='Equivalence')
            ax3.set_ylabel('Chi-squared Difference')
            ax3.set_title('Bootstrap 95% CI for Difference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No bootstrap results', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)

        # 4. LEE-corrected significance
        ax4 = axes[1, 1]
        if 'lee' in results:
            lee = results['lee']
            sigmas = [lee['sidak_sigma']]
            labels = ['LEE-corrected']
            colors = ['green' if s > 5 else 'orange' if s > 3 else 'red' for s in sigmas]

            ax4.barh(labels, sigmas, color=colors, height=0.4)
            ax4.axvline(5, color='green', linestyle='--', label='5 sigma (discovery)')
            ax4.axvline(3, color='orange', linestyle='--', label='3 sigma (evidence)')
            ax4.set_xlabel('Significance (sigma)')
            ax4.set_title('Global Significance (LEE-corrected)')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'No LEE results', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)

        plt.suptitle('GIFT Framework Uniqueness Test Summary', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / save_name)
        plt.close()

        print(f"Saved: {self.output_dir / save_name}")

    def plot_all(self):
        """Generate all visualizations."""
        print("\nGenerating visualizations...")

        results = self.load_results()

        if 'sobol' in results:
            print("\nPlotting Sobol results...")
            self.plot_chi2_distribution(results['sobol'],
                                       title="Sobol QMC: Chi-squared Distribution",
                                       save_name="sobol_chi2_distribution.png")
            self.plot_cumulative_distribution(results['sobol'],
                                             title="Sobol QMC: Cumulative Distribution",
                                             save_name="sobol_cumulative.png")

        if 'lhs' in results:
            print("\nPlotting LHS results...")
            self.plot_chi2_distribution(results['lhs'],
                                       title="Latin Hypercube: Chi-squared Distribution",
                                       save_name="lhs_chi2_distribution.png")

        if 'grid' in results:
            print("\nPlotting Grid Search results...")
            self.plot_parameter_space_heatmap(results['grid'],
                                             title="Grid Search: Chi-squared Landscape",
                                             save_name="grid_heatmap.png")
            self.plot_contour_map(results['grid'],
                                 title="Grid Search: Chi-squared Contours",
                                 save_name="grid_contours.png")

        if 'bootstrap' in results:
            print("\nPlotting Bootstrap results...")
            self.plot_bootstrap_distribution(results['bootstrap'],
                                            title="Bootstrap: Difference Distribution",
                                            save_name="bootstrap_distribution.png")

        # Summary plot
        print("\nGenerating summary plot...")
        self.plot_significance_summary(results)

        print(f"\nAll visualizations saved to: {self.output_dir}")


def generate_all_plots(results_dir: str = None):
    """Convenience function to generate all plots."""
    visualizer = UniquenessVisualizer(results_dir)
    visualizer.plot_all()


if __name__ == "__main__":
    generate_all_plots()
