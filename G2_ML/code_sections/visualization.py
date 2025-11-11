"""
Visualization Functions for GIFT v0.9
======================================

This module contains all plotting and visualization functions:
- Training history (all losses over time)
- Validation metrics table
- Regional heatmaps (M₁, Neck, M₂)
- Convergence plots
- Phase transitions visualization
- Cohomology spectrum plots

Uses matplotlib and seaborn for publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import torch


# ============================================================================
# Plotting Style Setup
# ============================================================================

def setup_plot_style():
    """
    Setup publication-quality plot style.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# Training History Plots
# ============================================================================

def plot_training_history(history, save_path=None, show_phases=True):
    """
    Plot complete training history with all loss components.

    Creates 3×3 grid:
    - Total loss
    - Torsion loss
    - Volume loss
    - Harmonic orthogonality
    - Harmonic determinant
    - Separation loss
    - Boundary loss
    - Decay loss
    - Learning rate

    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        show_phases: If True, show vertical lines for phase transitions
    """
    setup_plot_style()

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    epochs = history['epoch']

    # Phase boundaries
    phase_boundaries = [2000, 5000, 8000] if show_phases else []

    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs, history['loss'], linewidth=1.5, label='Total Loss')
    for pb in phase_boundaries:
        ax1.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Total Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Torsion Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(epochs, history['torsion'], linewidth=1.5, color='orange', label='Torsion')
    for pb in phase_boundaries:
        ax2.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Torsion (log scale)')
    ax2.set_title('Torsion Loss (||∇φ||)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Volume Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(epochs, history['volume'], linewidth=1.5, color='green', label='Volume')
    for pb in phase_boundaries:
        ax3.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Volume Loss (log scale)')
    ax3.set_title('Volume Constraint (det(g) = 1)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Harmonic Orthogonality
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(epochs, history['harmonic_ortho'], linewidth=1.5, color='purple', label='Harmonic Ortho')
    for pb in phase_boundaries:
        ax4.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Orthogonality Loss (log scale)')
    ax4.set_title('Harmonic Orthogonality (||Gram - I||)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Harmonic Determinant
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['det_gram'], linewidth=1.5, color='brown', label='det(Gram)')
    ax5.axhline(0.995, color='red', linestyle='--', label='Target (0.995)', alpha=0.5)
    for pb in phase_boundaries:
        ax5.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('det(Gram)')
    ax5.set_title('Harmonic Determinant')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # 6. Separation Loss
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(epochs, history['separation'], linewidth=1.5, color='cyan', label='Separation')
    for pb in phase_boundaries:
        ax6.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Separation Loss (log scale)')
    ax6.set_title('Diagonal/Off-Diagonal Separation')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # 7. Boundary Loss
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.semilogy(epochs, history['boundary'], linewidth=1.5, color='magenta', label='Boundary')
    for pb in phase_boundaries:
        ax7.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Boundary Loss (log scale)')
    ax7.set_title('Boundary Matching Loss')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # 8. Decay Loss
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.semilogy(epochs, history['decay'], linewidth=1.5, color='olive', label='Decay')
    for pb in phase_boundaries:
        ax8.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Decay Loss (log scale)')
    ax8.set_title('ACyl Decay Loss (exp(-γ|t|/T))')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # 9. Learning Rate
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.semilogy(epochs, history['lr'], linewidth=1.5, color='navy', label='Learning Rate')
    for pb in phase_boundaries:
        ax9.axvline(pb, color='red', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Learning Rate (log scale)')
    ax9.set_title('Learning Rate (Cosine Annealing)')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    fig.suptitle('GIFT v0.9 Training History', fontsize=16, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def plot_test_metrics(test_history, save_path=None):
    """
    Plot test/validation metrics over time.

    Args:
        test_history: Test history dictionary
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = test_history['epoch']

    # 1. Test Torsion
    axes[0, 0].semilogy(epochs, test_history['test_torsion'], 'o-', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Test Torsion (log scale)')
    axes[0, 0].set_title('Test Torsion (||∇φ||)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Test det(Gram)
    axes[0, 1].plot(epochs, test_history['test_det_gram'], 'o-', linewidth=2, markersize=5, color='green')
    axes[0, 1].axhline(0.995, color='red', linestyle='--', label='Target (0.995)', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('det(Gram)')
    axes[0, 1].set_title('Test Harmonic Determinant')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Test ||dφ||_L²
    axes[1, 0].semilogy(epochs, test_history['test_dphi_L2'], 'o-', linewidth=2, markersize=5, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('||dφ||_L² (log scale)')
    axes[1, 0].set_title('Closedness Residual (dφ = 0)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Test ||δφ||_L²
    if 'test_dstar_phi_L2' in test_history and test_history['test_dstar_phi_L2']:
        axes[1, 1].semilogy(epochs, test_history['test_dstar_phi_L2'], 'o-', linewidth=2, markersize=5, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('||δφ||_L² (log scale)')
        axes[1, 1].set_title('Co-closedness Residual (δφ = 0)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'δφ data not available', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Co-closedness Residual')

    fig.suptitle('GIFT v0.9 Test Metrics', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Test metrics plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Validation Metrics Table
# ============================================================================

def create_validation_table(validation_summary, save_path=None):
    """
    Create formatted table of validation metrics.

    Args:
        validation_summary: Dictionary from create_validation_summary()
        save_path: Path to save table image (optional)

    Returns:
        df: Pandas DataFrame with validation metrics
    """
    data = {
        'Metric': [],
        'Value': [],
        'Target': [],
        'Status': []
    }

    # PDE Residuals
    dphi = validation_summary['pde_residuals']['dphi_L2']
    delta_phi = validation_summary['pde_residuals']['delta_phi_L2']

    data['Metric'].extend(['||dφ||_L²', '||δφ||_L²'])
    data['Value'].extend([f'{dphi:.3e}', f'{delta_phi:.3e}'])
    data['Target'].extend(['< 1e-6', '< 1e-6'])
    data['Status'].extend([
        '✓' if dphi < 1e-6 else '⚠',
        '✓' if delta_phi < 1e-6 else '⚠'
    ])

    # Ricci Curvature
    ricci = validation_summary['ricci_norm']
    data['Metric'].append('||Ricc||')
    data['Value'].append(f'{ricci:.3e}')
    data['Target'].append('< 1e-4')
    data['Status'].append('✓' if ricci < 1e-4 else '⚠')

    # Cohomology
    b2 = validation_summary['cohomology']['b2']
    data['Metric'].append('b₂')
    data['Value'].append(f'{b2}')
    data['Target'].append('21')
    data['Status'].append('✓' if b2 == 21 else '⚠')

    # Final metrics
    final = validation_summary['final_metrics']
    data['Metric'].extend(['Train Loss', 'Test Torsion', 'Test det(Gram)'])
    data['Value'].extend([
        f"{final['train_loss']:.3e}" if final['train_loss'] else 'N/A',
        f"{final['test_torsion']:.3e}" if final['test_torsion'] else 'N/A',
        f"{final['test_det_gram']:.6f}" if final['test_det_gram'] else 'N/A'
    ])
    data['Target'].extend(['< 1e-4', '< 1e-6', '≈ 0.995'])
    data['Status'].extend([
        '✓' if final['train_loss'] and final['train_loss'] < 1e-4 else '⚠',
        '✓' if final['test_torsion'] and final['test_torsion'] < 1e-6 else '⚠',
        '✓' if final['test_det_gram'] and abs(final['test_det_gram'] - 0.995) < 0.01 else '⚠'
    ])

    df = pd.DataFrame(data)

    print("\n" + "="*80)
    print("VALIDATION METRICS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    # Create table figure
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # Color code status
        for i in range(1, len(df) + 1):
            status = df.iloc[i-1]['Status']
            color = 'lightgreen' if status == '✓' else 'lightyellow'
            table[(i, 3)].set_facecolor(color)

        plt.title('GIFT v0.9 Validation Metrics', fontsize=14, pad=20)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation table saved: {save_path}")

        plt.show()

    return df


# ============================================================================
# Regional Heatmaps
# ============================================================================

def plot_regional_heatmaps(regional_metrics, save_path=None):
    """
    Plot heatmaps for regional analysis (M₁, Neck, M₂).

    Args:
        regional_metrics: Dictionary from analyze_regions()
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    # Extract data
    regions = ['M1', 'Neck', 'M2']
    metrics = ['phi_norm', 'torsion', 'det_metric', 'condition_number']
    metric_labels = ['||φ||', 'Torsion', 'det(g)', 'Condition #']

    data = np.zeros((len(metrics), len(regions)))
    for i, metric in enumerate(metrics):
        for j, region in enumerate(regions):
            data[i, j] = regional_metrics[region][metric]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(regions)
    ax.set_yticklabels(metric_labels)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    # Annotate cells with values
    for i in range(len(metrics)):
        for j in range(len(regions)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11)

    ax.set_title('Regional Analysis: M₁, Neck, M₂', fontsize=14, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regional heatmap saved: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Convergence Plots
# ============================================================================

def plot_convergence(history, window=100, save_path=None):
    """
    Plot convergence diagnostics with moving average and variance.

    Args:
        history: Training history dictionary
        window: Window size for moving statistics
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    epochs = np.array(history['epoch'])
    losses = np.array(history['loss'])

    # Compute moving average and std
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    moving_std = np.array([np.std(losses[max(0, i-window):i+1]) for i in range(len(losses))])

    # Relative change
    rel_change = np.abs(np.diff(losses)) / (losses[:-1] + 1e-10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. Loss with moving average
    axes[0].semilogy(epochs, losses, alpha=0.5, linewidth=0.5, label='Loss')
    axes[0].semilogy(epochs[window-1:], moving_avg, linewidth=2, color='red', label=f'Moving Avg (window={window})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Loss with Moving Average')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Moving standard deviation
    axes[1].semilogy(epochs, moving_std, linewidth=1.5, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Std(Loss) (log scale)')
    axes[1].set_title('Loss Variability (Moving Std Dev)')
    axes[1].grid(True, alpha=0.3)

    # 3. Relative change
    axes[2].semilogy(epochs[1:], rel_change, linewidth=1, alpha=0.7, color='green')
    axes[2].axhline(0.01, color='red', linestyle='--', label='1% threshold', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Relative Change (log scale)')
    axes[2].set_title('Epoch-to-Epoch Relative Change')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle('Convergence Diagnostics', fontsize=16, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Cohomology Spectrum
# ============================================================================

def plot_cohomology_spectrum(b2_eigenvalues, save_path=None):
    """
    Plot b₂ cohomology spectrum (Gram matrix eigenvalues).

    Args:
        b2_eigenvalues: Sorted eigenvalues from extract_b2_cohomology()
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Eigenvalue spectrum
    ax1.plot(range(len(b2_eigenvalues)), b2_eigenvalues, 'o-', linewidth=2, markersize=6)
    ax1.axhline(0.1, color='red', linestyle='--', label='Threshold (0.1)', alpha=0.5)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('b₂ Cohomology Spectrum (Gram Matrix Eigenvalues)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Log scale
    ax2.semilogy(range(len(b2_eigenvalues)), np.maximum(b2_eigenvalues, 1e-10), 'o-', linewidth=2, markersize=6, color='green')
    ax2.axhline(0.1, color='red', linestyle='--', label='Threshold (0.1)', alpha=0.5)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title('b₂ Spectrum (Log Scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(f'Harmonic 2-Forms Spectrum (b₂ = {(b2_eigenvalues > 0.1).sum()})', fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cohomology spectrum plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Phase Transition Visualization
# ============================================================================

def plot_phase_transitions(history, save_path=None):
    """
    Visualize 4-phase curriculum with loss components.

    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
    """
    setup_plot_style()

    epochs = np.array(history['epoch'])

    # Phase boundaries and names
    phases = [
        (0, 2000, 'Phase 1: Establish Structure'),
        (2000, 5000, 'Phase 2: Impose Torsion'),
        (5000, 8000, 'Phase 3: Refine b₃ + ACyl'),
        (8000, 10000, 'Phase 4: Polish Final')
    ]

    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot total loss
    ax.semilogy(epochs, history['loss'], linewidth=2, color='navy', label='Total Loss')

    # Shade phase regions
    for (start, end, name), color in zip(phases, colors):
        ax.axvspan(start, end, alpha=0.2, color=color, label=name)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Total Loss (log scale)', fontsize=14)
    ax.set_title('4-Phase Curriculum Training', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase transition plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Comprehensive Summary Plot
# ============================================================================

def plot_comprehensive_summary(history, test_history, regional_metrics,
                               b2_eigenvalues, save_dir=None):
    """
    Create comprehensive summary with all key visualizations.

    Args:
        history: Training history
        test_history: Test history
        regional_metrics: Regional analysis results
        b2_eigenvalues: b₂ cohomology eigenvalues
        save_dir: Directory to save individual plots (optional)
    """
    from pathlib import Path

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating comprehensive summary plots...")

    # 1. Training history
    plot_training_history(
        history,
        save_path=save_dir / 'training_history.png' if save_dir else None
    )

    # 2. Test metrics
    plot_test_metrics(
        test_history,
        save_path=save_dir / 'test_metrics.png' if save_dir else None
    )

    # 3. Regional heatmaps
    plot_regional_heatmaps(
        regional_metrics,
        save_path=save_dir / 'regional_heatmaps.png' if save_dir else None
    )

    # 4. Convergence
    plot_convergence(
        history,
        save_path=save_dir / 'convergence.png' if save_dir else None
    )

    # 5. Cohomology spectrum
    plot_cohomology_spectrum(
        b2_eigenvalues,
        save_path=save_dir / 'cohomology_spectrum.png' if save_dir else None
    )

    # 6. Phase transitions
    plot_phase_transitions(
        history,
        save_path=save_dir / 'phase_transitions.png' if save_dir else None
    )

    print("All summary plots generated!")
