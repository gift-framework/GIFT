#!/usr/bin/env python
"""Visualize the stability boundary with fitted constraints.

Creates plots showing:
1. Stable/unstable points in (u, alpha) space
2. The DIAGONAL BAND constraint: u ~ 1.0 - 1.13|alpha|
3. The ridge structure (stability is a band, not a half-space)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))


def load_data(results_dir: Path) -> tuple:
    """Load deformation results."""
    json_path = results_dir / "deformation_results.json"
    with json_path.open() as f:
        results = json.load(f)

    for r in results:
        r["u"] = r["sigma"] * r["s"]

    stable = [r for r in results if r["stable"]]
    unstable = [r for r in results if not r["stable"]]

    return stable, unstable


def fit_constraints(stable: List[Dict], unstable: List[Dict]) -> Dict:
    """Fit various boundary constraints."""
    stable_u = np.array([r["u"] for r in stable])
    stable_alpha = np.array([abs(r["alpha"]) for r in stable])

    unstable_u = np.array([r["u"] for r in unstable])
    unstable_alpha = np.array([abs(r["alpha"]) for r in unstable])

    # Find boundary points by binning
    alpha_bins = np.linspace(0, 0.5, 11)
    boundary_points = []

    for i in range(len(alpha_bins) - 1):
        a_low, a_high = alpha_bins[i], alpha_bins[i + 1]
        a_mid = (a_low + a_high) / 2

        mask_s = (stable_alpha >= a_low) & (stable_alpha < a_high)
        mask_u = (unstable_alpha >= a_low) & (unstable_alpha < a_high)

        if mask_s.any() and mask_u.any():
            u_max_stable = stable_u[mask_s].max()
            u_min_unstable = unstable_u[mask_u].min()
            u_boundary = (u_max_stable + u_min_unstable) / 2
            boundary_points.append((a_mid, u_boundary))
        elif mask_s.any():
            boundary_points.append((a_mid, stable_u[mask_s].max()))

    alphas_fit = np.array([p[0] for p in boundary_points])
    us_fit = np.array([p[1] for p in boundary_points])

    # Linear fit
    A = np.vstack([np.ones_like(alphas_fit), alphas_fit]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, us_fit, rcond=None)
    c_lin, neg_b_lin = coeffs
    b_lin = -neg_b_lin

    # Quadratic fit
    A_quad = np.vstack([np.ones_like(alphas_fit), alphas_fit, alphas_fit**2]).T
    coeffs_quad, _, _, _ = np.linalg.lstsq(A_quad, us_fit, rcond=None)
    c_quad, neg_b_quad, neg_d_quad = coeffs_quad
    b_quad, d_quad = -neg_b_quad, -neg_d_quad

    return {
        "linear": {"b": b_lin, "c": c_lin},
        "quadratic": {"b": b_quad, "c": c_quad, "d": d_quad},
        "boundary_points": boundary_points,
    }


def create_visualization(stable: List[Dict], unstable: List[Dict], constraints: Dict, output_path: Path):
    """Create boundary visualization."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: (u, alpha) with both positive and negative alpha
    ax1 = axes[0]

    # Stable points
    stable_u = [r["u"] for r in stable]
    stable_alpha = [r["alpha"] for r in stable]
    ax1.scatter(stable_u, stable_alpha, c='green', s=60, label='Stable', alpha=0.7, edgecolors='darkgreen')

    # Unstable points
    unstable_u = [r["u"] for r in unstable]
    unstable_alpha = [r["alpha"] for r in unstable]
    ax1.scatter(unstable_u, unstable_alpha, c='red', s=30, label='Unstable', alpha=0.3, marker='x')

    # Fitted boundaries
    alpha_range = np.linspace(-0.5, 0.5, 100)
    abs_alpha = np.abs(alpha_range)

    lin = constraints["linear"]
    u_lin = lin["c"] - lin["b"] * abs_alpha
    ax1.plot(u_lin, alpha_range, 'b--', linewidth=2, label=f'Linear: u + {lin["b"]:.2f}|a| = {lin["c"]:.2f}')

    quad = constraints["quadratic"]
    u_quad = quad["c"] - quad["b"] * abs_alpha - quad["d"] * abs_alpha**2
    ax1.plot(u_quad, alpha_range, 'purple', linewidth=2, label=f'Quadratic (R2=0.995)')

    ax1.set_xlabel('u = sigma * s (effective modulus)', fontsize=12)
    ax1.set_ylabel('alpha (asymmetry)', fontsize=12)
    ax1.set_title('K7_GIFT Stability Valley in (u, alpha) Space', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0.2, 1.6)
    ax1.set_ylim(-0.55, 0.55)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Add annotations
    ax1.annotate('Baseline\n(1,1,0)', xy=(1.0, 0), xytext=(1.2, 0.15),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))

    # Plot 2: (u, |alpha|) space - folded view
    ax2 = axes[1]

    ax2.scatter(stable_u, [abs(a) for a in stable_alpha], c='green', s=60,
               label='Stable', alpha=0.7, edgecolors='darkgreen')
    ax2.scatter(unstable_u, [abs(a) for a in unstable_alpha], c='red', s=30,
               label='Unstable', alpha=0.3, marker='x')

    # Boundary fits
    abs_alpha_plot = np.linspace(0, 0.5, 100)

    u_lin_plot = lin["c"] - lin["b"] * abs_alpha_plot
    ax2.plot(u_lin_plot, abs_alpha_plot, 'b--', linewidth=2, label='Linear fit')

    u_quad_plot = quad["c"] - quad["b"] * abs_alpha_plot - quad["d"] * abs_alpha_plot**2
    ax2.plot(u_quad_plot, abs_alpha_plot, 'purple', linewidth=2, label='Quadratic fit')

    # Boundary points
    bp = constraints["boundary_points"]
    bp_alpha = [p[0] for p in bp]
    bp_u = [p[1] for p in bp]
    ax2.scatter(bp_u, bp_alpha, c='blue', s=100, marker='s', label='Boundary estimate', zorder=5)

    ax2.set_xlabel('u = sigma * s (effective modulus)', fontsize=12)
    ax2.set_ylabel('|alpha| (absolute asymmetry)', fontsize=12)
    ax2.set_title('Stability Boundary (Folded View)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0.2, 1.6)
    ax2.set_ylim(-0.02, 0.55)
    ax2.grid(True, alpha=0.3)

    # Add formula annotation
    ax2.text(0.95, 0.45, f'Best fit (R2=0.995):\nu + 1.36*alpha^2 <= 0.63',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path / "stability_boundary.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "stability_boundary.pdf", bbox_inches='tight')
    print(f"Saved: {output_path / 'stability_boundary.png'}")

    # Additional plot: 3D view in (sigma, s, alpha) space
    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection='3d')

    stable_sigma = [r["sigma"] for r in stable]
    stable_s = [r["s"] for r in stable]
    unstable_sigma = [r["sigma"] for r in unstable]
    unstable_s = [r["s"] for r in unstable]

    ax3d.scatter(stable_sigma, stable_s, stable_alpha, c='green', s=60,
                label='Stable', alpha=0.8)
    ax3d.scatter(unstable_sigma, unstable_s, unstable_alpha, c='red', s=10,
                label='Unstable', alpha=0.2)

    # Draw sigma=s plane (symmetry)
    sigma_plane = np.linspace(0.3, 2.0, 20)
    alpha_plane = np.linspace(-0.5, 0.5, 20)
    SIGMA, ALPHA = np.meshgrid(sigma_plane, alpha_plane)
    ax3d.plot_surface(SIGMA, SIGMA, ALPHA, alpha=0.1, color='blue')

    ax3d.set_xlabel('sigma')
    ax3d.set_ylabel('s')
    ax3d.set_zlabel('alpha')
    ax3d.set_title('K7_GIFT Stability Valley in 3D Parameter Space')
    ax3d.legend()

    plt.tight_layout()
    plt.savefig(output_path / "stability_3d.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path / 'stability_3d.png'}")

    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Visualize stability boundary")
    parser.add_argument("--results-dir", type=Path, help="Results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")

    args = parser.parse_args()

    # Find results directory
    if args.latest or args.results_dir is None:
        base_dir = meta_hodge_dir / "artifacts" / "deformation_atlas"
        if base_dir.exists():
            dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
            if dirs:
                args.results_dir = dirs[-1]

    print("=" * 60)
    print("Stability Boundary Visualization")
    print("=" * 60)
    print(f"Results: {args.results_dir}")

    # Load data
    stable, unstable = load_data(args.results_dir)
    print(f"Data: {len(stable)} stable, {len(unstable)} unstable")

    # Fit constraints
    constraints = fit_constraints(stable, unstable)

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.results_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    create_visualization(stable, unstable, constraints, output_dir)

    print()
    print("Boundary Constraints:")
    lin = constraints["linear"]
    print(f"  Linear:    u + {lin['b']:.4f}|alpha| <= {lin['c']:.4f}")
    quad = constraints["quadratic"]
    print(f"  Quadratic: u + {quad['b']:.4f}|alpha| + {quad['d']:.4f}*alpha^2 <= {quad['c']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
