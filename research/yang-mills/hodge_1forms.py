#!/usr/bin/env python3
"""
Hodge Laplacian on 1-Forms for TCS Spectral Study
=================================================

This module implements the discrete Hodge Laplacian on 1-forms,
which is closer to the Yang-Mills relevant operator than the scalar Laplacian.

Background:
- Yang-Mills mass gap concerns gauge fields (1-forms with values in Lie algebra)
- Scalar Laplacian Δ₀ acts on functions (0-forms)
- Hodge Laplacian Δ₁ = dd* + d*d acts on 1-forms
- For G₂ manifolds with Ric=0: Δ₁ = ∇*∇ (Weitzenböck identity)

Key question:
- Does λ₁(Δ₁) × H* show the same "13 regime" as λ₁(Δ₀)?
- Is the 1-form spectrum more natural for the universal law?

Author: GIFT Project
Date: January 2026
Status: IMPLEMENTATION
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime


# =============================================================================
# DISCRETE DIFFERENTIAL GEOMETRY
# =============================================================================

def build_edge_list(W: np.ndarray, threshold: float = 1e-10) -> List[Tuple[int, int, float]]:
    """
    Extract edge list from adjacency matrix.

    Returns list of (i, j, weight) tuples with i < j.
    """
    n = W.shape[0]
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > threshold:
                edges.append((i, j, float(W[i, j])))

    return edges


def build_incidence_matrix(n_vertices: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    Build vertex-edge incidence matrix d₀.

    d₀: C⁰(vertices) → C¹(edges)
    (d₀f)(e) = f(target(e)) - f(source(e))

    For edge e = (i,j) with i < j:
    d₀[e, i] = -1
    d₀[e, j] = +1

    Returns (m, n) matrix where m = #edges, n = #vertices.
    """
    m = len(edges)
    d0 = np.zeros((m, n_vertices), dtype=np.float32)

    for e_idx, (i, j, w) in enumerate(edges):
        d0[e_idx, i] = -1.0
        d0[e_idx, j] = +1.0

    return d0


def build_edge_weight_matrix(edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    Build diagonal edge weight matrix.

    W_e[e, e] = w(e) = edge weight
    """
    m = len(edges)
    weights = np.array([w for (i, j, w) in edges], dtype=np.float32)
    return np.diag(weights)


# =============================================================================
# HODGE LAPLACIANS
# =============================================================================

def hodge_laplacian_0(W: np.ndarray) -> np.ndarray:
    """
    Standard graph Laplacian (Hodge Laplacian on 0-forms).

    Δ₀ = d₀* d₀ = D - W (unnormalized)

    For comparison with 1-form Laplacian.
    """
    d = W.sum(axis=1)
    return np.diag(d) - W


def hodge_laplacian_1_unweighted(n_vertices: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    Hodge Laplacian on 1-forms (unweighted version).

    For a graph without 2-simplices (no triangles considered):
    Δ₁ = d₀ᵀ d₀

    This is the "up-Laplacian" Δ₁⁺ = d₀ᵀ d₀.
    The "down-Laplacian" Δ₁⁻ = d₁ d₁ᵀ requires 2-forms (triangles).

    For now, we use Δ₁ = Δ₁⁺ (good approximation for sparse graphs).
    """
    d0 = build_incidence_matrix(n_vertices, edges)

    # Δ₁ = d₀ᵀ d₀ acts on edge space
    L1 = d0 @ d0.T

    return L1


def hodge_laplacian_1_weighted(
    n_vertices: int,
    edges: List[Tuple[int, int, float]],
    vertex_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Weighted Hodge Laplacian on 1-forms.

    With vertex measure μ and edge weights w:
    Δ₁ = d₀ᵀ M_μ⁻¹ d₀ W_e

    where M_μ = diag(μ(v)) and W_e = diag(w(e)).

    For uniform vertex measure: M_μ = I.
    """
    m = len(edges)
    d0 = build_incidence_matrix(n_vertices, edges)

    # Edge weights
    W_e = build_edge_weight_matrix(edges)

    # Vertex weights (default: degree-based or uniform)
    if vertex_weights is None:
        # Use degree as vertex measure
        vertex_weights = np.zeros(n_vertices)
        for (i, j, w) in edges:
            vertex_weights[i] += w
            vertex_weights[j] += w
        vertex_weights = np.maximum(vertex_weights, 1e-10)

    M_mu_inv = np.diag(1.0 / vertex_weights)

    # Weighted Laplacian
    # Δ₁ = W_e^{1/2} d₀ M_μ⁻¹ d₀ᵀ W_e^{1/2}  (symmetric form)
    W_e_sqrt = np.sqrt(W_e)
    L1 = W_e_sqrt @ d0 @ M_mu_inv @ d0.T @ W_e_sqrt

    return L1


def hodge_laplacian_1_normalized(
    n_vertices: int,
    edges: List[Tuple[int, int, float]]
) -> np.ndarray:
    """
    Normalized Hodge Laplacian on 1-forms.

    Analogous to normalized graph Laplacian for 0-forms.
    Spectrum in [0, 2] for connected graph.
    """
    L1 = hodge_laplacian_1_unweighted(n_vertices, edges)

    # Degree on edge space: sum of weights of adjacent edges
    m = len(edges)
    d_edge = np.maximum(np.sum(np.abs(L1), axis=1), 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(d_edge)

    # Normalized: L̃₁ = D_e^{-1/2} L₁ D_e^{-1/2}
    L1_norm = np.diag(d_inv_sqrt) @ L1 @ np.diag(d_inv_sqrt)

    return L1_norm


# =============================================================================
# SPECTRAL COMPUTATION
# =============================================================================

def compute_1form_spectrum(
    L1: np.ndarray,
    n_eigenvalues: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smallest eigenvalues of Hodge 1-form Laplacian.

    Note: Δ₁ may have many zero eigenvalues corresponding to
    harmonic 1-forms (cycles in the graph).
    """
    L1_sparse = csr_matrix(L1)

    try:
        eigenvalues, eigenvectors = eigsh(L1_sparse, k=min(n_eigenvalues, L1.shape[0]-1),
                                          which='SM', tol=1e-8)
    except Exception as e:
        # Fallback to dense
        eigenvalues, eigenvectors = np.linalg.eigh(L1)
        eigenvalues = eigenvalues[:n_eigenvalues]
        eigenvectors = eigenvectors[:, :n_eigenvalues]

    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


# =============================================================================
# TCS 1-FORM ANALYSIS
# =============================================================================

@dataclass
class OneFormResult:
    """Result of 1-form spectral analysis."""
    n_vertices: int
    n_edges: int
    eigenvalues_0form: List[float]
    eigenvalues_1form: List[float]
    lambda1_0form: float
    lambda1_1form: float
    ratio_1form_0form: float
    product_0form: float
    product_1form: float
    H_star: int
    config: Dict

    def to_dict(self) -> dict:
        return {
            "n_vertices": self.n_vertices,
            "n_edges": self.n_edges,
            "eigenvalues_0form": self.eigenvalues_0form,
            "eigenvalues_1form": self.eigenvalues_1form,
            "lambda1_0form": self.lambda1_0form,
            "lambda1_1form": self.lambda1_1form,
            "ratio_1form_0form": self.ratio_1form_0form,
            "product_0form": self.product_0form,
            "product_1form": self.product_1form,
            "H_star": self.H_star,
            "config": self.config
        }


def analyze_TCS_1forms(
    N: int = 3000,
    k: int = 40,
    ratio: float = 1.18,
    seed: int = 42,
    H_star: int = 99
) -> OneFormResult:
    """
    Compare 0-form and 1-form Laplacian spectra on TCS.

    This is the key test for Move #4.
    """
    from mode_localization import (
        sample_TCS_with_coords,
        tcs_distance_matrix
    )

    # Sample TCS
    theta, q1, q2, _ = sample_TCS_with_coords(N, seed, ratio)

    # Distance matrix
    D = tcs_distance_matrix(theta, q1, q2, ratio)

    # Sigma (median kNN)
    knn_dists = np.partition(D, k, axis=1)[:, 1:k+1]
    sigma = float(np.median(knn_dists))

    # Build adjacency matrix
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # k-NN sparsification
    for i in range(N):
        threshold = np.partition(W[i], -k)[-k]
        W[i, W[i] < threshold] = 0

    W = (W + W.T) / 2

    # Edge list
    edges = build_edge_list(W)
    n_edges = len(edges)

    # 0-form Laplacian (for comparison)
    d = W.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-10))
    L0 = np.eye(N) - np.diag(d_inv_sqrt) @ W @ np.diag(d_inv_sqrt)

    eigenvalues_0, _ = eigsh(csr_matrix(L0), k=10, which='SM', tol=1e-8)
    eigenvalues_0 = np.sort(eigenvalues_0)

    # First non-zero eigenvalue (0-form)
    lambda1_0 = 0.0
    for ev in eigenvalues_0:
        if ev > 1e-8:
            lambda1_0 = float(ev)
            break
    if lambda1_0 == 0 and len(eigenvalues_0) > 1:
        lambda1_0 = float(eigenvalues_0[1])

    # 1-form Laplacian
    L1 = hodge_laplacian_1_normalized(N, edges)

    eigenvalues_1, _ = compute_1form_spectrum(L1, n_eigenvalues=10)

    # First non-zero eigenvalue (1-form)
    # Note: 1-form Laplacian may have zero eigenvalues for each independent cycle
    lambda1_1 = 0.0
    for ev in eigenvalues_1:
        if ev > 1e-6:  # Slightly higher threshold for 1-forms
            lambda1_1 = float(ev)
            break
    if lambda1_1 == 0 and len(eigenvalues_1) > 1:
        lambda1_1 = float(eigenvalues_1[min(5, len(eigenvalues_1)-1)])  # Skip more zeros

    # ε-rescaling
    lambda1_0_hat = lambda1_0 / (sigma**2)
    lambda1_1_hat = lambda1_1 / (sigma**2)

    return OneFormResult(
        n_vertices=N,
        n_edges=n_edges,
        eigenvalues_0form=eigenvalues_0[:5].tolist(),
        eigenvalues_1form=eigenvalues_1[:5].tolist(),
        lambda1_0form=lambda1_0_hat,
        lambda1_1form=lambda1_1_hat,
        ratio_1form_0form=lambda1_1_hat / lambda1_0_hat if lambda1_0_hat > 0 else float('inf'),
        product_0form=lambda1_0_hat * H_star,
        product_1form=lambda1_1_hat * H_star,
        H_star=H_star,
        config={
            "N": N,
            "k": k,
            "ratio": ratio,
            "seed": seed,
            "sigma": sigma
        }
    )


def sweep_1form_analysis(
    ratios: List[float],
    N: int = 3000,
    k: int = 40,
    seed: int = 42,
    H_star: int = 99
) -> List[OneFormResult]:
    """
    Sweep ratios comparing 0-form and 1-form spectra.
    """
    results = []

    print(f"Sweeping {len(ratios)} ratios (N={N}, k={k})...")
    print("-" * 70)

    for ratio in ratios:
        print(f"  ratio={ratio:.2f}...", end=" ", flush=True)

        result = analyze_TCS_1forms(N, k, ratio, seed, H_star)

        print(f"λ₁(Δ₀)×H*={result.product_0form:6.2f}, "
              f"λ₁(Δ₁)×H*={result.product_1form:6.2f}, "
              f"ratio={result.ratio_1form_0form:.3f}")

        results.append(result)

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_1form_comparison(results: List[OneFormResult], save_path: Optional[Path] = None):
    """
    Compare 0-form and 1-form spectral products.
    """
    import matplotlib.pyplot as plt

    ratios = [r.config["ratio"] for r in results]
    products_0 = [r.product_0form for r in results]
    products_1 = [r.product_1form for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Both products vs ratio
    ax = axes[0]
    ax.plot(ratios, products_0, 'o-', markersize=8, linewidth=2, label='Δ₀ (scalar)')
    ax.plot(ratios, products_1, 's--', markersize=8, linewidth=2, label='Δ₁ (1-form)')
    ax.axhline(y=13, color='red', linestyle=':', alpha=0.7, label='13')
    ax.axhline(y=21, color='blue', linestyle=':', alpha=0.7, label='21')
    ax.axvline(x=99/84, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('λ₁ × H*')
    ax.set_title('Spectral Product: 0-Form vs 1-Form')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Ratio λ₁(Δ₁)/λ₁(Δ₀)
    ax = axes[1]
    ratios_1_0 = [r.ratio_1form_0form for r in results]
    ax.plot(ratios, ratios_1_0, 'D-', markersize=8, linewidth=2, color='purple')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('TCS Ratio')
    ax.set_ylabel('λ₁(Δ₁) / λ₁(Δ₀)')
    ax.set_title('1-Form vs 0-Form Eigenvalue Ratio')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HODGE 1-FORM LAPLACIAN ANALYSIS - Move #4 Implementation")
    print("=" * 70 + "\n")

    # Test ratios
    ratios = [0.8, 1.0, 1.18, 1.4, 1.6, 2.0]

    # Run analysis
    results = sweep_1form_analysis(
        ratios,
        N=2000,  # Smaller for quick test
        k=30,
        seed=42,
        H_star=99
    )

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    results_dicts = [r.to_dict() for r in results]
    with open(output_dir / "hodge_1form_results.json", "w") as f:
        json.dump(results_dicts, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'hodge_1form_results.json'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: 0-Form vs 1-Form Comparison")
    print("=" * 60)

    print(f"\n{'Ratio':>6} | {'Δ₀×H*':>8} | {'Δ₁×H*':>8} | {'Δ₁/Δ₀':>7}")
    print("-" * 40)
    for r in results:
        print(f"{r.config['ratio']:6.2f} | {r.product_0form:8.2f} | {r.product_1form:8.2f} | {r.ratio_1form_0form:7.3f}")

    # Key insight
    print("\n" + "=" * 60)
    print("KEY QUESTION:")
    print("Is the '13 regime' more natural for Δ₁ than for Δ₀?")
    print("=" * 60)

    # Find where each is closest to 13
    closest_0 = min(results, key=lambda r: abs(r.product_0form - 13))
    closest_1 = min(results, key=lambda r: abs(r.product_1form - 13))

    print(f"\nΔ₀ closest to 13: ratio={closest_0.config['ratio']:.2f}, product={closest_0.product_0form:.2f}")
    print(f"Δ₁ closest to 13: ratio={closest_1.config['ratio']:.2f}, product={closest_1.product_1form:.2f}")

    # Plot if matplotlib available
    try:
        plot_1form_comparison(results, output_dir / "hodge_1form_comparison.png")
    except ImportError:
        print("\nMatplotlib not available, skipping plot")
