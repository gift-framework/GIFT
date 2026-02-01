#!/usr/bin/env python3
"""
Mode Localization Analysis for TCS Spectral Study
==================================================

This module analyzes WHERE the first eigenmode v₁ lives on the TCS manifold.

Key question: Does the mode localize on different factors (S¹, S³₁, S³₂)
depending on the anisotropy ratio?

Hypothesis:
- ratio ≈ 1.18: mode is "neck-dominant" → spectral product ≈ 13
- ratio ≈ 1.4: mode is "S³₂-dominant" → spectral product ≈ 21

Diagnostics:
1. Participation ratio (localization vs delocalization)
2. Factor correlation (which factor dominates)
3. Spatial variance decomposition

Author: GIFT Project
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags, eye
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# LOCALIZATION METRICS
# =============================================================================

def participation_ratio(v: np.ndarray) -> float:
    """
    Compute participation ratio (inverse participation ratio normalized).

    PR = 1 / (N × Σᵢ |vᵢ|⁴)

    Interpretation:
    - PR → 1/N: fully localized (lives on ~1 site)
    - PR → 1: fully delocalized (uniform distribution)

    For random delocalized state: PR ≈ 1/3 (due to fluctuations)
    """
    v = np.asarray(v).flatten()
    v_normalized = v / np.linalg.norm(v)
    N = len(v)

    ipr = np.sum(v_normalized**4)  # Inverse participation ratio
    pr = 1.0 / (N * ipr)

    return float(pr)


def entropy_localization(v: np.ndarray) -> float:
    """
    Shannon entropy of |v|².

    S = -Σᵢ pᵢ log(pᵢ)  where pᵢ = |vᵢ|²

    Maximum entropy = log(N) for uniform distribution.
    Returns normalized entropy S / log(N) ∈ [0, 1].
    """
    v = np.asarray(v).flatten()
    p = np.abs(v)**2
    p = p / np.sum(p)  # Normalize to probability

    # Avoid log(0)
    p_nonzero = p[p > 1e-15]
    S = -np.sum(p_nonzero * np.log(p_nonzero))

    S_max = np.log(len(v))
    return float(S / S_max)


def localization_length(v: np.ndarray, coords: np.ndarray) -> float:
    """
    Estimate localization length from mode profile.

    Uses the second moment of the probability distribution.
    ξ = sqrt(⟨r²⟩ - ⟨r⟩²) where averaging is over |v|²

    coords: (N, d) array of spatial coordinates
    """
    v = np.asarray(v).flatten()
    p = np.abs(v)**2
    p = p / np.sum(p)

    # Center of mass
    r_mean = np.sum(p[:, None] * coords, axis=0)

    # Second moment
    r2_mean = np.sum(p[:, None] * coords**2, axis=0)

    # Variance in each direction
    var = r2_mean - r_mean**2

    # Total localization length
    xi = np.sqrt(np.sum(np.maximum(var, 0)))

    return float(xi)


# =============================================================================
# FACTOR CORRELATION (TCS-specific)
# =============================================================================

@dataclass
class FactorDecomposition:
    """Decomposition of mode onto TCS factors."""
    # Fractions (sum to 1)
    s1_fraction: float = 0.0
    s3_1_fraction: float = 0.0
    s3_2_fraction: float = 0.0

    # Variances
    s1_variance: float = 0.0
    s3_1_variance: float = 0.0
    s3_2_variance: float = 0.0

    # Localization center
    s1_center: float = 0.0
    s3_1_center: List[float] = None
    s3_2_center: List[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def weighted_circular_variance(theta: np.ndarray, weights: np.ndarray) -> float:
    """
    Circular variance for S¹ coordinates.

    Uses the resultant length R = |Σ wᵢ exp(i θᵢ)|
    Variance = 1 - R (ranges from 0 to 1)
    """
    weights = weights / np.sum(weights)
    z = np.sum(weights * np.exp(1j * theta))
    R = np.abs(z)
    return float(1 - R)


def weighted_spherical_variance(Q: np.ndarray, weights: np.ndarray) -> float:
    """
    Spherical variance for S³ coordinates.

    Uses the mean direction and dispersion around it.
    """
    weights = weights / np.sum(weights)

    # Weighted mean direction (not normalized)
    Q_mean = np.sum(weights[:, None] * Q, axis=0)
    Q_mean_norm = np.linalg.norm(Q_mean)

    if Q_mean_norm < 1e-10:
        return 1.0  # Uniform distribution

    Q_mean = Q_mean / Q_mean_norm

    # Dispersion: average squared distance to mean
    # d²(q, q_mean) = 2(1 - |q·q_mean|) for S³
    dots = np.abs(Q @ Q_mean)
    variance = np.sum(weights * 2 * (1 - dots))

    return float(variance)


def factor_correlation(
    v: np.ndarray,
    theta: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray
) -> FactorDecomposition:
    """
    Decompose mode v onto TCS factors S¹ × S³₁ × S³₂.

    The idea: measure how much of v's "structure" lives on each factor
    by computing the weighted variance on each.

    High variance on a factor = mode varies a lot there = mode "sees" that factor
    Low variance = mode is constant there = mode doesn't "see" that factor
    """
    v = np.asarray(v).flatten()
    p = np.abs(v)**2
    p = p / np.sum(p)

    # Compute variances on each factor
    var_s1 = weighted_circular_variance(theta, p)
    var_s3_1 = weighted_spherical_variance(q1, p)
    var_s3_2 = weighted_spherical_variance(q2, p)

    # Normalize to fractions
    total_var = var_s1 + var_s3_1 + var_s3_2
    if total_var < 1e-10:
        # Mode is constant everywhere
        frac_s1 = frac_s3_1 = frac_s3_2 = 1/3
    else:
        frac_s1 = var_s1 / total_var
        frac_s3_1 = var_s3_1 / total_var
        frac_s3_2 = var_s3_2 / total_var

    # Compute centers (weighted means)
    z_s1 = np.sum(p * np.exp(1j * theta))
    center_s1 = float(np.angle(z_s1))

    center_s3_1 = np.sum(p[:, None] * q1, axis=0)
    center_s3_1 = center_s3_1 / (np.linalg.norm(center_s3_1) + 1e-10)

    center_s3_2 = np.sum(p[:, None] * q2, axis=0)
    center_s3_2 = center_s3_2 / (np.linalg.norm(center_s3_2) + 1e-10)

    return FactorDecomposition(
        s1_fraction=frac_s1,
        s3_1_fraction=frac_s3_1,
        s3_2_fraction=frac_s3_2,
        s1_variance=var_s1,
        s3_1_variance=var_s3_1,
        s3_2_variance=var_s3_2,
        s1_center=center_s1,
        s3_1_center=center_s3_1.tolist(),
        s3_2_center=center_s3_2.tolist()
    )


# =============================================================================
# FULL MODE ANALYSIS
# =============================================================================

@dataclass
class ModeAnalysis:
    """Complete analysis of an eigenmode."""
    # Identity
    mode_index: int = 1
    eigenvalue: float = 0.0

    # Localization metrics
    participation_ratio: float = 0.0
    entropy_normalized: float = 0.0
    localization_length: float = 0.0

    # Factor decomposition
    factor_decomposition: Optional[FactorDecomposition] = None

    # Mode statistics
    mode_mean: float = 0.0
    mode_std: float = 0.0
    mode_skewness: float = 0.0
    mode_kurtosis: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.factor_decomposition:
            d['factor_decomposition'] = self.factor_decomposition.to_dict()
        return d


def analyze_mode(
    v: np.ndarray,
    eigenvalue: float,
    theta: Optional[np.ndarray] = None,
    q1: Optional[np.ndarray] = None,
    q2: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    mode_index: int = 1
) -> ModeAnalysis:
    """
    Perform complete analysis of an eigenmode.

    Args:
        v: Eigenvector (N,)
        eigenvalue: Corresponding eigenvalue
        theta, q1, q2: TCS coordinates (optional, for factor analysis)
        coords: Generic coordinates (optional, for localization length)
        mode_index: Which mode this is (1 = first non-zero)
    """
    v = np.asarray(v).flatten()

    # Basic localization
    pr = participation_ratio(v)
    entropy = entropy_localization(v)

    # Localization length (if coords provided)
    if coords is not None:
        xi = localization_length(v, coords)
    else:
        xi = 0.0

    # Factor decomposition (if TCS coordinates provided)
    if theta is not None and q1 is not None and q2 is not None:
        factor = factor_correlation(v, theta, q1, q2)
    else:
        factor = None

    # Mode statistics
    from scipy.stats import skew, kurtosis
    mode_mean = float(np.mean(v))
    mode_std = float(np.std(v))
    mode_skew = float(skew(v))
    mode_kurt = float(kurtosis(v))

    return ModeAnalysis(
        mode_index=mode_index,
        eigenvalue=eigenvalue,
        participation_ratio=pr,
        entropy_normalized=entropy,
        localization_length=xi,
        factor_decomposition=factor,
        mode_mean=mode_mean,
        mode_std=mode_std,
        mode_skewness=mode_skew,
        mode_kurtosis=mode_kurt
    )


# =============================================================================
# TCS ANALYSIS PIPELINE
# =============================================================================

def sample_TCS_with_coords(
    N: int,
    seed: int,
    ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample TCS and return coordinates."""
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2*np.pi, N).astype(np.float32)

    # S³ sampling
    q1 = rng.standard_normal((N, 4)).astype(np.float32)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)

    q2 = rng.standard_normal((N, 4)).astype(np.float32)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    return theta, q1, q2, None


def geodesic_S1(theta: np.ndarray) -> np.ndarray:
    """Geodesic distance on S¹."""
    diff = np.abs(theta[:, None] - theta[None, :])
    return np.minimum(diff, 2*np.pi - diff)


def geodesic_S3(Q: np.ndarray) -> np.ndarray:
    """Geodesic distance on S³."""
    dot = np.clip(np.abs(Q @ Q.T), 0, 1)
    return 2 * np.arccos(dot)


def tcs_distance_matrix(
    theta: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    ratio: float = 1.0,
    det_g: float = 65/32
) -> np.ndarray:
    """Compute TCS distance matrix."""
    d_s1 = geodesic_S1(theta)
    d_s3_1 = geodesic_S3(q1)
    d_s3_2 = geodesic_S3(q2)

    alpha = det_g / (ratio**3)
    d_sq = alpha * d_s1**2 + d_s3_1**2 + (ratio**2) * d_s3_2**2

    return np.sqrt(np.maximum(d_sq, 0))


def build_laplacian_with_eigenvectors(
    D: np.ndarray,
    k: int,
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Build Laplacian and return eigenvalues + eigenvectors."""
    n = D.shape[0]
    k_actual = min(k, n - 1)

    # Gaussian weights
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # k-NN sparsification
    for i in range(n):
        threshold = np.partition(W[i], -k_actual)[-k_actual]
        W[i, W[i] < threshold] = 0

    W = (W + W.T) / 2

    # Symmetric normalized Laplacian
    d = np.maximum(W.sum(axis=1), 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(n) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    # Eigendecomposition
    eigenvalues, eigenvectors = eigsh(csr_matrix(L), k=10, which='SM', tol=1e-8)
    idx = np.argsort(eigenvalues)

    return eigenvalues[idx], eigenvectors[:, idx]


def analyze_TCS_modes(
    N: int = 5000,
    k: int = 50,
    ratio: float = 1.18,
    seed: int = 42,
    H_star: int = 99
) -> Dict:
    """
    Full mode analysis pipeline for TCS at given ratio.

    Returns comprehensive analysis including:
    - Spectral data (eigenvalues, product)
    - Mode localization (PR, entropy)
    - Factor decomposition (S¹, S³₁, S³₂ fractions)
    """
    import time
    t0 = time.time()

    # Sample TCS
    theta, q1, q2, _ = sample_TCS_with_coords(N, seed, ratio)

    # Distance matrix
    D = tcs_distance_matrix(theta, q1, q2, ratio)

    # Sigma (median kNN)
    knn_dists = np.partition(D, k, axis=1)[:, 1:k+1]
    sigma = float(np.median(knn_dists))

    # Laplacian and eigenvectors
    eigenvalues, eigenvectors = build_laplacian_with_eigenvectors(D, k, sigma)

    # Find first non-zero mode
    for i, ev in enumerate(eigenvalues):
        if ev > 1e-8:
            mu1 = ev
            v1 = eigenvectors[:, i]
            mode_idx = i
            break
    else:
        mu1 = eigenvalues[1]
        v1 = eigenvectors[:, 1]
        mode_idx = 1

    # Analyze mode
    mode_analysis = analyze_mode(
        v1, mu1,
        theta=theta, q1=q1, q2=q2,
        mode_index=mode_idx
    )

    # Calibrated eigenvalue
    lambda1_hat = mu1 / (sigma**2)

    elapsed = time.time() - t0

    return {
        "config": {
            "N": N,
            "k": k,
            "ratio": ratio,
            "seed": seed,
            "H_star": H_star
        },
        "spectral": {
            "sigma": sigma,
            "mu1_raw": float(mu1),
            "lambda1_hat": float(lambda1_hat),
            "product_raw": float(mu1 * H_star),
            "product_calibrated": float(lambda1_hat * H_star),
            "eigenvalues": eigenvalues[:5].tolist()
        },
        "mode_analysis": mode_analysis.to_dict(),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# RATIO SWEEP WITH MODE ANALYSIS
# =============================================================================

def sweep_ratios(
    ratios: List[float],
    N: int = 5000,
    k: int = 50,
    seed: int = 42,
    H_star: int = 99
) -> List[Dict]:
    """
    Sweep over ratios and analyze mode structure.

    This is THE diagnostic for understanding the landscape.
    """
    results = []

    print(f"Sweeping {len(ratios)} ratios (N={N}, k={k})...")
    print("-" * 60)

    for ratio in ratios:
        result = analyze_TCS_modes(N, k, ratio, seed, H_star)

        # Extract key metrics
        pr = result["mode_analysis"]["participation_ratio"]
        s1_frac = result["mode_analysis"]["factor_decomposition"]["s1_fraction"]
        s3_1_frac = result["mode_analysis"]["factor_decomposition"]["s3_1_fraction"]
        s3_2_frac = result["mode_analysis"]["factor_decomposition"]["s3_2_fraction"]
        product = result["spectral"]["product_calibrated"]

        print(f"  ratio={ratio:.2f}: λ̂₁×H*={product:6.2f}, "
              f"PR={pr:.3f}, "
              f"S¹={s1_frac:.2f} S³₁={s3_1_frac:.2f} S³₂={s3_2_frac:.2f}")

        results.append(result)

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_mode_analysis(results: List[Dict], save_path: Optional[Path] = None):
    """
    Visualize mode analysis across ratios.

    Creates a 2×2 figure:
    - Top left: λ₁×H* vs ratio
    - Top right: Participation ratio vs ratio
    - Bottom left: Factor fractions vs ratio
    - Bottom right: Entropy vs ratio
    """
    import matplotlib.pyplot as plt

    ratios = [r["config"]["ratio"] for r in results]
    products = [r["spectral"]["product_calibrated"] for r in results]
    prs = [r["mode_analysis"]["participation_ratio"] for r in results]
    entropies = [r["mode_analysis"]["entropy_normalized"] for r in results]

    s1_fracs = [r["mode_analysis"]["factor_decomposition"]["s1_fraction"] for r in results]
    s3_1_fracs = [r["mode_analysis"]["factor_decomposition"]["s3_1_fraction"] for r in results]
    s3_2_fracs = [r["mode_analysis"]["factor_decomposition"]["s3_2_fraction"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Spectral product
    ax = axes[0, 0]
    ax.plot(ratios, products, 'o-', markersize=8, linewidth=2)
    ax.axhline(y=13, color='red', linestyle='--', label='13 (dim G₂ - 1)')
    ax.axhline(y=21, color='blue', linestyle='--', label='21 (b₂)')
    ax.axvline(x=99/84, color='green', linestyle=':', alpha=0.7, label='H*/84')
    ax.set_xlabel('Ratio')
    ax.set_ylabel('λ̂₁ × H*')
    ax.set_title('Spectral Product vs Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Participation ratio
    ax = axes[0, 1]
    ax.plot(ratios, prs, 's-', markersize=8, linewidth=2, color='purple')
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Mode Delocalization vs Ratio')
    ax.grid(True, alpha=0.3)

    # Bottom left: Factor fractions
    ax = axes[1, 0]
    ax.stackplot(ratios, s1_fracs, s3_1_fracs, s3_2_fracs,
                 labels=['S¹', 'S³₁', 'S³₂'],
                 colors=['#ff7f0e', '#2ca02c', '#1f77b4'],
                 alpha=0.7)
    ax.plot(ratios, s1_fracs, 'o-', color='#ff7f0e', markersize=4)
    ax.plot(ratios, [f1+f2 for f1,f2 in zip(s1_fracs, s3_1_fracs)],
            'o-', color='#2ca02c', markersize=4)
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Fraction')
    ax.set_title('Mode Factor Decomposition')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Bottom right: Entropy
    ax = axes[1, 1]
    ax.plot(ratios, entropies, 'd-', markersize=8, linewidth=2, color='brown')
    ax.set_xlabel('Ratio')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Mode Entropy vs Ratio')
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
    print("  MODE LOCALIZATION ANALYSIS - Move #2 Implementation")
    print("=" * 70 + "\n")

    # Key ratios to test
    ratios = [0.8, 1.0, 1.18, 1.3, 1.4, 1.6, 2.0]

    # Run sweep
    results = sweep_ratios(
        ratios,
        N=3000,  # Smaller for quick test
        k=40,
        seed=42,
        H_star=99
    )

    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "mode_localization_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'mode_localization_results.json'}")

    # Plot (if matplotlib available)
    try:
        plot_mode_analysis(results, output_dir / "mode_localization_analysis.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Mode Structure vs Ratio")
    print("=" * 60)

    for r in results:
        ratio = r["config"]["ratio"]
        product = r["spectral"]["product_calibrated"]
        pr = r["mode_analysis"]["participation_ratio"]
        fd = r["mode_analysis"]["factor_decomposition"]

        # Dominant factor
        fracs = [("S¹", fd["s1_fraction"]),
                 ("S³₁", fd["s3_1_fraction"]),
                 ("S³₂", fd["s3_2_fraction"])]
        dominant = max(fracs, key=lambda x: x[1])

        print(f"  ratio={ratio:.2f}: product={product:5.1f}, PR={pr:.3f}, "
              f"dominant={dominant[0]} ({dominant[1]:.0%})")
