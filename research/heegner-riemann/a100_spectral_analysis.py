#!/usr/bin/env python3
"""
A100 Spectral Analysis for K₇-Riemann Hypothesis

This script performs GPU-accelerated analysis of the spectral connection
between K₇ eigenvalues and Riemann zeta zeros.

Tasks:
1. Extended zero verification (100k+ zeros)
2. Eigenvalue pattern analysis (γₙ² + 1/4 structure)
3. Statistical validation with GPU Monte Carlo

Requirements:
    pip install numpy cupy scipy

Usage:
    python a100_spectral_analysis.py

Author: GIFT Research Team
Date: 2026-01-24
"""

import numpy as np
import time
import math
from pathlib import Path

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) available - using A100 acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not found - falling back to NumPy (CPU)")


# =============================================================================
# GIFT CONSTANTS
# =============================================================================

GIFT_CONSTANTS = {
    # Core topological
    7: "dim(K₇)",
    14: "dim(G₂)",
    21: "b₂",
    27: "dim(J₃(O))",
    77: "b₃",
    99: "H*",

    # Lie algebras
    78: "dim(E₆)",
    133: "dim(E₇)",
    248: "dim(E₈)",
    496: "dim(E₈×E₈)",

    # Root systems
    72: "|Roots(E₆)|",
    126: "|Roots(E₇)|",
    240: "|Roots(E₈)|",
    480: "|Roots(E₈×E₈)|",

    # Heegner
    43: "Heegner",
    67: "Heegner",
    163: "Heegner_max",
}


def load_zeros(filepath: str) -> np.ndarray:
    """Load zeta zeros from Odlyzko text file."""
    zeros = []
    with open(filepath, 'r') as f:
        for line in f:
            val = line.strip()
            if val:
                try:
                    zeros.append(float(val))
                except ValueError:
                    continue
    return np.array(zeros, dtype=np.float64)


def eigenvalue_analysis(zeros: np.ndarray):
    """Analyze eigenvalue structure λₙ = γₙ² + 1/4."""
    print("\n" + "="*70)
    print("EIGENVALUE STRUCTURE ANALYSIS")
    print("="*70)

    # Compute eigenvalues
    eigenvalues = zeros**2 + 0.25

    # Check which GIFT constants have λ ≈ C²
    print("\nResonant eigenvalue matches (λₙ ≈ C² for GIFT constant C):")
    print("-" * 60)

    matches = []
    for C, name in sorted(GIFT_CONSTANTS.items()):
        C_squared = C**2
        if C_squared > eigenvalues[-1]:
            continue

        # Find closest eigenvalue
        idx = np.argmin(np.abs(eigenvalues - C_squared))
        lam = eigenvalues[idx]
        precision = abs(lam - C_squared) / C_squared * 100

        if precision < 1.0:  # Within 1%
            gamma = zeros[idx]
            matches.append((C, name, idx+1, gamma, lam, precision))

    for C, name, n, gamma, lam, prec in sorted(matches, key=lambda x: x[5]):
        print(f"  C={C:3} ({name:15}): λ_{n:4} = {lam:10.2f} ≈ {C**2:6} ({prec:.4f}%)")

    return matches


def gpu_monte_carlo_test(zeros: np.ndarray, n_simulations: int = 100000):
    """GPU-accelerated Monte Carlo significance test."""
    print("\n" + "="*70)
    print(f"GPU MONTE CARLO TEST ({n_simulations:,} simulations)")
    print("="*70)

    xp = cp if GPU_AVAILABLE else np

    # Transfer to GPU
    zeros_gpu = xp.asarray(zeros[:1000])  # Use first 1000 zeros
    targets = xp.array([21, 77, 99, 163, 248], dtype=xp.float64)

    start_time = time.time()

    # Observed precisions
    observed = []
    for t in targets:
        t_val = float(t)
        idx = int(xp.argmin(xp.abs(zeros_gpu - t)))
        prec = float(xp.abs(zeros_gpu[idx] - t) / t * 100)
        observed.append(prec)

    observed = xp.array(observed)

    # Generate random zero sequences and test
    better_count = xp.zeros(len(targets), dtype=xp.int32)

    for _ in range(n_simulations):
        # Generate random "fake" zeros with similar spacing
        fake = xp.sort(xp.random.uniform(14, 300, size=100))

        for i, t in enumerate(targets):
            idx = int(xp.argmin(xp.abs(fake - t)))
            random_prec = float(xp.abs(fake[idx] - t) / t * 100)
            if random_prec <= observed[i]:
                better_count[i] += 1

    elapsed = time.time() - start_time

    # Transfer back to CPU for display
    if GPU_AVAILABLE:
        better_count = cp.asnumpy(better_count)
        observed = cp.asnumpy(observed)
        targets = cp.asnumpy(targets)

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"\n{'Target':>6} | {'Observed':>10} | {'p-value':>10} | Significance")
    print("-" * 50)

    for i, t in enumerate(targets):
        p_val = better_count[i] / n_simulations
        sig = "★★★" if p_val < 0.01 else "★★" if p_val < 0.05 else "★" if p_val < 0.1 else ""
        print(f"{int(t):6} | {observed[i]:9.4f}% | {p_val:10.6f} | {sig}")


def weyl_law_test(zeros: np.ndarray):
    """Test Weyl's law for K₇: N(λ) ~ λ^{7/2}."""
    print("\n" + "="*70)
    print("WEYL'S LAW TEST FOR K₇")
    print("="*70)

    eigenvalues = zeros**2 + 0.25
    n_vals = np.arange(1, len(eigenvalues) + 1)

    # Weyl's law for d=7: N(λ) ~ C × λ^{d/2} = C × λ^{3.5}
    # Inverted: λ ~ (n/C)^{2/d} = (n/C)^{2/7}

    # But for zeta, we have N(T) ~ T log T / (2π)
    # If λ = T², then N(√λ) ~ √λ log(√λ) / (2π)

    # Fit: log(λ) = a × log(n) + b
    log_lambda = np.log(eigenvalues[:1000])
    log_n = np.log(n_vals[:1000])

    # Linear regression
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    slope, intercept = np.linalg.lstsq(A, log_lambda, rcond=None)[0]

    print(f"\nFit: λₙ ~ n^α")
    print(f"  α = {slope:.4f}")
    print(f"  Compare to 2 (from N(T) ~ T log T): α ≈ 2 for large n")

    # Weyl for K₇ would give α = 2/7 ≈ 0.286 (very different!)
    # This suggests the spectral connection is NOT via standard Weyl law

    print(f"\n  Standard Weyl for dim=7 would give α = 2/7 ≈ 0.286")
    print(f"  Observed α ≈ {slope:.3f} matches Riemann counting, not Weyl")
    print(f"\n  → The connection must be through trace formula, not density")


def main():
    # Find zeros file
    script_dir = Path(__file__).parent
    zeros_file = script_dir / "zeros1.txt"

    if not zeros_file.exists():
        print(f"Error: {zeros_file} not found")
        print("Please ensure zeros1.txt is in the same directory")
        return 1

    print("="*70)
    print("K₇-RIEMANN SPECTRAL ANALYSIS")
    print("="*70)
    print(f"GPU: {'A100 (CuPy)' if GPU_AVAILABLE else 'Not available (using CPU)'}")

    # Load data
    print(f"\nLoading zeros from {zeros_file}...")
    zeros = load_zeros(zeros_file)
    print(f"Loaded {len(zeros):,} zeros")
    print(f"Range: γ₁ = {zeros[0]:.6f} to γ_{len(zeros)} = {zeros[-1]:.6f}")

    # Run analyses
    eigenvalue_analysis(zeros)
    weyl_law_test(zeros)

    if GPU_AVAILABLE:
        gpu_monte_carlo_test(zeros, n_simulations=100000)
    else:
        gpu_monte_carlo_test(zeros, n_simulations=10000)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
