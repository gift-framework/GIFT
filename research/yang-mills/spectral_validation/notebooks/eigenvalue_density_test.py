"""
Test de la formule de densité de Langlais: Λ(s) = 2(b^{q-1} + b^q)√s

Ce script extrait les 50 premières valeurs propres et teste si
N(λ) ~ A√λ avec A = 2(b₂ + b₃) = 196.

Référence: arXiv:2301.03513 (Langlais 2024)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONSTANTS
# =============================================================================

DET_G = 65/32  # G₂ metric determinant
H_STAR = 99    # b₂ + b₃ + 1
B2 = 21
B3 = 77

# Langlais prediction
A_LANGLAIS = 2 * (B2 + B3)  # = 196

print("="*60)
print("  EIGENVALUE DENSITY TEST")
print("  Testing Λ(s) = A√s (Langlais 2024)")
print("="*60)
print(f"H* = {H_STAR}, b₂ = {B2}, b₃ = {B3}")
print(f"Langlais prediction: A = 2(b₂+b₃) = {A_LANGLAIS}")
print()

# =============================================================================
# TCS CONSTRUCTION
# =============================================================================

def sample_S3(n: int, seed: int) -> np.ndarray:
    """Uniform sampling on S³."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)

def geodesic_S3_batched(Q: np.ndarray, batch_size: int = 2000) -> np.ndarray:
    """Geodesic distance matrix on S³."""
    n = Q.shape[0]
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            dot = np.clip(np.abs(Q[i:end_i] @ Q[j:end_j].T), 0, 1)
            D[i:end_i, j:end_j] = 2 * np.arccos(dot)

    return D

def tcs_distance_matrix(n: int, ratio: float, seed: int) -> np.ndarray:
    """TCS distance matrix for K₇ ≈ S¹ × S³ × S³."""
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2*np.pi, n).astype(np.float32)
    theta_diff = np.abs(theta[:, None] - theta[None, :])
    d_S1_sq = np.minimum(theta_diff, 2*np.pi - theta_diff)**2

    q1 = sample_S3(n, seed + 1000).astype(np.float32)
    q2 = sample_S3(n, seed + 2000).astype(np.float32)

    batch_size = min(2000, n)
    d1 = geodesic_S3_batched(q1, batch_size)
    d2 = geodesic_S3_batched(q2, batch_size)

    alpha = DET_G / (ratio**3)
    d_sq = alpha * d_S1_sq + d1**2 + (ratio**2) * d2**2

    return np.sqrt(np.maximum(d_sq, 0))

def ratio_formula(H_star: int) -> float:
    return max(H_star / 84, 0.8)

# =============================================================================
# EIGENVALUE EXTRACTION (MULTIPLE)
# =============================================================================

def compute_eigenvalues(D: np.ndarray, k: int = 25,
                       num_eigenvalues: int = 50) -> np.ndarray:
    """Compute first num_eigenvalues eigenvalues."""
    n = D.shape[0]
    k = min(k, n - 1)

    # k-NN sigma
    knn_dists = np.partition(D, k, axis=1)[:, :k]
    sigma = max(np.median(knn_dists), 1e-10)

    # Gaussian weights
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # Keep k nearest
    for i in range(n):
        idx = np.argpartition(W[i], -k)[-k:]
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        W[i, mask] = 0

    W = (W + W.T) / 2
    d = np.maximum(W.sum(axis=1), 1e-10)

    # Symmetric Laplacian
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(n) - np.diag(d_inv_sqrt) @ W @ np.diag(d_inv_sqrt)

    # Extract eigenvalues
    L_sparse = sp.csr_matrix(L)
    num_to_extract = min(num_eigenvalues + 5, n - 1)
    eigs, _ = eigsh(L_sparse, k=num_to_extract, which='SM', tol=1e-8)
    eigs = np.sort(np.real(eigs))

    # Filter non-zero
    eigs = eigs[eigs > 1e-8]

    return eigs[:num_eigenvalues]

# =============================================================================
# DENSITY ANALYSIS
# =============================================================================

def counting_function(eigenvalues: np.ndarray) -> tuple:
    """
    Compute N(λ) = number of eigenvalues ≤ λ.
    Returns (lambda_values, N_values).
    """
    sorted_eigs = np.sort(eigenvalues)
    N_values = np.arange(1, len(sorted_eigs) + 1)
    return sorted_eigs, N_values

def fit_density(lambda_values: np.ndarray, N_values: np.ndarray) -> dict:
    """
    Fit N(λ) = A√λ + B to the data.
    """
    def model(x, A, B):
        return A * np.sqrt(x) + B

    try:
        popt, pcov = curve_fit(model, lambda_values, N_values, p0=[100, 0])
        A_fit, B_fit = popt
        A_err, B_err = np.sqrt(np.diag(pcov))

        # R² calculation
        y_pred = model(lambda_values, A_fit, B_fit)
        ss_res = np.sum((N_values - y_pred)**2)
        ss_tot = np.sum((N_values - np.mean(N_values))**2)
        r_squared = 1 - ss_res / ss_tot

        return {
            "A": A_fit,
            "A_err": A_err,
            "B": B_fit,
            "B_err": B_err,
            "R_squared": r_squared,
            "success": True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# =============================================================================
# MAIN TEST
# =============================================================================

def run_density_test(N: int = 5000, k: int = 25, num_eigs: int = 50, seed: int = 42):
    """Run the full density test."""
    print(f"Configuration: N={N}, k={k}, extracting {num_eigs} eigenvalues")
    print()

    # Build TCS
    print("Building TCS distance matrix...")
    ratio = ratio_formula(H_STAR)
    D = tcs_distance_matrix(N, ratio, seed)
    print(f"  Done. Shape: {D.shape}")

    # Extract eigenvalues
    print(f"Extracting {num_eigs} eigenvalues...")
    eigenvalues = compute_eigenvalues(D, k, num_eigs)
    print(f"  Got {len(eigenvalues)} eigenvalues")
    print(f"  λ₁ = {eigenvalues[0]:.6f}")
    print(f"  λ₁ × H* = {eigenvalues[0] * H_STAR:.4f}")
    print(f"  λ_max = {eigenvalues[-1]:.6f}")
    print()

    # Counting function
    lambda_vals, N_vals = counting_function(eigenvalues)

    # Fit density
    print("Fitting N(λ) = A√λ + B...")
    fit_result = fit_density(lambda_vals, N_vals)

    if fit_result["success"]:
        print(f"  A = {fit_result['A']:.2f} ± {fit_result['A_err']:.2f}")
        print(f"  B = {fit_result['B']:.2f} ± {fit_result['B_err']:.2f}")
        print(f"  R² = {fit_result['R_squared']:.4f}")
        print()

        # Compare to Langlais
        A_ratio = fit_result['A'] / A_LANGLAIS
        print(f"Comparison to Langlais (A = {A_LANGLAIS}):")
        print(f"  A_fit / A_Langlais = {A_ratio:.4f}")

        if abs(A_ratio - 1) < 0.2:
            print("  → COMPATIBLE with Langlais formula! ✓")
        else:
            print(f"  → DEVIATION: factor {A_ratio:.2f}x from prediction")
    else:
        print(f"  Fit failed: {fit_result['error']}")

    return {
        "N": N,
        "k": k,
        "num_eigenvalues": len(eigenvalues),
        "eigenvalues": eigenvalues.tolist(),
        "lambda1": float(eigenvalues[0]),
        "lambda1_x_Hstar": float(eigenvalues[0] * H_STAR),
        "fit": fit_result,
        "A_Langlais": A_LANGLAIS,
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_density(results: dict, output_path: str = "eigenvalue_density.png"):
    """Plot N(λ) vs √λ."""
    eigs = np.array(results["eigenvalues"])
    lambda_vals, N_vals = counting_function(eigs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: N(λ) vs λ
    ax1.scatter(lambda_vals, N_vals, c='blue', alpha=0.7, s=30, label='Data')

    if results["fit"]["success"]:
        A = results["fit"]["A"]
        B = results["fit"]["B"]
        x_fit = np.linspace(0.001, lambda_vals.max(), 100)
        ax1.plot(x_fit, A * np.sqrt(x_fit) + B, 'r-', lw=2,
                label=f'Fit: {A:.1f}√λ + {B:.1f}')

        # Langlais prediction
        A_L = results["A_Langlais"]
        ax1.plot(x_fit, A_L * np.sqrt(x_fit), 'g--', lw=2,
                label=f'Langlais: {A_L}√λ')

    ax1.set_xlabel('λ (eigenvalue)', fontsize=12)
    ax1.set_ylabel('N(λ) = #{eigenvalues ≤ λ}', fontsize=12)
    ax1.set_title('Eigenvalue Counting Function', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: N(λ) vs √λ (should be linear if Λ(s) ~ √s)
    sqrt_lambda = np.sqrt(lambda_vals)
    ax2.scatter(sqrt_lambda, N_vals, c='blue', alpha=0.7, s=30, label='Data')

    if results["fit"]["success"]:
        A = results["fit"]["A"]
        B = results["fit"]["B"]
        ax2.plot(sqrt_lambda, A * sqrt_lambda + B, 'r-', lw=2,
                label=f'Fit: {A:.1f}x + {B:.1f}')
        ax2.plot(sqrt_lambda, A_L * sqrt_lambda, 'g--', lw=2,
                label=f'Langlais: {A_L}x')

    ax2.set_xlabel('√λ', fontsize=12)
    ax2.set_ylabel('N(λ)', fontsize=12)
    ax2.set_title('Linearized: N(λ) vs √λ', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Run test with modest N for speed
    results = run_density_test(N=5000, k=25, num_eigs=50, seed=42)

    # Save results
    output_file = "outputs/eigenvalue_density_results.json"
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {output_file}")
    except Exception as e:
        print(f"Could not save to {output_file}: {e}")

    # Plot
    try:
        plot_density(results, "outputs/eigenvalue_density.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"λ₁ × H* = {results['lambda1_x_Hstar']:.4f} (target: 13)")
    if results["fit"]["success"]:
        ratio = results["fit"]["A"] / A_LANGLAIS
        print(f"Density coefficient: A = {results['fit']['A']:.1f}")
        print(f"Langlais prediction: A = {A_LANGLAIS}")
        print(f"Ratio: {ratio:.3f}")
