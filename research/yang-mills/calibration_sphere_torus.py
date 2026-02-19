#!/usr/bin/env python3
"""
Calibration of Graph Laplacian on S^7 and T^7
=============================================

Purpose: Determine the normalization factor between graph Laplacian eigenvalues
and continuous Laplacian eigenvalues on manifolds with KNOWN spectra.

Known exact values:
- S^7 (unit sphere): lambda_1 = 7 (multiplicity 8)
- T^7 (flat torus):  lambda_1 = (2*pi)^2 = 4*pi^2 ~ 39.48 (for unit side length)
                     or lambda_1 = 1 for side length 2*pi

If graph Laplacian gives lambda_1^graph ~ 46 on K7 where we expect 14/99 ~ 0.141,
then the ratio is ~326x. We need to understand this normalization.

Author: GIFT Project
Date: 2026-01-21
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: SAMPLE GENERATION
# ============================================================================

def sample_sphere_S7(n_points: int, seed: int = 42) -> np.ndarray:
    """Sample uniformly from unit 7-sphere S^7 in R^8."""
    np.random.seed(seed)
    # Sample from standard normal, then normalize
    points = np.random.randn(n_points, 8)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def sample_torus_T7(n_points: int, side_length: float = 2*np.pi, seed: int = 42) -> np.ndarray:
    """Sample uniformly from flat 7-torus T^7 = (R/L*Z)^7."""
    np.random.seed(seed)
    return np.random.uniform(0, side_length, size=(n_points, 7))

def torus_distance(p1: np.ndarray, p2: np.ndarray, L: float) -> float:
    """Geodesic distance on flat torus with period L."""
    diff = np.abs(p1 - p2)
    diff = np.minimum(diff, L - diff)  # Periodic boundary
    return np.sqrt(np.sum(diff**2))

# ============================================================================
# PART 2: GRAPH LAPLACIAN CONSTRUCTION
# ============================================================================

def build_graph_laplacian_sphere(points: np.ndarray, k: int = 20) -> csr_matrix:
    """
    Build graph Laplacian for points on S^7.
    Uses k-nearest neighbors with Gaussian kernel.
    """
    n = len(points)

    # Compute pairwise distances (Euclidean in R^8, approximates geodesic for nearby points)
    distances = cdist(points, points, 'euclidean')

    # k-NN: for each point, find k nearest neighbors
    # Set bandwidth based on typical neighbor distance
    knn_dists = np.sort(distances, axis=1)[:, 1:k+1]
    sigma = np.mean(knn_dists)  # Adaptive bandwidth

    # Gaussian kernel
    W = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)  # No self-loops

    # Degree matrix
    D = np.diag(np.sum(W, axis=1))

    # Unnormalized Laplacian: L = D - W
    L = D - W

    # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-10))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    return csr_matrix(L_norm), sigma

def build_graph_laplacian_torus(points: np.ndarray, L: float, k: int = 20) -> csr_matrix:
    """
    Build graph Laplacian for points on T^7.
    Uses periodic distance.
    """
    n = len(points)

    # Compute pairwise periodic distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = torus_distance(points[i], points[j], L)
            distances[i, j] = d
            distances[j, i] = d

    # k-NN bandwidth
    knn_dists = np.sort(distances, axis=1)[:, 1:k+1]
    sigma = np.mean(knn_dists)

    # Gaussian kernel
    W = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # Normalized Laplacian
    D = np.sum(W, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-10)
    L_norm = np.eye(n) - np.outer(D_inv_sqrt, D_inv_sqrt) * W

    return csr_matrix(L_norm), sigma

# ============================================================================
# PART 3: EIGENVALUE COMPUTATION
# ============================================================================

def compute_spectrum(L: csr_matrix, n_eigenvalues: int = 10) -> np.ndarray:
    """Compute smallest eigenvalues of Laplacian."""
    eigenvalues, _ = eigsh(L, k=n_eigenvalues, which='SM')
    return np.sort(eigenvalues)

# ============================================================================
# PART 4: CALIBRATION ANALYSIS
# ============================================================================

def calibrate_sphere(n_points_list: list = [500, 1000, 2000, 4000]):
    """
    Calibrate graph Laplacian on S^7.

    Exact spectrum of S^7 (unit sphere):
    - lambda_0 = 0 (constant function)
    - lambda_1 = 7 (coordinate functions, multiplicity 8)
    - lambda_2 = 16 (quadratic harmonics)
    """
    print("=" * 60)
    print("CALIBRATION ON S^7 (Unit 7-Sphere)")
    print("=" * 60)
    print(f"Exact lambda_1(S^7) = n = 7")
    print(f"(First non-zero eigenvalue of Laplace-Beltrami on S^n is n)")
    print()

    results = []

    for n_points in n_points_list:
        print(f"n_points = {n_points}")

        # Sample sphere
        points = sample_sphere_S7(n_points)

        # Build Laplacian
        L, sigma = build_graph_laplacian_sphere(points, k=min(30, n_points//10))

        # Compute spectrum
        eigenvalues = compute_spectrum(L, n_eigenvalues=15)

        # First non-zero eigenvalue (skip lambda_0 ~ 0)
        lambda_1_graph = eigenvalues[1]

        # Calibration factor
        lambda_1_exact = 7.0
        factor = lambda_1_graph / lambda_1_exact if lambda_1_exact > 0 else np.nan

        print(f"  sigma (bandwidth) = {sigma:.4f}")
        print(f"  lambda_1 (graph)  = {lambda_1_graph:.6f}")
        print(f"  lambda_1 (exact)  = {lambda_1_exact:.6f}")
        print(f"  Ratio graph/exact = {factor:.4f}")
        print(f"  First 5 eigenvalues: {eigenvalues[:5]}")
        print()

        results.append({
            'n_points': n_points,
            'lambda_1_graph': lambda_1_graph,
            'lambda_1_exact': lambda_1_exact,
            'ratio': factor,
            'sigma': sigma
        })

    return results

def calibrate_torus(n_points_list: list = [500, 1000, 2000]):
    """
    Calibrate graph Laplacian on T^7.

    Exact spectrum of T^7 (flat torus with period 2*pi):
    - lambda_0 = 0 (constant)
    - lambda_1 = 1 (sin/cos of single coordinate, multiplicity 14)
    """
    print("=" * 60)
    print("CALIBRATION ON T^7 (Flat 7-Torus, period 2*pi)")
    print("=" * 60)
    print(f"Exact lambda_1(T^7) = 1 (for period 2*pi)")
    print(f"(First harmonics: sin(x_i), cos(x_i))")
    print()

    L_period = 2 * np.pi
    results = []

    for n_points in n_points_list:
        print(f"n_points = {n_points}")

        # Sample torus
        points = sample_torus_T7(n_points, side_length=L_period)

        # Build Laplacian (expensive for torus due to periodic distance)
        L, sigma = build_graph_laplacian_torus(points, L_period, k=min(30, n_points//10))

        # Compute spectrum
        eigenvalues = compute_spectrum(L, n_eigenvalues=15)

        # First non-zero eigenvalue
        lambda_1_graph = eigenvalues[1]

        # Calibration factor
        lambda_1_exact = 1.0
        factor = lambda_1_graph / lambda_1_exact

        print(f"  sigma (bandwidth) = {sigma:.4f}")
        print(f"  lambda_1 (graph)  = {lambda_1_graph:.6f}")
        print(f"  lambda_1 (exact)  = {lambda_1_exact:.6f}")
        print(f"  Ratio graph/exact = {factor:.4f}")
        print(f"  First 5 eigenvalues: {eigenvalues[:5]}")
        print()

        results.append({
            'n_points': n_points,
            'lambda_1_graph': lambda_1_graph,
            'lambda_1_exact': lambda_1_exact,
            'ratio': factor,
            'sigma': sigma
        })

    return results

# ============================================================================
# PART 5: MAIN ANALYSIS
# ============================================================================

def main():
    print()
    print("*" * 70)
    print("*  GRAPH LAPLACIAN CALIBRATION: S^7 AND T^7                         *")
    print("*" * 70)
    print()
    print("Goal: Determine normalization factor between graph and continuous")
    print("      Laplacian eigenvalues to interpret K7 results correctly.")
    print()

    # Calibrate on sphere
    sphere_results = calibrate_sphere([500, 1000, 2000])

    # Calibrate on torus (fewer points due to O(n^2) distance computation)
    torus_results = calibrate_torus([500, 1000])

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    print("S^7 Results:")
    for r in sphere_results:
        print(f"  n={r['n_points']:4d}: lambda_1^graph = {r['lambda_1_graph']:.4f}, "
              f"ratio = {r['ratio']:.4f}")

    avg_ratio_sphere = np.mean([r['ratio'] for r in sphere_results])
    print(f"  Average ratio (graph/exact): {avg_ratio_sphere:.4f}")
    print()

    print("T^7 Results:")
    for r in torus_results:
        print(f"  n={r['n_points']:4d}: lambda_1^graph = {r['lambda_1_graph']:.4f}, "
              f"ratio = {r['ratio']:.4f}")

    avg_ratio_torus = np.mean([r['ratio'] for r in torus_results])
    print(f"  Average ratio (graph/exact): {avg_ratio_torus:.4f}")
    print()

    # Implications for K7
    print("=" * 60)
    print("IMPLICATIONS FOR K7")
    print("=" * 60)
    print()
    print("If graph Laplacian on K7 gives lambda_1 ~ 46:")
    print()

    # Using sphere calibration
    lambda_1_K7_corrected_sphere = 46.0 / avg_ratio_sphere
    print(f"  Using S^7 calibration (ratio={avg_ratio_sphere:.3f}):")
    print(f"    lambda_1(K7) ~ 46 / {avg_ratio_sphere:.3f} = {lambda_1_K7_corrected_sphere:.4f}")
    print(f"    GIFT prediction: 14/99 = {14/99:.4f}")
    print(f"    Match: {'YES' if abs(lambda_1_K7_corrected_sphere - 14/99) < 0.05 else 'NO'}")
    print()

    # Using torus calibration
    lambda_1_K7_corrected_torus = 46.0 / avg_ratio_torus
    print(f"  Using T^7 calibration (ratio={avg_ratio_torus:.3f}):")
    print(f"    lambda_1(K7) ~ 46 / {avg_ratio_torus:.3f} = {lambda_1_K7_corrected_torus:.4f}")
    print(f"    GIFT prediction: 14/99 = {14/99:.4f}")
    print(f"    Match: {'YES' if abs(lambda_1_K7_corrected_torus - 14/99) < 0.05 else 'NO'}")
    print()

    # Key insight
    print("=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print()
    print("The normalized graph Laplacian has eigenvalues in [0, 2].")
    print("This is INDEPENDENT of the manifold's intrinsic scale!")
    print()
    print("To recover continuous lambda_1, we need:")
    print("  lambda_1^continuous = lambda_1^graph * (n_points / Vol)^(2/dim)")
    print()
    print("or use a different Laplacian normalization (unnormalized, random walk, etc.)")
    print()

if __name__ == "__main__":
    main()
