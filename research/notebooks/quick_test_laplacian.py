#!/usr/bin/env python3
"""
Quick test for curved Laplacian - uses smaller grid sizes for faster validation.
"""

import numpy as np
import time
from curved_laplacian_7d import (
    build_curved_laplacian_optimized,
    compute_eigenvalues,
    flat_metric,
    g2_inspired_metric,
)


def quick_tests():
    """Run quick validation tests with small grids."""
    print("=" * 60)
    print("Quick Validation Tests for Curved 7D Laplacian")
    print("=" * 60)

    N = 3  # 3^7 = 2187 points - very small grid for quick testing
    L = np.ones(7)

    # Test 1: Build and check properties for flat metric
    print("\n1. Building flat-space Laplacian (N=3)...")
    t0 = time.time()
    L_mat, points = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')
    print(f"   Build time: {time.time() - t0:.2f}s")

    # Check symmetry
    diff = L_mat - L_mat.T
    max_asym = np.max(np.abs(diff.toarray()))
    print(f"   Symmetry check: max|L - L^T| = {max_asym:.2e} {'PASS' if max_asym < 1e-14 else 'FAIL'}")

    # Check row sums
    row_sums = np.array(L_mat.sum(axis=1)).flatten()
    max_row = np.max(np.abs(row_sums))
    print(f"   Row sum check: max|sum| = {max_row:.2e} {'PASS' if max_row < 1e-12 else 'FAIL'}")

    # Compute eigenvalues
    print("\n2. Computing eigenvalues (flat space)...")
    t0 = time.time()
    eigenvalues, eigenvectors = compute_eigenvalues(L_mat, k=8)
    print(f"   Eigensolve time: {time.time() - t0:.2f}s")
    print(f"   Eigenvalues: {eigenvalues}")

    # Check zero mode
    zero_error = abs(eigenvalues[0])
    print(f"   Zero mode: λ_0 = {eigenvalues[0]:.6e} {'PASS' if zero_error < 1e-10 else 'FAIL'}")

    # For N=3 on unit domain with periodic BC, h = 1/3
    # Finite difference Laplacian eigenvalue: λ = -2/h² * (1 - cos(2πn/N)) for each dimension
    # First non-zero: n=1 in one dimension → -2*9*(1-cos(2π/3)) = -2*9*1.5 = -27
    h = 1.0 / N
    expected_first = -2 / h**2 * (1 - np.cos(2 * np.pi / N))
    print(f"   Expected first non-zero (FD): {expected_first:.4f}")
    print(f"   Computed first non-zero: {eigenvalues[1]:.4f}")
    rel_err = abs(eigenvalues[1] - expected_first) / abs(expected_first)
    print(f"   Relative error: {rel_err:.2e} {'PASS' if rel_err < 0.01 else 'FAIL'}")

    # Test 3: Curved metric
    print("\n3. Testing curved metric (G2-inspired)...")

    def curved_metric(x):
        return g2_inspired_metric(x, scale=0.2)

    t0 = time.time()
    L_curved, _ = build_curved_laplacian_optimized(N, L, curved_metric, boundary='periodic')
    print(f"   Build time: {time.time() - t0:.2f}s")

    t0 = time.time()
    ev_curved, _ = compute_eigenvalues(L_curved, k=5)
    print(f"   Eigensolve time: {time.time() - t0:.2f}s")
    print(f"   Eigenvalues: {ev_curved}")

    # Curved metric should still give negative semi-definite
    nsd_pass = np.max(ev_curved) < 1e-6
    print(f"   Negative semi-definite: {'PASS' if nsd_pass else 'FAIL'}")

    # Test 4: Dirichlet BC
    print("\n4. Testing Dirichlet boundary conditions...")
    L_dirichlet, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='dirichlet')
    ev_dirichlet, _ = compute_eigenvalues(L_dirichlet, k=5)
    print(f"   Eigenvalues: {ev_dirichlet}")

    # All eigenvalues should be strictly negative for Dirichlet
    strict_neg = np.max(ev_dirichlet) < -1
    print(f"   All strictly negative: {'PASS' if strict_neg else 'FAIL'}")

    # Test 5: Larger grid (N=4)
    print("\n5. Testing N=4 grid (16,384 points)...")
    N4 = 4
    t0 = time.time()
    L_mat4, _ = build_curved_laplacian_optimized(N4, L, flat_metric, boundary='periodic')
    print(f"   Build time: {time.time() - t0:.2f}s")

    t0 = time.time()
    ev4, _ = compute_eigenvalues(L_mat4, k=5)
    print(f"   Eigensolve time: {time.time() - t0:.2f}s")
    print(f"   Eigenvalues: {ev4}")

    # Expected for N=4: -2*16*(1-cos(π/2)) = -2*16*1 = -32
    expected_4 = -2 / (1/4)**2 * (1 - np.cos(2 * np.pi / 4))
    print(f"   Expected first non-zero (FD): {expected_4:.4f}")
    rel_err4 = abs(ev4[1] - expected_4) / abs(expected_4)
    print(f"   Relative error: {rel_err4:.2e} {'PASS' if rel_err4 < 0.01 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Quick validation complete!")
    print("=" * 60)


def demonstrate_usage():
    """Demonstrate typical usage patterns."""
    print("\n" + "=" * 60)
    print("Usage Demonstration")
    print("=" * 60)

    # Define a custom metric
    def my_metric(x):
        """Custom diagonal metric: conformal with radial dependence."""
        r = np.linalg.norm(x - 0.5)  # Distance from center
        omega = 1.0 + 0.5 * np.exp(-4 * r**2)  # Conformal factor
        return omega**2 * np.ones(7)

    N = 3
    domain = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    print(f"\nBuilding curved Laplacian on [0,1]^7 with conformal metric...")
    L, points = build_curved_laplacian_optimized(
        grid_points_per_dim=N,
        domain_size=domain,
        metric_func=my_metric,
        boundary='periodic'
    )

    print(f"\nComputing first 5 eigenvalues...")
    eigenvalues, eigenvectors = compute_eigenvalues(L, k=5)

    print(f"\nResults:")
    print(f"  Grid: {N}^7 = {N**7} points")
    print(f"  Matrix: {L.shape[0]} x {L.shape[1]}, {L.nnz} non-zeros")
    print(f"  Eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"    λ_{i} = {ev:.6f}")

    # Spectral gap
    gap = eigenvalues[0] - eigenvalues[1]
    print(f"  Spectral gap: {gap:.6f}")


if __name__ == "__main__":
    quick_tests()
    demonstrate_usage()
