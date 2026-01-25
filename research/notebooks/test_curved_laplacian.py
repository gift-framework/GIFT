#!/usr/bin/env python3
"""
Test script for curved Laplacian on 7D domain.

Verifies correctness against known analytical results and demonstrates usage.
"""

import numpy as np
import time
from curved_laplacian_7d import (
    build_curved_laplacian,
    build_curved_laplacian_optimized,
    compute_eigenvalues,
    compute_spectrum_without_matrix,
    flat_metric,
    g2_inspired_metric,
    validate_flat_space,
)


def test_index_conversion():
    """Test that index conversion is consistent."""
    print("Testing index conversion...")

    N = 4
    dim = 7
    strides = np.array([N**(dim - 1 - d) for d in range(dim)])

    # Test a few conversions
    for lin_idx in [0, 100, 1000, N**dim - 1]:
        multi_idx = np.array([(lin_idx // strides[d]) % N for d in range(dim)])
        recovered = np.sum(multi_idx * strides)
        assert recovered == lin_idx, f"Index mismatch: {lin_idx} -> {multi_idx} -> {recovered}"

    print("  PASSED: Index conversion is consistent")


def test_symmetry():
    """Test that the Laplacian matrix is symmetric."""
    print("Testing matrix symmetry...")

    N = 4
    L = np.ones(7)

    L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')

    # Check symmetry
    diff = L_mat - L_mat.T
    max_asymmetry = np.max(np.abs(diff.toarray()))

    print(f"  Maximum asymmetry: {max_asymmetry:.2e}")
    assert max_asymmetry < 1e-14, f"Matrix not symmetric: max diff = {max_asymmetry}"
    print("  PASSED: Matrix is symmetric")


def test_row_sum_periodic():
    """Test that row sums are zero for periodic BC (constant is null eigenvector)."""
    print("Testing row sums for periodic BC...")

    N = 4
    L = np.ones(7)

    L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')

    # Row sums should be zero
    row_sums = np.array(L_mat.sum(axis=1)).flatten()
    max_row_sum = np.max(np.abs(row_sums))

    print(f"  Maximum |row sum|: {max_row_sum:.2e}")
    assert max_row_sum < 1e-12, f"Row sums not zero: max = {max_row_sum}"
    print("  PASSED: Row sums are zero (constant is null eigenvector)")


def test_negative_semidefinite():
    """Test that the Laplacian is negative semi-definite."""
    print("Testing negative semi-definiteness...")

    N = 4
    L = np.ones(7)

    L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')

    # Compute a few eigenvalues
    k = min(10, N**7 - 2)
    eigenvalues, _ = compute_eigenvalues(L_mat, k=k)

    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Maximum eigenvalue: {np.max(eigenvalues):.6e}")

    # For negative semi-definite, all eigenvalues should be <= 0
    # Allow small numerical tolerance
    assert np.max(eigenvalues) < 1e-6, f"Not negative semi-definite: max eigenvalue = {np.max(eigenvalues)}"
    print("  PASSED: Laplacian is negative semi-definite")


def test_flat_space_eigenvalues():
    """Compare computed eigenvalues with analytical flat-space values."""
    print("Testing flat-space eigenvalues...")

    N = 5
    L = np.ones(7)  # Unit 7-torus

    L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')

    k = 15
    eigenvalues, _ = compute_eigenvalues(L_mat, k=k)

    # Theoretical: λ = -4π² Σ_i n_i² for integers n_i
    # First non-zero: one n_i = ±1, rest 0 → λ = -4π² ≈ -39.478
    # Second: two n_i = ±1, rest 0 → λ = -8π² ≈ -78.957

    lambda_theory_1 = -4 * np.pi**2
    lambda_theory_2 = -8 * np.pi**2

    print(f"  Computed λ_0: {eigenvalues[0]:.6f} (theory: 0)")
    print(f"  Computed λ_1: {eigenvalues[1]:.6f} (theory: {lambda_theory_1:.6f})")

    # Note: Finite difference has discretization error
    # For N=5 on unit domain, h = 0.2, and error is O(h²)
    zero_error = abs(eigenvalues[0])
    first_error = abs(eigenvalues[1] - lambda_theory_1) / abs(lambda_theory_1)

    print(f"  Zero mode error: {zero_error:.6e}")
    print(f"  First mode relative error: {first_error:.2%}")

    # With coarse grid, expect significant error but correct order of magnitude
    assert zero_error < 1.0, f"Zero mode error too large: {zero_error}"
    assert first_error < 0.5, f"First mode error too large: {first_error}"

    print("  PASSED: Eigenvalues have correct order of magnitude")


def test_curved_metric():
    """Test with a non-flat metric."""
    print("Testing curved metric...")

    N = 4
    L = np.ones(7)

    def curved_metric(x):
        return g2_inspired_metric(x, scale=0.1)

    L_mat, points = build_curved_laplacian_optimized(N, L, curved_metric, boundary='periodic')

    # Check symmetry
    diff = L_mat - L_mat.T
    max_asymmetry = np.max(np.abs(diff.toarray()))
    assert max_asymmetry < 1e-13, f"Curved metric matrix not symmetric"

    # Compute eigenvalues
    k = 8
    eigenvalues, _ = compute_eigenvalues(L_mat, k=k)

    print(f"  Eigenvalues: {eigenvalues}")

    # Should still be negative semi-definite
    assert np.max(eigenvalues) < 1e-6, f"Not negative semi-definite"

    print("  PASSED: Curved metric produces valid Laplacian")


def test_dirichlet_bc():
    """Test Dirichlet boundary conditions."""
    print("Testing Dirichlet boundary conditions...")

    N = 4
    L = np.ones(7)

    L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='dirichlet')

    # For Dirichlet BC, there should be no zero eigenvalue
    # (constant is not an eigenvector since it doesn't satisfy BC)
    k = 8
    eigenvalues, _ = compute_eigenvalues(L_mat, k=k)

    print(f"  Eigenvalues: {eigenvalues}")

    # All eigenvalues should be strictly negative
    assert eigenvalues[0] < -1, f"Largest eigenvalue should be negative for Dirichlet BC"

    print("  PASSED: Dirichlet BC produces strictly negative eigenvalues")


def test_both_implementations_match():
    """Test that both implementations give the same result."""
    print("Testing that implementations match...")

    N = 4
    L = np.ones(7)

    def metric(x):
        return g2_inspired_metric(x, scale=0.2)

    L1, points1 = build_curved_laplacian(N, L, metric, boundary='periodic')
    L2, points2 = build_curved_laplacian_optimized(N, L, metric, boundary='periodic')

    # Check matrices match
    diff = L1 - L2
    max_diff = np.max(np.abs(diff.toarray()))

    print(f"  Maximum difference between implementations: {max_diff:.2e}")
    assert max_diff < 1e-14, f"Implementations differ: max diff = {max_diff}"

    print("  PASSED: Both implementations produce identical results")


def benchmark_performance():
    """Benchmark performance for different grid sizes."""
    print("\nPerformance benchmark:")
    print("-" * 50)

    L = np.ones(7)

    for N in [4, 5, 6]:
        total = N**7

        start = time.time()
        L_mat, _ = build_curved_laplacian_optimized(N, L, flat_metric, boundary='periodic')
        build_time = time.time() - start

        start = time.time()
        eigenvalues, _ = compute_eigenvalues(L_mat, k=5)
        eigen_time = time.time() - start

        print(f"  N={N}: {total:,} points")
        print(f"    Build time: {build_time:.2f}s")
        print(f"    Eigenvalue time: {eigen_time:.2f}s")
        print(f"    Memory: ~{L_mat.data.nbytes / 1e6:.1f} MB")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running all tests for curved 7D Laplacian")
    print("=" * 60)

    tests = [
        test_index_conversion,
        test_symmetry,
        test_row_sum_periodic,
        test_negative_semidefinite,
        test_flat_space_eigenvalues,
        test_curved_metric,
        test_dirichlet_bc,
        test_both_implementations_match,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print()
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\nRunning performance benchmark...")
        benchmark_performance()

    exit(0 if success else 1)
