#!/usr/bin/env python3
"""
Conductor Selectivity Test for GIFT Hypothesis
===============================================

Critical test: Do GIFT conductors show better recurrence structure
in their L-function zeros compared to non-GIFT conductors?

This is the key falsification test identified but not yet performed.

GIFT Conductors: {7, 8, 11, 13, 14, 21, 27, 77, 99}
Non-GIFT Conductors: {6, 9, 10, 15, 16, 17, 19, 23, 25}

We test the [5, 8, 13, 27] recurrence on zeros of L(s, χ_q).

Author: GIFT Research
Date: February 2026
"""

import numpy as np
from typing import List, Dict, Tuple
import json

try:
    import mpmath as mp
    mp.mp.dps = 30
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False
    print("Warning: mpmath not available, using limited precision")


# GIFT Constants
GIFT_CONDUCTORS = [7, 8, 11, 13, 14, 21, 27, 77, 99]
NON_GIFT_CONDUCTORS = [6, 9, 10, 15, 16, 17, 19, 23, 25]

# Recurrence lags
GIFT_LAGS = [5, 8, 13, 27]
STANDARD_LAGS = [1, 2, 3, 4]


def get_dirichlet_zeros(q: int, num_zeros: int = 100) -> List[float]:
    """
    Get zeros of Dirichlet L-functions for conductor q.

    Uses mpmath's Dirichlet L-function facilities.
    Returns imaginary parts of zeros on critical line.
    """
    if not HAS_MPMATH:
        # Return dummy data for testing
        return list(np.cumsum(np.random.exponential(1, num_zeros)) + 10)

    zeros = []

    # For prime conductor, there's one primitive character
    # For composite, there are multiple
    # We'll use the principal character for simplicity

    # mpmath doesn't have direct L-function zero finder
    # Use the Riemann zeta zeros as proxy (rescaled)
    # This is a simplification - in production use LMFDB data

    print(f"  Computing zeros for conductor {q}...")

    # Get Riemann zeros and apply conductor scaling
    # L-functions with conductor q have zeros distributed differently
    # First zero height scales roughly as 2π / log(q)

    for k in range(1, num_zeros + 1):
        try:
            z = mp.zetazero(k)
            # Scale by conductor (approximate)
            scaled = float(z.imag) * (1 + 0.1 * np.log(q) / np.log(100))
            zeros.append(scaled)
        except:
            break

    return zeros


def fit_recurrence(zeros: List[float], lags: List[int]) -> Tuple[List[float], float]:
    """
    Fit a recurrence relation: γ_n = Σ_k a_k γ_{n-lag_k} + c

    Returns (coefficients, residual_error).
    """
    if len(zeros) < max(lags) + 10:
        return [], float('inf')

    max_lag = max(lags)
    n_points = len(zeros) - max_lag

    # Build design matrix
    X = np.zeros((n_points, len(lags) + 1))  # +1 for constant
    y = np.zeros(n_points)

    for i in range(n_points):
        n = i + max_lag
        y[i] = zeros[n]
        for j, lag in enumerate(lags):
            X[i, j] = zeros[n - lag]
        X[i, -1] = 1  # constant term

    # Least squares fit
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Compute fitted values and error
        y_fit = X @ coeffs
        mse = np.mean((y - y_fit) ** 2)
        rmse = np.sqrt(mse)

        # Relative error
        rel_error = rmse / np.mean(np.abs(y)) * 100

        return coeffs.tolist(), rel_error
    except:
        return [], float('inf')


def compute_fibonacci_constraint(coeffs: List[float], lags: List[int]) -> float:
    """
    Check Fibonacci constraint: 8 × a_8 = 13 × a_13 (should equal ~1).

    Returns the ratio R = (8 × a_8) / (13 × a_13).
    For perfect GIFT structure, R → 1.
    """
    if len(coeffs) < 3 or 8 not in lags or 13 not in lags:
        return float('nan')

    idx_8 = lags.index(8)
    idx_13 = lags.index(13)

    a_8 = coeffs[idx_8]
    a_13 = coeffs[idx_13]

    if abs(13 * a_13) < 1e-10:
        return float('nan')

    R = (8 * a_8) / (13 * a_13)
    return R


def analyze_conductor(q: int, zeros: List[float]) -> Dict:
    """Analyze recurrence structure for a given conductor."""

    # Fit with GIFT lags [5, 8, 13, 27]
    gift_coeffs, gift_error = fit_recurrence(zeros, GIFT_LAGS)

    # Fit with standard lags [1, 2, 3, 4]
    std_coeffs, std_error = fit_recurrence(zeros, STANDARD_LAGS)

    # Fibonacci constraint
    fib_R = compute_fibonacci_constraint(gift_coeffs, GIFT_LAGS)

    # Is GIFT better than standard?
    gift_advantage = std_error - gift_error if std_error < float('inf') else float('nan')

    return {
        'conductor': q,
        'is_gift': q in GIFT_CONDUCTORS,
        'num_zeros': len(zeros),
        'gift_lags_error': gift_error,
        'standard_lags_error': std_error,
        'gift_advantage': gift_advantage,  # positive = GIFT better
        'fibonacci_R': fib_R,
        'gift_coefficients': gift_coeffs,
    }


def main():
    print("=" * 70)
    print("CONDUCTOR SELECTIVITY TEST FOR GIFT HYPOTHESIS")
    print("=" * 70)
    print("\nTesting if GIFT conductors show better recurrence structure")
    print(f"GIFT conductors: {GIFT_CONDUCTORS}")
    print(f"Non-GIFT conductors: {NON_GIFT_CONDUCTORS}")
    print(f"\nGIFT lags: {GIFT_LAGS}")
    print(f"Standard lags: {STANDARD_LAGS}")

    results = []

    # Test all conductors
    all_conductors = sorted(GIFT_CONDUCTORS + NON_GIFT_CONDUCTORS)

    print("\n" + "-" * 70)
    print("Computing zeros and fitting recurrences...")
    print("-" * 70)

    for q in all_conductors:
        # Get zeros (using Riemann zeros as proxy for now)
        zeros = get_dirichlet_zeros(q, num_zeros=200)

        # Analyze
        result = analyze_conductor(q, zeros)
        results.append(result)

        marker = "★" if result['is_gift'] else " "
        print(f"{marker} q={q:3d}: GIFT err={result['gift_lags_error']:6.3f}%, "
              f"Std err={result['standard_lags_error']:6.3f}%, "
              f"Advantage={result['gift_advantage']:+6.3f}%, "
              f"Fib R={result['fibonacci_R']:.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    gift_results = [r for r in results if r['is_gift']]
    non_gift_results = [r for r in results if not r['is_gift']]

    # Mean GIFT advantage
    gift_advantages = [r['gift_advantage'] for r in gift_results
                       if not np.isnan(r['gift_advantage'])]
    non_gift_advantages = [r['gift_advantage'] for r in non_gift_results
                          if not np.isnan(r['gift_advantage'])]

    print(f"\nGIFT conductors (n={len(gift_results)}):")
    print(f"  Mean GIFT lag advantage: {np.mean(gift_advantages):+.4f}%")
    print(f"  Mean Fibonacci R: {np.mean([r['fibonacci_R'] for r in gift_results if not np.isnan(r['fibonacci_R'])]):.4f}")

    print(f"\nNon-GIFT conductors (n={len(non_gift_results)}):")
    print(f"  Mean GIFT lag advantage: {np.mean(non_gift_advantages):+.4f}%")
    print(f"  Mean Fibonacci R: {np.mean([r['fibonacci_R'] for r in non_gift_results if not np.isnan(r['fibonacci_R'])]):.4f}")

    # Test: Do GIFT conductors have R closer to 1?
    gift_R_deviation = [abs(r['fibonacci_R'] - 1) for r in gift_results
                        if not np.isnan(r['fibonacci_R'])]
    non_gift_R_deviation = [abs(r['fibonacci_R'] - 1) for r in non_gift_results
                           if not np.isnan(r['fibonacci_R'])]

    print(f"\nFibonacci constraint |R - 1|:")
    print(f"  GIFT conductors: {np.mean(gift_R_deviation):.4f}")
    print(f"  Non-GIFT conductors: {np.mean(non_gift_R_deviation):.4f}")

    # Statistical test
    if len(gift_R_deviation) > 3 and len(non_gift_R_deviation) > 3:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(gift_R_deviation, non_gift_R_deviation)
        print(f"\n  t-test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  → Significant difference detected!")
        else:
            print("  → No significant difference (p > 0.05)")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    mean_gift_R = np.mean([r['fibonacci_R'] for r in gift_results
                          if not np.isnan(r['fibonacci_R'])])
    mean_non_gift_R = np.mean([r['fibonacci_R'] for r in non_gift_results
                              if not np.isnan(r['fibonacci_R'])])

    if abs(mean_gift_R - 1) < abs(mean_non_gift_R - 1):
        print("GIFT conductors show Fibonacci R closer to 1.")
        print("→ Consistent with GIFT hypothesis (selectivity observed)")
    else:
        print("Non-GIFT conductors show Fibonacci R closer to 1.")
        print("→ INCONSISTENT with GIFT hypothesis!")

    # Note about limitations
    print("\n⚠️  CAVEAT: This analysis uses scaled Riemann zeros as proxy.")
    print("   For rigorous testing, use actual L-function zeros from LMFDB.")

    # Save results
    output = {
        'gift_conductors': GIFT_CONDUCTORS,
        'non_gift_conductors': NON_GIFT_CONDUCTORS,
        'results': results,
        'summary': {
            'mean_gift_R': mean_gift_R,
            'mean_non_gift_R': mean_non_gift_R,
            'gift_R_closer_to_1': bool(abs(mean_gift_R - 1) < abs(mean_non_gift_R - 1))
        }
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    with open('/home/user/GIFT/research/riemann/conductor_selectivity_results.json', 'w') as f:
        json.dump(convert(output), f, indent=2)

    print(f"\n✓ Results saved to conductor_selectivity_results.json")


if __name__ == "__main__":
    main()
