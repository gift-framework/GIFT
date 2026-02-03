#!/usr/bin/env python3
"""
FALSIFICATION BATTERY: Ultimate Tests for Fibonacci-Riemann Recurrence
========================================================================

This script implements ALL falsification tests demanded by the AI council:

1. OUT-OF-SAMPLE: Train on zeros 1-50k, test on 50k-100k (and high zeros)
2. COEFFICIENT ROBUSTNESS: Is 3/2 a sharp minimum or a plateau?
3. UNFOLDED FLUCTUATIONS: Test on detrended residuals (GPT's critical test)
4. GUE RANDOM MATRICES: Does the same recurrence work on random spectra?
5. BASELINE COMPARISON: Can ANY smooth monotone sequence achieve R²>0.999?

If the recurrence is real structure, it must:
- Maintain performance out-of-sample
- Show sharp optimum at 3/2 (not plateau)
- Work on FLUCTUATIONS, not just trend
- FAIL on GUE (or be generic, which deflates the claim)
- Significantly outperform naive baselines

Author: Falsification Protocol
Date: 2026-02-03
"""

import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2
PSI = 1 - PHI
SQRT5 = np.sqrt(5)

# GIFT constants
B2 = 21  # Second Betti number
DIM_G2 = 14  # G2 dimension
GIFT_RATIO = B2 / DIM_G2  # = 3/2

# Fibonacci numbers
def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

FIBS = [fib(i) for i in range(1, 25)]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_zeros(max_zeros=100000) -> np.ndarray:
    """Load Riemann zeros from local files."""
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 6):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    if line.strip():
                        zeros.append(float(line.strip()))
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

def load_high_zeros() -> Dict[str, np.ndarray]:
    """Load zeros at very high heights (10^12, 10^21, 10^22)."""
    zeros_dir = Path(__file__).parent
    high_zeros = {}

    # zeros2: around 10^12
    z2_file = zeros_dir / "zeros2"
    if z2_file.exists():
        with open(z2_file) as f:
            high_zeros['10^12'] = np.array([float(line.strip()) for line in f if line.strip()])

    # zeros3: around 10^21
    z3_file = zeros_dir / "zeros3"
    if z3_file.exists():
        with open(z3_file) as f:
            high_zeros['10^21'] = np.array([float(line.strip()) for line in f if line.strip()])

    return high_zeros

# ============================================================================
# CORE FITTING FUNCTIONS
# ============================================================================

def fit_recurrence(zeros: np.ndarray, lag1: int, lag2: int,
                   start_idx: int = 0, end_idx: int = None) -> Dict:
    """
    Fit γ_n = a × γ_{n-lag1} + b × γ_{n-lag2} + c

    Returns dict with coefficients, R², errors, predictions.
    """
    if end_idx is None:
        end_idx = len(zeros)

    max_lag = max(lag1, lag2)
    actual_start = max(start_idx, max_lag)

    n_samples = end_idx - actual_start
    if n_samples < 10:
        return None

    # Build design matrix
    X1 = zeros[actual_start - lag1:end_idx - lag1]
    X2 = zeros[actual_start - lag2:end_idx - lag2]
    X = np.column_stack([X1, X2, np.ones(n_samples)])
    y = zeros[actual_start:end_idx]

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coeffs

    # Predictions and metrics
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    errors = np.abs(y - y_pred)
    rel_errors = errors / y * 100

    return {
        'a': a, 'b': b, 'c': c,
        'a_plus_b': a + b,
        'r2': r2,
        'mean_abs_error': np.mean(errors),
        'mean_rel_error': np.mean(rel_errors),
        'max_rel_error': np.max(rel_errors),
        'std_error': np.std(errors),
        'n_samples': n_samples,
        'predictions': y_pred,
        'actuals': y,
        'residuals': y - y_pred
    }

def fit_constrained(zeros: np.ndarray, lag1: int, lag2: int,
                    fixed_a: float, start_idx: int = 0, end_idx: int = None) -> Dict:
    """
    Fit with FIXED coefficient a (to test robustness).
    γ_n = a × γ_{n-lag1} + (1-a) × γ_{n-lag2} + c
    """
    if end_idx is None:
        end_idx = len(zeros)

    max_lag = max(lag1, lag2)
    actual_start = max(start_idx, max_lag)
    n_samples = end_idx - actual_start

    if n_samples < 10:
        return None

    X1 = zeros[actual_start - lag1:end_idx - lag1]
    X2 = zeros[actual_start - lag2:end_idx - lag2]
    y = zeros[actual_start:end_idx]

    # Fixed a, b = 1-a, solve only for c
    b = 1 - fixed_a
    y_adjusted = y - fixed_a * X1 - b * X2
    c = np.mean(y_adjusted)

    y_pred = fixed_a * X1 + b * X2 + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    errors = np.abs(y - y_pred)
    rel_errors = errors / y * 100

    return {
        'a': fixed_a, 'b': b, 'c': c,
        'r2': r2,
        'mean_rel_error': np.mean(rel_errors),
        'n_samples': n_samples
    }

# ============================================================================
# TEST 1: OUT-OF-SAMPLE VALIDATION
# ============================================================================

def test_out_of_sample(zeros: np.ndarray) -> Dict:
    """
    Critical test: Train on first 50k, test on next 50k.
    If overfitting, performance will degrade significantly.
    """
    print("\n" + "=" * 70)
    print("TEST 1: OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)

    results = {}

    # Training set: zeros 0-50000
    train_fit = fit_recurrence(zeros, 8, 21, start_idx=0, end_idx=50000)

    # Test set: zeros 50000-100000 using TRAINING coefficients
    max_lag = 21
    test_start = 50000
    test_end = min(100000, len(zeros))

    X1_test = zeros[test_start - 8:test_end - 8]
    X2_test = zeros[test_start - 21:test_end - 21]
    y_test = zeros[test_start:test_end]

    # Apply training coefficients to test set
    y_pred_test = train_fit['a'] * X1_test + train_fit['b'] * X2_test + train_fit['c']

    ss_res_test = np.sum((y_test - y_pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test

    test_errors = np.abs(y_test - y_pred_test)
    test_rel_errors = test_errors / y_test * 100

    results['training'] = {
        'a': train_fit['a'], 'b': train_fit['b'], 'c': train_fit['c'],
        'r2': train_fit['r2'],
        'mean_rel_error': train_fit['mean_rel_error'],
        'n_samples': train_fit['n_samples']
    }

    results['test'] = {
        'r2': r2_test,
        'mean_rel_error': float(np.mean(test_rel_errors)),
        'max_rel_error': float(np.max(test_rel_errors)),
        'std_rel_error': float(np.std(test_rel_errors)),
        'n_samples': len(y_test)
    }

    # Performance degradation
    r2_drop = train_fit['r2'] - r2_test
    error_ratio = results['test']['mean_rel_error'] / results['training']['mean_rel_error']

    results['degradation'] = {
        'r2_drop': r2_drop,
        'error_increase_ratio': error_ratio
    }

    print(f"\nTRAINING (zeros 1-50k):")
    print(f"  Coefficients: a = {train_fit['a']:.6f}, b = {train_fit['b']:.6f}")
    print(f"  a + b = {train_fit['a'] + train_fit['b']:.6f}")
    print(f"  R² = {train_fit['r2']:.8f}")
    print(f"  Mean relative error = {train_fit['mean_rel_error']:.4f}%")

    print(f"\nTEST (zeros 50k-100k, using TRAINING coefficients):")
    print(f"  R² = {r2_test:.8f}")
    print(f"  Mean relative error = {results['test']['mean_rel_error']:.4f}%")
    print(f"  Max relative error = {results['test']['max_rel_error']:.4f}%")

    print(f"\nDEGRADATION ANALYSIS:")
    print(f"  R² drop: {r2_drop:.8f}")
    print(f"  Error increase ratio: {error_ratio:.4f}x")

    # Verdict
    if r2_drop < 0.0001 and error_ratio < 1.5:
        verdict = "PASS - Excellent generalization"
        results['verdict'] = 'PASS'
    elif r2_drop < 0.001 and error_ratio < 2.0:
        verdict = "PASS - Good generalization"
        results['verdict'] = 'PASS'
    elif r2_drop < 0.01:
        verdict = "MARGINAL - Some overfitting detected"
        results['verdict'] = 'MARGINAL'
    else:
        verdict = "FAIL - Significant overfitting"
        results['verdict'] = 'FAIL'

    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 2: COEFFICIENT ROBUSTNESS
# ============================================================================

def test_coefficient_robustness(zeros: np.ndarray) -> Dict:
    """
    Test if a = 3/2 is a SHARP minimum or just a plateau.
    If sharp: perturbations should cause rapid degradation.
    If plateau: any nearby value works equally well (suspicious).
    """
    print("\n" + "=" * 70)
    print("TEST 2: COEFFICIENT ROBUSTNESS")
    print("=" * 70)

    results = {}

    # Test range around 3/2 = 1.5
    a_values = np.linspace(1.3, 1.7, 41)  # Fine grid around 3/2

    errors_by_a = []
    for a in a_values:
        fit = fit_constrained(zeros, 8, 21, fixed_a=a, end_idx=50000)
        errors_by_a.append({
            'a': a,
            'r2': fit['r2'],
            'mean_rel_error': fit['mean_rel_error']
        })

    # Find optimal a
    errors = [e['mean_rel_error'] for e in errors_by_a]
    optimal_idx = np.argmin(errors)
    optimal_a = errors_by_a[optimal_idx]['a']
    optimal_error = errors_by_a[optimal_idx]['mean_rel_error']

    # Error at exactly 3/2
    idx_3_2 = np.argmin(np.abs(a_values - 1.5))
    error_at_3_2 = errors_by_a[idx_3_2]['mean_rel_error']

    # Sharpness: second derivative of error curve at optimum
    # Approximate using finite differences
    error_array = np.array(errors)
    da = a_values[1] - a_values[0]

    # Second derivative at optimum
    if optimal_idx > 0 and optimal_idx < len(errors) - 1:
        d2_error = (error_array[optimal_idx + 1] - 2 * error_array[optimal_idx] +
                    error_array[optimal_idx - 1]) / (da ** 2)
    else:
        d2_error = 0

    # Error increase at ±0.1 from optimum
    idx_low = np.argmin(np.abs(a_values - (optimal_a - 0.1)))
    idx_high = np.argmin(np.abs(a_values - (optimal_a + 0.1)))
    error_sensitivity = max(errors[idx_low], errors[idx_high]) - optimal_error

    results['scan'] = errors_by_a
    results['optimal_a'] = float(optimal_a)
    results['optimal_error'] = float(optimal_error)
    results['error_at_3_2'] = float(error_at_3_2)
    results['diff_from_3_2'] = float(optimal_a - 1.5)
    results['sharpness'] = float(d2_error)
    results['error_sensitivity_0.1'] = float(error_sensitivity)

    print(f"\nCoefficient scan around 3/2:")
    print(f"  Optimal a = {optimal_a:.6f}")
    print(f"  Error at optimal = {optimal_error:.4f}%")
    print(f"  Error at 3/2 = {error_at_3_2:.4f}%")
    print(f"  Difference from 3/2 = {optimal_a - 1.5:.6f}")

    print(f"\nSharpness analysis:")
    print(f"  Second derivative at optimum: {d2_error:.4f}")
    print(f"  Error increase at ±0.1: {error_sensitivity:.4f}%")

    # Compare specific values
    print(f"\nComparison of candidate values:")
    candidates = [1.4, 1.45, 1.4646, 1.5, PHI, 1.55, 1.6]
    for cand in candidates:
        fit = fit_constrained(zeros, 8, 21, fixed_a=cand, end_idx=50000)
        label = ""
        if abs(cand - 1.5) < 0.001:
            label = " ← 3/2 = b₂/dim(G₂)"
        elif abs(cand - PHI) < 0.001:
            label = " ← φ (golden ratio)"
        print(f"  a = {cand:.4f}: error = {fit['mean_rel_error']:.4f}%{label}")

    # Verdict
    if abs(optimal_a - 1.5) < 0.02 and d2_error > 0.5:
        verdict = "PASS - Sharp minimum near 3/2"
        results['verdict'] = 'PASS'
    elif abs(optimal_a - 1.5) < 0.05:
        verdict = "PASS - Minimum near 3/2 (moderate sharpness)"
        results['verdict'] = 'PASS'
    elif d2_error < 0.1:
        verdict = "FAIL - Plateau, not sharp minimum (suspicious)"
        results['verdict'] = 'FAIL'
    else:
        verdict = "MARGINAL - Minimum not at 3/2"
        results['verdict'] = 'MARGINAL'

    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 3: UNFOLDED FLUCTUATIONS (GPT's Critical Test)
# ============================================================================

def test_unfolded_fluctuations(zeros: np.ndarray) -> Dict:
    """
    GPT's critical test: Does the recurrence work on FLUCTUATIONS?

    The smooth trend N(T) ~ T/(2π) log(T/(2πe)) makes ANY linear stencil
    achieve high R² on raw zeros. The real test is on the residuals.

    Unfolding: x_n = N(γ_n) - n ≈ fluctuation around expected position
    If recurrence captures real structure, it should work on x_n too.
    """
    print("\n" + "=" * 70)
    print("TEST 3: UNFOLDED FLUCTUATIONS")
    print("=" * 70)

    results = {}

    n_zeros = min(50000, len(zeros))
    gamma = zeros[:n_zeros]
    n = np.arange(1, n_zeros + 1)

    # Riemann-von Mangoldt formula: N(T) ~ T/(2π) * log(T/(2πe)) + 7/8
    def counting_function(T):
        return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7/8

    # Expected position for each zero
    N_gamma = counting_function(gamma)

    # Fluctuations (unfolded)
    x = N_gamma - n  # Should be O(1) fluctuations

    print(f"Unfolding statistics:")
    print(f"  Mean fluctuation: {np.mean(x):.4f}")
    print(f"  Std fluctuation: {np.std(x):.4f}")
    print(f"  Min/Max: [{np.min(x):.4f}, {np.max(x):.4f}]")

    # Test 1: Recurrence on RAW zeros (should have high R² - this is the baseline)
    fit_raw = fit_recurrence(zeros, 8, 21, end_idx=n_zeros)

    # Test 2: Recurrence on UNFOLDED fluctuations
    # x_n = a × x_{n-8} + b × x_{n-21} + c ?
    max_lag = 21
    X1_unfold = x[max_lag - 8:-8] if n_zeros > max_lag + 8 else x[max_lag - 8:]
    X2_unfold = x[:-21] if n_zeros > 21 else x[:max_lag]

    # Proper alignment
    start = max_lag
    end = n_zeros
    X1_unfold = x[start - 8:end - 8]
    X2_unfold = x[start - 21:end - 21]
    y_unfold = x[start:end]

    n_samples = len(y_unfold)
    X_unfold = np.column_stack([X1_unfold[:n_samples], X2_unfold[:n_samples], np.ones(n_samples)])

    coeffs_unfold, _, _, _ = np.linalg.lstsq(X_unfold, y_unfold, rcond=None)
    a_unfold, b_unfold, c_unfold = coeffs_unfold

    y_pred_unfold = X_unfold @ coeffs_unfold
    ss_res_unfold = np.sum((y_unfold - y_pred_unfold) ** 2)
    ss_tot_unfold = np.sum((y_unfold - np.mean(y_unfold)) ** 2)
    r2_unfold = 1 - ss_res_unfold / ss_tot_unfold

    # Test 3: Baseline - can we predict x_n from x_{n-1}? (trivial autoregression)
    X_ar1 = np.column_stack([x[start-1:end-1], np.ones(n_samples)])
    coeffs_ar1, _, _, _ = np.linalg.lstsq(X_ar1, y_unfold, rcond=None)
    y_pred_ar1 = X_ar1 @ coeffs_ar1
    ss_res_ar1 = np.sum((y_unfold - y_pred_ar1) ** 2)
    r2_ar1 = 1 - ss_res_ar1 / ss_tot_unfold

    results['raw_zeros'] = {
        'r2': fit_raw['r2'],
        'mean_rel_error': fit_raw['mean_rel_error'],
        'a': fit_raw['a'],
        'b': fit_raw['b']
    }

    results['unfolded'] = {
        'r2': float(r2_unfold),
        'a': float(a_unfold),
        'b': float(b_unfold),
        'c': float(c_unfold),
        'a_plus_b': float(a_unfold + b_unfold),
        'mean_fluctuation': float(np.mean(x)),
        'std_fluctuation': float(np.std(x))
    }

    results['ar1_baseline'] = {
        'r2': float(r2_ar1),
        'description': 'Simple autoregression x_n ~ x_{n-1}'
    }

    print(f"\nRaw zeros (baseline):")
    print(f"  R² = {fit_raw['r2']:.8f}")
    print(f"  a = {fit_raw['a']:.6f}, b = {fit_raw['b']:.6f}")

    print(f"\nUnfolded fluctuations (THE REAL TEST):")
    print(f"  R² = {r2_unfold:.8f}")
    print(f"  a = {a_unfold:.6f}, b = {b_unfold:.6f}")
    print(f"  a + b = {a_unfold + b_unfold:.6f}")

    print(f"\nAR(1) baseline on fluctuations:")
    print(f"  R² = {r2_ar1:.8f}")

    # Key insight: ratio of explained variance
    improvement = (r2_unfold - r2_ar1) / (1 - r2_ar1) if r2_ar1 < 1 else 0
    results['improvement_over_ar1'] = float(improvement)

    print(f"\nImprovement over AR(1): {improvement:.4f}")

    # Verdict
    if r2_unfold > 0.5:
        verdict = "PASS - Strong structure in fluctuations"
        results['verdict'] = 'PASS'
    elif r2_unfold > 0.2:
        verdict = "MARGINAL - Some structure in fluctuations"
        results['verdict'] = 'MARGINAL'
    elif r2_unfold > r2_ar1 * 1.5:
        verdict = "MARGINAL - Better than AR(1) but weak"
        results['verdict'] = 'MARGINAL'
    else:
        verdict = "FAIL - No significant structure beyond trend"
        results['verdict'] = 'FAIL'

    print(f"\n→ VERDICT: {verdict}")

    # Additional: error in units of local spacing
    local_spacing = np.diff(gamma)
    mean_spacing = np.mean(local_spacing[start:end-1])
    error_in_spacings = np.std(y_unfold - y_pred_unfold) / mean_spacing
    results['error_in_spacings'] = float(error_in_spacings)
    print(f"  Error in units of mean spacing: {error_in_spacings:.4f}")

    return results

# ============================================================================
# TEST 4: GUE RANDOM MATRICES
# ============================================================================

def test_gue_comparison(zeros: np.ndarray, n_trials: int = 10) -> Dict:
    """
    Critical test: Does the Fibonacci recurrence work on GUE eigenvalues?

    GUE (Gaussian Unitary Ensemble) eigenvalues share many statistical
    properties with Riemann zeros (Montgomery-Odlyzko law).

    If the recurrence works equally well on GUE → it's a GENERIC property
    of determinantal point processes, NOT specific to Riemann.
    """
    print("\n" + "=" * 70)
    print("TEST 4: GUE RANDOM MATRIX COMPARISON")
    print("=" * 70)

    results = {}

    # Riemann zeros baseline
    n_test = min(10000, len(zeros))
    riemann_fit = fit_recurrence(zeros, 8, 21, end_idx=n_test)

    print(f"Riemann zeros (n={n_test}):")
    print(f"  R² = {riemann_fit['r2']:.8f}")
    print(f"  Mean rel error = {riemann_fit['mean_rel_error']:.4f}%")

    results['riemann'] = {
        'r2': riemann_fit['r2'],
        'mean_rel_error': riemann_fit['mean_rel_error'],
        'a': riemann_fit['a'],
        'b': riemann_fit['b']
    }

    # Generate GUE eigenvalues
    print(f"\nGenerating {n_trials} GUE trials (n={n_test} eigenvalues each)...")

    gue_results = []
    for trial in range(n_trials):
        # GUE matrix: H = (A + A†) / 2 where A has complex Gaussian entries
        N = n_test + 50  # Extra to ensure enough after sorting
        A = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        H = (A + A.conj().T) / 2
        eigenvalues = np.sort(np.linalg.eigvalsh(H))

        # Unfold to match Riemann zero density (semicircle → uniform)
        # For GUE, eigenvalues follow semicircle law, unfold via:
        # x = (2N/π) * arcsin(λ / (2√N))
        scaled = eigenvalues / (2 * np.sqrt(N))
        scaled = np.clip(scaled, -0.999, 0.999)
        unfolded = (2 * N / np.pi) * np.arcsin(scaled)

        # Shift to positive and scale like Riemann zeros
        unfolded = unfolded - unfolded.min() + 14  # Start near γ_1 ≈ 14
        unfolded = unfolded[:n_test]

        # Fit recurrence
        gue_fit = fit_recurrence(unfolded, 8, 21)
        if gue_fit:
            gue_results.append({
                'r2': gue_fit['r2'],
                'mean_rel_error': gue_fit['mean_rel_error'],
                'a': gue_fit['a'],
                'b': gue_fit['b']
            })

    if gue_results:
        gue_r2 = [r['r2'] for r in gue_results]
        gue_errors = [r['mean_rel_error'] for r in gue_results]
        gue_a = [r['a'] for r in gue_results]
        gue_b = [r['b'] for r in gue_results]

        results['gue'] = {
            'mean_r2': float(np.mean(gue_r2)),
            'std_r2': float(np.std(gue_r2)),
            'mean_rel_error': float(np.mean(gue_errors)),
            'mean_a': float(np.mean(gue_a)),
            'std_a': float(np.std(gue_a)),
            'mean_b': float(np.mean(gue_b)),
            'std_b': float(np.std(gue_b)),
            'n_trials': n_trials
        }

        print(f"\nGUE eigenvalues (mean over {n_trials} trials):")
        print(f"  R² = {np.mean(gue_r2):.8f} ± {np.std(gue_r2):.8f}")
        print(f"  Mean rel error = {np.mean(gue_errors):.4f}% ± {np.std(gue_errors):.4f}%")
        print(f"  a = {np.mean(gue_a):.6f} ± {np.std(gue_a):.6f}")
        print(f"  b = {np.mean(gue_b):.6f} ± {np.std(gue_b):.6f}")

        # Key comparison
        r2_diff = riemann_fit['r2'] - np.mean(gue_r2)
        a_diff = abs(riemann_fit['a'] - np.mean(gue_a))

        results['comparison'] = {
            'r2_difference': float(r2_diff),
            'a_difference': float(a_diff),
            'riemann_a_minus_gue_a': float(riemann_fit['a'] - np.mean(gue_a))
        }

        print(f"\nCOMPARISON:")
        print(f"  R² difference (Riemann - GUE): {r2_diff:.8f}")
        print(f"  |a| difference: {a_diff:.6f}")
        print(f"  Riemann a vs GUE a: {riemann_fit['a']:.6f} vs {np.mean(gue_a):.6f}")

        # Verdict: Does Riemann have SPECIAL structure?
        # If coefficients are similar AND R² is similar → generic property
        if a_diff > 0.1 and r2_diff > 0.001:
            verdict = "PASS - Riemann shows DISTINCT structure from GUE"
            results['verdict'] = 'PASS'
        elif r2_diff > 0.01:
            verdict = "PASS - Riemann has significantly better fit than GUE"
            results['verdict'] = 'PASS'
        elif a_diff < 0.05 and abs(r2_diff) < 0.001:
            verdict = "FAIL - Recurrence is GENERIC (works equally on GUE)"
            results['verdict'] = 'FAIL'
        else:
            verdict = "MARGINAL - Some distinction but not definitive"
            results['verdict'] = 'MARGINAL'

        print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 5: BASELINE COMPARISON
# ============================================================================

def test_baseline_comparison(zeros: np.ndarray) -> Dict:
    """
    Can ANY smooth monotone sequence achieve similar R²?

    We test:
    1. Perfect log curve: y = a * log(n)
    2. Perfect power law: y = a * n^b
    3. Random smooth monotone sequence

    If all achieve R² > 0.999, then high R² on Riemann is meaningless.
    """
    print("\n" + "=" * 70)
    print("TEST 5: BASELINE COMPARISON")
    print("=" * 70)

    results = {}
    n_test = min(50000, len(zeros))

    # Riemann baseline
    riemann_fit = fit_recurrence(zeros, 8, 21, end_idx=n_test)
    results['riemann'] = {
        'r2': riemann_fit['r2'],
        'mean_rel_error': riemann_fit['mean_rel_error'],
        'a': riemann_fit['a'],
        'b': riemann_fit['b']
    }

    print(f"Riemann zeros:")
    print(f"  R² = {riemann_fit['r2']:.8f}")
    print(f"  a = {riemann_fit['a']:.6f}, b = {riemann_fit['b']:.6f}")

    # Baseline 1: Log curve (similar growth to Riemann)
    n = np.arange(1, n_test + 1)
    # Fit Riemann zeros to get parameters
    # γ_n ~ (2πn / log(n)) by prime counting approximation
    log_sequence = 2 * np.pi * n / np.log(n + 1)
    # Scale to match Riemann
    scale = zeros[n_test-1] / log_sequence[-1]
    log_sequence = log_sequence * scale

    log_fit = fit_recurrence(log_sequence, 8, 21)
    results['log_curve'] = {
        'r2': log_fit['r2'],
        'mean_rel_error': log_fit['mean_rel_error'],
        'a': log_fit['a'],
        'b': log_fit['b']
    }

    print(f"\nLog curve y = 2πn/log(n):")
    print(f"  R² = {log_fit['r2']:.8f}")
    print(f"  a = {log_fit['a']:.6f}, b = {log_fit['b']:.6f}")

    # Baseline 2: Power law
    power_sequence = n ** 0.95 * 2  # Approximately matches Riemann growth
    scale = zeros[n_test-1] / power_sequence[-1]
    power_sequence = power_sequence * scale

    power_fit = fit_recurrence(power_sequence, 8, 21)
    results['power_law'] = {
        'r2': power_fit['r2'],
        'mean_rel_error': power_fit['mean_rel_error'],
        'a': power_fit['a'],
        'b': power_fit['b']
    }

    print(f"\nPower law y = n^0.95:")
    print(f"  R² = {power_fit['r2']:.8f}")
    print(f"  a = {power_fit['a']:.6f}, b = {power_fit['b']:.6f}")

    # Baseline 3: Smooth random monotone (cumsum of positive randoms)
    np.random.seed(42)
    random_increments = np.abs(np.random.randn(n_test)) + 0.1
    random_monotone = np.cumsum(random_increments)
    scale = zeros[n_test-1] / random_monotone[-1]
    random_monotone = random_monotone * scale

    random_fit = fit_recurrence(random_monotone, 8, 21)
    results['random_monotone'] = {
        'r2': random_fit['r2'],
        'mean_rel_error': random_fit['mean_rel_error'],
        'a': random_fit['a'],
        'b': random_fit['b']
    }

    print(f"\nRandom monotone (cumsum):")
    print(f"  R² = {random_fit['r2']:.8f}")
    print(f"  a = {random_fit['a']:.6f}, b = {random_fit['b']:.6f}")

    # KEY: Do baselines converge to SAME coefficient?
    print(f"\nCOEFFICIENT COMPARISON:")
    print(f"  Riemann:       a = {riemann_fit['a']:.6f}")
    print(f"  Log curve:     a = {log_fit['a']:.6f}")
    print(f"  Power law:     a = {power_fit['a']:.6f}")
    print(f"  Random:        a = {random_fit['a']:.6f}")
    print(f"  GIFT 3/2:      a = 1.500000")

    # Distance from 3/2
    riemann_dist = abs(riemann_fit['a'] - 1.5)
    log_dist = abs(log_fit['a'] - 1.5)
    power_dist = abs(power_fit['a'] - 1.5)
    random_dist = abs(random_fit['a'] - 1.5)

    results['distance_from_3_2'] = {
        'riemann': float(riemann_dist),
        'log_curve': float(log_dist),
        'power_law': float(power_dist),
        'random': float(random_dist)
    }

    print(f"\nDistance from 3/2:")
    print(f"  Riemann: {riemann_dist:.6f}")
    print(f"  Log:     {log_dist:.6f}")
    print(f"  Power:   {power_dist:.6f}")
    print(f"  Random:  {random_dist:.6f}")

    # Verdict
    if riemann_dist < log_dist and riemann_dist < power_dist and riemann_dist < 0.05:
        verdict = "PASS - Riemann uniquely close to 3/2"
        results['verdict'] = 'PASS'
    elif all(r['r2'] > 0.999 for r in [log_fit, power_fit, random_fit]):
        if riemann_dist < min(log_dist, power_dist, random_dist):
            verdict = "MARGINAL - All have high R², but Riemann closest to 3/2"
            results['verdict'] = 'MARGINAL'
        else:
            verdict = "FAIL - Generic high R² on any smooth sequence"
            results['verdict'] = 'FAIL'
    else:
        verdict = "PASS - Riemann has distinctive properties"
        results['verdict'] = 'PASS'

    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 6: HIGH ZEROS VALIDATION (Bonus)
# ============================================================================

def test_high_zeros(zeros: np.ndarray) -> Dict:
    """
    Test on zeros at VERY high heights (10^12, 10^21).
    If the recurrence is asymptotically exact, it should work better at height.
    """
    print("\n" + "=" * 70)
    print("TEST 6: HIGH ZEROS VALIDATION (BONUS)")
    print("=" * 70)

    results = {}

    # Low zeros baseline
    low_fit = fit_recurrence(zeros, 8, 21, end_idx=10000)
    results['low_zeros'] = {
        'r2': low_fit['r2'],
        'mean_rel_error': low_fit['mean_rel_error'],
        'a': low_fit['a'],
        'b': low_fit['b'],
        'height_range': 'γ_1 to γ_10000 (14 to ~9877)'
    }

    print(f"Low zeros (γ_1 to γ_10000):")
    print(f"  R² = {low_fit['r2']:.8f}")
    print(f"  a = {low_fit['a']:.6f}, b = {low_fit['b']:.6f}")

    # High zeros
    high_zeros = load_high_zeros()

    for label, hz in high_zeros.items():
        if len(hz) > 100:
            hfit = fit_recurrence(hz, 8, 21)
            results[f'high_{label}'] = {
                'r2': hfit['r2'],
                'mean_rel_error': hfit['mean_rel_error'],
                'a': hfit['a'],
                'b': hfit['b'],
                'n_zeros': len(hz),
                'height_range': f'Around {label}'
            }

            print(f"\nHigh zeros around {label} (n={len(hz)}):")
            print(f"  R² = {hfit['r2']:.8f}")
            print(f"  a = {hfit['a']:.6f}, b = {hfit['b']:.6f}")
            print(f"  Mean rel error = {hfit['mean_rel_error']:.4f}%")

    # Verdict
    if high_zeros:
        # Check if coefficients converge toward 3/2 at height
        low_a = low_fit['a']
        high_a_values = [results[k]['a'] for k in results if k.startswith('high_')]

        if high_a_values:
            mean_high_a = np.mean(high_a_values)
            if abs(mean_high_a - 1.5) < abs(low_a - 1.5):
                verdict = "PASS - Coefficients converge toward 3/2 at height"
                results['verdict'] = 'PASS'
            else:
                verdict = "MARGINAL - Coefficients stable but not converging to 3/2"
                results['verdict'] = 'MARGINAL'

            print(f"\nAsymptotic behavior:")
            print(f"  Low zeros a = {low_a:.6f}")
            print(f"  High zeros mean a = {mean_high_a:.6f}")
            print(f"  Target (3/2) = 1.500000")
            print(f"\n→ VERDICT: {verdict}")
    else:
        results['verdict'] = 'SKIPPED'
        print("\n→ High zeros data not available, test skipped")

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_battery():
    """Run all falsification tests and generate report."""
    print("\n" + "█" * 70)
    print("█  FALSIFICATION BATTERY: FIBONACCI-RIEMANN RECURRENCE")
    print("█  γₙ = (3/2)γₙ₋₈ - (1/2)γₙ₋₂₁ + c(N)")
    print("█" * 70)

    # Load data
    print("\nLoading Riemann zeros...")
    zeros = load_zeros(100000)
    print(f"✓ Loaded {len(zeros)} zeros")
    print(f"  Range: γ_1 = {zeros[0]:.6f} to γ_{len(zeros)} = {zeros[-1]:.6f}")

    all_results = {}
    verdicts = []

    # Run all tests
    all_results['test1_out_of_sample'] = test_out_of_sample(zeros)
    verdicts.append(('Out-of-Sample', all_results['test1_out_of_sample'].get('verdict', 'N/A')))

    all_results['test2_robustness'] = test_coefficient_robustness(zeros)
    verdicts.append(('Coefficient Robustness', all_results['test2_robustness'].get('verdict', 'N/A')))

    all_results['test3_unfolded'] = test_unfolded_fluctuations(zeros)
    verdicts.append(('Unfolded Fluctuations', all_results['test3_unfolded'].get('verdict', 'N/A')))

    all_results['test4_gue'] = test_gue_comparison(zeros, n_trials=10)
    verdicts.append(('GUE Comparison', all_results['test4_gue'].get('verdict', 'N/A')))

    all_results['test5_baseline'] = test_baseline_comparison(zeros)
    verdicts.append(('Baseline Comparison', all_results['test5_baseline'].get('verdict', 'N/A')))

    all_results['test6_high_zeros'] = test_high_zeros(zeros)
    verdicts.append(('High Zeros', all_results['test6_high_zeros'].get('verdict', 'N/A')))

    # Final summary
    print("\n" + "█" * 70)
    print("█  FINAL SUMMARY")
    print("█" * 70)

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)

    pass_count = 0
    fail_count = 0
    marginal_count = 0

    for test_name, verdict in verdicts:
        symbol = "✓" if verdict == 'PASS' else ("✗" if verdict == 'FAIL' else "~")
        print(f"  {symbol} {test_name}: {verdict}")
        if verdict == 'PASS':
            pass_count += 1
        elif verdict == 'FAIL':
            fail_count += 1
        elif verdict == 'MARGINAL':
            marginal_count += 1

    print(f"\nSCORE: {pass_count} PASS / {fail_count} FAIL / {marginal_count} MARGINAL")

    # Overall verdict
    print("\n" + "=" * 50)
    if fail_count == 0 and pass_count >= 4:
        overall = "STRONG EVIDENCE - Recurrence appears to be REAL STRUCTURE"
        all_results['overall_verdict'] = 'STRONG'
    elif fail_count <= 1 and pass_count >= 3:
        overall = "MODERATE EVIDENCE - Recurrence likely real but needs more investigation"
        all_results['overall_verdict'] = 'MODERATE'
    elif fail_count >= 2:
        overall = "WEAK EVIDENCE - Significant concerns about validity"
        all_results['overall_verdict'] = 'WEAK'
    else:
        overall = "INCONCLUSIVE - More testing needed"
        all_results['overall_verdict'] = 'INCONCLUSIVE'

    print(f"OVERALL: {overall}")
    print("=" * 50)

    all_results['summary'] = {
        'pass_count': pass_count,
        'fail_count': fail_count,
        'marginal_count': marginal_count,
        'verdicts': verdicts
    }

    # Save results
    output_path = Path(__file__).parent / "falsification_results.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(v) for v in obj)
        return obj

    # Remove large arrays before saving
    results_to_save = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            results_to_save[k] = {
                kk: vv for kk, vv in v.items()
                if kk not in ['predictions', 'actuals', 'residuals', 'scan']
            }
        else:
            results_to_save[k] = v

    results_to_save = convert_to_native(results_to_save)

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return all_results

if __name__ == "__main__":
    results = run_full_battery()
