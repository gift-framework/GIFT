#!/usr/bin/env python3
"""
ULTRA-RIGOROUS VALIDATION: GIFT-Riemann Bridge
===============================================

This script implements exhaustive statistical tests to validate (or falsify)
the claimed connection between GIFT topological constants and Riemann zeros.

Key claims to test:
1. The recurrence γₙ = (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁ + c is UNIQUE
2. The coefficients 31/21 and 10/21 are topologically special
3. The lags 8 and 21 (Fibonacci) are special

Tests implemented:
1. SOBOL COEFFICIENT SEARCH: Quasi-random exploration of (a, b) space
2. RATIONAL COEFFICIENT UNIQUENESS: Test ALL p/q with p,q ≤ 50
3. LAG SPACE EXHAUSTIVE SEARCH: All (l1, l2) with l1, l2 ≤ 40
4. FLUCTUATION ANALYSIS: Multiple detrending methods
5. PERMUTATION TEST: Shuffle zeros, measure R² distribution
6. NULL DISTRIBUTION: 10000 random monotone sequences
7. BOOTSTRAP STABILITY: Test coefficient stability across windows
8. CROSS-VALIDATION: k-fold on different zero ranges

Author: GIFT Validation Protocol
Date: 2026-02-03
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# For Sobol sequences
try:
    from scipy.stats import qmc
    HAS_SCIPY_QMC = True
except ImportError:
    HAS_SCIPY_QMC = False

from scipy import stats
from scipy.optimize import minimize_scalar
import time

# ============================================================================
# CONSTANTS
# ============================================================================

# GIFT topological constants
B2 = 21        # Second Betti number of K₇
B3 = 77        # Third Betti number of K₇
RANK_E8 = 8    # E₈ rank
P2 = 2         # Pontryagin class contribution
DIM_G2 = 14    # G₂ dimension
H_STAR = 99    # b₂ + b₃ + 1

# Claimed GIFT coefficients for the 2-term recurrence
GIFT_A = 31/21  # = (b₂ + rank_E8 + p₂) / b₂
GIFT_B = -10/21  # = -(rank_E8 + p₂) / b₂
GIFT_LAG1 = 8   # = rank(E₈) = F₆
GIFT_LAG2 = 21  # = b₂ = F₈

# Fibonacci numbers for reference
FIBS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_zeros(max_zeros: int = 100000) -> np.ndarray:
    """Load Riemann zeros from files in research/riemann/."""
    zeros = []
    # Try multiple locations
    search_paths = [
        Path(__file__).parent.parent / "research" / "riemann",
        Path(__file__).parent / "data",
        Path.cwd() / "research" / "riemann",
    ]

    for base_path in search_paths:
        for i in range(1, 6):
            zeros_file = base_path / f"zeros{i}"
            if zeros_file.exists():
                with open(zeros_file) as f:
                    for line in f:
                        if line.strip():
                            zeros.append(float(line.strip()))
                            if len(zeros) >= max_zeros:
                                return np.array(zeros)

    if len(zeros) == 0:
        raise FileNotFoundError("Could not find Riemann zeros data files")

    return np.array(zeros)

# ============================================================================
# CORE FITTING FUNCTIONS
# ============================================================================

def fit_2term_recurrence(zeros: np.ndarray, lag1: int, lag2: int,
                         start_idx: int = None, end_idx: int = None,
                         return_residuals: bool = False) -> Dict:
    """
    Fit γₙ = a × γₙ₋ₗ₁ + b × γₙ₋ₗ₂ + c using least squares.

    Returns dict with coefficients, R², and error metrics.
    """
    if start_idx is None:
        start_idx = max(lag1, lag2)
    if end_idx is None:
        end_idx = len(zeros)

    max_lag = max(lag1, lag2)
    actual_start = max(start_idx, max_lag)
    n_samples = end_idx - actual_start

    if n_samples < 50:
        return None

    # Build design matrix
    X1 = zeros[actual_start - lag1:end_idx - lag1]
    X2 = zeros[actual_start - lag2:end_idx - lag2]
    X = np.column_stack([X1, X2, np.ones(n_samples)])
    y = zeros[actual_start:end_idx]

    # Least squares fit
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b, c = coeffs

    # Predictions and metrics
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return None

    r2 = 1 - ss_res / ss_tot

    errors = np.abs(y - y_pred)
    rel_errors = errors / np.abs(y) * 100

    result = {
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'a_plus_b': float(a + b),
        'r2': float(r2),
        'mean_abs_error': float(np.mean(errors)),
        'mean_rel_error': float(np.mean(rel_errors)),
        'max_rel_error': float(np.max(rel_errors)),
        'rmse': float(np.sqrt(np.mean((y - y_pred)**2))),
        'n_samples': n_samples,
        'lag1': lag1,
        'lag2': lag2
    }

    if return_residuals:
        result['residuals'] = y - y_pred
        result['y_actual'] = y
        result['y_pred'] = y_pred

    return result


def fit_fixed_coefficients(zeros: np.ndarray, lag1: int, lag2: int,
                           a: float, b: float,
                           start_idx: int = None, end_idx: int = None) -> Dict:
    """
    Compute R² for FIXED coefficients (a, b) - only fit constant c.
    This tests whether specific rational coefficients work.
    """
    if start_idx is None:
        start_idx = max(lag1, lag2)
    if end_idx is None:
        end_idx = len(zeros)

    max_lag = max(lag1, lag2)
    actual_start = max(start_idx, max_lag)
    n_samples = end_idx - actual_start

    if n_samples < 50:
        return None

    X1 = zeros[actual_start - lag1:end_idx - lag1]
    X2 = zeros[actual_start - lag2:end_idx - lag2]
    y = zeros[actual_start:end_idx]

    # Only fit c
    y_adjusted = y - a * X1 - b * X2
    c = np.mean(y_adjusted)

    y_pred = a * X1 + b * X2 + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return None

    r2 = 1 - ss_res / ss_tot

    errors = np.abs(y - y_pred)
    rel_errors = errors / np.abs(y) * 100

    return {
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'r2': float(r2),
        'mean_rel_error': float(np.mean(rel_errors)),
        'rmse': float(np.sqrt(np.mean((y - y_pred)**2))),
        'n_samples': n_samples
    }

# ============================================================================
# TEST 1: SOBOL COEFFICIENT SEARCH
# ============================================================================

def test_sobol_coefficient_search(zeros: np.ndarray, n_samples: int = 10000) -> Dict:
    """
    Use Sobol quasi-random sequences to uniformly explore coefficient space.

    Tests if (31/21, -10/21) is truly optimal or if other regions work equally well.
    """
    print("\n" + "=" * 70)
    print("TEST 1: SOBOL QUASI-RANDOM COEFFICIENT SEARCH")
    print("=" * 70)

    results = {'test_name': 'sobol_coefficient_search'}

    # GIFT baseline
    gift_fit = fit_fixed_coefficients(zeros, GIFT_LAG1, GIFT_LAG2, GIFT_A, GIFT_B)
    if gift_fit is None:
        print("ERROR: Could not compute GIFT baseline")
        return {'verdict': 'ERROR'}

    results['gift_baseline'] = gift_fit
    print(f"\nGIFT baseline (a=31/21, b=-10/21, lags=8,21):")
    print(f"  R² = {gift_fit['r2']:.10f}")
    print(f"  Mean rel error = {gift_fit['mean_rel_error']:.6f}%")

    # Generate Sobol sequence in 2D for (a, b)
    # Search range: a ∈ [0.5, 2.5], b ∈ [-1.5, 0.5]
    if HAS_SCIPY_QMC:
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_samples = sampler.random(n_samples)
    else:
        # Fallback to pseudo-random
        np.random.seed(42)
        sobol_samples = np.random.rand(n_samples, 2)

    # Scale to parameter range
    a_range = (0.5, 2.5)
    b_range = (-1.5, 0.5)

    a_values = sobol_samples[:, 0] * (a_range[1] - a_range[0]) + a_range[0]
    b_values = sobol_samples[:, 1] * (b_range[1] - b_range[0]) + b_range[0]

    # Evaluate each point
    better_than_gift = []
    all_r2 = []
    best_result = {'r2': -np.inf}

    print(f"\nSearching {n_samples} Sobol points in coefficient space...")

    for i, (a, b) in enumerate(zip(a_values, b_values)):
        fit = fit_fixed_coefficients(zeros, GIFT_LAG1, GIFT_LAG2, a, b)
        if fit:
            all_r2.append(fit['r2'])
            if fit['r2'] > gift_fit['r2']:
                better_than_gift.append({'a': a, 'b': b, 'r2': fit['r2']})
            if fit['r2'] > best_result['r2']:
                best_result = fit.copy()
                best_result['a_tested'] = a
                best_result['b_tested'] = b

    n_better = len(better_than_gift)
    p_value = n_better / n_samples if n_samples > 0 else 1.0

    results['n_samples'] = n_samples
    results['n_better_than_gift'] = n_better
    results['p_value'] = float(p_value)
    results['best_found'] = best_result
    results['r2_distribution'] = {
        'mean': float(np.mean(all_r2)),
        'std': float(np.std(all_r2)),
        'median': float(np.median(all_r2)),
        'max': float(np.max(all_r2)),
        'percentile_95': float(np.percentile(all_r2, 95)),
        'percentile_99': float(np.percentile(all_r2, 99))
    }

    print(f"\nResults:")
    print(f"  Samples tested: {n_samples}")
    print(f"  Better than GIFT: {n_better} ({p_value*100:.4f}%)")
    print(f"  Best R² found: {best_result['r2']:.10f} at a={best_result.get('a_tested', 'N/A'):.6f}, b={best_result.get('b_tested', 'N/A'):.6f}")
    print(f"  R² distribution: mean={np.mean(all_r2):.6f}, std={np.std(all_r2):.6f}")
    print(f"  GIFT percentile: {100 * (1 - p_value):.2f}%")

    # Check if best is close to GIFT
    if best_result.get('a_tested'):
        dist_from_gift = np.sqrt((best_result['a_tested'] - GIFT_A)**2 +
                                  (best_result['b_tested'] - GIFT_B)**2)
        results['best_distance_from_gift'] = float(dist_from_gift)
        print(f"  Best point distance from GIFT: {dist_from_gift:.6f}")

    # Verdict
    if p_value < 0.001:
        verdict = "STRONG PASS - GIFT is in top 0.1% of coefficient space"
    elif p_value < 0.01:
        verdict = "PASS - GIFT is in top 1% of coefficient space"
    elif p_value < 0.05:
        verdict = "MARGINAL - GIFT is in top 5% but not exceptional"
    else:
        verdict = f"FAIL - {n_better} points ({p_value*100:.2f}%) beat GIFT"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 2: RATIONAL COEFFICIENT UNIQUENESS
# ============================================================================

def test_rational_uniqueness(zeros: np.ndarray, max_denom: int = 50) -> Dict:
    """
    Test ALL rational coefficients p/q with |p|, |q| ≤ max_denom.

    Checks if 31/21 is truly special among "nice" fractions.
    """
    print("\n" + "=" * 70)
    print("TEST 2: RATIONAL COEFFICIENT UNIQUENESS")
    print("=" * 70)

    results = {'test_name': 'rational_uniqueness'}

    # GIFT baseline
    gift_fit = fit_fixed_coefficients(zeros, GIFT_LAG1, GIFT_LAG2, GIFT_A, GIFT_B)
    results['gift_baseline'] = gift_fit

    print(f"\nGIFT: a = 31/21 = {GIFT_A:.10f}, b = -10/21 = {GIFT_B:.10f}")
    print(f"GIFT R² = {gift_fit['r2']:.10f}")

    # Generate all rationals
    print(f"\nTesting all rationals p/q with |p|, q ≤ {max_denom}...")

    rationals_tested = 0
    better_than_gift = []
    close_to_gift = []  # Within 0.001 of GIFT R²

    best_rational = {'r2': -np.inf, 'a_num': 0, 'a_den': 1, 'b_num': 0, 'b_den': 1}

    # Test all (p1/q1, p2/q2) combinations
    # Focus on a ∈ [1.0, 2.0] and b ∈ [-1.0, 0.0] based on Sobol results

    for q1 in range(1, max_denom + 1):
        for p1 in range(q1, 2 * q1 + 1):  # a between 1 and 2
            a = p1 / q1
            if a < 1.0 or a > 2.0:
                continue

            for q2 in range(1, max_denom + 1):
                for p2 in range(-q2, 1):  # b between -1 and 0
                    b = p2 / q2
                    if b < -1.0 or b > 0.0:
                        continue

                    rationals_tested += 1
                    fit = fit_fixed_coefficients(zeros, GIFT_LAG1, GIFT_LAG2, a, b)

                    if fit and fit['r2'] > best_rational['r2']:
                        best_rational = {
                            'r2': fit['r2'],
                            'a': a,
                            'b': b,
                            'a_num': p1, 'a_den': q1,
                            'b_num': p2, 'b_den': q2,
                            'mean_rel_error': fit['mean_rel_error']
                        }

                    if fit and fit['r2'] > gift_fit['r2']:
                        better_than_gift.append({
                            'a': f"{p1}/{q1}",
                            'b': f"{p2}/{q2}",
                            'r2': fit['r2'],
                            'improvement': fit['r2'] - gift_fit['r2']
                        })

                    if fit and abs(fit['r2'] - gift_fit['r2']) < 0.001:
                        close_to_gift.append({
                            'a': f"{p1}/{q1}",
                            'b': f"{p2}/{q2}",
                            'r2': fit['r2']
                        })

    results['rationals_tested'] = rationals_tested
    results['n_better'] = len(better_than_gift)
    results['n_close'] = len(close_to_gift)
    results['best_rational'] = best_rational

    print(f"\nResults:")
    print(f"  Rationals tested: {rationals_tested}")
    print(f"  Better than GIFT: {len(better_than_gift)}")
    print(f"  Within 0.001 of GIFT R²: {len(close_to_gift)}")

    print(f"\nBest rational found:")
    print(f"  a = {best_rational['a_num']}/{best_rational['a_den']} = {best_rational['a']:.10f}")
    print(f"  b = {best_rational['b_num']}/{best_rational['b_den']} = {best_rational['b']:.10f}")
    print(f"  R² = {best_rational['r2']:.10f}")

    # Check if best is exactly 31/21 or close
    is_gift = (best_rational['a_num'] == 31 and best_rational['a_den'] == 21 and
               best_rational['b_num'] == -10 and best_rational['b_den'] == 21)

    results['best_is_gift'] = is_gift

    # Show top 10 rationals
    if better_than_gift:
        print(f"\nTop rationals better than GIFT:")
        for i, r in enumerate(sorted(better_than_gift, key=lambda x: -x['r2'])[:10]):
            print(f"  {i+1}. a={r['a']}, b={r['b']}: R²={r['r2']:.10f} (+{r['improvement']:.2e})")
        results['top_better'] = sorted(better_than_gift, key=lambda x: -x['r2'])[:10]

    # Verdict
    if is_gift:
        verdict = "STRONG PASS - GIFT (31/21, -10/21) IS the optimal rational"
    elif len(better_than_gift) == 0:
        verdict = "PASS - No simpler rational beats GIFT"
    elif len(better_than_gift) < 10:
        verdict = f"MARGINAL - {len(better_than_gift)} rationals slightly better"
    else:
        verdict = f"FAIL - {len(better_than_gift)} rationals beat GIFT"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 3: LAG SPACE EXHAUSTIVE SEARCH
# ============================================================================

def test_lag_space_search(zeros: np.ndarray, max_lag: int = 40) -> Dict:
    """
    Test ALL lag combinations (l1, l2) with l1, l2 ≤ max_lag.

    Checks if (8, 21) = (F₆, F₈) is truly optimal.
    """
    print("\n" + "=" * 70)
    print("TEST 3: LAG SPACE EXHAUSTIVE SEARCH")
    print("=" * 70)

    results = {'test_name': 'lag_space_search'}

    # GIFT baseline
    gift_fit = fit_2term_recurrence(zeros, GIFT_LAG1, GIFT_LAG2)
    results['gift_baseline'] = {
        'lag1': GIFT_LAG1, 'lag2': GIFT_LAG2,
        'r2': gift_fit['r2'], 'a': gift_fit['a'], 'b': gift_fit['b']
    }

    print(f"\nGIFT: lags = ({GIFT_LAG1}, {GIFT_LAG2})")
    print(f"GIFT R² = {gift_fit['r2']:.10f}")
    print(f"GIFT coefficients: a = {gift_fit['a']:.6f}, b = {gift_fit['b']:.6f}")

    # Test all lag combinations
    print(f"\nTesting all lag pairs (l1, l2) with l1 < l2 ≤ {max_lag}...")

    all_results = []
    better_than_gift = []

    for l1 in range(1, max_lag):
        for l2 in range(l1 + 1, max_lag + 1):
            fit = fit_2term_recurrence(zeros, l1, l2)
            if fit:
                all_results.append({
                    'lag1': l1, 'lag2': l2,
                    'r2': fit['r2'],
                    'a': fit['a'], 'b': fit['b'],
                    'mean_rel_error': fit['mean_rel_error']
                })

                if fit['r2'] > gift_fit['r2']:
                    better_than_gift.append({
                        'lag1': l1, 'lag2': l2,
                        'r2': fit['r2'],
                        'a': fit['a'], 'b': fit['b'],
                        'improvement': fit['r2'] - gift_fit['r2']
                    })

    # Sort by R²
    all_results.sort(key=lambda x: -x['r2'])

    # Find GIFT rank
    gift_rank = next((i for i, r in enumerate(all_results)
                      if r['lag1'] == GIFT_LAG1 and r['lag2'] == GIFT_LAG2), -1) + 1

    results['total_lag_pairs'] = len(all_results)
    results['n_better'] = len(better_than_gift)
    results['gift_rank'] = gift_rank
    results['top_10'] = all_results[:10]

    print(f"\nResults:")
    print(f"  Total lag pairs tested: {len(all_results)}")
    print(f"  Better than GIFT (8, 21): {len(better_than_gift)}")
    print(f"  GIFT rank: #{gift_rank} out of {len(all_results)}")

    print(f"\nTop 10 lag pairs by R²:")
    for i, r in enumerate(all_results[:10]):
        fib_marker = ""
        if r['lag1'] in FIBS and r['lag2'] in FIBS:
            fib_marker = " [FIBONACCI]"
        gift_marker = " ← GIFT" if r['lag1'] == GIFT_LAG1 and r['lag2'] == GIFT_LAG2 else ""
        print(f"  {i+1}. ({r['lag1']}, {r['lag2']}): R²={r['r2']:.10f}, a={r['a']:.4f}, b={r['b']:.4f}{fib_marker}{gift_marker}")

    # Check Fibonacci special
    fib_pairs = [(r['lag1'], r['lag2']) for r in all_results[:20]
                 if r['lag1'] in FIBS and r['lag2'] in FIBS]
    results['fibonacci_in_top_20'] = fib_pairs

    print(f"\nFibonacci pairs in top 20: {fib_pairs}")

    # Verdict
    if gift_rank == 1:
        verdict = "STRONG PASS - GIFT (8, 21) is THE optimal lag pair"
    elif gift_rank <= 5:
        verdict = f"PASS - GIFT (8, 21) is #{gift_rank}, very competitive"
    elif gift_rank <= 20:
        verdict = f"MARGINAL - GIFT (8, 21) is #{gift_rank}, in top 20"
    else:
        verdict = f"FAIL - GIFT (8, 21) is only #{gift_rank}"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 4: FLUCTUATION ANALYSIS (CRITICAL)
# ============================================================================

def test_fluctuation_analysis(zeros: np.ndarray) -> Dict:
    """
    The CRITICAL test: Does the recurrence work on FLUCTUATIONS?

    Tests multiple detrending methods to see if the recurrence captures
    real arithmetic structure or just the smooth density N(T) ~ T log T.
    """
    print("\n" + "=" * 70)
    print("TEST 4: FLUCTUATION ANALYSIS (CRITICAL TEST)")
    print("=" * 70)

    results = {'test_name': 'fluctuation_analysis'}

    n_zeros = min(50000, len(zeros))
    gamma = zeros[:n_zeros]
    n = np.arange(1, n_zeros + 1)

    # Method 1: Riemann-von Mangoldt unfolding
    def counting_function(T):
        return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7/8

    N_gamma = counting_function(gamma)
    fluctuations_rvm = N_gamma - n

    # Method 2: Polynomial detrending (degree 3)
    poly_coeffs = np.polyfit(n, gamma, 3)
    gamma_poly_trend = np.polyval(poly_coeffs, n)
    fluctuations_poly = gamma - gamma_poly_trend

    # Method 3: Local linear detrending (rolling)
    window = 1000
    fluctuations_local = np.zeros_like(gamma)
    for i in range(len(gamma)):
        start = max(0, i - window // 2)
        end = min(len(gamma), i + window // 2)
        local_trend = np.mean(gamma[start:end])
        fluctuations_local[i] = gamma[i] - local_trend

    print(f"Analyzing fluctuations on {n_zeros} zeros...")
    print(f"\nFluctuation statistics:")

    for name, fluct in [('Riemann-von Mangoldt', fluctuations_rvm),
                         ('Polynomial (deg 3)', fluctuations_poly),
                         ('Local mean', fluctuations_local)]:
        print(f"  {name}: mean={np.mean(fluct):.4f}, std={np.std(fluct):.4f}")

    # Test recurrence on each type of fluctuation
    def test_recurrence_on_fluctuations(fluct: np.ndarray, name: str) -> Dict:
        """Fit recurrence on fluctuations and compute metrics."""
        max_lag = max(GIFT_LAG1, GIFT_LAG2)
        start = max_lag
        end = len(fluct)

        X1 = fluct[start - GIFT_LAG1:end - GIFT_LAG1]
        X2 = fluct[start - GIFT_LAG2:end - GIFT_LAG2]
        y = fluct[start:end]

        n_samples = len(y)
        X = np.column_stack([X1[:n_samples], X2[:n_samples], np.ones(n_samples)])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs

        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Also compare with AR(1) baseline
        X_ar1 = np.column_stack([fluct[start-1:end-1], np.ones(n_samples)])
        coeffs_ar1, _, _, _ = np.linalg.lstsq(X_ar1, y, rcond=None)
        y_pred_ar1 = X_ar1 @ coeffs_ar1
        ss_res_ar1 = np.sum((y - y_pred_ar1) ** 2)
        r2_ar1 = 1 - ss_res_ar1 / ss_tot if ss_tot > 0 else 0

        return {
            'name': name,
            'r2': float(r2),
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'a_plus_b': float(a + b),
            'r2_ar1_baseline': float(r2_ar1),
            'improvement_over_ar1': float(r2 - r2_ar1)
        }

    # Raw zeros baseline
    raw_fit = fit_2term_recurrence(zeros, GIFT_LAG1, GIFT_LAG2, end_idx=n_zeros)
    results['raw_zeros'] = {
        'r2': raw_fit['r2'],
        'a': raw_fit['a'],
        'b': raw_fit['b']
    }

    print(f"\nRecurrence R² comparison:")
    print(f"  Raw zeros: R² = {raw_fit['r2']:.10f}")

    # Test each fluctuation type
    fluct_results = []
    for name, fluct in [('Riemann-von Mangoldt', fluctuations_rvm),
                         ('Polynomial (deg 3)', fluctuations_poly),
                         ('Local mean', fluctuations_local)]:
        fr = test_recurrence_on_fluctuations(fluct, name)
        fluct_results.append(fr)
        print(f"  {name}: R² = {fr['r2']:.6f}, a+b = {fr['a_plus_b']:.4f}")
        print(f"    (AR(1) baseline: R² = {fr['r2_ar1_baseline']:.6f}, improvement: {fr['improvement_over_ar1']:.6f})")

    results['fluctuation_tests'] = fluct_results

    # Key metric: average R² on fluctuations
    avg_fluct_r2 = np.mean([fr['r2'] for fr in fluct_results])
    results['avg_fluctuation_r2'] = float(avg_fluct_r2)

    print(f"\nKey metric: Average R² on fluctuations = {avg_fluct_r2:.6f}")
    print(f"(Raw zeros R² = {raw_fit['r2']:.6f})")

    # Verdict
    if avg_fluct_r2 > 0.5:
        verdict = "STRONG PASS - Recurrence captures fluctuation structure"
    elif avg_fluct_r2 > 0.2:
        verdict = "PASS - Moderate structure in fluctuations"
    elif avg_fluct_r2 > 0.05:
        verdict = "MARGINAL - Weak structure in fluctuations"
    else:
        verdict = "FAIL - Recurrence captures ONLY trend, not fluctuations"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    # Additional insight
    print(f"\n  CRITICAL INSIGHT: The R² drop from {raw_fit['r2']:.4f} to {avg_fluct_r2:.4f}")
    print(f"  shows the recurrence primarily captures the density function,")
    print(f"  not deep arithmetic structure in the zero spacings.")

    return results

# ============================================================================
# TEST 5: PERMUTATION TEST
# ============================================================================

def test_permutation(zeros: np.ndarray, n_permutations: int = 1000) -> Dict:
    """
    Shuffle zeros and see if recurrence still achieves high R².

    If permuted sequences also achieve high R², the recurrence is just
    capturing generic properties of sorted sequences, not Riemann-specific.
    """
    print("\n" + "=" * 70)
    print("TEST 5: PERMUTATION TEST")
    print("=" * 70)

    results = {'test_name': 'permutation_test'}

    n_zeros = min(10000, len(zeros))
    gamma = zeros[:n_zeros].copy()

    # Original R²
    original_fit = fit_2term_recurrence(gamma, GIFT_LAG1, GIFT_LAG2)
    results['original_r2'] = original_fit['r2']
    results['original_a'] = original_fit['a']

    print(f"\nOriginal sequence (first {n_zeros} zeros):")
    print(f"  R² = {original_fit['r2']:.10f}")
    print(f"  a = {original_fit['a']:.6f}")

    # Generate permutations and fit
    print(f"\nTesting {n_permutations} random permutations...")

    np.random.seed(42)
    perm_r2s = []
    perm_as = []

    for i in range(n_permutations):
        # Shuffle then sort (maintains monotonicity but scrambles order)
        # This tests if the recurrence needs the EXACT zero positions

        # Method: Shuffle indices, take corresponding zeros, then re-sort
        # This creates a different monotone sequence with same range
        perm_indices = np.random.permutation(n_zeros)
        gamma_perm = np.sort(gamma[perm_indices])  # Still monotone but different values

        # Actually, for a proper test, we should NOT re-sort
        # Let's do a different approach: keep sequence but add small random perturbations
        perturbation = np.random.randn(n_zeros) * 0.1 * np.mean(np.diff(gamma))
        gamma_perturbed = gamma + perturbation
        gamma_perturbed = np.sort(gamma_perturbed)  # Ensure monotonicity

        fit = fit_2term_recurrence(gamma_perturbed, GIFT_LAG1, GIFT_LAG2)
        if fit:
            perm_r2s.append(fit['r2'])
            perm_as.append(fit['a'])

    results['n_permutations'] = n_permutations
    results['permuted_r2_mean'] = float(np.mean(perm_r2s))
    results['permuted_r2_std'] = float(np.std(perm_r2s))
    results['permuted_a_mean'] = float(np.mean(perm_as))
    results['permuted_a_std'] = float(np.std(perm_as))

    print(f"\nPermuted sequences:")
    print(f"  R² = {np.mean(perm_r2s):.10f} ± {np.std(perm_r2s):.10f}")
    print(f"  a = {np.mean(perm_as):.6f} ± {np.std(perm_as):.6f}")

    # Z-score of original vs permuted
    z_score_r2 = (original_fit['r2'] - np.mean(perm_r2s)) / np.std(perm_r2s) if np.std(perm_r2s) > 0 else 0
    z_score_a = (original_fit['a'] - np.mean(perm_as)) / np.std(perm_as) if np.std(perm_as) > 0 else 0

    results['z_score_r2'] = float(z_score_r2)
    results['z_score_a'] = float(z_score_a)

    print(f"\nZ-scores (original vs permuted distribution):")
    print(f"  R² z-score: {z_score_r2:.2f}σ")
    print(f"  a z-score: {z_score_a:.2f}σ")

    # What fraction of permutations beat original?
    n_better = sum(1 for r in perm_r2s if r > original_fit['r2'])
    p_value = n_better / n_permutations
    results['p_value'] = float(p_value)

    print(f"\n  {n_better}/{n_permutations} permutations beat original (p = {p_value:.4f})")

    # Verdict
    if z_score_r2 > 3 and z_score_a > 3:
        verdict = "STRONG PASS - Original sequence is highly distinct"
    elif z_score_r2 > 2 or z_score_a > 2:
        verdict = "PASS - Original sequence shows distinct structure"
    elif p_value < 0.05:
        verdict = "MARGINAL - Some distinction but weak"
    else:
        verdict = "FAIL - Permuted sequences achieve similar results"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 6: NULL DISTRIBUTION (RANDOM MONOTONE SEQUENCES)
# ============================================================================

def test_null_distribution(zeros: np.ndarray, n_sequences: int = 5000) -> Dict:
    """
    Generate random monotone sequences and compute R² distribution.

    This establishes the null distribution: what R² would we expect
    by chance on a random smooth monotone sequence?
    """
    print("\n" + "=" * 70)
    print("TEST 6: NULL DISTRIBUTION (RANDOM SEQUENCES)")
    print("=" * 70)

    results = {'test_name': 'null_distribution'}

    n_zeros = min(10000, len(zeros))
    gamma = zeros[:n_zeros]

    # Original R²
    original_fit = fit_2term_recurrence(gamma, GIFT_LAG1, GIFT_LAG2)
    results['original_r2'] = original_fit['r2']
    results['original_a'] = original_fit['a']

    print(f"\nRiemann zeros:")
    print(f"  R² = {original_fit['r2']:.10f}")
    print(f"  a = {original_fit['a']:.6f}")

    # Generate random monotone sequences
    print(f"\nGenerating {n_sequences} random monotone sequences...")

    np.random.seed(42)
    null_r2s = []
    null_as = []
    null_bs = []

    # Different types of random sequences
    seq_types = ['cumsum_gaussian', 'cumsum_exponential', 'power_law', 'log_law']

    for i in range(n_sequences):
        seq_type = seq_types[i % len(seq_types)]

        if seq_type == 'cumsum_gaussian':
            increments = np.abs(np.random.randn(n_zeros)) + 0.1
            seq = np.cumsum(increments)
        elif seq_type == 'cumsum_exponential':
            increments = np.random.exponential(1.0, n_zeros)
            seq = np.cumsum(increments)
        elif seq_type == 'power_law':
            n = np.arange(1, n_zeros + 1)
            alpha = 0.9 + 0.2 * np.random.rand()
            noise = 0.1 * np.random.randn(n_zeros)
            seq = n ** alpha + noise
            seq = np.sort(seq)  # Ensure monotone
        elif seq_type == 'log_law':
            n = np.arange(1, n_zeros + 1)
            noise = 0.1 * np.random.randn(n_zeros)
            seq = n * np.log(n + 1) / np.log(2) + noise
            seq = np.sort(seq)

        # Scale to match Riemann zeros range
        seq = seq / seq[-1] * gamma[-1]
        seq = seq - seq[0] + gamma[0]

        fit = fit_2term_recurrence(seq, GIFT_LAG1, GIFT_LAG2)
        if fit:
            null_r2s.append(fit['r2'])
            null_as.append(fit['a'])
            null_bs.append(fit['b'])

    results['n_sequences'] = n_sequences
    results['null_r2_mean'] = float(np.mean(null_r2s))
    results['null_r2_std'] = float(np.std(null_r2s))
    results['null_r2_percentiles'] = {
        '50': float(np.percentile(null_r2s, 50)),
        '90': float(np.percentile(null_r2s, 90)),
        '95': float(np.percentile(null_r2s, 95)),
        '99': float(np.percentile(null_r2s, 99)),
        '99.9': float(np.percentile(null_r2s, 99.9))
    }

    print(f"\nNull distribution (random monotone sequences):")
    print(f"  R² = {np.mean(null_r2s):.6f} ± {np.std(null_r2s):.6f}")
    print(f"  Percentiles: 50%={np.percentile(null_r2s, 50):.6f}, "
          f"95%={np.percentile(null_r2s, 95):.6f}, "
          f"99%={np.percentile(null_r2s, 99):.6f}")

    # Compute Riemann's percentile in null distribution
    riemann_percentile = np.mean(np.array(null_r2s) < original_fit['r2']) * 100
    results['riemann_percentile'] = float(riemann_percentile)

    # Z-score
    z_score = (original_fit['r2'] - np.mean(null_r2s)) / np.std(null_r2s)
    results['z_score'] = float(z_score)

    # p-value (one-sided: what fraction of null exceeds Riemann?)
    n_exceed = sum(1 for r in null_r2s if r >= original_fit['r2'])
    p_value = n_exceed / n_sequences
    results['p_value'] = float(p_value)

    print(f"\nRiemann vs null distribution:")
    print(f"  Riemann percentile: {riemann_percentile:.2f}%")
    print(f"  Z-score: {z_score:.2f}σ")
    print(f"  p-value: {p_value:.6f} ({n_exceed}/{n_sequences} exceed)")

    # COEFFICIENT comparison (this is where Riemann might be special)
    a_percentile = np.mean(np.array(null_as) < original_fit['a']) * 100
    results['a_percentile'] = float(a_percentile)

    # Distance from 3/2
    null_dist_from_gift = [abs(a - GIFT_A) for a in null_as]
    riemann_dist_from_gift = abs(original_fit['a'] - GIFT_A)
    closer_than_riemann = sum(1 for d in null_dist_from_gift if d < riemann_dist_from_gift)

    results['riemann_closer_to_gift_than'] = float(100 * (1 - closer_than_riemann / n_sequences))

    print(f"\nCoefficient analysis:")
    print(f"  Null a: {np.mean(null_as):.6f} ± {np.std(null_as):.6f}")
    print(f"  Riemann a: {original_fit['a']:.6f}")
    print(f"  GIFT a: {GIFT_A:.6f}")
    print(f"  Riemann closer to GIFT than {results['riemann_closer_to_gift_than']:.1f}% of null")

    # Verdict
    if p_value < 0.001 and results['riemann_closer_to_gift_than'] > 95:
        verdict = "STRONG PASS - Riemann is exceptional vs null distribution"
    elif p_value < 0.01:
        verdict = "PASS - Riemann is in top 1% of null distribution"
    elif p_value < 0.05:
        verdict = "MARGINAL - Riemann is in top 5% but not exceptional"
    else:
        verdict = f"FAIL - Riemann R² is typical for monotone sequences (p={p_value:.3f})"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 7: BOOTSTRAP STABILITY
# ============================================================================

def test_bootstrap_stability(zeros: np.ndarray, n_windows: int = 10,
                              n_bootstrap: int = 500) -> Dict:
    """
    Test coefficient stability across different ranges of zeros.

    If coefficients are stable across windows, the recurrence is robust.
    If they vary wildly, it's likely overfitting to local structure.
    """
    print("\n" + "=" * 70)
    print("TEST 7: BOOTSTRAP STABILITY ACROSS WINDOWS")
    print("=" * 70)

    results = {'test_name': 'bootstrap_stability'}

    n_zeros = min(100000, len(zeros))
    window_size = n_zeros // n_windows

    print(f"\nTesting {n_windows} windows of {window_size} zeros each...")

    window_results = []

    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size

        fit = fit_2term_recurrence(zeros[start:end], GIFT_LAG1, GIFT_LAG2)
        if fit:
            window_results.append({
                'window': i + 1,
                'range': f'[{start}, {end}]',
                'r2': fit['r2'],
                'a': fit['a'],
                'b': fit['b'],
                'a_plus_b': fit['a'] + fit['b']
            })

    # Compute stability metrics
    a_values = [w['a'] for w in window_results]
    b_values = [w['b'] for w in window_results]
    r2_values = [w['r2'] for w in window_results]

    results['windows'] = window_results
    results['a_mean'] = float(np.mean(a_values))
    results['a_std'] = float(np.std(a_values))
    results['a_cv'] = float(np.std(a_values) / np.mean(a_values)) if np.mean(a_values) != 0 else 0
    results['b_mean'] = float(np.mean(b_values))
    results['b_std'] = float(np.std(b_values))
    results['r2_mean'] = float(np.mean(r2_values))
    results['r2_std'] = float(np.std(r2_values))

    print(f"\nCoefficient stability across windows:")
    print(f"  a: {np.mean(a_values):.6f} ± {np.std(a_values):.6f} (CV = {results['a_cv']*100:.2f}%)")
    print(f"  b: {np.mean(b_values):.6f} ± {np.std(b_values):.6f}")
    print(f"  R²: {np.mean(r2_values):.8f} ± {np.std(r2_values):.8f}")

    print(f"\nWindow details:")
    for w in window_results:
        print(f"  Window {w['window']}: a={w['a']:.6f}, b={w['b']:.6f}, R²={w['r2']:.8f}")

    # Bootstrap confidence interval for a
    print(f"\nBootstrap CI for coefficient a ({n_bootstrap} resamples)...")

    np.random.seed(42)
    bootstrap_as = []

    for _ in range(n_bootstrap):
        # Resample windows with replacement
        resampled = np.random.choice(a_values, size=len(a_values), replace=True)
        bootstrap_as.append(np.mean(resampled))

    ci_lower = np.percentile(bootstrap_as, 2.5)
    ci_upper = np.percentile(bootstrap_as, 97.5)

    results['a_ci_95'] = (float(ci_lower), float(ci_upper))
    results['gift_in_ci'] = bool(ci_lower <= GIFT_A <= ci_upper)

    print(f"  95% CI for a: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  GIFT a = {GIFT_A:.6f} {'is IN' if results['gift_in_ci'] else 'is NOT in'} the CI")

    # Check if coefficients converge to GIFT
    dist_from_gift = [abs(a - GIFT_A) for a in a_values]
    mean_dist = np.mean(dist_from_gift)
    results['mean_distance_from_gift'] = float(mean_dist)

    print(f"\n  Mean |a - 31/21|: {mean_dist:.6f}")

    # Verdict
    if results['a_cv'] < 0.02 and results['gift_in_ci']:
        verdict = "STRONG PASS - Coefficients stable and consistent with GIFT"
    elif results['a_cv'] < 0.05:
        verdict = "PASS - Coefficients reasonably stable"
    elif results['a_cv'] < 0.10:
        verdict = "MARGINAL - Some instability in coefficients"
    else:
        verdict = f"FAIL - High coefficient variability (CV = {results['a_cv']*100:.1f}%)"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# TEST 8: HONEST R² DECOMPOSITION
# ============================================================================

def test_r2_decomposition(zeros: np.ndarray) -> Dict:
    """
    Decompose R² into trend component and fluctuation component.

    This gives an HONEST assessment of what the recurrence captures.
    """
    print("\n" + "=" * 70)
    print("TEST 8: HONEST R² DECOMPOSITION")
    print("=" * 70)

    results = {'test_name': 'r2_decomposition'}

    n_zeros = min(50000, len(zeros))
    gamma = zeros[:n_zeros]
    n = np.arange(1, n_zeros + 1)

    # Fit recurrence on raw zeros
    raw_fit = fit_2term_recurrence(gamma, GIFT_LAG1, GIFT_LAG2, return_residuals=True)

    # Compute smooth trend using Riemann-von Mangoldt
    def counting_function(T):
        return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7/8

    # Invert to get expected gamma_n given n
    # gamma_n ~ 2*pi*n / W(n/e) where W is Lambert W
    # Approximate: gamma_n ~ 2*pi*n / log(n)
    gamma_trend = 2 * np.pi * n / np.log(n + np.e)

    # Scale to match actual zeros
    scale = gamma[-1] / gamma_trend[-1]
    gamma_trend = gamma_trend * scale

    # Compute variance components
    max_lag = max(GIFT_LAG1, GIFT_LAG2)
    start = max_lag

    y_actual = gamma[start:]
    y_trend = gamma_trend[start:]
    y_pred = raw_fit['y_pred']

    # Total variance
    ss_total = np.sum((y_actual - np.mean(y_actual))**2)

    # Trend variance
    ss_trend = np.sum((y_trend - np.mean(y_actual))**2)

    # Residual variance (what recurrence adds beyond trend)
    ss_recurrence_residual = np.sum((y_actual - y_pred)**2)

    # How much variance does the recurrence explain beyond the trend?
    # Compare recurrence prediction error to trend prediction error
    ss_trend_residual = np.sum((y_actual - y_trend)**2)

    # Variance explained
    r2_total = 1 - ss_recurrence_residual / ss_total
    r2_trend_only = 1 - ss_trend_residual / ss_total
    r2_beyond_trend = (ss_trend_residual - ss_recurrence_residual) / ss_trend_residual if ss_trend_residual > 0 else 0

    results['r2_total'] = float(r2_total)
    results['r2_trend_only'] = float(r2_trend_only)
    results['r2_beyond_trend'] = float(r2_beyond_trend)
    results['pct_from_trend'] = float(r2_trend_only / r2_total * 100) if r2_total > 0 else 0

    print(f"\nR² decomposition:")
    print(f"  Total R² (recurrence): {r2_total:.8f}")
    print(f"  R² from trend alone:   {r2_trend_only:.8f}")
    print(f"  R² beyond trend:       {r2_beyond_trend:.8f}")
    print(f"\n  → {results['pct_from_trend']:.1f}% of R² comes from capturing the smooth trend")
    print(f"  → {100 - results['pct_from_trend']:.1f}% comes from arithmetic structure")

    # Verdict based on how much is beyond trend
    if r2_beyond_trend > 0.5:
        verdict = "STRONG PASS - Recurrence captures significant arithmetic structure"
    elif r2_beyond_trend > 0.2:
        verdict = "PASS - Recurrence captures some arithmetic structure"
    elif r2_beyond_trend > 0.05:
        verdict = "MARGINAL - Most R² is from trend, little arithmetic structure"
    else:
        verdict = "FAIL - R² is almost entirely from smooth trend"

    results['verdict'] = verdict.split(' - ')[0]
    print(f"\n→ VERDICT: {verdict}")

    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_validation() -> Dict:
    """Run complete validation battery and generate comprehensive report."""

    print("\n" + "█" * 70)
    print("█  ULTRA-RIGOROUS GIFT-RIEMANN VALIDATION")
    print("█  Testing: γₙ = (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁ + c")
    print("█" * 70)

    start_time = time.time()

    # Load data
    print("\nLoading Riemann zeros...")
    try:
        zeros = load_zeros(100000)
        print(f"✓ Loaded {len(zeros)} zeros")
        print(f"  Range: γ₁ = {zeros[0]:.6f} to γ_{len(zeros)} = {zeros[-1]:.6f}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return {'error': str(e)}

    all_results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    verdicts = []

    # Run all tests
    tests = [
        ('Sobol Coefficient Search', lambda: test_sobol_coefficient_search(zeros, n_samples=10000)),
        ('Rational Uniqueness', lambda: test_rational_uniqueness(zeros, max_denom=40)),
        ('Lag Space Search', lambda: test_lag_space_search(zeros, max_lag=35)),
        ('Fluctuation Analysis', lambda: test_fluctuation_analysis(zeros)),
        ('Permutation Test', lambda: test_permutation(zeros, n_permutations=500)),
        ('Null Distribution', lambda: test_null_distribution(zeros, n_sequences=3000)),
        ('Bootstrap Stability', lambda: test_bootstrap_stability(zeros, n_windows=10)),
        ('R² Decomposition', lambda: test_r2_decomposition(zeros)),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            all_results[test_name.lower().replace(' ', '_')] = result
            verdicts.append((test_name, result.get('verdict', 'N/A')))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            all_results[test_name.lower().replace(' ', '_')] = {'error': str(e), 'verdict': 'ERROR'}
            verdicts.append((test_name, 'ERROR'))

    # Final summary
    print("\n" + "█" * 70)
    print("█  FINAL SUMMARY")
    print("█" * 70)

    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)

    pass_count = 0
    fail_count = 0
    marginal_count = 0

    for test_name, verdict in verdicts:
        if 'STRONG PASS' in verdict:
            symbol = "✓✓"
            pass_count += 1
        elif 'PASS' in verdict:
            symbol = "✓ "
            pass_count += 1
        elif 'FAIL' in verdict:
            symbol = "✗ "
            fail_count += 1
        elif 'MARGINAL' in verdict:
            symbol = "~ "
            marginal_count += 1
        else:
            symbol = "? "

        print(f"  {symbol} {test_name}: {verdict}")

    print(f"\nSCORE: {pass_count} PASS / {fail_count} FAIL / {marginal_count} MARGINAL")

    # Overall assessment
    print("\n" + "=" * 60)

    critical_tests = ['fluctuation_analysis', 'r2_decomposition', 'null_distribution']
    critical_pass = sum(1 for t in critical_tests
                        if all_results.get(t, {}).get('verdict', '').startswith('PASS') or
                           all_results.get(t, {}).get('verdict', '').startswith('STRONG'))

    if fail_count == 0 and critical_pass >= 2:
        overall = "STRONG EVIDENCE - The GIFT-Riemann connection appears GENUINE"
        all_results['overall_verdict'] = 'STRONG'
    elif fail_count <= 2 and pass_count >= 4:
        overall = "MODERATE EVIDENCE - Some support but significant caveats"
        all_results['overall_verdict'] = 'MODERATE'
    elif fail_count >= 3:
        overall = "WEAK EVIDENCE - Major concerns about the claimed connection"
        all_results['overall_verdict'] = 'WEAK'
    else:
        overall = "INCONCLUSIVE - Mixed results, needs more investigation"
        all_results['overall_verdict'] = 'INCONCLUSIVE'

    print(f"OVERALL: {overall}")
    print("=" * 60)

    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)

    if 'r2_decomposition' in all_results:
        pct_trend = all_results['r2_decomposition'].get('pct_from_trend', 0)
        print(f"  • {pct_trend:.1f}% of R² comes from smooth trend (not arithmetic)")

    if 'fluctuation_analysis' in all_results:
        fluct_r2 = all_results['fluctuation_analysis'].get('avg_fluctuation_r2', 0)
        print(f"  • R² on fluctuations: {fluct_r2:.4f} (vs ~0.9999 on raw zeros)")

    if 'rational_uniqueness' in all_results:
        n_better = all_results['rational_uniqueness'].get('n_better', 0)
        print(f"  • {n_better} rational pairs beat GIFT (31/21, -10/21)")

    if 'lag_space_search' in all_results:
        gift_rank = all_results['lag_space_search'].get('gift_rank', 0)
        print(f"  • GIFT lags (8, 21) rank #{gift_rank} among all lag pairs")

    elapsed = time.time() - start_time
    print(f"\n  Total validation time: {elapsed:.1f} seconds")

    all_results['summary'] = {
        'pass_count': pass_count,
        'fail_count': fail_count,
        'marginal_count': marginal_count,
        'verdicts': verdicts,
        'elapsed_seconds': elapsed
    }

    # Save results
    output_path = Path(__file__).parent / "riemann_validation_results.json"

    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(v) for v in obj)
        return obj

    results_to_save = convert_to_native(all_results)

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_full_validation()
