#!/usr/bin/env python3
"""
Statistical Validation: Is k=6 Uniquely Locked?
================================================

This script performs rigorous statistical tests to verify:
1. Does k=6 (h_G₂) give the best fit among all k values?
2. Is the exact 31/21 coefficient locked by the data?
3. Does the recurrence work on fluctuations (unfolded) or just trend?
4. Bootstrap confidence intervals for precision

Key tests from GPT's critique:
- "k=5,6,7 donnent tous ~1.47, il faut verrouiller spécifiquement 31/21"
- Test on unfolded zeros (x_n = u_n - n) to check structural vs trend
"""

import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize_scalar
import json

# =============================================================================
# FIBONACCI AND FORMULAS
# =============================================================================

def fib(n):
    """Return the n-th Fibonacci number (F_0=0, F_1=1)."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def get_coefficients_for_k(k):
    """
    Return (a, b, lag1, lag2) for a given k using the derived formula.

    a = (F_{k+3} - F_{k-2}) / F_{k+2}
    b = -(F_{k+1} - F_{k-2}) / F_{k+2}
    lag1 = F_k
    lag2 = F_{k+2}
    """
    if k < 2:
        return None

    F_k = fib(k)
    F_k_plus_1 = fib(k + 1)
    F_k_plus_2 = fib(k + 2)
    F_k_plus_3 = fib(k + 3)
    F_k_minus_2 = fib(k - 2)

    a = (F_k_plus_3 - F_k_minus_2) / F_k_plus_2
    b = -(F_k_plus_1 - F_k_minus_2) / F_k_plus_2

    return a, b, F_k, F_k_plus_2

# =============================================================================
# LOAD ZEROS
# =============================================================================

def load_zeros(max_zeros=100000):
    """Load Riemann zeros from data files."""
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 11):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        zeros.append(float(line))
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

# =============================================================================
# FIT AND EVALUATE RECURRENCE
# =============================================================================

def fit_recurrence_free(zeros, lag1, lag2, n_samples=None):
    """
    Fit recurrence with FREE coefficients.
    Returns (a_fit, b_fit, c_fit, R², residuals, predictions).
    """
    max_lag = max(lag1, lag2)
    if n_samples is None:
        n_samples = len(zeros) - max_lag
    else:
        n_samples = min(n_samples, len(zeros) - max_lag)

    # Build design matrix
    X1 = zeros[max_lag - lag1:max_lag - lag1 + n_samples]
    X2 = zeros[max_lag - lag2:max_lag - lag2 + n_samples]
    ones = np.ones(n_samples)
    X = np.column_stack([X1, X2, ones])
    y = zeros[max_lag:max_lag + n_samples]

    # Least squares fit
    coeffs, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a_fit, b_fit, c_fit = coeffs

    # Predictions and R²
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    residuals = y - y_pred

    return a_fit, b_fit, c_fit, r_squared, residuals, y_pred

def evaluate_fixed_coefficients(zeros, a, b, lag1, lag2, n_samples=None):
    """
    Evaluate recurrence with FIXED coefficients a, b.
    Fit only the constant c.
    Returns (c_fit, R², residuals, MAE).
    """
    max_lag = max(lag1, lag2)
    if n_samples is None:
        n_samples = len(zeros) - max_lag
    else:
        n_samples = min(n_samples, len(zeros) - max_lag)

    X1 = zeros[max_lag - lag1:max_lag - lag1 + n_samples]
    X2 = zeros[max_lag - lag2:max_lag - lag2 + n_samples]
    y = zeros[max_lag:max_lag + n_samples]

    # With fixed a, b: y = a*X1 + b*X2 + c
    # Solve for c: c = mean(y - a*X1 - b*X2)
    prediction_no_c = a * X1 + b * X2
    c_fit = np.mean(y - prediction_no_c)

    y_pred = prediction_no_c + c_fit
    residuals = y - y_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(residuals))

    return c_fit, r_squared, residuals, mae

# =============================================================================
# UNFOLDED ZEROS
# =============================================================================

def unfold_zeros(zeros):
    """
    Unfold zeros to remove the smooth trend.
    The unfolded sequence u_n = N(γ_n) where N(T) ~ (T/2π)log(T/2πe)
    Then x_n = u_n - n measures pure fluctuations.
    """
    # Smooth counting function N(T) ≈ (T/2π)log(T/2πe) + 7/8
    def N_smooth(T):
        if T <= 0:
            return 0
        return (T / (2 * np.pi)) * np.log(T / (2 * np.pi * np.e)) + 7/8

    # Unfolded: u_n = N(γ_n)
    u = np.array([N_smooth(g) for g in zeros])

    # Fluctuations: x_n = u_n - n
    n_indices = np.arange(1, len(zeros) + 1)
    x = u - n_indices

    return x, u

# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_coefficients(zeros, lag1, lag2, n_bootstrap=1000, block_size=100):
    """
    Bootstrap confidence intervals for fitted coefficients.
    Uses block bootstrap to preserve local structure.
    """
    max_lag = max(lag1, lag2)
    n = len(zeros) - max_lag

    # Full fit first
    a_full, b_full, c_full, _, _, _ = fit_recurrence_free(zeros, lag1, lag2)

    # Bootstrap
    a_samples = []
    b_samples = []

    n_blocks = n // block_size

    for _ in range(n_bootstrap):
        # Block bootstrap: sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        # Build bootstrap sample
        boot_indices = []
        for bi in block_indices:
            start = bi * block_size + max_lag
            end = min(start + block_size, len(zeros))
            boot_indices.extend(range(start, end))

        boot_indices = np.array(boot_indices[:n])  # Trim to original size

        # Fit on bootstrap sample
        X1 = zeros[boot_indices - lag1]
        X2 = zeros[boot_indices - lag2]
        y = zeros[boot_indices]
        X = np.column_stack([X1, X2, np.ones(len(boot_indices))])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            a_samples.append(coeffs[0])
            b_samples.append(coeffs[1])
        except:
            continue

    a_samples = np.array(a_samples)
    b_samples = np.array(b_samples)

    return {
        'a_mean': np.mean(a_samples),
        'a_std': np.std(a_samples),
        'a_ci_95': (np.percentile(a_samples, 2.5), np.percentile(a_samples, 97.5)),
        'b_mean': np.mean(b_samples),
        'b_std': np.std(b_samples),
        'b_ci_95': (np.percentile(b_samples, 2.5), np.percentile(b_samples, 97.5)),
    }

# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():
    print("=" * 70)
    print("STATISTICAL VALIDATION: IS k=6 UNIQUELY LOCKED?")
    print("=" * 70)

    # Load zeros
    zeros = load_zeros(50000)
    print(f"\n✓ Loaded {len(zeros)} Riemann zeros")

    # ==========================================================================
    # TEST 1: Compare k values (4, 5, 6, 7, 8)
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 1: COMPARING k VALUES (Fibonacci formula)")
    print("-" * 70)

    k_values = [4, 5, 6, 7, 8]
    k_results = []

    print(f"\n{'k':<4} {'h_G':<6} {'Lag1':<6} {'Lag2':<6} {'a':<12} {'b':<12} {'a+b':<8} {'R² (exact)':<12} {'R² (free)':<12}")
    print("-" * 90)

    for k in k_values:
        result = get_coefficients_for_k(k)
        if result is None:
            continue

        a, b, lag1, lag2 = result

        # Skip if lag2 > available data
        if lag2 > len(zeros) - 1000:
            continue

        # Evaluate with exact coefficients
        c_exact, r2_exact, res_exact, mae_exact = evaluate_fixed_coefficients(
            zeros, a, b, lag1, lag2
        )

        # Fit with free coefficients
        a_free, b_free, c_free, r2_free, res_free, _ = fit_recurrence_free(
            zeros, lag1, lag2
        )

        # Group name
        group = "G₂" if k == 6 else f"k={k}"

        print(f"{k:<4} {group:<6} {lag1:<6} {lag2:<6} {a:<12.6f} {b:<12.6f} {a+b:<8.4f} {r2_exact*100:<12.6f} {r2_free*100:<12.6f}")

        k_results.append({
            'k': k,
            'lag1': lag1,
            'lag2': lag2,
            'a_theory': a,
            'b_theory': b,
            'a_free': a_free,
            'b_free': b_free,
            'r2_exact': r2_exact,
            'r2_free': r2_free,
            'mae_exact': mae_exact
        })

    # Find best k by R²
    best_k_exact = max(k_results, key=lambda x: x['r2_exact'])
    best_k_free = max(k_results, key=lambda x: x['r2_free'])

    print(f"\n→ Best k by R² (exact): k = {best_k_exact['k']} (R² = {best_k_exact['r2_exact']*100:.6f}%)")
    print(f"→ Best k by R² (free):  k = {best_k_free['k']} (R² = {best_k_free['r2_free']*100:.6f}%)")

    # ==========================================================================
    # TEST 2: Is 31/21 specifically locked vs nearby rationals?
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 2: IS 31/21 SPECIFICALLY LOCKED?")
    print("-" * 70)

    # Candidates from GPT
    candidates = [
        ('31/21 (k=6)', 31/21, -10/21),
        ('3/2 (simple)', 3/2, -1/2),
        ('φ - φ⁻⁴ (limit)', 1.4721, -0.4721),
        ('19/13 (k=5)', 19/13, -6/13),
        ('50/34 (k=7)', 50/34, -16/34),
        ('1.5014 (old fit)', 1.5014, -0.5014),
    ]

    lag1, lag2 = 8, 21  # Fixed lags for comparison

    print(f"\nFixed lags: ({lag1}, {lag2})")
    print(f"\n{'Candidate':<20} {'a':<12} {'b':<12} {'R²':<14} {'MAE':<12}")
    print("-" * 70)

    for name, a, b in candidates:
        c, r2, res, mae = evaluate_fixed_coefficients(zeros, a, b, lag1, lag2)
        print(f"{name:<20} {a:<12.6f} {b:<12.6f} {r2*100:<14.8f} {mae:<12.6f}")

    # Also fit freely and compare
    a_free, b_free, c_free, r2_free, _, _ = fit_recurrence_free(zeros, lag1, lag2)
    print(f"{'FREE FIT':<20} {a_free:<12.6f} {b_free:<12.6f} {r2_free*100:<14.8f} {'—':<12}")

    # ==========================================================================
    # TEST 3: Bootstrap confidence intervals
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 3: BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 70)

    print("\nRunning bootstrap (1000 samples, block size 100)...")
    boot_results = bootstrap_coefficients(zeros, lag1, lag2, n_bootstrap=1000)

    print(f"\nCoefficient a:")
    print(f"  Mean: {boot_results['a_mean']:.6f}")
    print(f"  Std:  {boot_results['a_std']:.6f}")
    print(f"  95% CI: [{boot_results['a_ci_95'][0]:.6f}, {boot_results['a_ci_95'][1]:.6f}]")
    print(f"  Theory (31/21): {31/21:.6f}")
    print(f"  → In CI: {'✓ YES' if boot_results['a_ci_95'][0] <= 31/21 <= boot_results['a_ci_95'][1] else '✗ NO'}")

    print(f"\nCoefficient b:")
    print(f"  Mean: {boot_results['b_mean']:.6f}")
    print(f"  Std:  {boot_results['b_std']:.6f}")
    print(f"  95% CI: [{boot_results['b_ci_95'][0]:.6f}, {boot_results['b_ci_95'][1]:.6f}]")
    print(f"  Theory (-10/21): {-10/21:.6f}")
    print(f"  → In CI: {'✓ YES' if boot_results['b_ci_95'][0] <= -10/21 <= boot_results['b_ci_95'][1] else '✗ NO'}")

    # ==========================================================================
    # TEST 4: Unfolded zeros (fluctuations only)
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 4: UNFOLDED ZEROS (FLUCTUATIONS ONLY)")
    print("-" * 70)

    x_unfolded, u = unfold_zeros(zeros)

    print(f"\nUnfolded statistics:")
    print(f"  Mean fluctuation: {np.mean(x_unfolded):.4f}")
    print(f"  Std fluctuation:  {np.std(x_unfolded):.4f}")
    print(f"  Range: [{np.min(x_unfolded):.2f}, {np.max(x_unfolded):.2f}]")

    # Fit recurrence on unfolded
    # Note: we need to be careful about what this means
    # The recurrence on unfolded would be: x_n ≈ a*x_{n-8} + b*x_{n-21} + c

    a_unf, b_unf, c_unf, r2_unf, res_unf, _ = fit_recurrence_free(x_unfolded, lag1, lag2)

    print(f"\nRecurrence on UNFOLDED zeros (fluctuations):")
    print(f"  x_n ≈ {a_unf:.4f} * x_{{n-8}} + {b_unf:.4f} * x_{{n-21}} + {c_unf:.4f}")
    print(f"  R² = {r2_unf*100:.4f}%")
    print(f"  a + b = {a_unf + b_unf:.4f}")

    # Compare to raw zeros
    print(f"\nComparison:")
    print(f"  Raw zeros R²:      {r2_free*100:.6f}%")
    print(f"  Unfolded zeros R²: {r2_unf*100:.6f}%")

    if r2_unf > 0.5:
        print(f"\n  → Recurrence captures STRUCTURE (not just trend)")
    elif r2_unf > 0.1:
        print(f"\n  → Recurrence captures some structure beyond trend")
    else:
        print(f"\n  → Recurrence mainly captures trend, little structure")

    # ==========================================================================
    # TEST 5: Residual analysis
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 5: RESIDUAL ANALYSIS (k=6 exact)")
    print("-" * 70)

    c, r2, residuals, mae = evaluate_fixed_coefficients(zeros, 31/21, -10/21, 8, 21)

    print(f"\nResiduals for exact 31/21, -10/21:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std:  {np.std(residuals):.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Max:  {np.max(np.abs(residuals)):.6f}")

    # Test for normality
    _, p_normal = stats.normaltest(residuals[:5000])  # Use subset for speed
    print(f"\n  Normality test (D'Agostino-Pearson): p = {p_normal:.4f}")
    print(f"  → {'Residuals appear normal' if p_normal > 0.05 else 'Residuals non-normal (expected for structured data)'}")

    # Autocorrelation at key lags
    print(f"\n  Autocorrelation of residuals:")
    for lag in [1, 5, 8, 13, 21]:
        acf = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
        print(f"    Lag {lag:2d}: {acf:+.4f}")

    # ==========================================================================
    # TEST 6: Uniqueness - is k=6 statistically best?
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 6: UNIQUENESS - IS k=6 STATISTICALLY BEST?")
    print("-" * 70)

    # Compare AIC/BIC for different k models
    n_data = len(zeros) - 21  # Common sample size

    print(f"\nModel comparison using AIC (lower is better):")
    print(f"\n{'k':<4} {'Lag1':<6} {'Lag2':<6} {'R²':<12} {'RSS':<14} {'AIC':<14}")
    print("-" * 60)

    for res in k_results:
        k = res['k']
        lag1, lag2 = res['lag1'], res['lag2']

        # Recompute with common sample size where possible
        if lag2 <= 21:
            _, r2, residuals, _ = evaluate_fixed_coefficients(
                zeros, res['a_theory'], res['b_theory'], lag1, lag2, n_samples=n_data
            )
            rss = np.sum(residuals**2)
            # AIC = n*log(RSS/n) + 2k (k=3 parameters: a, b, c)
            aic = n_data * np.log(rss / n_data) + 2 * 3
            print(f"{k:<4} {lag1:<6} {lag2:<6} {r2*100:<12.6f} {rss:<14.2f} {aic:<14.2f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY: STATISTICAL VALIDATION RESULTS")
    print("=" * 70)

    # Determine if 31/21 is in bootstrap CI
    a_in_ci = boot_results['a_ci_95'][0] <= 31/21 <= boot_results['a_ci_95'][1]
    b_in_ci = boot_results['b_ci_95'][0] <= -10/21 <= boot_results['b_ci_95'][1]

    print(f"""
TEST RESULTS:

1. k COMPARISON:
   - Best k by R² (exact): k = {best_k_exact['k']}
   - k=6 gives R² = {[r['r2_exact'] for r in k_results if r['k']==6][0]*100:.6f}%
   - Verdict: {'✓ k=6 is optimal' if best_k_exact['k'] == 6 else '⚠ k=' + str(best_k_exact['k']) + ' is better'}

2. COEFFICIENT LOCKING:
   - Theoretical 31/21 = {31/21:.6f}
   - Bootstrap mean:     {boot_results['a_mean']:.6f}
   - 95% CI: [{boot_results['a_ci_95'][0]:.6f}, {boot_results['a_ci_95'][1]:.6f}]
   - 31/21 in CI: {'✓ YES' if a_in_ci else '✗ NO'}

3. UNFOLDED TEST:
   - R² on fluctuations: {r2_unf*100:.4f}%
   - Verdict: {'✓ Structural' if r2_unf > 0.3 else '⚠ Mainly trend' if r2_unf > 0.1 else '✗ Trend only'}

4. OVERALL VERDICT:
""")

    # Final verdict
    score = 0
    if best_k_exact['k'] == 6:
        score += 1
    if a_in_ci and b_in_ci:
        score += 1
    if r2_unf > 0.1:
        score += 1

    if score == 3:
        print("   ✅ STRONG VALIDATION: k=6 uniquely locked, coefficients exact, structural")
    elif score == 2:
        print("   ⚠️ PARTIAL VALIDATION: Most tests pass but some concerns remain")
    else:
        print("   ❌ WEAK VALIDATION: Results do not strongly support k=6 uniqueness")

    # Save results
    results = {
        'k_comparison': k_results,
        'bootstrap': {
            'a_mean': float(boot_results['a_mean']),
            'a_std': float(boot_results['a_std']),
            'a_ci_95': [float(x) for x in boot_results['a_ci_95']],
            'b_mean': float(boot_results['b_mean']),
            'b_std': float(boot_results['b_std']),
            'b_ci_95': [float(x) for x in boot_results['b_ci_95']],
            'a_31_21_in_ci': bool(a_in_ci),
            'b_10_21_in_ci': bool(b_in_ci)
        },
        'unfolded': {
            'r2': float(r2_unf),
            'a': float(a_unf),
            'b': float(b_unf)
        },
        'verdict': {
            'k6_optimal': best_k_exact['k'] == 6,
            'coefficients_locked': a_in_ci and b_in_ci,
            'structural': r2_unf > 0.1,
            'score': score
        }
    }

    with open(Path(__file__).parent / "validation_k6_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n✓ Results saved to validation_k6_results.json")

if __name__ == "__main__":
    main()
