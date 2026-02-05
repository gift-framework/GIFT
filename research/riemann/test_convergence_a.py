#!/usr/bin/env python3
"""
Test A: Does a*(n) converge to 3/2 or 31/21?
=============================================

The critical test suggested by the AI council:
- Fit coefficient 'a' on sliding windows at different heights n
- Track a*(n) as n → ∞
- Check if it converges to 3/2 = 1.5 or 31/21 ≈ 1.476

Key insight from Claude:
  31/21 - 3/2 = 1/42 = 1/(2×b₂)

If a*(n) → 3/2, then 31/21 is a finite-size correction.
If a*(n) → 31/21, then the Fibonacci formula is exact.
"""

import numpy as np
from pathlib import Path
import json

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
# FIT COEFFICIENT ON WINDOW
# =============================================================================

def fit_coefficient_on_window(zeros, start_idx, window_size, lag1=8, lag2=21):
    """
    Fit recurrence on a window [start_idx, start_idx + window_size].
    Return (a_fit, b_fit, a_std, b_std) with bootstrap uncertainty.
    """
    end_idx = min(start_idx + window_size, len(zeros))
    max_lag = max(lag1, lag2)

    if start_idx < max_lag:
        start_idx = max_lag

    n = end_idx - start_idx
    if n < 100:
        return None, None, None, None

    # Build design matrix
    X1 = zeros[start_idx - lag1:end_idx - lag1]
    X2 = zeros[start_idx - lag2:end_idx - lag2]
    y = zeros[start_idx:end_idx]
    X = np.column_stack([X1, X2, np.ones(n)])

    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a_fit, b_fit = coeffs[0], coeffs[1]

    # Bootstrap for uncertainty
    n_bootstrap = 200
    a_samples = []
    b_samples = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        try:
            coeffs_boot, _, _, _ = np.linalg.lstsq(X_boot, y_boot, rcond=None)
            a_samples.append(coeffs_boot[0])
            b_samples.append(coeffs_boot[1])
        except:
            continue

    a_std = np.std(a_samples) if a_samples else 0
    b_std = np.std(b_samples) if b_samples else 0

    return a_fit, b_fit, a_std, b_std

# =============================================================================
# MAIN CONVERGENCE TEST
# =============================================================================

def main():
    print("=" * 70)
    print("TEST A: COEFFICIENT CONVERGENCE a*(n) → 3/2 or 31/21?")
    print("=" * 70)

    # Load zeros
    zeros = load_zeros(100000)
    print(f"\n✓ Loaded {len(zeros)} Riemann zeros")

    # Key values to compare
    VAL_3_2 = 3/2           # 1.5 exactly
    VAL_31_21 = 31/21       # 1.476190...
    VAL_PHI = (1 + np.sqrt(5))/2  # 1.618...

    print(f"\nReference values:")
    print(f"  3/2     = {VAL_3_2:.6f}")
    print(f"  31/21   = {VAL_31_21:.6f}")
    print(f"  φ       = {VAL_PHI:.6f}")
    print(f"  31/21 - 3/2 = {VAL_31_21 - VAL_3_2:.6f} = 1/42 = {1/42:.6f}")

    # ==========================================================================
    # Test 1: Sliding windows at increasing heights
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 1: SLIDING WINDOWS AT INCREASING HEIGHTS")
    print("-" * 70)

    window_size = 5000

    # Windows starting at different heights
    window_starts = [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000,
                     60000, 70000, 80000, 90000]
    window_starts = [w for w in window_starts if w + window_size <= len(zeros)]

    print(f"\nWindow size: {window_size}")
    print(f"\n{'Start n':<10} {'a*':<12} {'±σ':<10} {'|a*-3/2|':<12} {'|a*-31/21|':<12} {'Closer to':<12}")
    print("-" * 70)

    results = []

    for start in window_starts:
        a_fit, b_fit, a_std, b_std = fit_coefficient_on_window(zeros, start, window_size)
        if a_fit is None:
            continue

        dist_3_2 = abs(a_fit - VAL_3_2)
        dist_31_21 = abs(a_fit - VAL_31_21)
        closer = "3/2" if dist_3_2 < dist_31_21 else "31/21"

        print(f"{start:<10} {a_fit:<12.6f} {a_std:<10.6f} {dist_3_2:<12.6f} {dist_31_21:<12.6f} {closer:<12}")

        results.append({
            'start_n': start,
            'center_n': start + window_size // 2,
            'a_fit': a_fit,
            'a_std': a_std,
            'b_fit': b_fit,
            'dist_3_2': dist_3_2,
            'dist_31_21': dist_31_21,
            'closer_to': closer
        })

    # ==========================================================================
    # Test 2: Cumulative fit (all zeros up to n)
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 2: CUMULATIVE FIT (all zeros up to n)")
    print("-" * 70)

    cumulative_ns = [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 70000, 90000]
    cumulative_ns = [n for n in cumulative_ns if n <= len(zeros)]

    print(f"\n{'n':<10} {'a*':<12} {'±σ':<10} {'|a*-3/2|':<12} {'|a*-31/21|':<12} {'Closer to':<12}")
    print("-" * 70)

    cumulative_results = []

    for n in cumulative_ns:
        a_fit, b_fit, a_std, b_std = fit_coefficient_on_window(zeros, 21, n - 21)
        if a_fit is None:
            continue

        dist_3_2 = abs(a_fit - VAL_3_2)
        dist_31_21 = abs(a_fit - VAL_31_21)
        closer = "3/2" if dist_3_2 < dist_31_21 else "31/21"

        print(f"{n:<10} {a_fit:<12.6f} {a_std:<10.6f} {dist_3_2:<12.6f} {dist_31_21:<12.6f} {closer:<12}")

        cumulative_results.append({
            'n': n,
            'a_fit': a_fit,
            'a_std': a_std,
            'dist_3_2': dist_3_2,
            'dist_31_21': dist_31_21,
            'closer_to': closer
        })

    # ==========================================================================
    # Analysis: Which value does it converge to?
    # ==========================================================================

    print("\n" + "-" * 70)
    print("CONVERGENCE ANALYSIS")
    print("-" * 70)

    # Use cumulative results for convergence
    if len(cumulative_results) >= 3:
        # Look at the last few values
        last_3 = cumulative_results[-3:]
        avg_a = np.mean([r['a_fit'] for r in last_3])
        avg_dist_3_2 = np.mean([r['dist_3_2'] for r in last_3])
        avg_dist_31_21 = np.mean([r['dist_31_21'] for r in last_3])

        print(f"\nAverage a* over last 3 cumulative fits (n ≥ 30k): {avg_a:.6f}")
        print(f"Average distance to 3/2:   {avg_dist_3_2:.6f}")
        print(f"Average distance to 31/21: {avg_dist_31_21:.6f}")

        # Trend analysis
        a_values = [r['a_fit'] for r in cumulative_results]
        n_values = [r['n'] for r in cumulative_results]

        # Linear fit to see trend
        slope, intercept = np.polyfit(np.log(n_values), a_values, 1)

        print(f"\nTrend: a*(n) ≈ {intercept:.6f} + {slope:.6f} × log(n)")

        # Extrapolate to n = 10^6, 10^9
        a_extrapolated_1M = intercept + slope * np.log(1e6)
        a_extrapolated_1B = intercept + slope * np.log(1e9)

        print(f"\nExtrapolated values:")
        print(f"  a*(10⁶) ≈ {a_extrapolated_1M:.6f}")
        print(f"  a*(10⁹) ≈ {a_extrapolated_1B:.6f}")

    # ==========================================================================
    # Test 3: Compare φ-approximants
    # ==========================================================================

    print("\n" + "-" * 70)
    print("TEST 3: φ-APPROXIMANT COMPARISON")
    print("-" * 70)

    # Different Fibonacci-related ratios
    phi_approximants = [
        ('3/2', 3/2),
        ('31/21', 31/21),
        ('5/3', 5/3),
        ('8/5', 8/5),
        ('13/8', 13/8),
        ('21/13', 21/13),
        ('φ', VAL_PHI),
        ('φ-φ⁻⁴', VAL_PHI - VAL_PHI**(-4)),
    ]

    # Evaluate each on full dataset
    print(f"\n{'Name':<12} {'Value':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 50)

    max_lag = 21
    n = len(zeros) - max_lag
    X1 = zeros[max_lag - 8:max_lag - 8 + n]
    X2 = zeros[max_lag - 21:max_lag - 21 + n]
    y = zeros[max_lag:max_lag + n]

    phi_results = []

    for name, a in phi_approximants:
        b = 1 - a  # Enforce a + b = 1
        pred_no_c = a * X1 + b * X2
        c = np.mean(y - pred_no_c)
        pred = pred_no_c + c
        errors = y - pred
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        print(f"{name:<12} {a:<12.6f} {mae:<12.6f} {rmse:<12.6f}")

        phi_results.append({
            'name': name,
            'a': float(a),
            'mae': float(mae),
            'rmse': float(rmse)
        })

    # Find best
    best_mae = min(phi_results, key=lambda x: x['mae'])
    print(f"\n→ Best by MAE: {best_mae['name']} (a = {best_mae['a']:.6f})")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count how many windows/cumulative fits are closer to each
    closer_3_2 = sum(1 for r in results if r['closer_to'] == '3/2')
    closer_31_21 = sum(1 for r in results if r['closer_to'] == '31/21')

    print(f"""
CONVERGENCE RESULTS:

1. SLIDING WINDOWS:
   - Closer to 3/2:   {closer_3_2} / {len(results)}
   - Closer to 31/21: {closer_31_21} / {len(results)}

2. CUMULATIVE FIT:
   - Final value (n={cumulative_results[-1]['n']}): a* = {cumulative_results[-1]['a_fit']:.6f}
   - Distance to 3/2:   {cumulative_results[-1]['dist_3_2']:.6f}
   - Distance to 31/21: {cumulative_results[-1]['dist_31_21']:.6f}

3. TREND:
   - Slope: {slope:+.6f} (per log(n))
   - Extrapolated a*(10⁶): {a_extrapolated_1M:.6f}
   - Extrapolated a*(10⁹): {a_extrapolated_1B:.6f}

4. φ-APPROXIMANTS:
   - Best performer: {best_mae['name']} (MAE = {best_mae['mae']:.6f})
""")

    # Verdict
    final_a = cumulative_results[-1]['a_fit']
    if abs(final_a - VAL_3_2) < abs(final_a - VAL_31_21):
        verdict = "3/2"
        explanation = "The data converges toward 3/2 = b₂/dim(G₂)"
    else:
        verdict = "31/21"
        explanation = "The data supports the exact Fibonacci formula"

    # But also check the trend
    if slope < -0.01:
        trend_direction = "DECREASING toward 31/21"
    elif slope > 0.01:
        trend_direction = "INCREASING toward 3/2"
    else:
        trend_direction = "STABLE (no clear trend)"

    print(f"VERDICT:")
    print(f"  Current best match: {verdict}")
    print(f"  Trend: {trend_direction}")
    print(f"  Interpretation: {explanation}")

    # Claude's insight
    print(f"""
CLAUDE'S INSIGHT:
  31/21 - 3/2 = {VAL_31_21 - VAL_3_2:.6f} ≈ 1/42 = {1/42:.6f}

  This suggests:
  - 3/2 is the asymptotic limit (N → ∞)
  - 31/21 includes a finite-size correction of 1/(2×b₂)
  - Both are "correct" at different scales
""")

    # Save results
    output = {
        'sliding_windows': results,
        'cumulative': cumulative_results,
        'trend': {
            'slope': float(slope),
            'intercept': float(intercept),
            'extrapolated_1M': float(a_extrapolated_1M),
            'extrapolated_1B': float(a_extrapolated_1B)
        },
        'phi_approximants': phi_results,
        'verdict': verdict
    }

    with open(Path(__file__).parent / "convergence_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✓ Results saved to convergence_test_results.json")

if __name__ == "__main__":
    main()
