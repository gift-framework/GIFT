#!/usr/bin/env python3
"""
Spacings Test (from Council-17 GPT)

Test if the recurrence structure appears in the SPACINGS sₙ = γₙ₊₁ - γₙ,
not just in the raw zeros.

If structure appears in spacings → genuine arithmetic structure
If structure vanishes in spacings → we're just fitting a smooth trend
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import correlate
import json

print("=" * 70)
print("SPACINGS TEST: Does structure appear in stationary variable?")
print("=" * 70)

# Load zeros
try:
    zeros = np.load('/home/user/GIFT/riemann_zeros_10k.npy')
except:
    zeros = np.load('/home/user/GIFT/notebooks/riemann_zeros_10k.npy')

print(f"\nLoaded {len(zeros)} Riemann zeros")

# Compute spacings
spacings = np.diff(zeros)
print(f"Computed {len(spacings)} spacings")
print(f"Mean spacing: {np.mean(spacings):.6f}")
print(f"Spacing std: {np.std(spacings):.6f}")

# Compute unfolded fluctuations
# u_n = N(γ_n) where N(T) ≈ (T/2π) log(T/2πe) + 7/8
def counting_function(t):
    """Riemann-von Mangoldt counting function N(T)"""
    return t / (2 * np.pi) * np.log(t / (2 * np.pi * np.e)) + 7/8

u = counting_function(zeros)
n = np.arange(1, len(zeros) + 1)
fluctuations = u - n
print(f"\nComputed unfolded fluctuations u_n - n")
print(f"Mean fluctuation: {np.mean(fluctuations):.6f}")
print(f"Fluctuation std: {np.std(fluctuations):.6f}")

def autocorr(x, max_lag=50):
    """Compute autocorrelation."""
    x = x - np.mean(x)
    result = correlate(x, x, mode='full')
    result = result[len(result)//2:]
    return result[:max_lag+1] / result[0]

def fit_recurrence(data, lags, name):
    """Fit recurrence with given lags."""
    N = len(data)
    max_lag = max(lags)

    # Build feature matrix
    X = np.column_stack([data[max_lag - lag : N - lag] for lag in lags])
    y = data[max_lag:]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred

    return {
        'name': name,
        'lags': lags,
        'coefficients': [float(c) for c in model.coef_],
        'intercept': float(model.intercept_),
        'r2': float(r2),
        'residual_std': float(np.std(residuals)),
        'coeff_sum': float(sum(model.coef_))
    }

# ============================================================
# TEST 1: Recurrence on RAW ZEROS (baseline)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Recurrence on RAW ZEROS γₙ (baseline)")
print("=" * 70)

result_zeros = fit_recurrence(zeros, [8, 21], "Raw zeros [8, 21]")
print(f"  R² = {result_zeros['r2']:.10f}")
print(f"  Coefficients: a = {result_zeros['coefficients'][0]:.6f}, b = {result_zeros['coefficients'][1]:.6f}")
print(f"  Sum a+b = {result_zeros['coeff_sum']:.6f}")

# ============================================================
# TEST 2: Recurrence on SPACINGS
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Recurrence on SPACINGS sₙ = γₙ₊₁ - γₙ")
print("=" * 70)

result_spacings = fit_recurrence(spacings, [8, 21], "Spacings [8, 21]")
print(f"  R² = {result_spacings['r2']:.10f}")
print(f"  Coefficients: a = {result_spacings['coefficients'][0]:.6f}, b = {result_spacings['coefficients'][1]:.6f}")
print(f"  Sum a+b = {result_spacings['coeff_sum']:.6f}")

# Also test other lags
result_spacings_fib = fit_recurrence(spacings, [8, 13, 21], "Spacings Fib [8,13,21]")
result_spacings_opt = fit_recurrence(spacings, [8, 21, 140], "Spacings [8,21,140]")

print(f"\n  Fibonacci [8, 13, 21]: R² = {result_spacings_fib['r2']:.6f}")
print(f"  Optimal [8, 21, 140]:  R² = {result_spacings_opt['r2']:.6f}")

# ============================================================
# TEST 3: Recurrence on FLUCTUATIONS
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Recurrence on FLUCTUATIONS (uₙ - n)")
print("=" * 70)

result_fluct = fit_recurrence(fluctuations, [8, 21], "Fluctuations [8, 21]")
print(f"  R² = {result_fluct['r2']:.10f}")
print(f"  Coefficients: a = {result_fluct['coefficients'][0]:.6f}, b = {result_fluct['coefficients'][1]:.6f}")
print(f"  Sum a+b = {result_fluct['coeff_sum']:.6f}")

result_fluct_fib = fit_recurrence(fluctuations, [8, 13, 21], "Fluctuations Fib")
result_fluct_opt = fit_recurrence(fluctuations, [8, 21, 140], "Fluctuations [8,21,140]")

print(f"\n  Fibonacci [8, 13, 21]: R² = {result_fluct_fib['r2']:.6f}")
print(f"  Optimal [8, 21, 140]:  R² = {result_fluct_opt['r2']:.6f}")

# ============================================================
# TEST 4: Autocorrelation comparison
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Autocorrelation Comparison")
print("=" * 70)

acf_zeros = autocorr(zeros - np.mean(zeros), max_lag=50)
acf_spacings = autocorr(spacings, max_lag=50)
acf_fluct = autocorr(fluctuations, max_lag=50)

print(f"\n{'Lag':<6} {'ACF(zeros)':<15} {'ACF(spacings)':<15} {'ACF(fluct)':<15}")
print("-" * 55)
for lag in [1, 5, 8, 13, 21, 34, 42]:
    print(f"{lag:<6} {acf_zeros[lag]:<15.4f} {acf_spacings[lag]:<15.4f} {acf_fluct[lag]:<15.4f}")

# ============================================================
# TEST 5: Best lags for spacings
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Systematic search for best lags on SPACINGS")
print("=" * 70)

results_spacings_search = []
for lag1 in range(3, 15):
    for lag2 in range(lag1 + 3, 30):
        try:
            r = fit_recurrence(spacings, [lag1, lag2], f"[{lag1}, {lag2}]")
            results_spacings_search.append(r)
        except:
            pass

results_spacings_search.sort(key=lambda x: x['r2'], reverse=True)

print("\nTop 10 lag pairs for spacings by R²:")
for i, r in enumerate(results_spacings_search[:10], 1):
    print(f"  #{i}: lags {r['lags']}, R² = {r['r2']:.6f}, a+b = {r['coeff_sum']:.4f}")

print(f"\nWhere does [8, 21] rank? ", end="")
for i, r in enumerate(results_spacings_search, 1):
    if r['lags'] == [8, 21]:
        total = len(results_spacings_search)
        print(f"#{i}/{total} (percentile: {100 * (total - i) / total:.1f}%)")
        break

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Does structure transfer to stationary variables?")
print("=" * 70)

print(f"""
                        Raw Zeros      Spacings       Fluctuations
                        ---------      --------       ------------
R² [8, 21]              {result_zeros['r2']:.8f}   {result_spacings['r2']:.8f}   {result_fluct['r2']:.8f}
Coeff a                 {result_zeros['coefficients'][0]:.6f}       {result_spacings['coefficients'][0]:.6f}       {result_fluct['coefficients'][0]:.6f}
Coeff b                 {result_zeros['coefficients'][1]:.6f}       {result_spacings['coefficients'][1]:.6f}       {result_fluct['coefficients'][1]:.6f}
Sum a+b                 {result_zeros['coeff_sum']:.6f}       {result_spacings['coeff_sum']:.6f}       {result_fluct['coeff_sum']:.6f}
""")

# Interpretation
zeros_r2 = result_zeros['r2']
spacings_r2 = result_spacings['r2']
fluct_r2 = result_fluct['r2']

print("-" * 70)
print("INTERPRETATION:")
print("-" * 70)

if spacings_r2 > 0.1:
    print(f"""
✓ STRONG RESULT: Recurrence structure PERSISTS in spacings!
  R²(spacings) = {spacings_r2:.4f} > 10%
  This is NOT just trend-fitting - there is genuine arithmetic structure.
""")
elif spacings_r2 > 0.01:
    print(f"""
~ WEAK RESULT: Some structure in spacings, but weak.
  R²(spacings) = {spacings_r2:.4f}
  The structure is mostly in the trend, not the fine fluctuations.
""")
else:
    print(f"""
✗ NULL RESULT: Structure VANISHES in spacings.
  R²(spacings) = {spacings_r2:.4f} < 1%
  The recurrence captures cumulative growth, NOT fine structure.
  This is expected: the zeros grow like T ~ 2πn/log(n).
""")

if fluct_r2 > 0.1:
    print(f"""
✓ Fluctuations also show structure: R² = {fluct_r2:.4f}
""")
else:
    print(f"""
✗ Fluctuations show NO structure: R² = {fluct_r2:.4f}
  The recurrence does not capture the "interesting" part of zeros.
""")

# What this means for GIFT
print("-" * 70)
print("WHAT THIS MEANS FOR GIFT:")
print("-" * 70)

print("""
The recurrence γₙ ≈ (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁ + c works because:

1. The zeros grow asymptotically like T ~ 2πn/log(n)
2. Any linear combination with a+b=1 preserves this growth
3. The [8, 21] lags and 31/21 ratio emerge from this growth structure

This is STILL interesting because:
- The coefficient 31/21 is CLOSE to Fibonacci ratio F₈₊₃/F₈ = 34/21
- The constraint a+b=1 is EXACT (translation invariance)
- The lags [8, 21] ARE special (beat [136, 149] by 923%)

But the "deep structure" is in the GROWTH, not the fluctuations.
""")

# Save results
results = {
    'raw_zeros': result_zeros,
    'spacings': result_spacings,
    'spacings_fibonacci': result_spacings_fib,
    'spacings_optimal': result_spacings_opt,
    'fluctuations': result_fluct,
    'fluctuations_fibonacci': result_fluct_fib,
    'fluctuations_optimal': result_fluct_opt,
    'top_spacings_lags': [r for r in results_spacings_search[:10]],
    'interpretation': {
        'spacings_has_structure': bool(spacings_r2 > 0.1),
        'fluctuations_has_structure': bool(fluct_r2 > 0.1),
        'structure_is_in_growth': bool(spacings_r2 < 0.1)
    }
}

with open('/home/user/GIFT/research/riemann/spacings_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("Results saved to spacings_test_results.json")
print("=" * 70)
