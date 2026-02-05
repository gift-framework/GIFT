#!/usr/bin/env python3
"""
Test second-order correction at lag 42 = 2 × 21 = 2 × b₂(K₇)

Hypothesis: The ACF(21) = 0.34 signal in residuals will be reduced
by adding a third term γ_{n-42} to the recurrence.

γₙ ≈ a·γₙ₋₈ + b·γₙ₋₂₁ + c·γₙ₋₄₂ + d
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import correlate
import json

# Load zeros
print("="*70)
print("SECOND-ORDER CORRECTION TEST: LAG 42 = 2 × b₂(K₇)")
print("="*70)

try:
    zeros = np.load('/home/user/GIFT/riemann_zeros_10k.npy')
except:
    zeros = np.load('/home/user/GIFT/notebooks/riemann_zeros_10k.npy')

print(f"\nLoaded {len(zeros)} Riemann zeros")

def autocorr(x, max_lag=50):
    """Compute autocorrelation."""
    x = x - np.mean(x)
    result = correlate(x, x, mode='full')
    result = result[len(result)//2:]
    return result[:max_lag] / result[0]

def fit_recurrence(zeros, lags, name):
    """Fit recurrence with given lags."""
    N = len(zeros)
    max_lag = max(lags)

    # Build feature matrix
    X = np.column_stack([zeros[max_lag - lag : N - lag] for lag in lags])
    y = zeros[max_lag:]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred

    acf = autocorr(residuals, max_lag=50)

    return {
        'name': name,
        'lags': lags,
        'coefficients': list(model.coef_),
        'intercept': model.intercept_,
        'r2': r2,
        'residual_std': np.std(residuals),
        'acf_8': acf[8],
        'acf_13': acf[13],
        'acf_21': acf[21],
        'acf_34': acf[34] if len(acf) > 34 else None,
        'acf_42': acf[42] if len(acf) > 42 else None,
        'residuals': residuals
    }

# ============================================================
# TEST 1: Original recurrence (lags 8, 21)
# ============================================================
print("\n" + "="*70)
print("TEST 1: Original recurrence (lags 8, 21)")
print("="*70)

result_2lag = fit_recurrence(zeros, [8, 21], "2-lag (8, 21)")

print(f"\nCoefficients:")
print(f"  a (lag 8):  {result_2lag['coefficients'][0]:.6f}  (target: 31/21 = {31/21:.6f})")
print(f"  b (lag 21): {result_2lag['coefficients'][1]:.6f}  (target: -10/21 = {-10/21:.6f})")
print(f"  a + b:      {sum(result_2lag['coefficients']):.6f}  (target: 1.0)")
print(f"\nR² = {result_2lag['r2']:.10f}")
print(f"Residual σ = {result_2lag['residual_std']:.4f}")
print(f"\nResidual ACF:")
print(f"  ACF(8):  {result_2lag['acf_8']:.4f}")
print(f"  ACF(13): {result_2lag['acf_13']:.4f}")
print(f"  ACF(21): {result_2lag['acf_21']:.4f} ← TARGET TO REDUCE")
print(f"  ACF(34): {result_2lag['acf_34']:.4f}")

# ============================================================
# TEST 2: Add lag 42 = 2 × 21
# ============================================================
print("\n" + "="*70)
print("TEST 2: Three-lag recurrence (8, 21, 42)")
print("="*70)

result_3lag = fit_recurrence(zeros, [8, 21, 42], "3-lag (8, 21, 42)")

print(f"\nCoefficients:")
print(f"  a (lag 8):  {result_3lag['coefficients'][0]:.6f}")
print(f"  b (lag 21): {result_3lag['coefficients'][1]:.6f}")
print(f"  c (lag 42): {result_3lag['coefficients'][2]:.6f} ← NEW")
print(f"  Sum:        {sum(result_3lag['coefficients']):.6f}")
print(f"\nR² = {result_3lag['r2']:.10f}")
print(f"Residual σ = {result_3lag['residual_std']:.4f}")
print(f"\nResidual ACF:")
print(f"  ACF(8):  {result_3lag['acf_8']:.4f}")
print(f"  ACF(13): {result_3lag['acf_13']:.4f}")
print(f"  ACF(21): {result_3lag['acf_21']:.4f} ← COMPARE TO BEFORE")
print(f"  ACF(34): {result_3lag['acf_34']:.4f}")
print(f"  ACF(42): {result_3lag['acf_42']:.4f}")

# ============================================================
# TEST 3: Full Fibonacci sequence (8, 21, 34, 55)
# ============================================================
print("\n" + "="*70)
print("TEST 3: Fibonacci sequence (8, 13, 21, 34)")
print("="*70)

result_fib = fit_recurrence(zeros, [8, 13, 21, 34], "Fibonacci (8,13,21,34)")

print(f"\nCoefficients:")
for i, lag in enumerate([8, 13, 21, 34]):
    print(f"  lag {lag}: {result_fib['coefficients'][i]:.6f}")
print(f"  Sum:    {sum(result_fib['coefficients']):.6f}")
print(f"\nR² = {result_fib['r2']:.10f}")
print(f"Residual σ = {result_fib['residual_std']:.4f}")
print(f"\nResidual ACF:")
print(f"  ACF(8):  {result_fib['acf_8']:.4f}")
print(f"  ACF(13): {result_fib['acf_13']:.4f}")
print(f"  ACF(21): {result_fib['acf_21']:.4f}")
print(f"  ACF(34): {result_fib['acf_34']:.4f}")

# ============================================================
# TEST 4: GIFT-motivated (8, 21, 77) with b₃ = 77
# ============================================================
print("\n" + "="*70)
print("TEST 4: GIFT topology (8, 21, 77) with b₃ = 77")
print("="*70)

result_gift = fit_recurrence(zeros, [8, 21, 77], "GIFT (8, 21, 77)")

print(f"\nCoefficients:")
print(f"  lag 8 (rank E₈):   {result_gift['coefficients'][0]:.6f}")
print(f"  lag 21 (b₂):       {result_gift['coefficients'][1]:.6f}")
print(f"  lag 77 (b₃):       {result_gift['coefficients'][2]:.6f} ← NEW")
print(f"  Sum:               {sum(result_gift['coefficients']):.6f}")
print(f"\nR² = {result_gift['r2']:.10f}")
print(f"Residual σ = {result_gift['residual_std']:.4f}")
print(f"\nResidual ACF:")
print(f"  ACF(8):  {result_gift['acf_8']:.4f}")
print(f"  ACF(13): {result_gift['acf_13']:.4f}")
print(f"  ACF(21): {result_gift['acf_21']:.4f}")

# ============================================================
# COMPARISON SUMMARY
# ============================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Model':<25} {'R²':<18} {'σ_resid':<10} {'ACF(21)':<10} {'ACF(21) reduced?':<15}")
print("-" * 80)

baseline_acf21 = result_2lag['acf_21']

for r in [result_2lag, result_3lag, result_fib, result_gift]:
    reduction = (baseline_acf21 - r['acf_21']) / baseline_acf21 * 100
    status = f"↓ {reduction:.1f}%" if reduction > 5 else "~same"
    print(f"{r['name']:<25} {r['r2']:<18.12f} {r['residual_std']:<10.4f} {r['acf_21']:<10.4f} {status:<15}")

# ============================================================
# KEY INSIGHT
# ============================================================
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

acf21_reduction = (result_2lag['acf_21'] - result_3lag['acf_21']) / result_2lag['acf_21'] * 100

print(f"""
1. ORIGINAL (8, 21):
   - ACF(21) = {result_2lag['acf_21']:.4f} (significant!)
   - This was our "mystery signal"

2. WITH LAG 42 = 2×b₂:
   - ACF(21) = {result_3lag['acf_21']:.4f}
   - Reduction: {acf21_reduction:.1f}%
   - New coefficient c(42) = {result_3lag['coefficients'][2]:.6f}

3. INTERPRETATION:
""")

if acf21_reduction > 20:
    print(f"   ✓ Adding lag 42 SIGNIFICANTLY reduces ACF(21)!")
    print(f"   ✓ This confirms the hypothesis: lag 21 had 'harmonic' structure")
    print(f"   ✓ The second Betti number b₂ = 21 appears TWICE in the recurrence")
elif acf21_reduction > 5:
    print(f"   ~ Partial reduction of ACF(21)")
    print(f"   ~ The lag 42 helps but doesn't fully explain the signal")
else:
    print(f"   ✗ Adding lag 42 does NOT reduce ACF(21)")
    print(f"   ✗ The signal at lag 21 has a different origin")

# Check if coefficient at lag 42 is Fibonacci-related
c42 = result_3lag['coefficients'][2]
print(f"""
4. COEFFICIENT ANALYSIS:
   c(42) = {c42:.6f}

   Fibonacci check:
   - F_9/F_8² = 34/441 = {34/441:.6f}
   - 1/42 = {1/42:.6f}
   - c(42) sign: {'positive' if c42 > 0 else 'negative'}
""")

# Save results
results = {
    '2_lag': {
        'lags': [8, 21],
        'coefficients': result_2lag['coefficients'],
        'r2': result_2lag['r2'],
        'acf_21': result_2lag['acf_21']
    },
    '3_lag_42': {
        'lags': [8, 21, 42],
        'coefficients': result_3lag['coefficients'],
        'r2': result_3lag['r2'],
        'acf_21': result_3lag['acf_21']
    },
    'fibonacci': {
        'lags': [8, 13, 21, 34],
        'coefficients': result_fib['coefficients'],
        'r2': result_fib['r2'],
        'acf_21': result_fib['acf_21']
    },
    'gift_topology': {
        'lags': [8, 21, 77],
        'coefficients': result_gift['coefficients'],
        'r2': result_gift['r2'],
        'acf_21': result_gift['acf_21']
    },
    'acf21_reduction_with_lag42': acf21_reduction
}

with open('/home/user/GIFT/research/riemann/lag42_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to lag42_test_results.json")
print("="*70)
