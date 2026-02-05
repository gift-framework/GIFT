#!/usr/bin/env python3
"""
Decisive Comparison Test (from Council-17 Kimi)

Compare three approaches on the SAME data:

Method A: lags [8, 21], coefficients CONSTRAINED to 31/21, -10/21
Method B: lags [136, 149], coefficients FREE-FIT
Method C: lags [8, 21], coefficients FREE-FIT

If A > C: The theoretical constraint IMPROVES performance
If B > A > C: Optimal lags beat theoretical lags
If A > B: Theoretical structure beats optimal lags (strong result!)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import json

print("=" * 70)
print("DECISIVE COMPARISON: Constrained vs Free-Fit")
print("=" * 70)

# Load zeros
try:
    zeros = np.load('/home/user/GIFT/riemann_zeros_10k.npy')
except:
    zeros = np.load('/home/user/GIFT/notebooks/riemann_zeros_10k.npy')

print(f"\nLoaded {len(zeros)} Riemann zeros")

def evaluate_method(zeros, lags, coeffs=None, name=""):
    """
    Evaluate a recurrence method.

    If coeffs is None, use free-fit (LinearRegression).
    If coeffs is provided, use those constrained coefficients.
    """
    N = len(zeros)
    max_lag = max(lags)

    # Build feature matrix
    X = np.column_stack([zeros[max_lag - lag : N - lag] for lag in lags])
    y = zeros[max_lag:]

    if coeffs is None:
        # Free fit
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X)
        fitted_coeffs = list(model.coef_)
        intercept = model.intercept_
    else:
        # Constrained coefficients
        fitted_coeffs = list(coeffs)
        # Still fit intercept
        y_pred_no_intercept = X @ np.array(coeffs)
        intercept = np.mean(y - y_pred_no_intercept)
        y_pred = y_pred_no_intercept + intercept

    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(np.mean(residuals**2))
    sigma = np.std(residuals)

    return {
        'name': name,
        'lags': lags,
        'coefficients': fitted_coeffs,
        'intercept': float(intercept),
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'sigma': float(sigma),
        'residuals': residuals
    }

# ============================================================
# METHOD A: Constrained (theoretical)
# ============================================================
print("\n" + "=" * 70)
print("METHOD A: Constrained [8, 21] with coeffs 31/21, -10/21")
print("=" * 70)

method_a = evaluate_method(
    zeros,
    lags=[8, 21],
    coeffs=[31/21, -10/21],
    name="A: Constrained (31/21, -10/21)"
)

print(f"  Lags: {method_a['lags']}")
print(f"  Coefficients: {method_a['coefficients'][0]:.6f}, {method_a['coefficients'][1]:.6f}")
print(f"  R² = {method_a['r2']:.10f}")
print(f"  RMSE = {method_a['rmse']:.6f}")
print(f"  σ_residual = {method_a['sigma']:.6f}")

# ============================================================
# METHOD B: Optimal lags (free-fit)
# ============================================================
print("\n" + "=" * 70)
print("METHOD B: Optimal lags [136, 149] with FREE-FIT")
print("=" * 70)

method_b = evaluate_method(
    zeros,
    lags=[136, 149],
    coeffs=None,  # Free fit
    name="B: Optimal lags (136, 149) free-fit"
)

print(f"  Lags: {method_b['lags']}")
print(f"  Coefficients: {method_b['coefficients'][0]:.6f}, {method_b['coefficients'][1]:.6f}")
print(f"  Sum: {sum(method_b['coefficients']):.6f}")
print(f"  R² = {method_b['r2']:.10f}")
print(f"  RMSE = {method_b['rmse']:.6f}")
print(f"  σ_residual = {method_b['sigma']:.6f}")

# ============================================================
# METHOD C: Theoretical lags, free-fit
# ============================================================
print("\n" + "=" * 70)
print("METHOD C: Theoretical lags [8, 21] with FREE-FIT")
print("=" * 70)

method_c = evaluate_method(
    zeros,
    lags=[8, 21],
    coeffs=None,  # Free fit
    name="C: Theoretical lags (8, 21) free-fit"
)

print(f"  Lags: {method_c['lags']}")
print(f"  Coefficients: {method_c['coefficients'][0]:.6f}, {method_c['coefficients'][1]:.6f}")
print(f"  Sum: {sum(method_c['coefficients']):.6f}")
print(f"  R² = {method_c['r2']:.10f}")
print(f"  RMSE = {method_c['rmse']:.6f}")
print(f"  σ_residual = {method_c['sigma']:.6f}")

# ============================================================
# METHOD D: GIFT topology [8, 21, 77] free-fit
# ============================================================
print("\n" + "=" * 70)
print("METHOD D: GIFT topology [8, 21, 77] with FREE-FIT")
print("=" * 70)

method_d = evaluate_method(
    zeros,
    lags=[8, 21, 77],
    coeffs=None,
    name="D: GIFT (8, 21, 77) free-fit"
)

print(f"  Lags: {method_d['lags']}")
print(f"  Coefficients: {[f'{c:.6f}' for c in method_d['coefficients']]}")
print(f"  Sum: {sum(method_d['coefficients']):.6f}")
print(f"  R² = {method_d['r2']:.10f}")
print(f"  RMSE = {method_d['rmse']:.6f}")
print(f"  σ_residual = {method_d['sigma']:.6f}")

# ============================================================
# METHOD E: "Optimal" 3-lag (top from systematic search)
# ============================================================
print("\n" + "=" * 70)
print("METHOD E: Top systematic 3-lag [8, 21, 140]")
print("=" * 70)

method_e = evaluate_method(
    zeros,
    lags=[8, 21, 140],
    coeffs=None,
    name="E: Systematic best (8, 21, 140)"
)

print(f"  Lags: {method_e['lags']}")
print(f"  Coefficients: {[f'{c:.6f}' for c in method_e['coefficients']]}")
print(f"  Sum: {sum(method_e['coefficients']):.6f}")
print(f"  R² = {method_e['r2']:.10f}")
print(f"  RMSE = {method_e['rmse']:.6f}")
print(f"  σ_residual = {method_e['sigma']:.6f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

methods = [method_a, method_b, method_c, method_d, method_e]

print(f"\n{'Method':<40} {'R²':<18} {'σ_resid':<10} {'RMSE':<10}")
print("-" * 80)

for m in methods:
    print(f"{m['name']:<40} {m['r2']:<18.12f} {m['sigma']:<10.4f} {m['rmse']:<10.4f}")

# ============================================================
# THE VERDICT
# ============================================================
print("\n" + "=" * 70)
print("THE VERDICT")
print("=" * 70)

# Sort by sigma (lower is better)
ranked = sorted(methods, key=lambda x: x['sigma'])

print("\nRanking by σ_residual (lower = better):")
for i, m in enumerate(ranked, 1):
    print(f"  #{i}: {m['name']} (σ = {m['sigma']:.4f})")

# Key comparisons
print("\n" + "-" * 70)
print("KEY COMPARISONS:")
print("-" * 70)

# A vs C: Does constraint help?
# Note: lower sigma is better
a_vs_c_pct = (method_a['sigma'] - method_c['sigma']) / method_c['sigma'] * 100
print(f"\n1. Constrained (A) vs Free-fit same lags (C):")
print(f"   A σ = {method_a['sigma']:.4f}, C σ = {method_c['sigma']:.4f}")
if method_a['sigma'] < method_c['sigma']:
    print(f"   → Constraint IMPROVES fit by {a_vs_c_pct:.2f}%")
    print(f"   → Theoretical 31/21 is OPTIMAL!")
else:
    print(f"   → Constraint DEGRADES fit by {a_vs_c_pct:.2f}%")
    print(f"   → Free-fit finds BETTER coefficients than 31/21")

# C vs B: Do theoretical lags beat optimal lags?
c_vs_b = (method_b['sigma'] - method_c['sigma']) / method_c['sigma'] * 100
print(f"\n2. Theoretical lags (C) vs Optimal lags (B):")
print(f"   C σ = {method_c['sigma']:.4f}, B σ = {method_b['sigma']:.4f}")
if c_vs_b > 0:
    print(f"   → Theoretical lags [8, 21] beat optimal [136, 149] by {c_vs_b:.2f}%!")
    print(f"   → The GIFT lags are genuinely special")
else:
    print(f"   → Optimal lags [136, 149] beat theoretical by {-c_vs_b:.2f}%")
    print(f"   → [8, 21] is not optimal, just theoretical")

# D vs E: Does GIFT (77) beat systematic (140)?
d_vs_e = (method_e['sigma'] - method_d['sigma']) / method_d['sigma'] * 100
print(f"\n3. GIFT 3-lag (D) vs Systematic best (E):")
print(f"   D σ = {method_d['sigma']:.4f}, E σ = {method_e['sigma']:.4f}")
if d_vs_e > 0:
    print(f"   → GIFT [8, 21, 77] beats systematic [8, 21, 140] by {d_vs_e:.2f}%!")
else:
    print(f"   → Systematic [8, 21, 140] beats GIFT by {-d_vs_e:.2f}%")

# Free-fit coefficient analysis
print("\n" + "-" * 70)
print("COEFFICIENT ANALYSIS:")
print("-" * 70)

print(f"\nTheoretical prediction: a = 31/21 = {31/21:.6f}")
print(f"Free-fit result:       a = {method_c['coefficients'][0]:.6f}")
print(f"Difference:            Δa = {method_c['coefficients'][0] - 31/21:.6f}")
print(f"Relative error:        {abs(method_c['coefficients'][0] - 31/21) / (31/21) * 100:.2f}%")

print(f"\nTheoretical prediction: b = -10/21 = {-10/21:.6f}")
print(f"Free-fit result:       b = {method_c['coefficients'][1]:.6f}")
print(f"Difference:            Δb = {method_c['coefficients'][1] - (-10/21):.6f}")
print(f"Relative error:        {abs(method_c['coefficients'][1] - (-10/21)) / abs(-10/21) * 100:.2f}%")

# ============================================================
# FINAL INTERPRETATION
# ============================================================
print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

# Check if free-fit converges to theory
coeff_a_close = abs(method_c['coefficients'][0] - 31/21) < 0.05
coeff_b_close = abs(method_c['coefficients'][1] - (-10/21)) < 0.05

if coeff_a_close and coeff_b_close:
    print("""
The free-fit coefficients are CLOSE to 31/21 and -10/21 (within 5%).
This means:
  - The constraint is not arbitrary - it reflects a real optimum
  - The 31/21 ratio has empirical support
  - The theoretical prediction is vindicated
""")
else:
    print("""
The free-fit coefficients DIVERGE from 31/21 and -10/21.
This means:
  - The constraint is NOT optimal for prediction
  - 31/21 may have theoretical meaning but is not the empirical best
  - Further investigation needed
""")

# Check if [8, 21] beats optimal
if method_c['sigma'] < method_b['sigma']:
    print("""
IMPORTANT: Theoretical lags [8, 21] outperform "optimal" lags [136, 149]!
This is strong evidence that [8, 21] captures genuine structure.
""")
else:
    print(f"""
IMPORTANT: "Optimal" lags [136, 149] outperform theoretical [8, 21].
The σ gap is {method_b['sigma'] - method_c['sigma']:.4f}.
[8, 21] is special for THEORY, not for raw prediction.
""")

# Save results
results = {
    'method_a': {k: v for k, v in method_a.items() if k != 'residuals'},
    'method_b': {k: v for k, v in method_b.items() if k != 'residuals'},
    'method_c': {k: v for k, v in method_c.items() if k != 'residuals'},
    'method_d': {k: v for k, v in method_d.items() if k != 'residuals'},
    'method_e': {k: v for k, v in method_e.items() if k != 'residuals'},
    'comparisons': {
        'constraint_vs_freefit': float(a_vs_c_pct),
        'theoretical_vs_optimal': float(c_vs_b),
        'gift_vs_systematic': float(d_vs_e)
    },
    'conclusion': {
        'freefit_matches_theory': bool(coeff_a_close and coeff_b_close),
        'theoretical_beats_optimal': bool(method_c['sigma'] < method_b['sigma'])
    }
}

with open('/home/user/GIFT/research/riemann/decisive_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("Results saved to decisive_comparison_results.json")
print("=" * 70)
