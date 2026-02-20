#!/usr/bin/env python3
"""
STRATEGY B NUMERICAL VALIDATION
================================
Three tests to validate the Bethe Ansatz approach for closing the gap
between spectral constraint and ordered recurrence.

Tests:
1. FFT of oscillatory corrections δₙ → Fibonacci frequency peaks?
2. Recurrence residual |Rₙ|/|δₙ| → Fibonacci projection saturation?
3. Spacing autocorrelation R(k) → anomaly at lags 8, 21?
"""

import numpy as np
from scipy.special import lambertw
import json

# =============================================================
# LOAD DATA
# =============================================================
zeros = np.load('/home/user/GIFT/riemann_zeros_10k.npy')
N = len(zeros)
phi = (1 + np.sqrt(5)) / 2  # golden ratio

print("=" * 70)
print("STRATEGY B: NUMERICAL VALIDATION OF GAP CLOSURE")
print("=" * 70)
print(f"Loaded {N} Riemann zeros, γ₁ = {zeros[0]:.6f}, γ_N = {zeros[-1]:.2f}")

# =============================================================
# TEST 1: FFT OF OSCILLATORY CORRECTIONS
# =============================================================
print("\n" + "=" * 70)
print("TEST 1: FFT OF OSCILLATORY CORRECTIONS δₙ")
print("=" * 70)

# Smooth approximation: Franca-LeClair Lambert W formula
# γₙ⁽⁰⁾ ≈ 2πn / W₀(n/e)
def smooth_zero(n):
    """Lambert W approximation for n-th Riemann zero."""
    w = np.real(lambertw(n / np.e))
    return 2 * np.pi * n / w

# Compute smooth approximation and corrections
ns = np.arange(1, N + 1)
gamma_smooth = np.array([smooth_zero(n) for n in ns])
delta = zeros - gamma_smooth  # oscillatory corrections

print(f"\nSmooth approximation quality:")
print(f"  Mean |γₙ - γₙ⁽⁰⁾|:     {np.mean(np.abs(delta)):.4f}")
print(f"  Mean |γₙ - γₙ⁽⁰⁾|/γₙ:  {np.mean(np.abs(delta/zeros)):.6f} ({np.mean(np.abs(delta/zeros))*100:.4f}%)")
print(f"  Max |δₙ|:              {np.max(np.abs(delta)):.4f}")

# FFT of corrections
delta_centered = delta - np.mean(delta)
fft_delta = np.abs(np.fft.rfft(delta_centered))
freqs = np.fft.rfftfreq(N, d=1.0)  # frequencies in cycles per sample

# Fibonacci frequencies: we expect peaks related to the geodesic structure
# The Fibonacci geodesic has ℓ₀ = 2 log φ
# Lag-k in the recurrence corresponds to frequency k/N in the FFT

print(f"\nFFT analysis of δₙ:")
print(f"  Total power: {np.sum(fft_delta**2):.2f}")

# Check power at specific lags (as fraction of N)
lags_to_check = [3, 5, 8, 13, 21, 34, 42, 55, 77, 99]
print(f"\n  Power at specific lag-frequencies:")
print(f"  {'Lag':>5} {'Freq':>10} {'|FFT|':>12} {'Rank':>6} {'Percentile':>12}")

# Compute ranks
sorted_fft = np.sort(fft_delta)[::-1]
for lag in lags_to_check:
    if lag < len(fft_delta):
        power = fft_delta[lag]
        rank = np.searchsorted(-sorted_fft, -power) + 1
        pctile = (1 - rank / len(fft_delta)) * 100
        marker = " <<<" if lag in [8, 21] else ""
        print(f"  {lag:>5} {lag/N:>10.6f} {power:>12.2f} {rank:>6} {pctile:>11.1f}%{marker}")

# Top 20 frequencies
top_indices = np.argsort(fft_delta[1:])[::-1][:20] + 1  # skip DC
print(f"\n  Top 20 frequencies (by |FFT|):")
for i, idx in enumerate(top_indices):
    fib_marker = " (Fibonacci!)" if idx in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89] else ""
    print(f"    #{i+1:>2}: lag={idx:>4}, |FFT|={fft_delta[idx]:>10.2f}{fib_marker}")

# =============================================================
# TEST 2: RECURRENCE ON CORRECTIONS
# =============================================================
print("\n" + "=" * 70)
print("TEST 2: RECURRENCE RESIDUAL ON CORRECTIONS δₙ")
print("=" * 70)

a = 31 / 21
b = -10 / 21

# Residual on RAW zeros (for comparison)
start = 21  # need at least 21 previous values
R_raw = zeros[start:] - a * zeros[start-8:-8] - b * zeros[start-21:-21]
print(f"\nRaw zero recurrence (γₙ - a·γₙ₋₈ - b·γₙ₋₂₁):")
print(f"  Mean residual:   {np.mean(R_raw):.4f}")
print(f"  Std residual:    {np.std(R_raw):.4f}")
print(f"  Mean |residual|: {np.mean(np.abs(R_raw)):.4f}")
print(f"  Relative error:  {np.mean(np.abs(R_raw) / zeros[start:]):.6f} ({np.mean(np.abs(R_raw) / zeros[start:])*100:.4f}%)")

# Residual on CORRECTIONS δₙ
R_delta = delta[start:] - a * delta[start-8:-8] - b * delta[start-21:-21]

print(f"\nCorrection recurrence (δₙ - a·δₙ₋₈ - b·δₙ₋₂₁):")
print(f"  Mean R_delta:   {np.mean(R_delta):.4f}")
print(f"  Std R_delta:    {np.std(R_delta):.4f}")
print(f"  Mean |R_delta|: {np.mean(np.abs(R_delta)):.4f}")
print(f"  Mean |δₙ|:      {np.mean(np.abs(delta[start:])):.4f}")
ratio = np.mean(np.abs(R_delta)) / np.mean(np.abs(delta[start:]))
print(f"\n  *** |R_delta| / |δₙ| = {ratio:.4f} ({ratio*100:.2f}%) ***")

if ratio < 0.5:
    print(f"  → PROMISING: Fibonacci projection captures >{(1-ratio)*100:.0f}% of corrections!")
elif ratio < 1.0:
    print(f"  → PARTIAL: Fibonacci projection captures some structure")
else:
    print(f"  → WEAK: Corrections not well-described by Fibonacci recurrence")

# Compare with other lag combinations
print(f"\n  Comparison with other lag pairs:")
lag_pairs = [(8, 21), (5, 13), (3, 8), (13, 34), (7, 19), (10, 25)]
for l1, l2 in lag_pairs:
    s = max(l1, l2)
    # Free fit for these lags
    X = np.column_stack([delta[s-l1:-l1], delta[s-l2:-l2]])
    y = delta[s:]
    # OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    R_free = y - X @ beta
    ratio_free = np.std(R_free) / np.std(y)
    marker = " <<<" if (l1, l2) == (8, 21) else ""
    print(f"    [{l1:>2}, {l2:>2}]: σ_residual/σ_δ = {ratio_free:.4f}, best-fit coeffs = ({beta[0]:.3f}, {beta[1]:.3f}){marker}")

# =============================================================
# TEST 3: SPACING AUTOCORRELATION
# =============================================================
print("\n" + "=" * 70)
print("TEST 3: SPACING AUTOCORRELATION R(k)")
print("=" * 70)

# Compute spacings
spacings = np.diff(zeros)

# Normalize spacings by local mean (unfolding)
# Local density: dn/dγ ≈ (1/2π) log(γ/2π)
local_density = (1 / (2 * np.pi)) * np.log(zeros[:-1] / (2 * np.pi))
normalized_spacings = spacings * local_density
normalized_spacings = normalized_spacings - np.mean(normalized_spacings)

# Autocorrelation function
max_lag = 35
R = np.zeros(max_lag + 1)
for k in range(max_lag + 1):
    if k == 0:
        R[k] = np.var(normalized_spacings)
    else:
        R[k] = np.mean(normalized_spacings[k:] * normalized_spacings[:-k])

R_normalized = R / R[0]  # normalize by variance

print(f"\nAutocorrelation R(k) of normalized spacings:")
print(f"  {'Lag k':>6} {'R(k)':>10} {'|R(k)|':>10} {'Rank':>6} {'Significance':>14}")

# Compute significance: under null (i.i.d.), |R(k)| ~ N(0, 1/N)
sig_threshold = 2 / np.sqrt(N)  # 2-sigma threshold

sorted_abs_R = np.sort(np.abs(R_normalized[1:]))[::-1]
for k in range(1, max_lag + 1):
    rank = np.searchsorted(-sorted_abs_R, -np.abs(R_normalized[k])) + 1
    z_score = np.abs(R_normalized[k]) / (1/np.sqrt(N))
    sig = "***" if z_score > 3 else "**" if z_score > 2 else "*" if z_score > 1.5 else ""
    marker = " <<<" if k in [8, 21] else ""
    print(f"  {k:>6} {R_normalized[k]:>10.6f} {np.abs(R_normalized[k]):>10.6f} {rank:>6} {f'z={z_score:.1f} {sig}':>14}{marker}")

# Specific check for lags 8 and 21
print(f"\n  2σ significance threshold: |R(k)| > {sig_threshold:.6f}")
print(f"  R(8)  = {R_normalized[8]:.6f}, z = {np.abs(R_normalized[8])*np.sqrt(N):.1f}")
print(f"  R(21) = {R_normalized[21]:.6f}, z = {np.abs(R_normalized[21])*np.sqrt(N):.1f}")

# Fibonacci lags vs non-Fibonacci
fib_lags = [1, 2, 3, 5, 8, 13, 21, 34]
non_fib_lags = [k for k in range(1, max_lag+1) if k not in fib_lags]

fib_power = np.mean([np.abs(R_normalized[k]) for k in fib_lags if k <= max_lag])
non_fib_power = np.mean([np.abs(R_normalized[k]) for k in non_fib_lags if k <= max_lag])

print(f"\n  Mean |R(k)| for Fibonacci lags {fib_lags}: {fib_power:.6f}")
print(f"  Mean |R(k)| for non-Fibonacci lags:        {non_fib_power:.6f}")
print(f"  Ratio (Fib/non-Fib): {fib_power/non_fib_power:.3f}")

# =============================================================
# SUMMARY
# =============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results = {
    "test1_fft": {
        "description": "FFT of oscillatory corrections",
        "delta_mean_abs": float(np.mean(np.abs(delta))),
        "delta_relative": float(np.mean(np.abs(delta/zeros))),
        "lag8_fft_power": float(fft_delta[8]),
        "lag21_fft_power": float(fft_delta[21]),
    },
    "test2_recurrence": {
        "description": "Recurrence on corrections",
        "R_delta_std": float(np.std(R_delta)),
        "delta_std": float(np.std(delta[start:])),
        "ratio_R_over_delta": float(ratio),
        "fibonacci_captures_pct": float((1-ratio)*100) if ratio < 1 else 0,
    },
    "test3_autocorrelation": {
        "description": "Spacing autocorrelation",
        "R_8": float(R_normalized[8]),
        "R_21": float(R_normalized[21]),
        "R_8_zscore": float(np.abs(R_normalized[8])*np.sqrt(N)),
        "R_21_zscore": float(np.abs(R_normalized[21])*np.sqrt(N)),
        "fib_mean_abs_R": float(fib_power),
        "nonfib_mean_abs_R": float(non_fib_power),
        "fib_to_nonfib_ratio": float(fib_power/non_fib_power),
    }
}

print(f"\nTest 1 (FFT): Corrections δₙ have mean {results['test1_fft']['delta_relative']*100:.3f}% relative to γₙ")
print(f"Test 2 (Recurrence): |R_δ|/|δ| = {results['test2_recurrence']['ratio_R_over_delta']:.4f}")
print(f"Test 3 (Autocorrelation): R(8) z-score = {results['test3_autocorrelation']['R_8_zscore']:.1f}, R(21) z-score = {results['test3_autocorrelation']['R_21_zscore']:.1f}")

# Save results
with open('/home/user/GIFT/research/riemann/strategy_B_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to strategy_B_validation_results.json")
print("=" * 70)
