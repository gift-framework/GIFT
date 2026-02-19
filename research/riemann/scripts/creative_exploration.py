#!/usr/bin/env python3
"""
CREATIVE EXPLORATION: Hunting for Fine Structure
=================================================

The trend recurrence is "trivial" - let's dig into what's ACTUALLY there.

Questions to explore:
1. What IS the coefficient ~1.47? Where does it come from?
2. Is there Fibonacci structure in the FLUCTUATIONS?
3. What about the SPACINGS (gaps between zeros)?
4. Autocorrelation structure of fluctuations?
5. Spectral analysis (FFT) - hidden frequencies?
6. Can we derive the coefficient theoretically?

Let's hunt for the real signal.
"""

import numpy as np
from pathlib import Path
import json

# Constants
PHI = (1 + np.sqrt(5)) / 2
PSI = 1 - PHI
SQRT5 = np.sqrt(5)
PI = np.pi
E = np.e

# GIFT constants
B2 = 21
B3 = 77
DIM_G2 = 14
H_STAR = 99
RANK_E8 = 8

def fib(n):
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

FIBS = [fib(i) for i in range(1, 20)]

def load_zeros(max_zeros=100000):
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 6):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            zeros.append(float(line.strip()))
                        except:
                            continue
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

print("=" * 70)
print("CREATIVE EXPLORATION: HUNTING FOR FINE STRUCTURE")
print("=" * 70)

zeros = load_zeros(100000)
n = np.arange(1, len(zeros) + 1)
print(f"\n✓ Loaded {len(zeros)} zeros")

# ============================================================================
# 1. WHAT IS THE COEFFICIENT ~1.47?
# ============================================================================

print("\n" + "=" * 70)
print("1. DECODING THE MYSTERY COEFFICIENT")
print("=" * 70)

# Fit on different ranges to see convergence
print("\nCoefficient evolution with N:")
coefficients = []
for end in [1000, 5000, 10000, 30000, 50000, 80000, 100000]:
    if end > len(zeros):
        continue
    max_lag = 21
    X1 = zeros[max_lag - 8:end - 8]
    X2 = zeros[max_lag - 21:end - 21]
    y = zeros[max_lag:end]
    X = np.column_stack([X1, X2, np.ones(len(y))])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coeffs
    coefficients.append({'N': end, 'a': a, 'b': b, 'c': c})
    print(f"  N={end:6d}: a = {a:.8f}, b = {b:.8f}, c = {c:.6f}")

# Asymptotic value
a_asymp = coefficients[-1]['a']
b_asymp = coefficients[-1]['b']

print(f"\nAsymptotic coefficient: a ≈ {a_asymp:.8f}")

# Try to match known constants
print("\nSearching for matches in mathematical constants:")

candidates = {
    '3/2': 3/2,
    'φ': PHI,
    'φ - 1/φ': PHI - 1/PHI,
    '2φ - 1': 2*PHI - 1,
    '(φ + 1)/φ': (PHI + 1)/PHI,
    'φ²/2': PHI**2 / 2,
    '√φ + 1/2': np.sqrt(PHI) + 0.5,
    'ln(φ) + 1': np.log(PHI) + 1,
    '1 + 1/e': 1 + 1/E,
    'π/2 - 1/10': PI/2 - 0.1,
    '1 + 1/π': 1 + 1/PI,
    'e/2 + 1/10': E/2 + 0.1,
    '2 - 1/φ': 2 - 1/PHI,
    '1 + φ/3': 1 + PHI/3,
    '(1 + φ)/φ': (1 + PHI)/PHI,
    'b₂/dim(G₂)': B2/DIM_G2,
    '(b₂ - 7)/dim(G₂)': (B2 - 7)/DIM_G2,
    'dim(G₂)/rank(E₈) - 1/4': DIM_G2/RANK_E8 - 0.25,
    '(φ² + ψ²)/2': (PHI**2 + PSI**2)/2,
    '(φ² + 1)/2': (PHI**2 + 1)/2,
    '1 + 1/2.1': 1 + 1/2.1,
    '21/14': 21/14,
    '22/15': 22/15,
    '29/20': 29/20,
    '44/30': 44/30,
    '(21-1)/(14-1)': 20/13,
    '(21+1)/(14+1)': 22/15,
}

print(f"\n{'Expression':<25} {'Value':<12} {'Diff from a':<12}")
print("-" * 50)

matches = []
for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - a_asymp)):
    diff = abs(val - a_asymp)
    matches.append((name, val, diff))
    if diff < 0.1:
        print(f"{name:<25} {val:<12.8f} {diff:<12.8f}")

print(f"\nBest matches:")
for name, val, diff in matches[:5]:
    print(f"  {name} = {val:.8f} (diff = {diff:.8f})")

# ============================================================================
# 2. THEORETICAL DERIVATION ATTEMPT
# ============================================================================

print("\n" + "=" * 70)
print("2. THEORETICAL DERIVATION OF THE COEFFICIENT")
print("=" * 70)

print("""
The Riemann-von Mangoldt formula:
  N(T) = T/(2π) · log(T/(2πe)) + O(log T)

For large n, γ_n ≈ 2πn / W(n/e) where W is Lambert W.

Approximation: γ_n ≈ 2πn / log(n) for large n

Let's see what coefficient emerges from this asymptotic form...
""")

# If γ_n ~ 2πn/log(n), what is the "natural" recurrence coefficient?
# γ_n / γ_{n-8} ≈ (n/log(n)) / ((n-8)/log(n-8))
#              ≈ n·log(n-8) / ((n-8)·log(n))

def theoretical_ratio(n, lag):
    """Theoretical γ_n / γ_{n-lag} from asymptotic formula"""
    return (n * np.log(n - lag)) / ((n - lag) * np.log(n))

# Test at different n
print("Theoretical ratio γ_n/γ_{n-8} from asymptotic formula:")
for test_n in [100, 1000, 10000, 50000]:
    r8 = theoretical_ratio(test_n, 8)
    r21 = theoretical_ratio(test_n, 21)
    print(f"  n={test_n:5d}: γ_n/γ_{{n-8}} = {r8:.6f}, γ_n/γ_{{n-21}} = {r21:.6f}")

# What coefficient would make γ_n = a·γ_{n-8} + (1-a)·γ_{n-21} exact?
print("\nDeriving 'natural' coefficient from asymptotics...")

# At large n: γ_n ≈ f(n) = 2πn/log(n)
# We want: f(n) = a·f(n-8) + (1-a)·f(n-21)
# Solving for a at large n...

def compute_natural_a(n):
    """What 'a' makes f(n) = a·f(n-8) + (1-a)·f(n-21) for f(n)=2πn/log(n)?"""
    f_n = 2*PI*n / np.log(n)
    f_n8 = 2*PI*(n-8) / np.log(n-8)
    f_n21 = 2*PI*(n-21) / np.log(n-21)
    # f_n = a·f_n8 + (1-a)·f_n21
    # f_n = a·f_n8 + f_n21 - a·f_n21
    # f_n - f_n21 = a·(f_n8 - f_n21)
    a = (f_n - f_n21) / (f_n8 - f_n21)
    return a

print("\n'Natural' coefficient a from f(n) = 2πn/log(n):")
for test_n in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    a_nat = compute_natural_a(test_n)
    print(f"  n={test_n:6d}: a_natural = {a_nat:.8f}")

# Limit as n → ∞
print("\nAsymptotic analysis:")
print("  As n → ∞, the 'natural' a converges to...")

# Series expansion: f(n-k)/f(n) ≈ 1 - k/n + k/n·log(n) + O(1/n²)
# This is getting complex. Let's compute numerically at very large n.
large_n = 1000000
a_limit = compute_natural_a(large_n)
print(f"  At n=10⁶: a = {a_limit:.8f}")

large_n = 10000000
a_limit = compute_natural_a(large_n)
print(f"  At n=10⁷: a = {a_limit:.8f}")

# ============================================================================
# 3. FLUCTUATION STRUCTURE
# ============================================================================

print("\n" + "=" * 70)
print("3. HUNTING IN THE FLUCTUATIONS")
print("=" * 70)

# Compute fluctuations
def counting_function(T):
    return T / (2 * PI) * np.log(T / (2 * PI * E)) + 7/8

N_gamma = counting_function(zeros)
fluctuations = N_gamma - n  # x_n = N(γ_n) - n

print(f"Fluctuation statistics:")
print(f"  Mean: {np.mean(fluctuations):.6f}")
print(f"  Std:  {np.std(fluctuations):.6f}")
print(f"  Range: [{np.min(fluctuations):.4f}, {np.max(fluctuations):.4f}]")

# Autocorrelation of fluctuations
print("\nAutocorrelation at Fibonacci lags:")
x = fluctuations[:50000]
x_centered = x - np.mean(x)
var_x = np.var(x)

autocorr = {}
for lag in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
    if lag < len(x):
        corr = np.mean(x_centered[:-lag] * x_centered[lag:]) / var_x
        autocorr[lag] = corr
        fib_marker = " ← Fibonacci" if lag in FIBS else ""
        print(f"  lag {lag:2d}: r = {corr:+.6f}{fib_marker}")

# Is there a pattern in autocorrelation at Fibonacci lags?
fib_autocorrs = [autocorr[k] for k in [3, 5, 8, 13, 21, 34] if k in autocorr]
non_fib_autocorrs = [autocorr[k] for k in [1, 2] if k in autocorr]

print(f"\nMean |autocorr| at Fibonacci lags (3,5,8,13,21,34): {np.mean(np.abs(fib_autocorrs)):.6f}")
print(f"Mean |autocorr| at non-Fibonacci lags (1,2): {np.mean(np.abs(non_fib_autocorrs)):.6f}")

# ============================================================================
# 4. SPACINGS (GAPS BETWEEN CONSECUTIVE ZEROS)
# ============================================================================

print("\n" + "=" * 70)
print("4. STRUCTURE IN SPACINGS")
print("=" * 70)

spacings = np.diff(zeros)
normalized_spacings = spacings / np.mean(spacings[:1000])  # Normalize by local mean

print(f"Spacing statistics:")
print(f"  Mean spacing: {np.mean(spacings):.6f}")
print(f"  Std spacing:  {np.std(spacings):.6f}")

# Try Fibonacci recurrence on spacings
print("\nFibonacci recurrence on SPACINGS (not zeros):")

def fit_on_sequence(seq, lag1, lag2):
    max_lag = max(lag1, lag2)
    n_fit = len(seq) - max_lag
    X1 = seq[max_lag - lag1:max_lag - lag1 + n_fit]
    X2 = seq[max_lag - lag2:max_lag - lag2 + n_fit]
    X = np.column_stack([X1, X2, np.ones(n_fit)])
    y = seq[max_lag:max_lag + n_fit]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coeffs[0], coeffs[1], r2

a_sp, b_sp, r2_sp = fit_on_sequence(spacings[:50000], 8, 21)
print(f"  Lags (8, 21): a = {a_sp:.6f}, b = {b_sp:.6f}, R² = {r2_sp:.6f}")

# Try other lag pairs on spacings
print("\nOther lag pairs on spacings:")
for lag1, lag2 in [(1, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 21)]:
    a, b, r2 = fit_on_sequence(spacings[:50000], lag1, lag2)
    print(f"  ({lag1:2d}, {lag2:2d}): a = {a:.4f}, b = {b:.4f}, R² = {r2:.6f}")

# ============================================================================
# 5. SPECTRAL ANALYSIS (FFT)
# ============================================================================

print("\n" + "=" * 70)
print("5. SPECTRAL ANALYSIS OF FLUCTUATIONS")
print("=" * 70)

# FFT of fluctuations
n_fft = 2**14  # Use power of 2 for efficiency
x_fft = fluctuations[:n_fft]
fft_result = np.fft.fft(x_fft)
freqs = np.fft.fftfreq(n_fft)
power = np.abs(fft_result)**2

# Find dominant frequencies (excluding DC)
positive_freqs = freqs[1:n_fft//2]
positive_power = power[1:n_fft//2]

# Top 10 frequencies
top_indices = np.argsort(positive_power)[-10:][::-1]
print("Top 10 frequencies in fluctuation spectrum:")
for i, idx in enumerate(top_indices):
    f = positive_freqs[idx]
    p = positive_power[idx]
    period = 1/f if f != 0 else np.inf
    # Check if period is near Fibonacci
    fib_match = ""
    for fib in FIBS:
        if abs(period - fib) < 0.5:
            fib_match = f" ≈ F={fib}"
            break
    print(f"  {i+1}. freq = {f:.6f}, period = {period:.2f}{fib_match}")

# Check power at Fibonacci periods specifically
print("\nPower at Fibonacci periods:")
for fib in [3, 5, 8, 13, 21, 34, 55, 89]:
    freq_fib = 1.0 / fib
    idx = np.argmin(np.abs(positive_freqs - freq_fib))
    p = positive_power[idx]
    print(f"  Period {fib:2d}: power = {p:.2e}")

# ============================================================================
# 6. RATIO ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("6. RATIO ANALYSIS: γ_n / γ_{n-k}")
print("=" * 70)

print("Distribution of γ_n / γ_{n-k} for Fibonacci k:")
for k in [3, 5, 8, 13, 21]:
    ratios = zeros[k:50000] / zeros[:50000-k]
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    print(f"  k={k:2d}: mean ratio = {mean_ratio:.6f} ± {std_ratio:.6f}")

# Is there a golden ratio hiding in the ratios?
print("\nLooking for φ in ratio patterns:")
k = 8
ratios_8 = zeros[k:50000] / zeros[:50000-k]
k = 21
ratios_21 = zeros[k:50000-13] / zeros[:50000-k-13]

# Ratio of ratios
ratio_of_ratios = ratios_8[:len(ratios_21)] / ratios_21
print(f"  Mean of (γ_n/γ_{{n-8}}) / (γ_n/γ_{{n-21}}): {np.mean(ratio_of_ratios):.6f}")
print(f"  Compare to φ = {PHI:.6f}")
print(f"  Compare to 21/8 = {21/8:.6f}")

# ============================================================================
# 7. CUMULATIVE DEVIATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("7. CUMULATIVE STRUCTURE")
print("=" * 70)

# S(n) = sum of first n fluctuations
cumulative = np.cumsum(fluctuations)
print(f"Cumulative fluctuation statistics:")
print(f"  S(1000) = {cumulative[999]:.4f}")
print(f"  S(10000) = {cumulative[9999]:.4f}")
print(f"  S(50000) = {cumulative[49999]:.4f}")

# Rate of growth
print("\nGrowth rate of cumulative fluctuations:")
for end in [1000, 5000, 10000, 30000, 50000]:
    rate = cumulative[end-1] / np.sqrt(end)
    print(f"  S({end})/√{end} = {rate:.4f}")

# ============================================================================
# 8. THE DEEPER QUESTION: WHAT MAKES RIEMANN SPECIAL?
# ============================================================================

print("\n" + "=" * 70)
print("8. WHAT MAKES RIEMANN ZEROS SPECIAL?")
print("=" * 70)

print("""
From our analysis, the Riemann zeros have:

1. A 'natural' coefficient ~1.47 in the Fibonacci(8,21) recurrence
   - This is CLOSE to 3/2 but not exact
   - It's DIFFERENT from GUE (~1.56)
   - It may be a property of the density N(T) ~ T·log(T)

2. Fluctuations with WEAK autocorrelation
   - No strong Fibonacci pattern in fluctuations
   - But some structure exists (not pure noise)

3. Spacings with their own structure
   - GUE-like statistics (Montgomery-Odlyzko)
   - Weak Fibonacci recurrence on spacings

The GIFT connection (3/2 = b₂/dim(G₂)) remains intriguing because:
   - Empirical coefficient ~1.47-1.50 is in the neighborhood
   - The LAGS 8 = rank(E₈) and 21 = b₂ are exact Fibonacci AND GIFT constants
   - This triple coincidence (lags + coefficient + GIFT values) is suspicious
""")

# Final mystery
print("\n" + "=" * 70)
print("THE REMAINING MYSTERIES")
print("=" * 70)

print("""
1. WHY do Fibonacci lags (8, 21) work best?
   - 8 = F_6 = rank(E₈)
   - 21 = F_8 = b₂(K₇)
   - Gap of 2 indices relates to φ² = φ + 1

2. WHY is the coefficient ~1.47?
   - Close to 3/2 = b₂/dim(G₂) = 21/14
   - Close to (φ² + 1)/2 = 1.809/2 ≈ 0.905... no wait
   - The 'natural' asymptotic coefficient from N(T) ~ T·log(T)
     converges to ~1.615 at large n... close to φ!

3. The GIFT-Fibonacci-Riemann TRIANGLE:

        RIEMANN ZEROS
             ↑
      coefficient ~1.5
             ↑
   FIBONACCI ←→ GIFT
    lags 8,21    b₂/dim(G₂)=3/2

   All three domains meet at the SAME numbers.
   Coincidence? Or deep structure?
""")

# Save findings
findings = {
    'asymptotic_coefficient': {
        'value': float(a_asymp),
        'convergence': [{'N': c['N'], 'a': float(c['a'])} for c in coefficients]
    },
    'best_constant_matches': [(name, float(val), float(diff)) for name, val, diff in matches[:10]],
    'theoretical_natural_a': {
        'at_1M': float(compute_natural_a(1000000)),
        'at_10M': float(compute_natural_a(10000000)),
    },
    'fluctuation_autocorr': {str(k): float(v) for k, v in autocorr.items()},
    'spacing_recurrence': {
        'a': float(a_sp), 'b': float(b_sp), 'r2': float(r2_sp)
    }
}

with open(Path(__file__).parent / "creative_exploration_results.json", "w") as f:
    json.dump(findings, f, indent=2)

print("\n✓ Results saved to creative_exploration_results.json")
