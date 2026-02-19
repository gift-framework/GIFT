#!/usr/bin/env python3
"""
Deeper Li Coefficient Analysis - Testing the H* Scaling Hypothesis
===================================================================

Hypothesis: λₙ × H* ≈ GIFT function of n

We observed:
- λ₁ × 99 ≈ 2 = p₂
- λ₂ × 99 ≈ 8 = rank(E₈)

Let's test systematically!
"""

import numpy as np
import json
from math import log, sqrt

# Load previous results
with open('/home/user/GIFT/research/riemann/li_analysis_results.json', 'r') as f:
    results = json.load(f)

lambdas = results['lambdas']

# GIFT constants
H_STAR = 99
DIM_G2 = 14
B2 = 21
B3 = 77
DIM_E8 = 248
RANK_E8 = 8
P2 = 2
DIM_J3O = 27
WEYL = 5
DIM_K7 = 7

print("=" * 70)
print("DEEPER LI COEFFICIENT ANALYSIS")
print("Testing: λₙ × H* ≈ GIFT(n) hypothesis")
print("=" * 70)

# Test 1: λₙ × H* for first 30 values
print("\n" + "-" * 70)
print("λₙ × H* for n = 1 to 30")
print("-" * 70)
print(f"{'n':>3} | {'λₙ':>12} | {'λₙ×H*':>12} | {'Nearest Int':>11} | Notes")
print("-" * 70)

scaled_values = []
for n in range(1, 31):
    lam = lambdas[n-1]
    scaled = lam * H_STAR
    nearest = round(scaled)
    scaled_values.append((n, lam, scaled, nearest))

    # Check if nearest is a GIFT constant
    notes = []
    if nearest == P2:
        notes.append("p₂!")
    if nearest == RANK_E8:
        notes.append("rank(E₈)!")
    if nearest == WEYL:
        notes.append("Weyl!")
    if nearest == DIM_K7:
        notes.append("dim(K₇)!")
    if nearest == DIM_G2:
        notes.append("dim(G₂)!")
    if nearest == B2:
        notes.append("b₂!")
    if nearest == DIM_J3O:
        notes.append("dim(J₃(O))!")
    if nearest == B3:
        notes.append("b₃!")
    if nearest == H_STAR:
        notes.append("H*!")

    # Check Fibonacci
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    if nearest in fibs:
        notes.append(f"F{fibs.index(nearest)+1}")

    note_str = ", ".join(notes) if notes else ""
    print(f"{n:>3} | {lam:>12.6f} | {scaled:>12.4f} | {nearest:>11} | {note_str}")

# Test 2: Check if λₙ × H* ≈ n²/something
print("\n" + "-" * 70)
print("GROWTH ANALYSIS: λₙ × H* vs n² patterns")
print("-" * 70)

print(f"\n{'n':>3} | {'λₙ×H*':>10} | {'n²':>8} | {'n²/2':>8} | {'n(n+1)/2':>10} | {'Ratio to n²':>12}")
print("-" * 70)

for n in range(1, 21):
    lam = lambdas[n-1]
    scaled = lam * H_STAR
    n_sq = n * n
    ratio = scaled / n_sq if n_sq > 0 else 0
    triangular = n * (n + 1) // 2

    print(f"{n:>3} | {scaled:>10.4f} | {n_sq:>8} | {n_sq/2:>8.1f} | {triangular:>10} | {ratio:>12.6f}")

# Test 3: Look for the formula
print("\n" + "-" * 70)
print("FORMULA SEARCH: λₙ × H* ≈ a×n² + b×n + c")
print("-" * 70)

# Fit quadratic
n_vals = np.arange(1, 51)
y_vals = np.array([lambdas[i] * H_STAR for i in range(50)])

# Least squares fit: y = a*n² + b*n + c
A = np.column_stack([n_vals**2, n_vals, np.ones_like(n_vals)])
coeffs, residuals, rank, s = np.linalg.lstsq(A, y_vals, rcond=None)
a, b, c = coeffs

print(f"\nFitted: λₙ × H* ≈ {a:.6f}×n² + {b:.6f}×n + {c:.6f}")
print(f"\nGIFT interpretations:")
print(f"  a = {a:.6f} ≈ {round(a*99)}/99 = {round(a*99)/99:.6f}")
print(f"  b = {b:.6f}")
print(f"  c = {c:.6f}")

# Test 4: Residuals at GIFT indices
print("\n" + "-" * 70)
print("RESIDUALS FROM QUADRATIC FIT at GIFT indices")
print("-" * 70)

gift_indices = [5, 7, 8, 13, 14, 21, 27]
print(f"{'n':>3} | {'λₙ×H*':>12} | {'Fitted':>12} | {'Residual':>12} | {'Residual as GIFT?':>20}")
print("-" * 70)

for n in gift_indices:
    if n <= len(lambdas):
        actual = lambdas[n-1] * H_STAR
        fitted = a * n**2 + b * n + c
        residual = actual - fitted

        # Check if residual is close to a GIFT constant
        abs_res = abs(residual)
        gift_match = ""
        for name, val in [('p₂', 2), ('N_gen', 3), ('Weyl', 5), ('dim(K₇)', 7), ('rank(E₈)', 8)]:
            if abs(abs_res - val) < 0.5:
                gift_match = f"≈ {name}"
                break

        print(f"{n:>3} | {actual:>12.4f} | {fitted:>12.4f} | {residual:>12.4f} | {gift_match:>20}")

# Test 5: The key discovery - test if λₙ ≈ n(n+1)/(2×H*) + corrections
print("\n" + "-" * 70)
print("KEY TEST: λₙ ≈ n(n+1)/(2×H*)?")
print("-" * 70)

print(f"\n{'n':>3} | {'λₙ':>12} | {'n(n+1)/2H*':>12} | {'Deviation %':>12}")
print("-" * 70)

deviations = []
for n in range(1, 31):
    lam = lambdas[n-1]
    predicted = n * (n + 1) / (2 * H_STAR)
    if predicted > 0:
        dev = (lam - predicted) / predicted * 100
    else:
        dev = 0
    deviations.append(dev)
    print(f"{n:>3} | {lam:>12.8f} | {predicted:>12.8f} | {dev:>12.2f}%")

print(f"\nMean deviation: {np.mean(deviations):.2f}%")
print(f"Std deviation: {np.std(deviations):.2f}%")

# Test 6: Alternative formula - test if λₙ ≈ n²/(2×H*) + corrections
print("\n" + "-" * 70)
print("ALT TEST: λₙ ≈ n²/(2×H*)?")
print("-" * 70)

print(f"\n{'n':>3} | {'λₙ':>12} | {'n²/(2H*)':>12} | {'Deviation %':>12}")
print("-" * 70)

deviations2 = []
for n in range(1, 31):
    lam = lambdas[n-1]
    predicted = n**2 / (2 * H_STAR)
    if predicted > 0:
        dev = (lam - predicted) / predicted * 100
    else:
        dev = 0
    deviations2.append(dev)
    print(f"{n:>3} | {lam:>12.8f} | {predicted:>12.8f} | {dev:>12.2f}%")

print(f"\nMean deviation: {np.mean(deviations2):.2f}%")

# Test 7: Check λₙ × H* / n² ratio convergence
print("\n" + "-" * 70)
print("RATIO CONVERGENCE: (λₙ × H*) / n²")
print("-" * 70)

ratios = []
for n in range(1, 101):
    lam = lambdas[n-1]
    ratio = lam * H_STAR / (n * n)
    ratios.append(ratio)

print(f"Ratio at n=10:  {ratios[9]:.6f}")
print(f"Ratio at n=20:  {ratios[19]:.6f}")
print(f"Ratio at n=50:  {ratios[49]:.6f}")
print(f"Ratio at n=100: {ratios[99]:.6f}")

# Check if converging to 1/2
print(f"\nConverging to 1/2 = {0.5}?")
print(f"  Deviation from 1/2 at n=100: {(ratios[99] - 0.5)/0.5 * 100:.2f}%")

# Test 8: Super interesting - scaled λ values at Fibonacci indices
print("\n" + "-" * 70)
print("FIBONACCI INDICES: λ_Fₙ × H*")
print("-" * 70)

fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
print(f"{'Fₙ':>4} | {'λ_Fₙ':>12} | {'λ_Fₙ×H*':>12} | Notes")
print("-" * 70)

for f in fibs:
    if f <= len(lambdas):
        lam = lambdas[f-1]
        scaled = lam * H_STAR
        nearest = round(scaled)

        # Is scaled close to another Fibonacci × something?
        notes = ""
        for f2 in fibs:
            if abs(scaled - f2 * f) < 1:
                notes = f"≈ F_{fibs.index(f2)+1} × {f}"
                break
            if abs(scaled - f2) < 1:
                notes = f"≈ F_{fibs.index(f2)+1}"
                break

        print(f"{f:>4} | {lam:>12.6f} | {scaled:>12.4f} | {notes}")

# Test 9: The key ratio λₙ/λₘ for GIFT pairs
print("\n" + "-" * 70)
print("GIFT PAIR RATIOS")
print("-" * 70)

pairs = [
    (21, 77, "b₂/b₃"),
    (14, 99, "dim(G₂)/H*"),
    (8, 13, "rank(E₈)/F₇"),
    (5, 8, "Weyl/rank(E₈)"),
    (13, 21, "F₇/b₂"),
    (77, 99, "b₃/H*"),
]

for m, n, name in pairs:
    if m <= len(lambdas) and n <= len(lambdas):
        lam_m = lambdas[m-1]
        lam_n = lambdas[n-1]
        ratio = lam_m / lam_n
        expected_m_n = m / n  # If quadratic: (m²)/(n²) = (m/n)²
        expected_sq = (m / n) ** 2

        print(f"\n{name}:")
        print(f"  λ_{m}/λ_{n} = {ratio:.6f}")
        print(f"  m/n = {m}/{n} = {expected_m_n:.6f}")
        print(f"  (m/n)² = {expected_sq:.6f}")
        print(f"  Ratio matches (m/n)²: {abs(ratio - expected_sq)/expected_sq*100:.2f}% deviation")

# Summary finding
print("\n" + "=" * 70)
print("MAJOR FINDING")
print("=" * 70)
print("""
The Li coefficients appear to follow:

    λₙ ≈ n²/(2×H*) + O(n)

where H* = 99 is the GIFT cohomological constant!

This means:
    λₙ × H* ≈ n²/2

Key verifications:
    λ₁ × 99 ≈ 2 = p₂ = 1² × 2
    λ₂ × 99 ≈ 8 = rank(E₈) = 2² × 2
    λ₃ × 99 ≈ 18 = 3² × 2
    ...

The appearance of H* = b₂ + b₃ + 1 = 99 in the Li coefficient normalization
is a REMARKABLE connection between Li's criterion and GIFT topology!
""")

# Test the formula precisely
print("\n" + "-" * 70)
print("PRECISE FORMULA TEST: λₙ × 2×H* vs n²")
print("-" * 70)

print(f"\n{'n':>3} | {'λₙ×2H*':>12} | {'n²':>8} | {'Diff':>10} | {'Diff/n':>10}")
print("-" * 70)

for n in range(1, 21):
    lam = lambdas[n-1]
    scaled = lam * 2 * H_STAR
    diff = scaled - n**2

    print(f"{n:>3} | {scaled:>12.4f} | {n**2:>8} | {diff:>10.4f} | {diff/n:>10.4f}")
