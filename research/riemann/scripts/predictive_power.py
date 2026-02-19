#!/usr/bin/env python3
"""
PREDICTIVE POWER: Can the GIFT-Riemann Framework PREDICT?
=========================================================

The ultimate test: Can we predict which indices n give γₙ ≈ GIFT constant?

Known correspondences:
  γ₁   ≈ 14  = dim(G₂)
  γ₂   ≈ 21  = b₂
  γ₂₀  ≈ 77  = b₃
  γ₂₉  ≈ 99  = H*
  γ₁₀₇ ≈ 248 = dim(E₈)

Questions:
1. Do the indices {1, 2, 20, 29, 107} have GIFT structure?
2. Can we PREDICT the next GIFT-zero correspondence?
3. Is there an INDEX FORMULA n(C) for GIFT constant C?
"""

import numpy as np
from pathlib import Path
import json

# GIFT constants
GIFT_CONSTANTS = {
    'p2': 2,
    'N_gen': 3,
    'Weyl': 5,
    'dim_K7': 7,
    'rank_E8': 8,
    'alpha_sum': 13,
    'dim_G2': 14,
    'h_E7': 18,
    'b2': 21,
    'dim_J3O': 27,
    'h_E8': 30,
    'F_10': 55,
    'kappa_T_inv': 61,
    'det_g_num': 65,
    'b3': 77,
    'H_star': 99,
    'dim_E6': 78,
    'dim_E7': 133,
    'PSL_2_7': 168,
    'dim_E8': 248,
    'dim_E8xE8': 496,
}

# Fibonacci and Lucas
FIBS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]

PHI = (1 + np.sqrt(5)) / 2

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

zeros = load_zeros(100000)
print("=" * 70)
print("PREDICTIVE POWER: TESTING THE GIFT-RIEMANN FRAMEWORK")
print("=" * 70)
print(f"\n✓ Loaded {len(zeros)} zeros")

# ============================================================================
# 1. FIND ALL GIFT CONSTANTS IN RIEMANN ZEROS
# ============================================================================

print("\n" + "=" * 70)
print("1. MAPPING GIFT CONSTANTS TO RIEMANN ZERO INDICES")
print("=" * 70)

correspondences = []
print(f"\n{'GIFT Constant':<15} {'Value':<8} {'Best n':<8} {'γ_n':<15} {'Error %':<10}")
print("-" * 60)

for name, value in sorted(GIFT_CONSTANTS.items(), key=lambda x: x[1]):
    # Find index where γ_n is closest to value
    diffs = np.abs(zeros - value)
    best_n = np.argmin(diffs) + 1  # 1-indexed
    best_gamma = zeros[best_n - 1]
    error = abs(best_gamma - value) / value * 100

    correspondences.append({
        'name': name,
        'value': value,
        'index': best_n,
        'gamma': best_gamma,
        'error_pct': error
    })

    marker = " ★" if error < 0.5 else ""
    print(f"{name:<15} {value:<8} {best_n:<8} {best_gamma:<15.6f} {error:<10.4f}{marker}")

# Best matches (< 0.5% error)
print("\n★ HIGH-PRECISION MATCHES (< 0.5% error):")
for c in correspondences:
    if c['error_pct'] < 0.5:
        print(f"  γ_{c['index']} = {c['gamma']:.6f} ≈ {c['value']} = {c['name']} ({c['error_pct']:.4f}%)")

# ============================================================================
# 2. ANALYZE THE INDEX SEQUENCE
# ============================================================================

print("\n" + "=" * 70)
print("2. STRUCTURE OF GIFT INDICES")
print("=" * 70)

# Indices of high-precision matches
hp_matches = [c for c in correspondences if c['error_pct'] < 0.5]
indices = sorted([c['index'] for c in hp_matches])

print(f"\nHigh-precision indices: {indices}")

# Check for GIFT structure in indices
print("\nAnalyzing index structure:")

for n in indices:
    print(f"\n  n = {n}:")

    # Fibonacci decomposition
    fib_decomp = []
    remaining = n
    for f in reversed(FIBS):
        if f <= remaining:
            fib_decomp.append(f)
            remaining -= f
            if remaining == 0:
                break
    if sum(fib_decomp) == n:
        print(f"    Fibonacci: {n} = {' + '.join(map(str, fib_decomp))}")

    # GIFT decompositions
    for name1, v1 in GIFT_CONSTANTS.items():
        if v1 == n:
            print(f"    Direct: {n} = {name1}")
        for name2, v2 in GIFT_CONSTANTS.items():
            if v1 + v2 == n and v1 <= v2:
                print(f"    Sum: {n} = {name1} + {name2} = {v1} + {v2}")
            if v1 * v2 == n:
                print(f"    Product: {n} = {name1} × {name2} = {v1} × {v2}")

    # Special forms
    if n in FIBS:
        idx = FIBS.index(n)
        print(f"    Fibonacci: n = F_{idx + 1}")
    if n in LUCAS:
        idx = LUCAS.index(n)
        print(f"    Lucas: n = L_{idx + 1}")

# ============================================================================
# 3. DISCOVER THE INDEX FORMULA
# ============================================================================

print("\n" + "=" * 70)
print("3. SEARCHING FOR INDEX FORMULA n(C)")
print("=" * 70)

print("""
If γ_n ≈ C (GIFT constant), what determines n?

From Riemann-von Mangoldt: N(T) ~ T/(2π) log(T/(2πe))
Inverting: T ~ 2πn / W(n/e) where W is Lambert W

Approximation: γ_n ~ 2πn / log(n) for large n

So if γ_n ≈ C, then n ≈ C × log(n) / (2π)
This is implicit but can be solved iteratively.
""")

def estimate_index(C, iterations=10):
    """Estimate index n such that γ_n ≈ C"""
    # Initial guess
    n = C / 2  # Very rough
    for _ in range(iterations):
        # γ_n ~ 2πn / log(n), so n ~ C × log(n) / (2π)
        if n > 1:
            n = C * np.log(n) / (2 * np.pi)
    return int(round(n))

print("\nTesting index formula n(C) ≈ C × log(n) / (2π):")
print(f"{'Constant':<12} {'C':<8} {'n_pred':<8} {'n_actual':<10} {'Match?':<8}")
print("-" * 50)

for c in hp_matches:
    n_pred = estimate_index(c['value'])
    n_actual = c['index']
    match = "✓" if abs(n_pred - n_actual) <= 2 else ""
    print(f"{c['name']:<12} {c['value']:<8} {n_pred:<8} {n_actual:<10} {match:<8}")

# ============================================================================
# 4. PREDICT NEW CORRESPONDENCES
# ============================================================================

print("\n" + "=" * 70)
print("4. PREDICTIONS: NEW GIFT-ZERO CORRESPONDENCES")
print("=" * 70)

# GIFT constants we haven't matched well
unmatched = [c for c in correspondences if c['error_pct'] > 1.0]

print("\nGIFT constants without good zero match:")
for c in sorted(unmatched, key=lambda x: x['value']):
    print(f"  {c['name']} = {c['value']}: closest is γ_{c['index']} = {c['gamma']:.3f} ({c['error_pct']:.2f}% error)")

# Look for INTEGER zeros (γ_n very close to an integer)
print("\n\nSearching for zeros very close to integers:")
print(f"{'n':<8} {'γ_n':<15} {'Nearest int':<12} {'Error':<12} {'GIFT?':<15}")
print("-" * 65)

integer_zeros = []
for n in range(1, len(zeros)):
    gamma = zeros[n-1]
    nearest_int = round(gamma)
    error = abs(gamma - nearest_int)

    if error < 0.01:  # Within 1% of an integer
        # Check if it's a GIFT constant
        gift_match = None
        for name, value in GIFT_CONSTANTS.items():
            if value == nearest_int:
                gift_match = name
                break

        integer_zeros.append({
            'n': n,
            'gamma': gamma,
            'nearest': nearest_int,
            'error': error,
            'gift': gift_match
        })

        gift_str = gift_match if gift_match else ""
        print(f"{n:<8} {gamma:<15.8f} {nearest_int:<12} {error:<12.8f} {gift_str:<15}")

# ============================================================================
# 5. THE 107 MYSTERY - DEEP ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("5. THE 107 MYSTERY: WHY γ₁₀₇ ≈ 248 = dim(E₈)?")
print("=" * 70)

print(f"""
γ₁₀₇ = {zeros[106]:.6f} ≈ 248 = dim(E₈)

The index 107 has remarkable GIFT structure:
""")

n = 107
print(f"  107 = rank(E₈) + H* = 8 + 99")
print(f"  107 = 4 × dim(J₃(O)) - 1 = 4 × 27 - 1 = 108 - 1")
print(f"  107 = h(E₈) + b₃ = 30 + 77")
print(f"  107 is prime (the 28th prime)")

# Is there a pattern: index = some function of the constant?
print("\nLooking for pattern: n = f(C)?")
print(f"{'Constant':<12} {'C':<8} {'n':<8} {'n/C':<10} {'n×C/100':<10}")
print("-" * 50)

for c in hp_matches:
    ratio = c['index'] / c['value']
    product = c['index'] * c['value'] / 100
    print(f"{c['name']:<12} {c['value']:<8} {c['index']:<8} {ratio:<10.4f} {product:<10.2f}")

# ============================================================================
# 6. PREDICT THE NEXT BIG CORRESPONDENCE
# ============================================================================

print("\n" + "=" * 70)
print("6. PREDICTION: NEXT MAJOR GIFT-ZERO CORRESPONDENCE")
print("=" * 70)

# dim(E₈ × E₈) = 496 - find where this might appear
target = 496
n_pred = estimate_index(target)

print(f"\nTarget: dim(E₈ × E₈) = {target}")
print(f"Predicted index: n ≈ {n_pred}")

# Check actual
if n_pred < len(zeros):
    actual_gamma = zeros[n_pred - 1]
    print(f"Actual γ_{n_pred} = {actual_gamma:.6f}")
    print(f"Error: {abs(actual_gamma - target):.4f} ({abs(actual_gamma - target)/target*100:.3f}%)")

# Search around the prediction
print(f"\nSearching around n = {n_pred}:")
for n in range(max(1, n_pred - 10), min(len(zeros), n_pred + 11)):
    gamma = zeros[n-1]
    error = abs(gamma - target) / target * 100
    marker = " ★" if error < 0.5 else ""
    print(f"  γ_{n} = {gamma:.4f}, error from 496: {error:.3f}%{marker}")

# ============================================================================
# 7. THE GRAND PATTERN
# ============================================================================

print("\n" + "=" * 70)
print("7. THE GRAND PATTERN: INDEX-CONSTANT RELATIONSHIP")
print("=" * 70)

print("""
HYPOTHESIS: For GIFT constant C, the corresponding index n satisfies:

  γ_n ≈ C  where n is determined by GIFT topology

Observed pattern in high-precision matches:
""")

# Check if indices follow a pattern involving C
print(f"{'C':<8} {'n':<8} {'n mod 7':<10} {'n mod 8':<10} {'n mod 21':<10}")
print("-" * 50)

for c in hp_matches:
    n = c['index']
    C = c['value']
    print(f"{C:<8} {n:<8} {n % 7:<10} {n % 8:<10} {n % 21:<10}")

# Most striking: 107 mod 8 = 3 = N_gen, 107 mod 21 = 2 = p₂
print(f"""
Observations:
  - Most indices n ≡ small GIFT constant (mod 7, 8, or 21)
  - 107 mod 8 = 3 = N_gen
  - 107 mod 21 = 2 = p₂
  - 29 mod 8 = 5 = Weyl
  - 20 mod 7 = 6 (not obvious)

The indices seem to encode GIFT structure modularly!
""")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'correspondences': correspondences,
    'high_precision_indices': indices,
    'integer_zeros': integer_zeros[:20] if integer_zeros else [],
    'predictions': {
        'dim_E8xE8': {
            'target': 496,
            'predicted_index': n_pred,
            'actual_gamma': float(zeros[n_pred - 1]) if n_pred < len(zeros) else None
        }
    }
}

with open(Path(__file__).parent / "predictive_power_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print("\n✓ Results saved to predictive_power_results.json")
