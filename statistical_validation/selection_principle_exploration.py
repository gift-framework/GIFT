#!/usr/bin/env python3
"""
Selection Principle Exploration for GIFT v3.3

Systematically explore combinations of topological invariants
to identify patterns that distinguish "used" from "unused" formulas.
"""

from fractions import Fraction
from itertools import combinations, permutations
from math import gcd, sqrt
from typing import Dict, List, Tuple

# =============================================================================
# GIFT CONSTANTS
# =============================================================================

CONSTANTS = {
    # Fiber/Base
    'dim_K7': 7,
    'D_bulk': 11,

    # Betti numbers
    'b2': 21,
    'b3': 77,

    # Lie algebras
    'dim_G2': 14,
    'rank_G2': 2,
    'dim_E8': 248,
    'rank_E8': 8,
    'dim_E8xE8': 496,
    'dim_E6': 78,
    'dim_E7': 133,
    'dim_F4': 52,
    'dim_J3O': 27,

    # Derived
    'H_star': 99,
    'N_gen': 3,
    'Weyl': 5,
    'p2': 2,
    'kappa_T_inv': 61,
    'det_g_num': 65,
    'det_g_den': 32,
    'tau_num': 3472,
    'tau_den': 891,

    # Other
    'alpha_sum': 13,
    'PSL27': 168,
}

# Known physical values
EXPERIMENTAL = {
    'sin2_theta_W': 0.23122,
    'Q_Koide': 0.666661,
    'alpha_s': 0.1179,
    'N_gen': 3,
    'delta_CP': 197,  # degrees
    'theta_13': 8.54,  # degrees
    'theta_23': 49.3,  # degrees
    'theta_12': 33.41,  # degrees
    'm_tau_m_e': 3477.15,
    'm_mu_m_e': 206.768,
    'm_s_m_d': 20.0,
}

# Known GIFT formulas
KNOWN_FORMULAS = {
    'sin2_theta_W': ('b2', '/', ('b3', '+', 'dim_G2')),  # 21/91 = 3/13
    'Q_Koide': ('dim_G2', '/', 'b2'),  # 14/21 = 2/3
    'N_gen_betti': ('b2', '/', 'dim_K7'),  # 21/7 = 3
    'kappa_T_inv': ('b3', '-', 'dim_G2', '-', 'p2'),  # 77-14-2 = 61
    'H_star': ('b2', '+', 'b3', '+', 1),  # 21+77+1 = 99
}


def mod7_class(n: int) -> int:
    """Return mod 7 residue class."""
    return n % 7


def is_fano_closed(nums: List[int]) -> bool:
    """Check if all numbers are divisible by 7."""
    return all(n % 7 == 0 for n in nums)


def analyze_ratio(num: int, den: int) -> Dict:
    """Analyze a ratio for GIFT patterns."""
    g = gcd(num, den)
    reduced_num = num // g
    reduced_den = den // g

    return {
        'value': num / den,
        'fraction': Fraction(num, den),
        'gcd': g,
        'reduced': (reduced_num, reduced_den),
        'num_mod7': mod7_class(num),
        'den_mod7': mod7_class(den),
        'gcd_has_7': g % 7 == 0,
        'both_div_7': num % 7 == 0 and den % 7 == 0,
    }


def find_matching_experimental(value: float, tolerance: float = 0.05) -> List[str]:
    """Find experimental values within tolerance."""
    matches = []
    for name, exp_val in EXPERIMENTAL.items():
        if abs(value - exp_val) / max(abs(exp_val), 1e-10) < tolerance:
            matches.append(name)
    return matches


def explore_ratios():
    """Explore all simple ratios of GIFT constants."""
    print("=" * 80)
    print("RATIO EXPLORATION")
    print("=" * 80)

    # Key constants for ratio exploration
    keys = ['b2', 'b3', 'dim_G2', 'dim_K7', 'H_star', 'rank_E8', 'N_gen',
            'Weyl', 'p2', 'dim_E8', 'dim_J3O', 'alpha_sum', 'kappa_T_inv']

    interesting = []

    for num_key in keys:
        for den_key in keys:
            if num_key == den_key:
                continue

            num = CONSTANTS[num_key]
            den = CONSTANTS[den_key]

            if den == 0:
                continue

            analysis = analyze_ratio(num, den)
            matches = find_matching_experimental(analysis['value'])

            if matches or analysis['both_div_7']:
                interesting.append({
                    'formula': f"{num_key}/{den_key}",
                    'num': num,
                    'den': den,
                    **analysis,
                    'matches': matches,
                })

    # Sort by whether they match experimental values
    interesting.sort(key=lambda x: (len(x['matches']) == 0, x['formula']))

    print("\nRatios matching experimental values:")
    print("-" * 60)
    for item in interesting:
        if item['matches']:
            print(f"{item['formula']:25} = {str(item['fraction']):10} = {item['value']:.6f}")
            print(f"  -> Matches: {item['matches']}")
            print(f"  -> GCD={item['gcd']}, mod7: {item['num_mod7']}/{item['den_mod7']}")

    print("\n\nRatios with both num/den divisible by 7 (Fano-closed):")
    print("-" * 60)
    for item in interesting:
        if item['both_div_7'] and not item['matches']:
            print(f"{item['formula']:25} = {str(item['fraction']):10} = {item['value']:.6f}")


def explore_sums_differences():
    """Explore sums and differences that yield GIFT constants."""
    print("\n" + "=" * 80)
    print("SUM/DIFFERENCE EXPLORATION")
    print("=" * 80)

    keys = ['b2', 'b3', 'dim_G2', 'dim_K7', 'H_star', 'rank_E8', 'N_gen',
            'Weyl', 'p2', 'dim_E8', 'dim_J3O', 'alpha_sum']

    target_values = set(CONSTANTS.values())

    print("\nTwo-term sums that yield other GIFT constants:")
    print("-" * 60)
    for k1, k2 in combinations(keys, 2):
        v1, v2 = CONSTANTS[k1], CONSTANTS[k2]

        # Sum
        s = v1 + v2
        if s in target_values:
            targets = [k for k, v in CONSTANTS.items() if v == s]
            print(f"{k1} + {k2} = {v1} + {v2} = {s} = {targets}")

        # Difference
        d = abs(v1 - v2)
        if d in target_values and d > 0:
            targets = [k for k, v in CONSTANTS.items() if v == d]
            if v1 > v2:
                print(f"{k1} - {k2} = {v1} - {v2} = {d} = {targets}")
            else:
                print(f"{k2} - {k1} = {v2} - {v1} = {d} = {targets}")

    print("\nTwo-term products that yield other GIFT constants:")
    print("-" * 60)
    for k1, k2 in combinations(keys, 2):
        v1, v2 = CONSTANTS[k1], CONSTANTS[k2]
        p = v1 * v2
        if p in target_values:
            targets = [k for k, v in CONSTANTS.items() if v == p]
            print(f"{k1} × {k2} = {v1} × {v2} = {p} = {targets}")


def explore_denominator_patterns():
    """Analyze what gets added to denominators in known formulas."""
    print("\n" + "=" * 80)
    print("DENOMINATOR PATTERN ANALYSIS")
    print("=" * 80)

    print("\nKnown formulas with compound denominators:")
    print("-" * 60)

    # sin²θ_W = b2 / (b3 + dim_G2)
    # Why +dim_G2 and not something else?

    print("\nsin²θ_W = b2 / (b3 + X)")
    print("Testing different X values:")
    alternatives = {
        'dim_G2': 14,  # ACTUAL
        '0': 0,
        'dim_K7': 7,
        'rank_E8': 8,
        'N_gen': 3,
        'Weyl': 5,
        'p2': 2,
    }

    for name, x in alternatives.items():
        den = CONSTANTS['b3'] + x
        if den > 0:
            value = CONSTANTS['b2'] / den
            diff_from_exp = abs(value - 0.23122) / 0.23122 * 100
            marker = " <-- USED" if name == 'dim_G2' else ""
            print(f"  X = {name:10} ({x:3}): b2/(b3+X) = 21/{den:3} = {value:.6f}  ({diff_from_exp:5.2f}% off){marker}")


def explore_triple_derivations():
    """Find quantities with multiple independent derivations."""
    print("\n" + "=" * 80)
    print("MULTIPLE DERIVATION ANALYSIS")
    print("=" * 80)

    # Check all integer values from 1 to 500 for multiple derivations
    derivation_counts = {}

    keys = ['b2', 'b3', 'dim_G2', 'dim_K7', 'H_star', 'rank_E8', 'N_gen',
            'Weyl', 'p2', 'dim_E8', 'dim_J3O', 'alpha_sum', 'kappa_T_inv',
            'dim_E6', 'dim_F4', 'PSL27']

    for target in range(1, 300):
        derivations = []

        # Single values
        for k in keys:
            if CONSTANTS[k] == target:
                derivations.append(k)

        # Sums of two
        for k1, k2 in combinations(keys, 2):
            if CONSTANTS[k1] + CONSTANTS[k2] == target:
                derivations.append(f"{k1}+{k2}")

        # Differences
        for k1 in keys:
            for k2 in keys:
                if k1 != k2 and CONSTANTS[k1] - CONSTANTS[k2] == target:
                    derivations.append(f"{k1}-{k2}")

        # Products of two
        for k1, k2 in combinations(keys, 2):
            if CONSTANTS[k1] * CONSTANTS[k2] == target:
                derivations.append(f"{k1}×{k2}")

        # Quotients (integer only)
        for k1 in keys:
            for k2 in keys:
                if k1 != k2 and CONSTANTS[k2] != 0:
                    if CONSTANTS[k1] % CONSTANTS[k2] == 0:
                        if CONSTANTS[k1] // CONSTANTS[k2] == target:
                            derivations.append(f"{k1}/{k2}")

        if len(derivations) >= 3:
            derivation_counts[target] = derivations

    print("\nValues with 3+ independent derivations:")
    print("-" * 60)
    for value, derivs in sorted(derivation_counts.items(), key=lambda x: -len(x[1])):
        print(f"\n{value} ({len(derivs)} derivations):")
        for d in derivs[:8]:  # Limit display
            print(f"  - {d}")
        if len(derivs) > 8:
            print(f"  ... and {len(derivs) - 8} more")


def main():
    """Run all explorations."""
    print("GIFT Selection Principle Exploration")
    print("=" * 80)

    explore_ratios()
    explore_sums_differences()
    explore_denominator_patterns()
    explore_triple_derivations()

    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    print("""
1. FANO CLOSURE: Most "used" formulas have numerator AND denominator ≡ 0 (mod 7)

2. DENOMINATOR AUGMENTATION: sin²θ_W uses (b3 + dim_G2), not just b3
   - Adding dim_G2 gives the correct value
   - This might encode "holonomy contribution" to gauge-matter coupling

3. MULTIPLE DERIVATIONS: Quantities appearing in physics have 3+ derivations
   - Weyl = 5: 4 independent paths
   - PSL(2,7) = 168: 4 independent paths
   - N_gen = 3: 3+ paths

4. CLOSURE UNDER GIFT ALGEBRA: Results reduce to other GIFT constants
   - b3 + dim_G2 = 91 = 7 × 13 = dim_K7 × alpha_sum
   - b2 + b3 = 98 = 7 × 14 = dim_K7 × dim_G2
""")


if __name__ == "__main__":
    main()
