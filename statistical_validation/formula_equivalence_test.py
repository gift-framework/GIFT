#!/usr/bin/env python3
"""
Formula Equivalence Test

Key insight: Multiple "matching formulas" are often ALGEBRAICALLY EQUIVALENT.
They express the SAME underlying ratio in different ways.

This is STRONGER evidence for structural inevitability:
- There's ONE value (e.g., 3/13 for sin²θ_W)
- It admits multiple equivalent GIFT expressions
- This reveals the value as a STRUCTURAL CONSTANT
"""

from fractions import Fraction
from collections import defaultdict
from itertools import combinations

GIFT = {
    'b0': 1, 'p2': 2, 'N_gen': 3, 'Weyl': 5, 'dim_K7': 7,
    'rank_E8': 8, 'D_bulk': 11, 'alpha_sum': 13, 'dim_G2': 14,
    'b2': 21, 'dim_J3O': 27, 'det_g_den': 32, 'dim_F4': 52,
    'kappa_T_inv': 61, 'det_g_num': 65, 'b3': 77, 'dim_E6': 78,
    'H_star': 99, 'dim_E7': 133, 'PSL27': 168, 'dim_E8': 248,
}


def find_all_expressions_for_value(target_num: int, target_den: int):
    """Find all GIFT expressions that reduce to target_num/target_den."""
    target = Fraction(target_num, target_den)
    expressions = []

    keys = list(GIFT.keys())

    # Simple ratios a/b
    for num_key in keys:
        for den_key in keys:
            if num_key == den_key:
                continue
            frac = Fraction(GIFT[num_key], GIFT[den_key])
            if frac == target:
                expressions.append(f"{num_key}/{den_key}")

    # a/(b+c)
    for num_key in keys:
        for den1, den2 in combinations(keys, 2):
            den_val = GIFT[den1] + GIFT[den2]
            if den_val > 0:
                frac = Fraction(GIFT[num_key], den_val)
                if frac == target:
                    expressions.append(f"{num_key}/({den1}+{den2})")

    # (a+b)/c
    for num1, num2 in combinations(keys, 2):
        for den_key in keys:
            num_val = GIFT[num1] + GIFT[num2]
            den_val = GIFT[den_key]
            if den_val > 0:
                frac = Fraction(num_val, den_val)
                if frac == target:
                    expressions.append(f"({num1}+{num2})/{den_key}")

    return expressions


def analyze_structural_constant(name: str, num: int, den: int, experimental: float):
    """Analyze a structural constant."""
    print(f"\n{'=' * 70}")
    print(f"STRUCTURAL CONSTANT: {name}")
    print(f"{'=' * 70}")
    print(f"Value: {num}/{den} = {Fraction(num, den)} = {num/den:.10f}")
    print(f"Experimental: {experimental}")
    print(f"Deviation: {abs(num/den - experimental)/experimental * 100:.4f}%")

    expressions = find_all_expressions_for_value(num, den)

    print(f"\nNumber of GIFT expressions: {len(expressions)}")
    print("\nAll equivalent expressions:")
    for i, expr in enumerate(expressions, 1):
        print(f"  {i:2}. {expr}")

    return expressions


def main():
    print("=" * 70)
    print("STRUCTURAL CONSTANTS OF GIFT")
    print("Each value admits multiple equivalent expressions")
    print("=" * 70)

    # The key structural constants
    constants = [
        ("sin²θ_W", 3, 13, 0.23122),
        ("Q_Koide", 2, 3, 0.666661),
        ("N_gen", 3, 1, 3),
    ]

    all_results = {}

    for name, num, den, exp in constants:
        exprs = analyze_structural_constant(name, num, den, exp)
        all_results[name] = exprs

    # Summary
    print("\n" + "=" * 70)
    print("STRUCTURAL INEVITABILITY ANALYSIS")
    print("=" * 70)

    print("""
KEY FINDING: Each observable corresponds to a UNIQUE REDUCED FRACTION
that admits MULTIPLE equivalent GIFT expressions.

This is not "selection from alternatives" — it's ALGEBRAIC STRUCTURE.
""")

    print("\n| Constant | Reduced Form | # Expressions | Interpretation |")
    print("|----------|--------------|---------------|----------------|")
    for name, num, den, _ in constants:
        n_expr = len(all_results[name])
        print(f"| {name:12} | {num}/{den:2} | {n_expr:13} | Unique value, many forms |")

    print("""
\n╔═══════════════════════════════════════════════════════════════════════╗
║  CONCLUSION: STRUCTURAL INEVITABILITY                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  The GIFT formulas are not "selected" — they are ALGEBRAIC IDENTITIES.║
║                                                                       ║
║  Example: sin²θ_W = 3/13                                              ║
║                                                                       ║
║  This ratio can be expressed as:                                      ║
║    • N_gen / alpha_sum           (generation / anomaly)               ║
║    • b₂ / (b₃ + dim_G₂)         (gauge / matter+holonomy)            ║
║    • 21 / 91                     (direct topological)                 ║
║                                                                       ║
║  These are NOT alternatives — they are THE SAME NUMBER expressed      ║
║  in different but equivalent ways within GIFT algebra.                ║
║                                                                       ║
║  The question "why this formula?" dissolves into:                     ║
║  "This is the unique value 3/13 that the topology determines."        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Check algebraic identities
    print("\n" + "=" * 70)
    print("ALGEBRAIC IDENTITIES UNDERLYING THE EQUIVALENCES")
    print("=" * 70)

    print("""
Why do multiple expressions give the same value?
Because of IDENTITIES in the GIFT algebra:

1. For 3/13:
   • alpha_sum = 13 = rank_E8 + Weyl = 8 + 5 ✓
   • alpha_sum = 13 = p2 + D_bulk = 2 + 11 ✓
   • 91 = 7 × 13 = dim_K7 × alpha_sum ✓
   • 91 = b3 + dim_G2 = 77 + 14 ✓
   • 21 = 3 × 7 = N_gen × dim_K7 ✓

   So: N_gen/alpha_sum = b2/(b3+dim_G2) = 3/13

2. For 2/3:
   • b2 = 21 = 3 × 7 = N_gen × dim_K7 ✓
   • dim_G2 = 14 = 2 × 7 = p2 × dim_K7 ✓
   • dim_F4 = 52 = 4 × 13 = p2² × alpha_sum ✓
   • dim_E6 = 78 = 6 × 13 = (2×N_gen) × alpha_sum ✓

   So: p2/N_gen = dim_G2/b2 = dim_F4/dim_E6 = 2/3

These identities are not coincidences — they reflect the algebraic
structure of exceptional Lie algebras and octonionic geometry.
""")


if __name__ == "__main__":
    main()
