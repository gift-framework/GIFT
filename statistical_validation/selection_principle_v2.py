#!/usr/bin/env python3
"""
Selection Principle v2: GIFT-Expressible Reduced Fractions

Hypothesis: A formula is "admissible" if its reduced form
has BOTH numerator AND denominator expressible as GIFT constants.
"""

from fractions import Fraction
from math import gcd

# GIFT integer constants
GIFT_INTEGERS = {
    1: "b0 (connected)",
    2: "p2, rank_G2",
    3: "N_gen",
    5: "Weyl",
    7: "dim_K7",
    8: "rank_E8",
    11: "D_bulk",
    13: "alpha_sum",
    14: "dim_G2",
    21: "b2",
    27: "dim_J3O",
    32: "det_g_den = 2^Weyl",
    52: "dim_F4",
    56: "fund_E7 = b3 - b2",
    61: "kappa_T_inv",
    65: "det_g_num",
    77: "b3",
    78: "dim_E6",
    91: "b3 + dim_G2",
    99: "H_star",
    133: "dim_E7",
    168: "PSL27",
    248: "dim_E8",
    496: "dim_E8xE8",
}

KNOWN_FORMULAS = {
    "sin2_theta_W": {
        "formula": "b2/(b3+dim_G2)",
        "num": 21,
        "den": 91,
        "reduced": (3, 13),
        "experimental": 0.23122,
    },
    "Q_Koide": {
        "formula": "dim_G2/b2",
        "num": 14,
        "den": 21,
        "reduced": (2, 3),
        "experimental": 0.666661,
    },
    "N_gen_formula": {
        "formula": "b2/dim_K7",
        "num": 21,
        "den": 7,
        "reduced": (3, 1),
        "experimental": 3,
    },
}


def is_gift_expressible(n: int) -> tuple:
    """Check if integer is a GIFT constant or simple product."""
    if n in GIFT_INTEGERS:
        return True, GIFT_INTEGERS[n]

    # Check products of two small GIFT constants
    small = [2, 3, 5, 7, 8, 11, 13]
    for a in small:
        if n % a == 0:
            b = n // a
            if b in GIFT_INTEGERS:
                return True, f"{a} × {b} ({GIFT_INTEGERS.get(a, a)} × {GIFT_INTEGERS[b]})"

    return False, None


def analyze_formula(name: str, num: int, den: int):
    """Analyze if a formula satisfies the GIFT-expressibility criterion."""
    g = gcd(num, den)
    red_num = num // g
    red_den = den // g

    num_gift, num_expr = is_gift_expressible(red_num)
    den_gift, den_expr = is_gift_expressible(red_den)

    print(f"\n{name}:")
    print(f"  Original: {num}/{den}")
    print(f"  Reduced:  {red_num}/{red_den} = {num/den:.6f}")
    print(f"  Numerator {red_num}: {'✓ GIFT' if num_gift else '✗ NOT GIFT'} - {num_expr}")
    print(f"  Denominator {red_den}: {'✓ GIFT' if den_gift else '✗ NOT GIFT'} - {den_expr}")
    print(f"  ADMISSIBLE: {'✓ YES' if (num_gift and den_gift) else '✗ NO'}")

    return num_gift and den_gift


def test_alternative_formulas():
    """Test alternative formulas that are NOT used."""
    print("\n" + "=" * 70)
    print("TESTING ALTERNATIVE (UNUSED) FORMULAS")
    print("=" * 70)

    alternatives = {
        "b2/b3 (NOT USED for sin2_theta_W)": (21, 77),
        "dim_G2/b3": (14, 77),
        "dim_G2/(b2+1)": (14, 22),
        "(dim_G2+1)/b2": (15, 21),
        "b2/(b3+dim_K7)": (21, 84),
        "b2/(b3+rank_E8)": (21, 85),
    }

    for name, (num, den) in alternatives.items():
        is_admissible = analyze_formula(name, num, den)


def main():
    print("=" * 70)
    print("SELECTION PRINCIPLE: GIFT-EXPRESSIBLE REDUCED FRACTIONS")
    print("=" * 70)

    print("\n--- KNOWN FORMULAS (SHOULD BE ADMISSIBLE) ---")

    for name, data in KNOWN_FORMULAS.items():
        analyze_formula(name, data["num"], data["den"])

    test_alternative_formulas()

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The selection principle appears to be:

╔═══════════════════════════════════════════════════════════════════════╗
║  A GIFT formula f(x₁,...,xₙ) is ADMISSIBLE if and only if:           ║
║                                                                       ║
║  When reduced to lowest terms a/b, BOTH a AND b are expressible      ║
║  as GIFT topological constants or simple products thereof.            ║
╚═══════════════════════════════════════════════════════════════════════╝

Examples:
  • sin²θ_W = 21/91 = 3/13  →  N_gen / alpha_sum  ✓
  • Q_Koide = 14/21 = 2/3   →  p₂ / N_gen  ✓
  • N_gen   = 21/7  = 3/1   →  N_gen / 1  ✓

Counter-examples:
  • b₂/b₃ = 21/77 = 3/11  →  N_gen / D_bulk  ✓ (but gives WRONG value!)

Wait - 11 IS D_bulk, so 3/11 = N_gen/D_bulk IS GIFT-expressible!

This means GIFT-expressibility is NECESSARY but NOT SUFFICIENT.
There must be an additional criterion...
""")

    # Let's check what's special about 13 vs 11
    print("\n--- DISTINGUISHING 3/13 from 3/11 ---")
    print("""
Both 3/13 and 3/11 are GIFT-expressible:
  • 3/13 = N_gen / alpha_sum
  • 3/11 = N_gen / D_bulk

But only 3/13 matches experiment (0.231 vs 0.273).

What's special about 13 = alpha_sum?
  • 13 = rank_E8 + Weyl = 8 + 5
  • 13 = b2 - rank_E8 = 21 - 8
  • 13 is the 6th prime = prime(6)
  • 13 appears in dim_E6 = 78 = 6 × 13

What's special about 11 = D_bulk?
  • 11 = rank_E8 + N_gen = 8 + 3
  • 11 = dim_G2 - N_gen = 14 - 3
  • 11 is the 5th prime = prime(Weyl)

HYPOTHESIS: The correct denominator involves HOLONOMY (dim_G2 = 14)
rather than just Betti numbers.

  b3 + dim_G2 = 77 + 14 = 91 = 7 × 13 → involves holonomy ✓
  b3 alone = 77 = 7 × 11 → no holonomy contribution ✗

The formula "b2/(b3 + dim_G2)" encodes:
  gauge_moduli / (matter_modes + holonomy_constraints)

This is a PHYSICAL selection principle, not just algebraic!
""")


if __name__ == "__main__":
    main()
