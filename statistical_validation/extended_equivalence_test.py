#!/usr/bin/env python3
"""
Extended Formula Equivalence Test

Verifies that the 15 new correspondences from Extended Observables Research
also exhibit structural inevitability (multiple equivalent GIFT expressions).
"""

from fractions import Fraction
from itertools import combinations
from math import gcd

# Complete GIFT constant dictionary (extended for new correspondences)
GIFT = {
    'b0': 1,
    'p2': 2,
    'N_gen': 3,
    'Weyl': 5,
    'dim_K7': 7,
    'rank_E8': 8,
    'D_bulk': 11,
    'alpha_sum': 13,
    'dim_G2': 14,
    'b2': 21,
    'dim_J3O': 27,
    'det_g_den': 32,
    'chi_K7': 42,  # Euler characteristic = 2 × 3 × 7
    'dim_F4': 52,
    'fund_E7': 56,
    'kappa_T_inv': 61,
    'det_g_num': 65,
    'b3': 77,
    'dim_E6': 78,
    'H_star': 99,
    'dim_E7': 133,
    'PSL27': 168,
    'dim_E8': 248,
    'dim_E8xE8': 496,
}


def find_all_expressions_for_value(target_num: int, target_den: int, max_complexity: int = 2):
    """
    Find all GIFT expressions that reduce to target_num/target_den.

    max_complexity:
        1 = simple ratios a/b only
        2 = include (a+b)/c and a/(b+c)
        3 = include (a-b)/c and a/(b-c)
    """
    target = Fraction(target_num, target_den)
    expressions = []
    keys = list(GIFT.keys())

    # Simple ratios a/b
    for num_key in keys:
        for den_key in keys:
            if num_key == den_key:
                continue
            if GIFT[den_key] == 0:
                continue
            frac = Fraction(GIFT[num_key], GIFT[den_key])
            if frac == target:
                expressions.append({
                    'expr': f"{num_key}/{den_key}",
                    'raw': f"{GIFT[num_key]}/{GIFT[den_key]}",
                    'type': 'simple'
                })

    if max_complexity >= 2:
        # (a+b)/c
        for num1, num2 in combinations(keys, 2):
            for den_key in keys:
                if den_key in [num1, num2]:
                    continue
                num_val = GIFT[num1] + GIFT[num2]
                den_val = GIFT[den_key]
                if den_val == 0:
                    continue
                frac = Fraction(num_val, den_val)
                if frac == target:
                    expressions.append({
                        'expr': f"({num1}+{num2})/{den_key}",
                        'raw': f"({GIFT[num1]}+{GIFT[num2]})/{GIFT[den_key]}",
                        'type': 'sum_num'
                    })

        # a/(b+c)
        for num_key in keys:
            for den1, den2 in combinations(keys, 2):
                if num_key in [den1, den2]:
                    continue
                num_val = GIFT[num_key]
                den_val = GIFT[den1] + GIFT[den2]
                if den_val == 0:
                    continue
                frac = Fraction(num_val, den_val)
                if frac == target:
                    expressions.append({
                        'expr': f"{num_key}/({den1}+{den2})",
                        'raw': f"{GIFT[num_key]}/({GIFT[den1]}+{GIFT[den2]})",
                        'type': 'sum_den'
                    })

    if max_complexity >= 3:
        # (a-b)/c where a > b
        for num1 in keys:
            for num2 in keys:
                if num1 == num2:
                    continue
                if GIFT[num1] <= GIFT[num2]:
                    continue
                for den_key in keys:
                    if den_key in [num1, num2]:
                        continue
                    num_val = GIFT[num1] - GIFT[num2]
                    den_val = GIFT[den_key]
                    if den_val == 0 or num_val <= 0:
                        continue
                    frac = Fraction(num_val, den_val)
                    if frac == target:
                        expressions.append({
                            'expr': f"({num1}-{num2})/{den_key}",
                            'raw': f"({GIFT[num1]}-{GIFT[num2]})/{GIFT[den_key]}",
                            'type': 'diff_num'
                        })

        # a/(b-c) where b > c
        for num_key in keys:
            for den1 in keys:
                for den2 in keys:
                    if den1 == den2:
                        continue
                    if GIFT[den1] <= GIFT[den2]:
                        continue
                    if num_key in [den1, den2]:
                        continue
                    num_val = GIFT[num_key]
                    den_val = GIFT[den1] - GIFT[den2]
                    if den_val <= 0:
                        continue
                    frac = Fraction(num_val, den_val)
                    if frac == target:
                        expressions.append({
                            'expr': f"{num_key}/({den1}-{den2})",
                            'raw': f"{GIFT[num_key]}/({GIFT[den1]}-{GIFT[den2]})",
                            'type': 'diff_den'
                        })

    return expressions


def analyze_observable(name: str, num: int, den: int, experimental: float,
                       interpretation: str = ""):
    """Analyze a single observable for structural inevitability."""
    reduced = Fraction(num, den)
    red_num = reduced.numerator
    red_den = reduced.denominator

    value = num / den
    deviation = abs(value - experimental) / experimental * 100

    print(f"\n{'='*70}")
    print(f"OBSERVABLE: {name}")
    print(f"{'='*70}")
    print(f"Proposed fraction: {num}/{den} = {red_num}/{red_den}")
    print(f"Decimal value: {value:.6f}")
    print(f"Experimental: {experimental:.6f}")
    print(f"Deviation: {deviation:.2f}%")
    if interpretation:
        print(f"Interpretation: {interpretation}")

    # Find equivalent expressions
    expressions = find_all_expressions_for_value(red_num, red_den, max_complexity=3)

    print(f"\nEquivalent GIFT expressions: {len(expressions)}")

    if expressions:
        # Group by type
        by_type = {}
        for e in expressions:
            t = e['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for t, exprs in by_type.items():
            print(f"\n  [{t}] ({len(exprs)} expressions):")
            for e in exprs[:5]:  # Show max 5 per type
                print(f"    • {e['expr']:40} = {e['raw']}")
            if len(exprs) > 5:
                print(f"    ... and {len(exprs)-5} more")
    else:
        print("  ⚠ NO EQUIVALENT EXPRESSIONS FOUND")
        print("  This fraction may not be structurally inevitable!")

    return {
        'name': name,
        'fraction': f"{red_num}/{red_den}",
        'value': value,
        'experimental': experimental,
        'deviation': deviation,
        'n_expressions': len(expressions),
        'expressions': expressions
    }


def main():
    print("="*70)
    print("EXTENDED FORMULA EQUIVALENCE TEST")
    print("Testing 15 new correspondences for structural inevitability")
    print("="*70)

    # The 15 new correspondences from Extended Observables Research
    observables = [
        # PMNS Matrix
        ("sin²θ₁₂_PMNS", 4, 13, 0.307, "(b₀+N_gen)/α_sum"),
        ("sin²θ₂₃_PMNS", 6, 11, 0.546, "(D_bulk-Weyl)/D_bulk"),
        ("sin²θ₁₃_PMNS", 11, 496, 0.0220, "D_bulk/dim(E₈×E₈)"),

        # Quark mass ratios
        ("m_s/m_d", 20, 1, 20.0, "(α_sum+dim_J₃O)/p₂"),
        ("m_c/m_s", 246, 21, 11.7, "(dim_E₈-p₂)/b₂"),
        ("m_b/m_t", 1, 42, 0.024, "1/χ(K₇)"),
        ("m_u/m_d", 233, 496, 0.47, "(det_g+PSL27)/dim(E₈×E₈)"),

        # Boson mass ratios
        ("m_H/m_W", 81, 52, 1.558, "(N_gen+dim_E₆)/dim_F₄"),
        ("m_H/m_t", 56, 77, 0.725, "fund(E₇)/b₃"),
        ("m_W/m_Z", 46, 52, 0.8815, "(dim_G₂+det_g_den)/dim_F₄"),

        # CKM
        ("sin²θ₁₂_CKM", 56, 248, 0.2250, "fund(E₇)/dim(E₈)"),

        # Cosmological
        ("Ω_b/Ω_m", 39, 248, 0.157, "(dim_F₄-α_sum)/dim(E₈)"),
        ("Ω_Λ/Ω_m", 25, 11, 2.27, "(det_g_den-dim_K₇)/D_bulk"),

        # Strong coupling
        ("α_s(M_Z)", 29, 248, 0.1179, "(fund_E₇-dim_J₃O)/dim(E₈)"),

        # Lepton ratio
        ("m_μ/m_τ", 10, 168, 0.0595, "(b₂-D_bulk)/PSL27"),
    ]

    results = []
    for obs in observables:
        result = analyze_observable(*obs)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: STRUCTURAL INEVITABILITY ANALYSIS")
    print("="*70)

    print("\n| Observable | Fraction | Deviation | # Expressions | Status |")
    print("|------------|----------|-----------|---------------|--------|")

    total_expressions = 0
    inevitable_count = 0

    for r in results:
        status = "✓ INEVITABLE" if r['n_expressions'] >= 2 else "⚠ UNIQUE"
        if r['n_expressions'] == 0:
            status = "✗ NOT GIFT"
        print(f"| {r['name']:14} | {r['fraction']:8} | {r['deviation']:7.2f}% | {r['n_expressions']:13} | {status} |")
        total_expressions += r['n_expressions']
        if r['n_expressions'] >= 2:
            inevitable_count += 1

    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print("="*70)

    print(f"""
Observables analyzed: {len(results)}
With multiple expressions (≥2): {inevitable_count}
Total equivalent expressions found: {total_expressions}
Mean expressions per observable: {total_expressions/len(results):.1f}

""")

    # Check for problematic cases
    problematic = [r for r in results if r['n_expressions'] < 2]
    if problematic:
        print("⚠ OBSERVABLES NEEDING ATTENTION:")
        for r in problematic:
            print(f"  • {r['name']}: Only {r['n_expressions']} expression(s)")
        print("\nThese may be:")
        print("  - Numerologically accidental matches")
        print("  - Requiring extended GIFT constant set")
        print("  - Actually structural but with complex expressions")
    else:
        print("✓ ALL OBSERVABLES SHOW STRUCTURAL INEVITABILITY")
        print("  Each fraction admits multiple equivalent GIFT expressions.")

    # Check the sin²θ_W vs m_W/m_Z consistency
    print(f"\n{'='*70}")
    print("CONSISTENCY CHECK: sin²θ_W vs m_W/m_Z")
    print("="*70)

    sin2_W = 3/13  # Known GIFT value
    cos2_W = 1 - sin2_W
    cos_W = cos2_W ** 0.5

    mW_mZ_gift = 46/52  # From extended observables

    print(f"""
Electroweak consistency requires: m_W/m_Z = cos(θ_W)

From GIFT sin²θ_W = 3/13:
  cos²θ_W = 1 - 3/13 = 10/13 = {10/13:.6f}
  cos θ_W = √(10/13) = {cos_W:.6f}

Proposed GIFT m_W/m_Z = 46/52 = {46/52:.6f}

Discrepancy: {abs(cos_W - 46/52)/cos_W * 100:.2f}%

This suggests either:
  1. m_W/m_Z = 46/52 is a numerical coincidence, not structural
  2. There are radiative corrections not captured
  3. The sin²θ_W = 3/13 is the "bare" value, 46/52 is "dressed"
""")


if __name__ == "__main__":
    main()
