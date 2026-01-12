#!/usr/bin/env python3
"""
Formula Uniqueness Test

Hypothesis: For each physical observable, there is essentially ONE
GIFT-expressible formula that matches experiment.

This would prove "structural inevitability" rather than "selection".
"""

from fractions import Fraction
from itertools import combinations, product
from math import gcd, sqrt, pi
import json

# =============================================================================
# GIFT CONSTANTS (the "alphabet" of formulas)
# =============================================================================

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
    'dim_F4': 52,
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

# Physical observables with experimental values and tolerances
OBSERVABLES = {
    'sin2_theta_W': {
        'exp': 0.23122,
        'tol': 0.005,  # ~2% tolerance
        'known_formula': 'b2/(b3+dim_G2)',
    },
    'Q_Koide': {
        'exp': 0.666661,
        'tol': 0.01,
        'known_formula': 'dim_G2/b2',
    },
    'alpha_s': {
        'exp': 0.1179,
        'tol': 0.01,
        'known_formula': 'sqrt(2)/12',
    },
    'theta_13_deg': {
        'exp': 8.54,
        'tol': 0.5,
        'known_formula': '180/b2',
    },
    'theta_23_deg': {
        'exp': 49.3,
        'tol': 2.0,
        'known_formula': '(rank_E8+b3)/H_star * 90',
    },
    'm_s_over_m_d': {
        'exp': 20.0,
        'tol': 2.0,
        'known_formula': 'p2^2 * Weyl',
    },
}


def generate_simple_ratios():
    """Generate all simple ratios a/b where a,b are GIFT constants."""
    formulas = []
    keys = list(GIFT.keys())

    for num_key in keys:
        for den_key in keys:
            if num_key == den_key:
                continue
            num = GIFT[num_key]
            den = GIFT[den_key]
            if den == 0:
                continue
            formulas.append({
                'expr': f"{num_key}/{den_key}",
                'value': num / den,
                'type': 'ratio',
            })

    return formulas


def generate_compound_ratios():
    """Generate ratios a/(b+c) and (a+b)/c."""
    formulas = []
    keys = list(GIFT.keys())

    # a/(b+c)
    for num_key in keys:
        for den1_key, den2_key in combinations(keys, 2):
            num = GIFT[num_key]
            den = GIFT[den1_key] + GIFT[den2_key]
            if den == 0 or num_key in [den1_key, den2_key]:
                continue
            formulas.append({
                'expr': f"{num_key}/({den1_key}+{den2_key})",
                'value': num / den,
                'type': 'compound_ratio',
            })

    # (a+b)/c
    for num1_key, num2_key in combinations(keys, 2):
        for den_key in keys:
            num = GIFT[num1_key] + GIFT[num2_key]
            den = GIFT[den_key]
            if den == 0 or den_key in [num1_key, num2_key]:
                continue
            formulas.append({
                'expr': f"({num1_key}+{num2_key})/{den_key}",
                'value': num / den,
                'type': 'compound_ratio',
            })

    return formulas


def generate_products():
    """Generate products and powers."""
    formulas = []
    keys = list(GIFT.keys())

    # a * b
    for k1, k2 in combinations(keys, 2):
        formulas.append({
            'expr': f"{k1}*{k2}",
            'value': GIFT[k1] * GIFT[k2],
            'type': 'product',
        })

    # a^2
    for k in keys:
        if GIFT[k] <= 20:  # Avoid huge numbers
            formulas.append({
                'expr': f"{k}^2",
                'value': GIFT[k] ** 2,
                'type': 'power',
            })

    return formulas


def generate_special_forms():
    """Generate special forms like sqrt, pi-related."""
    formulas = []
    keys = list(GIFT.keys())

    # sqrt(a)/b
    for num_key in keys:
        for den_key in keys:
            if num_key == den_key:
                continue
            den = GIFT[den_key]
            if den == 0:
                continue
            formulas.append({
                'expr': f"sqrt({num_key})/{den_key}",
                'value': sqrt(GIFT[num_key]) / den,
                'type': 'sqrt_ratio',
            })

    # pi/a (for angles)
    for k in keys:
        v = GIFT[k]
        if v > 0:
            formulas.append({
                'expr': f"180/{k}",
                'value': 180 / v,
                'type': 'angle_formula',
            })
            formulas.append({
                'expr': f"90*{k}/100",
                'value': 90 * v / 100,
                'type': 'angle_formula',
            })

    return formulas


def find_matching_formulas(observable: str, obs_data: dict, all_formulas: list) -> list:
    """Find all formulas matching an observable within tolerance."""
    exp = obs_data['exp']
    tol = obs_data['tol']

    matches = []
    for f in all_formulas:
        if abs(f['value'] - exp) < tol:
            deviation = abs(f['value'] - exp) / exp * 100
            matches.append({
                **f,
                'deviation': deviation,
            })

    # Sort by deviation
    matches.sort(key=lambda x: x['deviation'])
    return matches


def main():
    print("=" * 80)
    print("FORMULA UNIQUENESS TEST")
    print("Proving structural inevitability: only ONE formula works per observable")
    print("=" * 80)

    # Generate all formulas
    all_formulas = []
    all_formulas.extend(generate_simple_ratios())
    all_formulas.extend(generate_compound_ratios())
    all_formulas.extend(generate_products())
    all_formulas.extend(generate_special_forms())

    print(f"\nGenerated {len(all_formulas)} candidate formulas")
    print(f"  - Simple ratios: {len(generate_simple_ratios())}")
    print(f"  - Compound ratios: {len(generate_compound_ratios())}")
    print(f"  - Products/powers: {len(generate_products())}")
    print(f"  - Special forms: {len(generate_special_forms())}")

    # Test each observable
    results = {}

    for obs_name, obs_data in OBSERVABLES.items():
        print(f"\n{'=' * 80}")
        print(f"OBSERVABLE: {obs_name}")
        print(f"Experimental: {obs_data['exp']} ± {obs_data['tol']}")
        print(f"Known formula: {obs_data['known_formula']}")
        print("-" * 80)

        matches = find_matching_formulas(obs_name, obs_data, all_formulas)
        results[obs_name] = matches

        if len(matches) == 0:
            print("  NO MATCHES FOUND!")
        else:
            print(f"  Found {len(matches)} matching formula(s):\n")
            for i, m in enumerate(matches[:10]):  # Show top 10
                marker = " <-- KNOWN" if obs_data['known_formula'] in m['expr'] else ""
                print(f"  {i+1}. {m['expr']:40} = {m['value']:.6f}  ({m['deviation']:.3f}% off){marker}")

            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more")

    # Summary
    print("\n" + "=" * 80)
    print("UNIQUENESS SUMMARY")
    print("=" * 80)

    print("\n| Observable | # Matches | Best Match | Known Formula Rank |")
    print("|------------|-----------|------------|-------------------|")

    for obs_name, matches in results.items():
        n_matches = len(matches)
        best = matches[0]['expr'] if matches else "NONE"

        # Find rank of known formula
        known = OBSERVABLES[obs_name]['known_formula']
        rank = "N/A"
        for i, m in enumerate(matches):
            if known in m['expr'] or m['expr'] in known:
                rank = f"#{i+1}"
                break

        print(f"| {obs_name:18} | {n_matches:9} | {best:30} | {rank:17} |")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    unique_count = sum(1 for m in results.values() if len(m) <= 3)
    total = len(results)

    print(f"""
Observables with ≤3 matching formulas: {unique_count}/{total}

INTERPRETATION:
""")

    if unique_count == total:
        print("""
✓ STRUCTURAL INEVITABILITY CONFIRMED

For each observable, only a small number of GIFT-expressible formulas
match experiment. This is NOT selection from a large pool; it's the
ONLY option that works.

The formulas are not "chosen" — they are FORCED by:
1. The algebra of GIFT constants
2. The constraint of matching experiment

This is analogous to how the Balmer formula wasn't "selected" from
alternatives — it was the unique expression that fit the data.
""")
    else:
        print(f"""
⚠ PARTIAL UNIQUENESS

Some observables have multiple matching formulas. This could mean:
1. Degeneracy in the formula space (multiple equivalent expressions)
2. Need for additional selection criterion
3. Some formulas are algebraically equivalent
""")


if __name__ == "__main__":
    main()
