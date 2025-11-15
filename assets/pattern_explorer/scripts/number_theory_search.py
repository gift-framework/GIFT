#!/usr/bin/env python3
"""
GIFT Framework - Phase 8: Computational Number Theory Pattern Discovery

This script systematically tests advanced number-theoretic functions against
the 37 GIFT framework observables, including:
- Dirichlet L-functions (Catalan's constant)
- Polylogarithm functions
- Euler-Mascheroni constant patterns
- Apéry's constant (zeta(3)) patterns
- Prime zeta function
- Bernoulli numbers
- Ramanujan constants

Uses mpmath for high-precision arbitrary-precision arithmetic.
"""

import mpmath as mp
from mpmath import mp as mpf_context
import csv
from pathlib import Path
import sys
from typing import List, Tuple, Dict
import itertools
import math

# Set high precision for calculations
mp.dps = 50  # 50 decimal places

# ============================================================================
# OBSERVABLE DATA - All 37 GIFT framework observables
# ============================================================================

OBSERVABLES = {
    # Gauge Sector (3)
    'alpha_inv': 137.036,
    'alpha_s': 0.1179,
    'sin2_theta_W': 0.23122,

    # Neutrino Sector (4) - angles in degrees, convert to sin² for some tests
    'theta_12': 33.44,
    'theta_13': 8.57,
    'theta_23': 49.2,
    'delta_CP': 197.0,

    # Lepton Sector (3)
    'Q_Koide': 0.6667,
    'mu_e_ratio': 206.768,
    'tau_e_ratio': 3477.15,

    # Quark Masses (6) - in MeV except top
    'm_u': 2.16,
    'm_d': 4.67,
    'm_s': 93.4,
    'm_c': 1270.0,
    'm_b': 4180.0,
    'm_t': 172500.0,

    # Quark Mass Ratios (10)
    'm_s_m_d': 20.0,
    'm_b_m_u': 1935.19,
    'm_c_m_d': 271.94,
    'm_d_m_u': 2.162,
    'm_c_m_s': 13.6,
    'm_t_m_c': 135.83,
    'm_b_m_d': 895.07,
    'm_b_m_c': 3.29,
    'm_t_m_s': 1846.89,
    'm_b_m_s': 44.76,

    # CKM Matrix (1)
    'theta_C': 13.04,

    # Higgs Sector (3)
    'lambda_H': 0.1286,
    'v_EW': 246.22,  # GeV
    'm_H': 125.25,    # GeV

    # Cosmological (4)
    'Omega_DE': 0.6847,
    'Omega_DM': 0.120,
    'n_s': 0.9649,
    'H_0': 73.04,

    # Dark Matter (2)
    'm_chi1': 90.5,   # GeV
    'm_chi2': 352.7,  # GeV

    # Temporal (1)
    'D_H': 0.856220,
}

# ============================================================================
# NUMBER THEORY FUNCTIONS
# ============================================================================

def catalan_constant():
    """Catalan's constant G = L(2, χ₄) = 0.915965594..."""
    return mp.catalan

def euler_gamma():
    """Euler-Mascheroni constant γ = 0.5772156649..."""
    return mp.euler

def apery_constant():
    """Apéry's constant ζ(3) = 1.202056903..."""
    return mp.zeta(3)

def polylog(s, z):
    """Polylogarithm Li_s(z)"""
    return mp.polylog(s, z)

def dirichlet_beta(s):
    """
    Dirichlet beta function β(s) = L(s, χ₄)
    Related to Catalan's constant: β(2) = G
    """
    # β(s) = Σ (-1)^n / (2n+1)^s
    # For s=2, this is Catalan's constant
    if s == 2:
        return mp.catalan
    else:
        # Use series expansion for other values
        result = mp.mpf(0)
        for n in range(1000):
            term = mp.power(-1, n) / mp.power(2*n + 1, s)
            result += term
            if abs(term) < mp.mpf(10)**(-mp.dps):
                break
        return result

def prime_zeta(s):
    """
    Prime zeta function P(s) = Σ 1/p^s over primes
    """
    # Use first 1000 primes for approximation
    primes = []
    n = 2
    while len(primes) < 1000:
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
        n += 1

    result = mp.mpf(0)
    for p in primes:
        result += mp.power(p, -s)
    return result

def bernoulli_number(n):
    """Bernoulli number B_n"""
    return mp.bernoulli(n)

def ramanujan_constant():
    """e^(π√163) ≈ 262537412640768743.99999999999925..."""
    return mp.exp(mp.pi * mp.sqrt(163))

# ============================================================================
# PATTERN GENERATION
# ============================================================================

def generate_number_theory_patterns():
    """
    Generate systematic patterns using number theory functions.
    Returns list of (observable, formula, theoretical_value, function_type) tuples.
    """
    patterns = []

    # Pre-compute common constants
    G = catalan_constant()
    gamma = euler_gamma()
    zeta3 = apery_constant()
    exp_gamma = mp.exp(gamma)
    gamma_squared = gamma ** 2

    # Polylogarithm values
    Li2_half = polylog(2, mp.mpf(1)/2)
    Li2_third = polylog(2, mp.mpf(1)/3)
    Li2_two_thirds = polylog(2, mp.mpf(2)/3)
    Li3_half = polylog(3, mp.mpf(1)/2)
    Li3_third = polylog(3, mp.mpf(1)/3)

    # Prime zeta values
    P2 = prime_zeta(2)
    P3 = prime_zeta(3)

    # Bernoulli numbers (even indices only, odd ones are 0 except B_1)
    B2 = bernoulli_number(2)
    B4 = bernoulli_number(4)
    B6 = bernoulli_number(6)
    B8 = bernoulli_number(8)

    # Dirichlet beta values
    beta2 = dirichlet_beta(2)  # = G
    beta3 = dirichlet_beta(3)

    constants_dict = {
        'G': G,
        'gamma': gamma,
        'zeta3': zeta3,
        'exp_gamma': exp_gamma,
        'gamma^2': gamma_squared,
        'Li2(1/2)': Li2_half,
        'Li2(1/3)': Li2_third,
        'Li2(2/3)': Li2_two_thirds,
        'Li3(1/2)': Li3_half,
        'Li3(1/3)': Li3_third,
        'P(2)': P2,
        'P(3)': P3,
        'B_2': B2,
        'B_4': B4,
        'B_6': B6,
        'B_8': B8,
        'beta(2)': beta2,
        'beta(3)': beta3,
    }

    # ========================================================================
    # 1. Direct constant matches
    # ========================================================================
    for const_name, const_value in constants_dict.items():
        patterns.append(('direct', const_name, float(const_value), 'direct_constant'))

    # ========================================================================
    # 2. Simple powers and products
    # ========================================================================
    base_constants = {
        'G': G,
        'gamma': gamma,
        'zeta3': zeta3,
        'Li2(1/2)': Li2_half,
        'Li2(1/3)': Li2_third,
    }

    for name1, val1 in base_constants.items():
        # Powers
        for power in [2, 3, 1/2, 1/3, -1, -2]:
            result = mp.power(val1, power)
            if power == 1/2:
                formula = f'sqrt({name1})'
            elif power == 1/3:
                formula = f'{name1}^(1/3)'
            elif power == -1:
                formula = f'1/{name1}'
            elif power == -2:
                formula = f'1/{name1}^2'
            else:
                formula = f'{name1}^{power}'
            patterns.append(('power', formula, float(result), 'power'))

        # Exponentials
        if abs(float(val1)) < 10:  # Avoid overflow
            result = mp.exp(val1)
            patterns.append(('exponential', f'exp({name1})', float(result), 'exponential'))

            result = mp.exp(-val1)
            patterns.append(('exponential', f'exp(-{name1})', float(result), 'exponential'))

        # Logarithms
        if float(val1) > 0:
            result = mp.log(val1)
            patterns.append(('logarithm', f'ln({name1})', float(result), 'logarithm'))

    # ========================================================================
    # 3. Products and ratios of two constants
    # ========================================================================
    const_pairs = list(itertools.combinations(base_constants.items(), 2))
    for (name1, val1), (name2, val2) in const_pairs:
        # Products
        result = val1 * val2
        patterns.append(('product', f'{name1} × {name2}', float(result), 'product'))

        # Ratios
        result = val1 / val2
        patterns.append(('ratio', f'{name1}/{name2}', float(result), 'ratio'))

        result = val2 / val1
        patterns.append(('ratio', f'{name2}/{name1}', float(result), 'ratio'))

        # Sums
        result = val1 + val2
        patterns.append(('sum', f'{name1} + {name2}', float(result), 'sum'))

        # Differences
        result = val1 - val2
        patterns.append(('difference', f'{name1} - {name2}', float(result), 'difference'))

        result = val2 - val1
        patterns.append(('difference', f'{name2} - {name1}', float(result), 'difference'))

    # ========================================================================
    # 4. Catalan constant special patterns
    # ========================================================================
    # G with pi
    patterns.append(('catalan_pi', 'G × π', float(G * mp.pi), 'catalan_pattern'))
    patterns.append(('catalan_pi', 'G / π', float(G / mp.pi), 'catalan_pattern'))
    patterns.append(('catalan_pi', 'π / G', float(mp.pi / G), 'catalan_pattern'))
    patterns.append(('catalan_pi', 'G × π²', float(G * mp.pi**2), 'catalan_pattern'))
    patterns.append(('catalan_pi', 'G / π²', float(G / mp.pi**2), 'catalan_pattern'))

    # G with e
    patterns.append(('catalan_e', 'G × e', float(G * mp.e), 'catalan_pattern'))
    patterns.append(('catalan_e', 'G / e', float(G / mp.e), 'catalan_pattern'))
    patterns.append(('catalan_e', 'e / G', float(mp.e / G), 'catalan_pattern'))

    # G with ln(2)
    ln2 = mp.log(2)
    patterns.append(('catalan_ln2', 'G × ln(2)', float(G * ln2), 'catalan_pattern'))
    patterns.append(('catalan_ln2', 'G / ln(2)', float(G / ln2), 'catalan_pattern'))
    patterns.append(('catalan_ln2', 'ln(2) / G', float(ln2 / G), 'catalan_pattern'))

    # ========================================================================
    # 5. Euler gamma patterns with zeta
    # ========================================================================
    for n in [2, 3, 4, 5, 6, 7]:
        zeta_n = mp.zeta(n)
        patterns.append(('gamma_zeta', f'γ × ζ({n})', float(gamma * zeta_n), 'gamma_zeta'))
        patterns.append(('gamma_zeta', f'γ / ζ({n})', float(gamma / zeta_n), 'gamma_zeta'))
        patterns.append(('gamma_zeta', f'ζ({n}) / γ', float(zeta_n / gamma), 'gamma_zeta'))

    # ========================================================================
    # 6. Apéry constant powers
    # ========================================================================
    for power in [2, 3, 4, 1/2, 1/3, -1, -2]:
        result = mp.power(zeta3, power)
        if power == 1/2:
            formula = f'sqrt(ζ(3))'
        elif power == 1/3:
            formula = f'ζ(3)^(1/3)'
        elif power == -1:
            formula = f'1/ζ(3)'
        else:
            formula = f'ζ(3)^{power}'
        patterns.append(('apery_power', formula, float(result), 'apery_pattern'))

    # ========================================================================
    # 7. Polylog combinations
    # ========================================================================
    # Li2(1/2) = π²/12 - ln(2)²/2
    theoretical_Li2_half = mp.pi**2/12 - mp.log(2)**2/2
    patterns.append(('polylog', 'π²/12 - ln(2)²/2', float(theoretical_Li2_half), 'polylog'))

    # Li2 values with pi
    patterns.append(('polylog', 'Li2(1/2) × π', float(Li2_half * mp.pi), 'polylog'))
    patterns.append(('polylog', 'Li2(1/3) × π', float(Li2_third * mp.pi), 'polylog'))
    patterns.append(('polylog', 'Li3(1/2) × π', float(Li3_half * mp.pi), 'polylog'))

    # ========================================================================
    # 8. Prime zeta patterns
    # ========================================================================
    patterns.append(('prime_zeta', 'P(2) × π', float(P2 * mp.pi), 'prime_zeta'))
    patterns.append(('prime_zeta', 'P(2) × e', float(P2 * mp.e), 'prime_zeta'))
    patterns.append(('prime_zeta', 'P(3) × π', float(P3 * mp.pi), 'prime_zeta'))

    # ========================================================================
    # 9. Bernoulli ratios
    # ========================================================================
    patterns.append(('bernoulli', 'B_4/B_2', float(B4/B2), 'bernoulli'))
    patterns.append(('bernoulli', 'B_6/B_2', float(B6/B2), 'bernoulli'))
    patterns.append(('bernoulli', 'B_8/B_2', float(B8/B2), 'bernoulli'))
    patterns.append(('bernoulli', 'B_6/B_4', float(B6/B4), 'bernoulli'))
    patterns.append(('bernoulli', '|B_2|', float(abs(B2)), 'bernoulli'))
    patterns.append(('bernoulli', '|B_4|', float(abs(B4)), 'bernoulli'))

    # ========================================================================
    # 10. Mixed special patterns
    # ========================================================================
    # gamma with G
    patterns.append(('gamma_catalan', 'γ × G', float(gamma * G), 'gamma_catalan'))
    patterns.append(('gamma_catalan', 'γ / G', float(gamma / G), 'gamma_catalan'))
    patterns.append(('gamma_catalan', 'G / γ', float(G / gamma), 'gamma_catalan'))
    patterns.append(('gamma_catalan', 'γ² × G', float(gamma_squared * G), 'gamma_catalan'))

    # zeta(3) with G
    patterns.append(('zeta3_catalan', 'ζ(3) × G', float(zeta3 * G), 'zeta3_catalan'))
    patterns.append(('zeta3_catalan', 'ζ(3) / G', float(zeta3 / G), 'zeta3_catalan'))
    patterns.append(('zeta3_catalan', 'G / ζ(3)', float(G / zeta3), 'zeta3_catalan'))

    # polylog with G
    patterns.append(('polylog_catalan', 'Li2(1/2) × G', float(Li2_half * G), 'polylog_catalan'))
    patterns.append(('polylog_catalan', 'Li2(1/2) / G', float(Li2_half / G), 'polylog_catalan'))

    # Three-way products
    patterns.append(('triple', 'γ × ζ(3) × G', float(gamma * zeta3 * G), 'triple_product'))
    patterns.append(('triple', 'γ × ζ(3) × π', float(gamma * zeta3 * mp.pi), 'triple_product'))

    # ========================================================================
    # 11. Integer multiples and simple fractions
    # ========================================================================
    for const_name, const_value in [('G', G), ('gamma', gamma), ('Li2(1/2)', Li2_half),
                                     ('zeta3', zeta3), ('Li2(1/3)', Li2_third)]:
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 20, 21, 24, 30]:
            # Multiples
            patterns.append(('multiple', f'{n} × {const_name}', float(n * const_value), 'multiple'))
            # Divisions
            patterns.append(('division', f'{const_name}/{n}', float(const_value/n), 'division'))
            # Simple fractions
            for m in range(1, min(n, 20)):
                if math.gcd(m, n) == 1:  # coprime
                    patterns.append(('fraction', f'({m}/{n}) × {const_name}',
                                   float(mp.mpf(m)/n * const_value), 'fraction'))

    # ========================================================================
    # 12. Ramanujan constant patterns
    # ========================================================================
    R = ramanujan_constant()
    # Take logarithm to get more reasonable values
    log_R = mp.log(R)
    patterns.append(('ramanujan', 'ln(e^(π√163))', float(log_R), 'ramanujan'))
    patterns.append(('ramanujan', 'π√163', float(mp.pi * mp.sqrt(163)), 'ramanujan'))

    # Inverse and related
    patterns.append(('ramanujan', '1/e^(π√163)', float(1/R), 'ramanujan'))

    # ========================================================================
    # 13. Combinations with pi, e, ln(2)
    # ========================================================================
    fundamental = {
        'π': mp.pi,
        'e': mp.e,
        'ln(2)': mp.log(2),
        'sqrt(2)': mp.sqrt(2),
        'sqrt(3)': mp.sqrt(3),
        'sqrt(5)': mp.sqrt(5),
        'sqrt(7)': mp.sqrt(7),
    }

    # Number theory constants with fundamental constants
    nt_constants = {
        'G': G,
        'gamma': gamma,
        'zeta3': zeta3,
        'Li2(1/2)': Li2_half,
    }

    for (nt_name, nt_val), (fund_name, fund_val) in itertools.product(
        nt_constants.items(), fundamental.items()
    ):
        # Products
        patterns.append(('nt_fund_product', f'{nt_name} × {fund_name}',
                        float(nt_val * fund_val), 'nt_fundamental'))
        patterns.append(('nt_fund_ratio', f'{nt_name}/{fund_name}',
                        float(nt_val / fund_val), 'nt_fundamental'))
        patterns.append(('nt_fund_ratio', f'{fund_name}/{nt_name}',
                        float(fund_val / nt_val), 'nt_fundamental'))

    # ========================================================================
    # 14. More zeta values and combinations
    # ========================================================================
    for n in [2, 3, 4, 5, 6, 7, 9, 11]:
        zeta_n = mp.zeta(n)
        patterns.append(('zeta', f'ζ({n})', float(zeta_n), 'zeta'))
        patterns.append(('zeta', f'1/ζ({n})', float(1/zeta_n), 'zeta'))
        patterns.append(('zeta', f'sqrt(ζ({n}))', float(mp.sqrt(zeta_n)), 'zeta'))

        # Combinations with number theory constants
        for nt_name, nt_val in [('G', G), ('gamma', gamma)]:
            patterns.append(('zeta_nt', f'ζ({n}) × {nt_name}',
                           float(zeta_n * nt_val), 'zeta_nt'))
            patterns.append(('zeta_nt', f'ζ({n})/{nt_name}',
                           float(zeta_n / nt_val), 'zeta_nt'))

    # ========================================================================
    # 15. More polylog combinations
    # ========================================================================
    polylog_values = [
        ('Li2(1/2)', Li2_half),
        ('Li2(1/3)', Li2_third),
        ('Li2(2/3)', Li2_two_thirds),
        ('Li3(1/2)', Li3_half),
        ('Li3(1/3)', Li3_third),
    ]

    for name1, val1 in polylog_values:
        # With integers
        for n in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
            patterns.append(('polylog_mult', f'{n} × {name1}',
                           float(n * val1), 'polylog_multiple'))
            patterns.append(('polylog_div', f'{name1}/{n}',
                           float(val1/n), 'polylog_division'))

        # Powers
        for exp in [2, 3, 1/2, 1/3]:
            result = mp.power(val1, exp)
            if exp == 1/2:
                formula = f'sqrt({name1})'
            elif exp == 1/3:
                formula = f'{name1}^(1/3)'
            else:
                formula = f'{name1}^{exp}'
            patterns.append(('polylog_power', formula, float(result), 'polylog_power'))

    # ========================================================================
    # 16. Squared and cubed number theory constants
    # ========================================================================
    for const_name, const_value in nt_constants.items():
        # With pi and e
        patterns.append(('squared_comb', f'{const_name}² × π',
                        float(const_value**2 * mp.pi), 'squared_combo'))
        patterns.append(('squared_comb', f'{const_name}² × e',
                        float(const_value**2 * mp.e), 'squared_combo'))
        patterns.append(('squared_comb', f'{const_name}² / π',
                        float(const_value**2 / mp.pi), 'squared_combo'))

        # Inverse squared
        patterns.append(('inv_squared', f'1/{const_name}²',
                        float(1/const_value**2), 'inv_squared'))

    return patterns

# ============================================================================
# PATTERN MATCHING
# ============================================================================

def find_matches(patterns, observables, tolerance_pct=1.0):
    """
    Find matches between patterns and observables.

    Args:
        patterns: List of (type, formula, value, function_type) tuples
        observables: Dict of observable_name -> experimental_value
        tolerance_pct: Maximum deviation in percent

    Returns:
        List of match dictionaries
    """
    matches = []

    for obs_name, obs_value in observables.items():
        for pattern_type, formula, theoretical, func_type in patterns:
            # Skip if theoretical value is too extreme
            if abs(theoretical) > 1e6 or abs(theoretical) < 1e-6:
                continue

            deviation_pct = abs(theoretical - obs_value) / abs(obs_value) * 100

            if deviation_pct <= tolerance_pct:
                matches.append({
                    'observable': obs_name,
                    'formula': formula,
                    'experimental': obs_value,
                    'theoretical': theoretical,
                    'deviation_pct': deviation_pct,
                    'function_type': func_type,
                })

    return matches

# ============================================================================
# REPORTING
# ============================================================================

def save_results_csv(matches, filename):
    """Save matches to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'observable', 'formula', 'experimental', 'theoretical',
            'deviation_pct', 'function_type'
        ])
        writer.writeheader()
        writer.writerows(matches)

def generate_markdown_report(matches, filename):
    """Generate comprehensive markdown report."""

    # Sort by deviation
    matches_sorted = sorted(matches, key=lambda x: x['deviation_pct'])

    # Group by function type
    by_type = {}
    for m in matches:
        func_type = m['function_type']
        if func_type not in by_type:
            by_type[func_type] = []
        by_type[func_type].append(m)

    # Find Catalan matches
    catalan_matches = [m for m in matches if 'G' in m['formula'] or 'catalan' in m['function_type']]
    catalan_matches_sorted = sorted(catalan_matches, key=lambda x: x['deviation_pct'])

    with open(filename, 'w') as f:
        f.write("# GIFT Framework - Phase 8: Computational Number Theory Patterns\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"This analysis systematically tested advanced number-theoretic functions ")
        f.write(f"against the 37 GIFT framework observables.\n\n")

        f.write(f"**Total patterns tested**: {len(matches)}\n")
        f.write(f"**Patterns found (<1% deviation)**: {len([m for m in matches if m['deviation_pct'] < 1.0])}\n")
        f.write(f"**Patterns found (<0.5% deviation)**: {len([m for m in matches if m['deviation_pct'] < 0.5])}\n")
        f.write(f"**Patterns found (<0.1% deviation)**: {len([m for m in matches if m['deviation_pct'] < 0.1])}\n\n")

        f.write("### Function Types Explored\n\n")
        f.write("1. **Dirichlet L-functions**: L(s, χ) including Catalan's constant G = L(2, χ₄)\n")
        f.write("2. **Polylogarithm functions**: Li_s(z) for various s and z\n")
        f.write("3. **Euler-Mascheroni constant**: γ and combinations with zeta functions\n")
        f.write("4. **Apéry's constant**: ζ(3) and powers thereof\n")
        f.write("5. **Prime zeta function**: P(s) = Σ 1/p^s\n")
        f.write("6. **Bernoulli numbers**: Ratios of B_n for even n\n")
        f.write("7. **Ramanujan constants**: e^(π√163) and related values\n\n")

        f.write("## Patterns by Function Type\n\n")
        for func_type in sorted(by_type.keys()):
            count = len(by_type[func_type])
            best_dev = min(m['deviation_pct'] for m in by_type[func_type])
            f.write(f"- **{func_type}**: {count} patterns (best: {best_dev:.4f}%)\n")
        f.write("\n")

        f.write("## Catalan Constant Matches\n\n")
        f.write("Catalan's constant G = 0.915965594177219015...\n\n")
        if catalan_matches_sorted:
            f.write("| Observable | Formula | Experimental | Theoretical | Deviation (%) |\n")
            f.write("|------------|---------|--------------|-------------|---------------|\n")
            for m in catalan_matches_sorted[:20]:  # Top 20
                f.write(f"| {m['observable']} | {m['formula']} | {m['experimental']:.6f} | ")
                f.write(f"{m['theoretical']:.6f} | {m['deviation_pct']:.4f} |\n")
        else:
            f.write("No Catalan constant matches found within tolerance.\n")
        f.write("\n")

        f.write("## Top 10 Overall Patterns\n\n")
        f.write("| Observable | Formula | Experimental | Theoretical | Deviation (%) | Type |\n")
        f.write("|------------|---------|--------------|-------------|---------------|------|\n")
        for m in matches_sorted[:10]:
            f.write(f"| {m['observable']} | {m['formula']} | {m['experimental']:.6f} | ")
            f.write(f"{m['theoretical']:.6f} | {m['deviation_pct']:.4f} | {m['function_type']} |\n")
        f.write("\n")

        f.write("## Detailed Results by Observable\n\n")

        # Group by observable
        by_obs = {}
        for m in matches:
            obs = m['observable']
            if obs not in by_obs:
                by_obs[obs] = []
            by_obs[obs].append(m)

        for obs_name in sorted(by_obs.keys()):
            obs_matches = sorted(by_obs[obs_name], key=lambda x: x['deviation_pct'])
            if len(obs_matches) > 0:
                f.write(f"### {obs_name}\n\n")
                f.write(f"Experimental value: {OBSERVABLES[obs_name]:.6f}\n\n")
                f.write("| Formula | Theoretical | Deviation (%) | Function Type |\n")
                f.write("|---------|-------------|---------------|---------------|\n")
                for m in obs_matches[:5]:  # Top 5 per observable
                    f.write(f"| {m['formula']} | {m['theoretical']:.6f} | ")
                    f.write(f"{m['deviation_pct']:.4f} | {m['function_type']} |\n")
                f.write("\n")

        f.write("## Mathematical Constants Reference\n\n")
        f.write("| Constant | Symbol | Value (50 digits) |\n")
        f.write("|----------|--------|-------------------|\n")
        f.write(f"| Catalan | G | {float(mp.catalan):.40f} |\n")
        f.write(f"| Euler-Mascheroni | γ | {float(mp.euler):.40f} |\n")
        f.write(f"| Apéry | ζ(3) | {float(mp.zeta(3)):.40f} |\n")
        f.write(f"| e^γ | - | {float(mp.exp(mp.euler)):.40f} |\n")
        f.write(f"| Li₂(1/2) | - | {float(polylog(2, mp.mpf(1)/2)):.40f} |\n")
        f.write(f"| P(2) | - | {float(prime_zeta(2)):.40f} |\n")
        f.write("\n")

        f.write("## Methodology\n\n")
        f.write("All calculations performed using mpmath library with 50 decimal places precision. ")
        f.write("Patterns generated systematically through:\n\n")
        f.write("1. Direct constant evaluation\n")
        f.write("2. Powers and roots\n")
        f.write("3. Products and ratios of pairs\n")
        f.write("4. Sums and differences\n")
        f.write("5. Special combinations (e.g., γ × ζ(n))\n")
        f.write("6. Integer multiples and simple fractions\n\n")

        f.write("## References\n\n")
        f.write("- Catalan's constant: OEIS A006752\n")
        f.write("- Euler-Mascheroni constant: OEIS A001620\n")
        f.write("- Apéry's constant ζ(3): OEIS A002117\n")
        f.write("- Polylogarithm functions: Lewin, L. (1981). Polylogarithms and Associated Functions\n")
        f.write("- Prime zeta function: Fröberg, C.-E. (1968). On the Prime Zeta Function\n")

def print_summary(matches):
    """Print summary to console."""
    print("\n" + "="*70)
    print("GIFT Framework - Phase 8: Number Theory Pattern Discovery")
    print("="*70)

    print(f"\nTotal patterns found: {len(matches)}")
    print(f"Patterns <1% deviation: {len([m for m in matches if m['deviation_pct'] < 1.0])}")
    print(f"Patterns <0.5% deviation: {len([m for m in matches if m['deviation_pct'] < 0.5])}")
    print(f"Patterns <0.1% deviation: {len([m for m in matches if m['deviation_pct'] < 0.1])}")

    # Count by function type
    by_type = {}
    for m in matches:
        func_type = m['function_type']
        by_type[func_type] = by_type.get(func_type, 0) + 1

    print("\nPatterns by function type:")
    for func_type in sorted(by_type.keys()):
        print(f"  {func_type}: {by_type[func_type]}")

    # Catalan matches
    catalan_matches = [m for m in matches if 'G' in m['formula'] or 'catalan' in m['function_type']]
    print(f"\nCatalan constant patterns: {len(catalan_matches)}")
    if catalan_matches:
        best_catalan = min(catalan_matches, key=lambda x: x['deviation_pct'])
        print(f"Best Catalan match: {best_catalan['observable']} = {best_catalan['formula']}")
        print(f"  Deviation: {best_catalan['deviation_pct']:.4f}%")

    # Top 10
    print("\nTop 10 patterns:")
    matches_sorted = sorted(matches, key=lambda x: x['deviation_pct'])
    for i, m in enumerate(matches_sorted[:10], 1):
        print(f"{i:2d}. {m['observable']:20s} = {m['formula']:30s}  ({m['deviation_pct']:.4f}%)")

    print("\n" + "="*70)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("Generating number theory patterns...")
    patterns = generate_number_theory_patterns()
    print(f"Generated {len(patterns)} theoretical patterns")

    print("\nMatching against 37 GIFT observables...")
    matches = find_matches(patterns, OBSERVABLES, tolerance_pct=1.0)

    print(f"Found {len(matches)} matches within 1% tolerance")

    # Save results
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    reports_dir = Path(__file__).resolve().parent.parent / 'reports'
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_file = data_dir / 'number_theory_patterns.csv'
    print(f"\nSaving CSV to {csv_file}...")
    save_results_csv(matches, csv_file)

    md_file = reports_dir / 'NUMBER_THEORY_PATTERNS_REPORT.md'
    print(f"Generating markdown report: {md_file}...")
    generate_markdown_report(matches, md_file)

    # Print summary
    print_summary(matches)

    print(f"\nFiles generated:")
    print(f"  1. {csv_file}")
    print(f"  2. {md_file}")
    print(f"  3. {Path(__file__).resolve()} (this script)")

if __name__ == '__main__':
    main()
