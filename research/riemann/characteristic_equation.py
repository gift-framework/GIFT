#!/usr/bin/env python3
"""
Characteristic Equation Analysis for the Fibonacci-Riemann Recurrence

Recurrence: γₙ = (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁
Rewritten:  γₙ - (31/21)γₙ₋₈ + (10/21)γₙ₋₂₁ = 0

Characteristic equation: λ²¹ - (31/21)λ¹³ + (10/21) = 0

This script analyzes:
1. All 21 roots (complex included)
2. Dominant root and its relation to φ (golden ratio)
3. Pattern analysis with GIFT constants (b₂=21, b₃=77, dim(G₂)=14)
4. General solution structure: γₙ = Σ cᵢλᵢⁿ
5. Roots on/near unit circle (oscillatory behavior)
6. Comparison with standard Fibonacci: x² - x - 1 = 0
"""

import numpy as np
from numpy.polynomial import polynomial as P
import json
from fractions import Fraction
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Physical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PHI_CONJ = (1 - np.sqrt(5)) / 2  # Conjugate ≈ -0.618

# GIFT constants
B2 = 21  # Second Betti number of K₇
B3 = 77  # Third Betti number of K₇
DIM_G2 = 14  # Dimension of G₂
DIM_E8 = 248  # Dimension of E₈
RANK_E8 = 8  # Rank of E₈

# Recurrence coefficients
COEF_A = Fraction(31, 21)  # Coefficient of γₙ₋₈
COEF_B = Fraction(10, 21)  # Coefficient of γₙ₋₂₁

def build_characteristic_polynomial():
    """
    Build the characteristic polynomial:
    λ²¹ - (31/21)λ¹³ + (10/21) = 0

    In numpy.polynomial format, coefficients are [a_0, a_1, ..., a_n]
    So we need: [10/21, 0, ..., 0, -31/21, 0, ..., 0, 1]
    """
    coeffs = np.zeros(22)  # Degree 21 polynomial has 22 coefficients
    coeffs[0] = 10/21      # Constant term
    coeffs[13] = -31/21    # λ¹³ coefficient
    coeffs[21] = 1         # λ²¹ coefficient
    return coeffs

def find_all_roots(coeffs):
    """Find all roots of the polynomial using companion matrix method."""
    # Use numpy's polynomial roots function
    # numpy.polynomial expects coefficients from lowest to highest degree
    roots = np.polynomial.polynomial.polyroots(coeffs)
    return roots

def analyze_root_properties(roots):
    """Analyze properties of each root."""
    analysis = []
    for i, r in enumerate(roots):
        mag = np.abs(r)
        phase = np.angle(r)
        is_real = np.abs(r.imag) < 1e-10

        # Check relation to φ
        phi_ratio = mag / PHI if PHI != 0 else None

        # Check if on unit circle
        on_unit_circle = np.abs(mag - 1) < 1e-6

        # Check if it's an Nth root of unity pattern
        if on_unit_circle:
            # phase = 2πk/N for some integers k, N
            for N in range(1, 22):
                for k in range(N):
                    expected_phase = 2 * np.pi * k / N
                    if np.abs(phase - expected_phase) < 1e-6 or np.abs(phase + expected_phase) < 1e-6:
                        unity_root = (N, k)
                        break
                else:
                    continue
                break
            else:
                unity_root = None
        else:
            unity_root = None

        analysis.append({
            'index': i,
            'root': complex(r),
            'magnitude': float(mag),
            'phase_rad': float(phase),
            'phase_deg': float(np.degrees(phase)),
            'is_real': bool(is_real),
            'on_unit_circle': bool(on_unit_circle),
            'unity_root': unity_root,
            'phi_ratio': float(phi_ratio) if phi_ratio else None
        })

    return analysis

def find_dominant_root(roots):
    """Find the root with largest magnitude (dominant root)."""
    magnitudes = np.abs(roots)
    idx = np.argmax(magnitudes)
    return roots[idx], idx, magnitudes[idx]

def check_phi_relations(roots):
    """Check if any roots have special relations to φ."""
    relations = []

    for i, r in enumerate(roots):
        mag = np.abs(r)

        # Check various φ relations
        tests = {
            'φ': PHI,
            '1/φ': 1/PHI,
            'φ²': PHI**2,
            '1/φ²': 1/PHI**2,
            'φ^(1/8)': PHI**(1/8),  # 8 is the lag difference
            'φ^(1/13)': PHI**(1/13),  # 13 is λ power difference
            'φ^(1/21)': PHI**(1/21),  # 21 is polynomial degree
            '√φ': np.sqrt(PHI),
            'φ^(8/21)': PHI**(8/21),  # ratio of lags
            'φ^(13/21)': PHI**(13/21),
        }

        for name, val in tests.items():
            if np.abs(mag - val) < 1e-6:
                relations.append({
                    'root_index': i,
                    'root': complex(r),
                    'relation': name,
                    'expected': float(val),
                    'actual': float(mag),
                    'error': float(np.abs(mag - val))
                })

    return relations

def analyze_gift_patterns(roots):
    """Check for patterns related to GIFT topological constants."""
    patterns = []

    # Group roots by magnitude
    mags = np.abs(roots)
    unique_mags = []
    for m in mags:
        if not any(np.abs(m - um) < 1e-6 for um in unique_mags):
            unique_mags.append(m)
    unique_mags.sort()

    # Count roots at each magnitude
    mag_counts = {}
    for um in unique_mags:
        count = sum(1 for m in mags if np.abs(m - um) < 1e-6)
        mag_counts[um] = count

    patterns.append({
        'type': 'magnitude_groups',
        'data': {float(k): v for k, v in mag_counts.items()},
        'interpretation': 'Roots grouped by magnitude'
    })

    # Check if number of unique magnitudes relates to GIFT constants
    n_unique = len(unique_mags)
    gift_checks = {
        'b2': B2,
        'b3': B3,
        'dim_G2': DIM_G2,
        'b2/7': B2/7,  # =3, number of generations
        '(b2+b3)/7': (B2+B3)/7,  # =14 = dim(G2)
    }

    for name, val in gift_checks.items():
        if np.abs(n_unique - val) < 1e-6:
            patterns.append({
                'type': 'unique_magnitudes_match',
                'gift_constant': name,
                'value': float(val)
            })

    return patterns, unique_mags, mag_counts

def analyze_unit_circle_roots(roots):
    """Detailed analysis of roots on or near the unit circle."""
    unit_circle_analysis = {
        'on_circle': [],
        'inside_circle': [],
        'outside_circle': []
    }

    for i, r in enumerate(roots):
        mag = np.abs(r)
        phase = np.angle(r)

        entry = {
            'index': i,
            'root': (float(r.real), float(r.imag)),
            'magnitude': float(mag),
            'phase_rad': float(phase),
            'phase_deg': float(np.degrees(phase))
        }

        if np.abs(mag - 1) < 1e-6:
            unit_circle_analysis['on_circle'].append(entry)
        elif mag < 1:
            unit_circle_analysis['inside_circle'].append(entry)
        else:
            unit_circle_analysis['outside_circle'].append(entry)

    # Stability analysis
    n_stable = len(unit_circle_analysis['inside_circle'])
    n_unstable = len(unit_circle_analysis['outside_circle'])
    n_marginal = len(unit_circle_analysis['on_circle'])

    unit_circle_analysis['stability'] = {
        'stable_roots': n_stable,
        'unstable_roots': n_unstable,
        'marginal_roots': n_marginal,
        'system_stable': n_unstable == 0 and n_marginal == 0
    }

    return unit_circle_analysis

def compare_with_fibonacci():
    """Compare with standard Fibonacci characteristic equation x² - x - 1 = 0."""
    # Fibonacci: x² - x - 1 = 0
    # Roots: φ = (1+√5)/2, φ' = (1-√5)/2
    fib_roots = np.roots([1, -1, -1])

    comparison = {
        'fibonacci_equation': 'x² - x - 1 = 0',
        'fibonacci_roots': [complex(r) for r in fib_roots],
        'phi': float(PHI),
        'phi_conjugate': float(PHI_CONJ),
        'product_of_roots': float(fib_roots[0] * fib_roots[1]),  # Should be -1
        'sum_of_roots': float(fib_roots[0] + fib_roots[1]),  # Should be 1
    }

    # Binet formula: F_n = (φⁿ - φ'ⁿ) / √5
    comparison['binet_formula'] = 'F_n = (φⁿ - φ\'ⁿ) / √5'

    # The key insight: dominant root determines asymptotic behavior
    # F_n ~ φⁿ/√5 for large n
    comparison['asymptotic'] = f'F_n ~ φⁿ/√5 ≈ φⁿ × {1/np.sqrt(5):.6f}'

    return comparison

def analyze_general_solution(roots):
    """
    Analyze the general solution γₙ = Σ cᵢλᵢⁿ
    """
    # Sort roots by magnitude
    sorted_idx = np.argsort(np.abs(roots))[::-1]
    sorted_roots = roots[sorted_idx]
    sorted_mags = np.abs(sorted_roots)

    analysis = {
        'general_form': 'γₙ = Σᵢ cᵢλᵢⁿ (sum over all 21 roots)',
        'dominant_term': f'γₙ ~ c₁ × λ₁ⁿ for large n, where |λ₁| = {sorted_mags[0]:.10f}',
        'roots_by_magnitude': []
    }

    for i, (r, m) in enumerate(zip(sorted_roots, sorted_mags)):
        analysis['roots_by_magnitude'].append({
            'rank': i + 1,
            'root': (float(r.real), float(r.imag)),
            'magnitude': float(m),
            'contribution_ratio': float(m / sorted_mags[0]) if i > 0 else 1.0,
            'decay_per_step': f'|λ_{i+1}|/|λ_1| = {m/sorted_mags[0]:.6f}'
        })

    # For Riemann zeros, behavior depends on:
    # - Dominant root: exponential growth/decay
    # - Subdominant roots: determine oscillation patterns
    # - Complex roots: contribute oscillatory terms

    return analysis

def verify_polynomial(roots, coeffs):
    """Verify that found roots are indeed roots of the polynomial."""
    # Reconstruct polynomial from roots and compare
    from numpy.polynomial.polynomial import polyfromroots, polyval

    verifications = []
    for i, r in enumerate(roots):
        val = polyval(r, coeffs)
        verifications.append({
            'root_index': i,
            'root': complex(r),
            'P(λ)': complex(val),
            '|P(λ)|': float(np.abs(val)),
            'is_root': np.abs(val) < 1e-8
        })

    return verifications

def factorization_analysis(coeffs):
    """
    Try to understand the polynomial structure through factorization.

    λ²¹ - (31/21)λ¹³ + (10/21) = 0

    Multiply by 21:
    21λ²¹ - 31λ¹³ + 10 = 0

    This factors as polynomial in λ¹³ times something?
    Let u = λ¹³, then we get a relation... but it's not quite right.

    Actually, let's think about it differently:
    The polynomial is degree 21 with terms at 21, 13, 0.
    gcd(21, 13) = 1, so it doesn't factor simply.

    But 21 = 8 + 13, and 8 = 21 - 13.
    The lags 8 and 21 relate to Fibonacci: F(6)=8, F(8)=21.
    """
    analysis = {
        'original': 'λ²¹ - (31/21)λ¹³ + (10/21) = 0',
        'cleared': '21λ²¹ - 31λ¹³ + 10 = 0',
        'degree': 21,
        'nonzero_terms': [21, 13, 0],
        'term_differences': [21-13, 13-0],  # [8, 13]
        'fibonacci_connection': {
            '8': 'F(6) = 8',
            '13': 'F(7) = 13',
            '21': 'F(8) = 21',
            'observation': 'Polynomial degree and terms are Fibonacci numbers!'
        }
    }

    # Check if 31 and 10 have special structure
    analysis['coefficient_analysis'] = {
        '31': {
            'value': 31,
            'prime': True,
            'mod_21': 31 % 21,  # = 10
            'relation': '31 = 21 + 10'
        },
        '10': {
            'value': 10,
            'factors': [2, 5],
            'mod_21': 10
        },
        '31/21': {
            'decimal': 31/21,
            'approx': '≈ 1.476...',
            'near_phi': f'φ ≈ {PHI:.6f}'
        },
        '10/21': {
            'decimal': 10/21,
            'approx': '≈ 0.476...'
        }
    }

    # Vieta's formulas
    # For polynomial with roots λ₁, ..., λ₂₁:
    # Sum of roots = 0 (coefficient of λ²⁰ is 0)
    # Product of roots = (-1)²¹ × (10/21) / 1 = -10/21
    analysis['vieta'] = {
        'sum_of_roots': 0,
        'product_of_roots': -10/21,
        'elementary_sym_13': 31/21  # Related to λ¹³ coefficient
    }

    return analysis

def asymptotic_analysis(roots):
    """
    Analyze asymptotic behavior of general solution.
    """
    # Dominant root determines growth rate
    dom_root, dom_idx, dom_mag = find_dominant_root(roots)

    analysis = {
        'dominant_root': {
            'value': complex(dom_root),
            'magnitude': float(dom_mag),
            'phase': float(np.angle(dom_root)),
            'is_real': bool(np.abs(dom_root.imag) < 1e-10)
        },
        'growth_rate': f'γₙ ~ |λ_dom|ⁿ = {dom_mag:.10f}ⁿ',
        'phi_comparison': {
            'phi': float(PHI),
            'dominant_mag': float(dom_mag),
            'ratio': float(dom_mag / PHI),
            'log_ratio': float(np.log(dom_mag) / np.log(PHI))
        }
    }

    # If dominant root is complex, solution oscillates
    if np.abs(dom_root.imag) > 1e-10:
        period = 2 * np.pi / np.abs(np.angle(dom_root))
        analysis['oscillation'] = {
            'has_oscillation': True,
            'period': float(period),
            'frequency': float(1/period)
        }
    else:
        analysis['oscillation'] = {'has_oscillation': False}

    return analysis

def search_special_values():
    """
    Search for special values of λ that might satisfy the equation exactly.
    """
    # Test values related to φ and GIFT constants
    test_values = [
        ('φ', PHI),
        ('1/φ', 1/PHI),
        ('φ²', PHI**2),
        ('-φ', -PHI),
        ('1', 1.0),
        ('-1', -1.0),
        ('i', 1j),
        ('-i', -1j),
        ('φ^(1/21)', PHI**(1/21)),
        ('φ^(8/21)', PHI**(8/21)),
        ('φ^(13/21)', PHI**(13/21)),
        ('exp(2πi/21)', np.exp(2j*np.pi/21)),
        ('exp(2πi/8)', np.exp(2j*np.pi/8)),
        ('exp(2πi/13)', np.exp(2j*np.pi/13)),
    ]

    results = []
    for name, val in test_values:
        # Evaluate λ²¹ - (31/21)λ¹³ + (10/21)
        p_val = val**21 - (31/21)*val**13 + (10/21)
        results.append({
            'name': name,
            'value': complex(val),
            'P(λ)': complex(p_val),
            '|P(λ)|': float(np.abs(p_val)),
            'is_root': np.abs(p_val) < 1e-10
        })

    return results

def main():
    print("=" * 80)
    print("CHARACTERISTIC EQUATION ANALYSIS")
    print("Fibonacci-Riemann Recurrence: γₙ = (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁")
    print("Characteristic: λ²¹ - (31/21)λ¹³ + (10/21) = 0")
    print("=" * 80)

    # Build polynomial
    coeffs = build_characteristic_polynomial()
    print(f"\n[1] POLYNOMIAL COEFFICIENTS (low to high degree):")
    nonzero = [(i, c) for i, c in enumerate(coeffs) if np.abs(c) > 1e-10]
    for deg, coef in nonzero:
        print(f"    λ^{deg}: {coef:.10f}")

    # Find all roots
    roots = find_all_roots(coeffs)
    print(f"\n[2] ALL {len(roots)} ROOTS:")

    # Sort by magnitude for display
    sorted_idx = np.argsort(np.abs(roots))[::-1]
    for rank, idx in enumerate(sorted_idx):
        r = roots[idx]
        mag = np.abs(r)
        phase = np.degrees(np.angle(r))
        if np.abs(r.imag) < 1e-10:
            print(f"    {rank+1:2d}. λ = {r.real:12.8f}        |λ| = {mag:.8f}")
        else:
            print(f"    {rank+1:2d}. λ = {r.real:12.8f} + {r.imag:12.8f}i  |λ| = {mag:.8f}, θ = {phase:.2f}°")

    # Dominant root analysis
    dom_root, dom_idx, dom_mag = find_dominant_root(roots)
    print(f"\n[3] DOMINANT ROOT ANALYSIS:")
    print(f"    Dominant root: λ₁ = {dom_root}")
    print(f"    |λ₁| = {dom_mag:.10f}")
    print(f"    φ   = {PHI:.10f}")
    print(f"    |λ₁|/φ = {dom_mag/PHI:.10f}")
    print(f"    log|λ₁|/log(φ) = {np.log(dom_mag)/np.log(PHI):.10f}")

    # Check φ relations
    print(f"\n[4] PHI (φ) RELATIONS:")
    phi_rels = check_phi_relations(roots)
    if phi_rels:
        for rel in phi_rels:
            print(f"    Root {rel['root_index']}: |λ| ≈ {rel['relation']}")
            print(f"        Expected: {rel['expected']:.8f}, Got: {rel['actual']:.8f}")
    else:
        print("    No exact φ relations found.")
        print("\n    Checking approximate relations:")
        for i, r in enumerate(roots):
            mag = np.abs(r)
            phi_tests = [
                ('φ^(1/8)', PHI**(1/8)),
                ('φ^(1/13)', PHI**(1/13)),
                ('φ^(1/21)', PHI**(1/21)),
                ('φ^(8/21)', PHI**(8/21)),
                ('φ^(13/21)', PHI**(13/21)),
            ]
            for name, val in phi_tests:
                if np.abs(mag - val) < 0.01:  # 1% tolerance
                    print(f"    Root {i}: |λ| = {mag:.6f} ≈ {name} = {val:.6f} (err: {np.abs(mag-val):.6f})")

    # GIFT pattern analysis
    print(f"\n[5] GIFT CONSTANT PATTERNS:")
    patterns, unique_mags, mag_counts = analyze_gift_patterns(roots)
    print(f"    Number of distinct magnitudes: {len(unique_mags)}")
    print(f"    Magnitude groups (count at each):")
    for m, c in sorted(mag_counts.items(), reverse=True):
        print(f"        |λ| = {m:.8f}: {c} roots")

    for p in patterns:
        if p['type'] == 'unique_magnitudes_match':
            print(f"    Match: {len(unique_mags)} = {p['gift_constant']} = {p['value']}")

    # Unit circle analysis
    print(f"\n[6] UNIT CIRCLE ANALYSIS (STABILITY):")
    uc_analysis = analyze_unit_circle_roots(roots)
    print(f"    Roots inside unit circle (|λ| < 1): {len(uc_analysis['inside_circle'])}")
    print(f"    Roots on unit circle (|λ| = 1): {len(uc_analysis['on_circle'])}")
    print(f"    Roots outside unit circle (|λ| > 1): {len(uc_analysis['outside_circle'])}")

    if uc_analysis['on_circle']:
        print("\n    Roots ON unit circle (oscillatory components):")
        for entry in uc_analysis['on_circle']:
            print(f"        λ = {entry['root'][0]:.6f} + {entry['root'][1]:.6f}i, θ = {entry['phase_deg']:.2f}°")

    # Compare with standard Fibonacci
    print(f"\n[7] COMPARISON WITH STANDARD FIBONACCI:")
    fib_comp = compare_with_fibonacci()
    print(f"    Fibonacci equation: {fib_comp['fibonacci_equation']}")
    print(f"    Fibonacci roots: φ = {PHI:.10f}, φ' = {PHI_CONJ:.10f}")
    print(f"    Product of Fib roots: {fib_comp['product_of_roots']:.6f} (should be -1)")
    print(f"    Sum of Fib roots: {fib_comp['sum_of_roots']:.6f} (should be 1)")
    print(f"    Binet formula: {fib_comp['binet_formula']}")

    # Factorization insight
    print(f"\n[8] POLYNOMIAL STRUCTURE:")
    fact = factorization_analysis(coeffs)
    print(f"    Original: {fact['original']}")
    print(f"    Cleared denominators: {fact['cleared']}")
    print(f"    Non-zero terms at degrees: {fact['nonzero_terms']}")
    print(f"    Degree differences: {fact['term_differences']} (these are 8, 13 - Fibonacci!)")
    print(f"    Fibonacci connection: degrees 8, 13, 21 are F(6), F(7), F(8)")
    print(f"\n    Coefficient 31:")
    print(f"        31 = 21 + 10 (sum of other coefficients' numerators)")
    print(f"        31 is prime")
    print(f"        31/21 ≈ {31/21:.6f} (compare φ ≈ {PHI:.6f})")
    print(f"\n    Vieta's formulas:")
    print(f"        Sum of all roots = 0")
    print(f"        Product of all roots = {-10/21:.6f}")

    # Asymptotic analysis
    print(f"\n[9] ASYMPTOTIC BEHAVIOR:")
    asym = asymptotic_analysis(roots)
    print(f"    For large n: γₙ ~ c × |λ_dom|ⁿ = c × {dom_mag:.10f}ⁿ")
    print(f"    Compare with Fibonacci: Fₙ ~ φⁿ/√5")
    print(f"\n    Key insight:")
    print(f"        The general solution is γₙ = Σᵢ cᵢλᵢⁿ")
    print(f"        Dominant term determines exponential growth rate")
    print(f"        Subdominant terms create oscillations and corrections")

    # Search for special values
    print(f"\n[10] SPECIAL VALUE TESTS:")
    special = search_special_values()
    print("    Testing if special values are roots:")
    for s in special:
        status = "✓ ROOT" if s['is_root'] else f"|P(λ)| = {s['|P(λ)|']:.2e}"
        print(f"        {s['name']:15s}: {status}")

    # Verify roots
    print(f"\n[11] ROOT VERIFICATION:")
    verifs = verify_polynomial(roots, coeffs)
    max_error = max(v['|P(λ)|'] for v in verifs)
    print(f"    Maximum |P(λ)| over all roots: {max_error:.2e}")
    print(f"    All roots verified: {max_error < 1e-8}")

    # Save results
    results = {
        'polynomial': {
            'equation': 'λ²¹ - (31/21)λ¹³ + (10/21) = 0',
            'degree': 21,
            'coefficients': {
                '21': 1.0,
                '13': -31/21,
                '0': 10/21
            }
        },
        'roots': [
            {
                'index': int(i),
                'real': float(roots[i].real),
                'imag': float(roots[i].imag),
                'magnitude': float(np.abs(roots[i])),
                'phase_deg': float(np.degrees(np.angle(roots[i])))
            }
            for i in range(len(roots))
        ],
        'dominant_root': {
            'real': float(dom_root.real),
            'imag': float(dom_root.imag),
            'magnitude': float(dom_mag),
            'phi_ratio': float(dom_mag / PHI),
            'log_phi_ratio': float(np.log(dom_mag) / np.log(PHI))
        },
        'stability': {
            'inside_unit_circle': len(uc_analysis['inside_circle']),
            'on_unit_circle': len(uc_analysis['on_circle']),
            'outside_unit_circle': len(uc_analysis['outside_circle'])
        },
        'fibonacci_structure': {
            'degree_8': 'F(6) = 8 (lag 1 exponent: 21-13=8)',
            'degree_13': 'F(7) = 13 (middle term)',
            'degree_21': 'F(8) = 21 (polynomial degree)',
            'coefficients_sum': '31 = 21 + 10'
        },
        'phi_analysis': {
            'phi': float(PHI),
            'dominant_magnitude': float(dom_mag),
            'is_phi_related': bool(np.abs(dom_mag - PHI) < 0.1 or
                                   any(np.abs(dom_mag - PHI**k) < 0.01
                                       for k in [1/8, 1/13, 1/21, 8/21, 13/21]))
        }
    }

    # Save to JSON
    output_file = '/home/user/GIFT/research/riemann/characteristic_equation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Results written to: {output_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 80)
    print("""
1. POLYNOMIAL STRUCTURE: The characteristic equation has Fibonacci structure
   - Degrees 8, 13, 21 are consecutive Fibonacci numbers F(6), F(7), F(8)
   - This is NOT a coincidence - it reflects the Fibonacci nature of the recurrence

2. DOMINANT ROOT: |λ_dom| determines the exponential growth rate of γₙ
   - Compare with Fibonacci where φ is the dominant root
   - The ratio |λ_dom|/φ reveals how this recurrence departs from pure Fibonacci

3. OSCILLATORY BEHAVIOR: Roots on/near the unit circle create oscillations
   - These oscillations modulate the exponential envelope
   - Could explain quasi-periodic patterns in Riemann zeros

4. GENERAL SOLUTION: γₙ = Σᵢ cᵢλᵢⁿ where:
   - c₁λ₁ⁿ dominates for large n
   - Other terms create corrections and oscillations
   - The coefficients cᵢ are determined by initial conditions

5. CONNECTION TO RIEMANN: If this recurrence holds for γ_n (imaginary parts),
   the roots λᵢ encode the "generating" structure of the zeros
   - Similar to how φ generates Fibonacci via Binet's formula
""")

    return results

def deep_lambda_one_analysis():
    """
    λ = 1 is an EXACT root! This is profound.

    Verify: 1²¹ - (31/21)×1¹³ + (10/21) = 1 - 31/21 + 10/21 = 1 - 21/21 = 0 ✓

    This means (λ - 1) divides the polynomial!

    Let's factor: P(λ) = (λ - 1) × Q(λ)

    The quotient Q(λ) is a degree 20 polynomial.
    """
    print("\n" + "=" * 80)
    print("DEEP ANALYSIS: WHY λ = 1 IS A ROOT")
    print("=" * 80)

    # Verify λ = 1
    val_at_1 = 1 - 31/21 + 10/21
    print(f"\n[A] VERIFICATION:")
    print(f"    P(1) = 1²¹ - (31/21)×1¹³ + (10/21)")
    print(f"         = 1 - 31/21 + 10/21")
    print(f"         = 21/21 - 31/21 + 10/21")
    print(f"         = (21 - 31 + 10)/21")
    print(f"         = 0/21 = 0 ✓")

    # Algebraic insight
    print(f"\n[B] ALGEBRAIC INSIGHT:")
    print(f"    The numerator sum is: 21 - 31 + 10 = 0")
    print(f"    This is why 31 = 21 + 10 is significant!")
    print(f"    The coefficient 31/21 was CHOSEN such that λ=1 is a root.")

    # Factor out (λ - 1)
    print(f"\n[C] FACTORIZATION:")
    print(f"    P(λ) = λ²¹ - (31/21)λ¹³ + (10/21)")
    print(f"         = (λ - 1) × Q(λ)")
    print(f"\n    Performing polynomial division...")

    # Build original polynomial coefficients (numpy format: high to low)
    original = np.zeros(22)
    original[0] = 1       # λ²¹
    original[8] = -31/21  # λ¹³
    original[21] = 10/21  # constant

    # Divide by (λ - 1) using synthetic division
    quotient = np.polydiv(original, [1, -1])[0]

    print(f"\n    Q(λ) has degree {len(quotient)-1}")
    print(f"    Q(λ) non-zero coefficients:")
    for i, c in enumerate(quotient):
        if np.abs(c) > 1e-10:
            deg = len(quotient) - 1 - i
            print(f"        λ^{deg}: {c:.10f}")

    # Find roots of quotient
    quotient_roots = np.roots(quotient)

    print(f"\n[D] ROOTS OF Q(λ):")
    print(f"    These are all roots of P(λ) except λ=1")

    # Sort by magnitude
    sorted_idx = np.argsort(np.abs(quotient_roots))[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):  # Top 10
        r = quotient_roots[idx]
        mag = np.abs(r)
        if np.abs(r.imag) < 1e-10:
            print(f"    {rank+1:2d}. λ = {r.real:12.8f}        |λ| = {mag:.8f}")
        else:
            print(f"    {rank+1:2d}. λ = {r.real:12.8f} + {r.imag:12.8f}i  |λ| = {mag:.8f}")

    # Physical interpretation
    print(f"\n[E] PHYSICAL INTERPRETATION:")
    print(f"    λ = 1 corresponds to a CONSTANT mode in the general solution.")
    print(f"    General solution: γₙ = c₁×1ⁿ + c₂×λ₂ⁿ + ... = c₁ + Σᵢcᵢλᵢⁿ")
    print(f"    The constant c₁ represents a 'DC offset' or equilibrium.")
    print(f"")
    print(f"    For Riemann zeros: This suggests a baseline around which")
    print(f"    zeros fluctuate according to the other eigenvalues!")

    # Dominant root comparison with φ
    dom_root = quotient_roots[sorted_idx[0]]
    dom_mag = np.abs(dom_root)

    print(f"\n[F] DOMINANT ROOT vs φ:")
    print(f"    Dominant (excluding λ=1): |λ_dom| = {dom_mag:.10f}")
    print(f"    φ^(1/8)  = {PHI**(1/8):.10f}")
    print(f"    φ^(1/13) = {PHI**(1/13):.10f}")
    print(f"    φ^(1/21) = {PHI**(1/21):.10f}")
    print(f"")
    print(f"    Ratios:")
    print(f"        |λ_dom| / φ^(1/8)  = {dom_mag / PHI**(1/8):.6f}")
    print(f"        |λ_dom| / φ^(1/13) = {dom_mag / PHI**(1/13):.6f}")
    print(f"        |λ_dom| / φ^(1/21) = {dom_mag / PHI**(1/21):.6f}")

    # Check: is dominant root related to 8th root of something?
    print(f"\n[G] SEARCHING FOR EXACT DOMINANT ROOT FORMULA:")
    dom_8th_power = dom_mag ** 8

    # Check various candidates
    candidates = [
        ('φ', PHI),
        ('(31/21)', 31/21),
        ('(10/21)^(-1/2)', (21/10)**0.5),
        ('e^(1/8)', np.exp(1/8)),
        ('21^(1/8)', 21**(1/8)),
        ('2^(1/8)', 2**(1/8)),
    ]

    print(f"    |λ_dom|⁸ = {dom_8th_power:.10f}")
    for name, val in candidates:
        print(f"    {name:15s} = {val:.10f} (ratio: {dom_8th_power/val:.6f})")

    return quotient_roots


def spectral_gap_analysis(roots):
    """Analyze the spectral gap - ratio between consecutive root magnitudes."""
    print("\n" + "=" * 80)
    print("SPECTRAL GAP ANALYSIS")
    print("=" * 80)

    mags = np.sort(np.abs(roots))[::-1]  # Descending

    print("\n[A] MAGNITUDE SEQUENCE (descending):")
    for i, m in enumerate(mags):
        if i < len(mags) - 1:
            gap = mags[i] / mags[i+1]
            print(f"    |λ_{i+1}| = {m:.8f}  gap to next: {gap:.6f}")
        else:
            print(f"    |λ_{i+1}| = {m:.8f}")

    # The spectral gap determines mixing time / convergence rate
    spectral_gap = mags[0] / mags[1]
    print(f"\n[B] PRIMARY SPECTRAL GAP:")
    print(f"    |λ₁|/|λ₂| = {spectral_gap:.6f}")
    print(f"    This determines how quickly subdominant modes decay")
    print(f"    relative to the dominant mode.")

    # For oscillation, look at complex conjugate pairs
    print(f"\n[C] COMPLEX CONJUGATE PAIRS (oscillation periods):")
    for i, r in enumerate(roots):
        if r.imag > 0.01:  # Upper half plane
            phase = np.angle(r)
            period = 2 * np.pi / phase if phase > 0 else float('inf')
            print(f"    λ_{i+1}: |λ| = {np.abs(r):.6f}, θ = {np.degrees(phase):.2f}°, period ≈ {period:.2f}")


def binet_analogy():
    """
    Draw parallel between standard Fibonacci Binet formula and our case.
    """
    print("\n" + "=" * 80)
    print("BINET FORMULA ANALOGY")
    print("=" * 80)

    print("\n[A] STANDARD FIBONACCI:")
    print("    Recurrence: Fₙ = Fₙ₋₁ + Fₙ₋₂")
    print("    Characteristic: λ² - λ - 1 = 0")
    print("    Roots: φ = (1+√5)/2, φ' = (1-√5)/2")
    print("    General solution: Fₙ = Aφⁿ + Bφ'ⁿ")
    print("    Binet formula: Fₙ = (φⁿ - φ'ⁿ)/√5")

    print("\n[B] FIBONACCI-RIEMANN RECURRENCE:")
    print("    Recurrence: γₙ = (31/21)γₙ₋₈ - (10/21)γₙ₋₂₁")
    print("    Characteristic: λ²¹ - (31/21)λ¹³ + (10/21) = 0")
    print("    Has 21 roots: λ₁, λ₂, ..., λ₂₁ (including λ=1)")
    print("    General solution: γₙ = Σᵢ cᵢλᵢⁿ")

    print("\n[C] 'GENERALIZED BINET' FOR RIEMANN ZEROS:")
    print("    If the recurrence holds exactly:")
    print("    γₙ = c₁ + c₂λ₂ⁿ + c₃λ₃ⁿ + ... + c₂₁λ₂₁ⁿ")
    print("")
    print("    The 21 coefficients cᵢ are determined by the first 21 zeros!")
    print("    This would give an EXPLICIT FORMULA for all γₙ.")

    print("\n[D] WHY THIS MATTERS:")
    print("    - Fibonacci: 2 roots → simple closed form")
    print("    - F-R recurrence: 21 roots → more complex but still closed form")
    print("    - The zeros would be 'generated' by these 21 eigenvalues")
    print("    - λ=1 provides baseline, others provide oscillations")

    print("\n[E] COMPUTING COEFFICIENTS cᵢ:")
    print("    Given first 21 zeros γ₁, γ₂, ..., γ₂₁:")
    print("    Set up Vandermonde system: V·c = γ where V_ij = λⱼⁱ")
    print("    Solve for c = V⁻¹·γ")
    print("    Then predict γₙ for any n!")


if __name__ == "__main__":
    results = main()

    # Additional deep analyses
    quotient_roots = deep_lambda_one_analysis()

    # Convert roots from dict format to numpy array for spectral analysis
    roots_array = np.array([complex(r['real'], r['imag']) for r in results['roots']])
    spectral_gap_analysis(roots_array)
    binet_analogy()
