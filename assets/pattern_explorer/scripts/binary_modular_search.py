#!/usr/bin/env python3
"""
Binary and Modular Structure Analysis for GIFT Framework Observables
Analyzes 37 dimensionless observables for binary, modular, and base-conversion patterns
"""

import numpy as np
import math
from fractions import Fraction
from typing import List, Tuple, Dict
import csv

# Mathematical constants
PI = np.pi
E = np.e
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
EULER_GAMMA = 0.5772156649015329
DELTA_F = 4.669201609102990  # Feigenbaum constant

def zeta(n):
    """Riemann zeta function for specific values"""
    zeta_values = {
        2: np.pi**2 / 6,
        3: 1.2020569031595942,
        4: np.pi**4 / 90,
        5: 1.0369277551433699,
        6: np.pi**6 / 945,
        7: 1.0083492773819228,
        9: 1.0020083928260822,
        11: 1.0004941886041194,
    }
    return zeta_values.get(n, 1.0)

# Framework topological parameters
dim_K7 = 7
dim_G2 = 14
b2 = 21
b3 = 77
H_star = 99
dim_E8 = 248
rank_E8 = 8
p2 = 2
Weyl = 5
N_gen = 3
M2 = 3  # Mersenne 2^2-1
M3 = 7  # Mersenne 2^3-1
M5 = 31  # Mersenne 2^5-1
M13 = 8191  # Mersenne 2^13-1
F2 = 17  # Fermat 2^(2^2)+1
tau = 10416 / 2673
gamma_GIFT = 511 / 884
delta = 2 * PI / 25
xi = 5 * PI / 16
beta0 = PI / 8

# Define all 37 observables with experimental values
observables = {
    # Gauge sector (3)
    'alpha_inv': 137.036,
    'alpha_s': 0.1179,
    'sin2_theta_W': 0.23122,

    # Neutrino sector (4)
    'theta_12': 33.44,  # degrees
    'theta_13': 8.57,   # degrees
    'theta_23': 49.2,   # degrees
    'delta_CP': 197.0,  # degrees

    # Lepton sector (3)
    'Q_Koide': 0.6667,
    'm_mu_over_me': 206.768,
    'm_tau_over_me': 3477.15,

    # Quark masses (6) in MeV
    'm_u': 2.16,
    'm_d': 4.67,
    'm_s': 93.4,
    'm_c': 1270.0,
    'm_b': 4180.0,
    'm_t': 172500.0,

    # Quark mass ratios (10)
    'm_s_over_md': 20.0,
    'm_b_over_mu': 1935.19,
    'm_c_over_md': 271.94,
    'm_d_over_mu': 2.162,
    'm_c_over_ms': 13.6,
    'm_t_over_mc': 135.83,
    'm_b_over_md': 895.07,
    'm_b_over_mc': 3.29,
    'm_t_over_ms': 1846.89,
    'm_b_over_ms': 44.76,

    # CKM matrix (1)
    'theta_C': 13.04,  # degrees

    # Higgs sector (3)
    'lambda_H': 0.1286,
    'v_EW': 246.22,  # GeV
    'm_H': 125.25,   # GeV

    # Cosmology (4)
    'Omega_DE': 0.6847,
    'Omega_DM': 0.120,
    'n_s': 0.9649,
    'H_0': 73.04,  # km/s/Mpc

    # Dark matter (2)
    'm_chi1': 90.5,   # GeV
    'm_chi2': 352.7,  # GeV

    # Temporal structure (1)
    'D_H': 0.856220,
}


class BinaryAnalyzer:
    """Analyzes binary representations and patterns"""

    @staticmethod
    def to_binary_fraction(value, max_bits=20):
        """Convert decimal to binary fraction representation"""
        integer_part = int(value)
        fractional_part = value - integer_part

        # Integer part to binary
        int_binary = bin(integer_part)[2:] if integer_part > 0 else '0'

        # Fractional part to binary
        frac_binary = ''
        seen = set()
        while fractional_part > 0 and len(frac_binary) < max_bits:
            if fractional_part in seen:
                break
            seen.add(fractional_part)
            fractional_part *= 2
            bit = int(fractional_part)
            frac_binary += str(bit)
            fractional_part -= bit

        return f"{int_binary}.{frac_binary}" if frac_binary else int_binary

    @staticmethod
    def is_dyadic_rational(value, max_q=20):
        """Test if value is approximately p/2^q for integers p, q"""
        best_match = None
        best_dev = float('inf')

        for q in range(1, max_q + 1):
            denominator = 2**q
            p = round(value * denominator)
            approx = p / denominator
            deviation = abs(value - approx) / value if value != 0 else abs(value - approx)

            if deviation < best_dev:
                best_dev = deviation
                best_match = (p, q, approx, deviation * 100)

        return best_match

    @staticmethod
    def test_power_of_2_pattern(value, n_range=(-10, 10), m_range=(-10, 10)):
        """Test 2^n / 2^m patterns"""
        best_matches = []

        for n in range(n_range[0], n_range[1] + 1):
            for m in range(m_range[0], m_range[1] + 1):
                if n == m:
                    continue
                approx = 2**n / 2**m if m != 0 else 2**n
                deviation = abs(value - approx) / value if value != 0 else abs(value - approx)

                if deviation < 0.01:  # Less than 1%
                    best_matches.append({
                        'formula': f'2^{n}/2^{m}',
                        'value': approx,
                        'deviation_pct': deviation * 100
                    })

        return best_matches

    @staticmethod
    def test_power_of_2_with_constants(value, n_range=(-10, 10)):
        """Test 2^n × constant patterns"""
        constants = {
            'pi': PI,
            'e': E,
            'phi': PHI,
            'gamma': EULER_GAMMA,
            'delta_F': DELTA_F,
            'ln(2)': np.log(2),
            'sqrt(2)': np.sqrt(2),
            'zeta(3)': zeta(3),
            'zeta(5)': zeta(5),
            'zeta(7)': zeta(7),
            'tau': tau,
            'M3': M3,
            'M5': M5,
        }

        best_matches = []
        for n in range(n_range[0], n_range[1] + 1):
            for const_name, const_val in constants.items():
                approx = (2**n) * const_val
                deviation = abs(value - approx) / value if value != 0 else abs(value - approx)

                if deviation < 0.01:
                    best_matches.append({
                        'formula': f'2^{n} × {const_name}',
                        'value': approx,
                        'deviation_pct': deviation * 100
                    })

                # Also test division
                if const_val != 0:
                    approx = (2**n) / const_val
                    deviation = abs(value - approx) / value if value != 0 else abs(value - approx)

                    if deviation < 0.01:
                        best_matches.append({
                            'formula': f'2^{n} / {const_name}',
                            'value': approx,
                            'deviation_pct': deviation * 100
                        })

        return best_matches


class ModularAnalyzer:
    """Analyzes modular arithmetic patterns"""

    @staticmethod
    def find_rational_approximation(value, max_denominator=1000):
        """Find best rational approximation"""
        frac = Fraction(value).limit_denominator(max_denominator)
        return frac.numerator, frac.denominator, float(frac)

    @staticmethod
    def test_mersenne_patterns(value):
        """Test patterns involving Mersenne numbers"""
        mersenne_exponents = [2, 3, 5, 7, 13, 17, 19]
        mersenne_numbers = {exp: 2**exp - 1 for exp in mersenne_exponents}

        patterns = []
        for exp1, M1 in mersenne_numbers.items():
            for exp2, M2 in mersenne_numbers.items():
                if M2 == 0:
                    continue

                # Test M1/M2
                approx = M1 / M2
                deviation = abs(value - approx) / value if value != 0 else abs(value - approx)
                if deviation < 0.01:
                    patterns.append({
                        'formula': f'M_{exp1}/M_{exp2} = {M1}/{M2}',
                        'value': approx,
                        'deviation_pct': deviation * 100
                    })

                # Test (M1 mod M2) / M2
                if M2 > 1:
                    mod_result = M1 % M2
                    approx = mod_result / M2
                    deviation = abs(value - approx) / value if value != 0 else abs(value - approx)
                    if deviation < 0.01:
                        patterns.append({
                            'formula': f'(M_{exp1} mod M_{exp2})/M_{exp2} = {mod_result}/{M2}',
                            'value': approx,
                            'deviation_pct': deviation * 100
                        })

        return patterns

    @staticmethod
    def test_fermat_patterns(value):
        """Test patterns involving Fermat primes"""
        fermat_primes = {
            0: 3,   # F0 = 2^(2^0) + 1
            1: 5,   # F1 = 2^(2^1) + 1
            2: 17,  # F2 = 2^(2^2) + 1
        }

        patterns = []
        for n1, F1 in fermat_primes.items():
            for n2, F2 in fermat_primes.items():
                if F2 == 0:
                    continue

                approx = F1 / F2
                deviation = abs(value - approx) / value if value != 0 else abs(value - approx)
                if deviation < 0.01:
                    patterns.append({
                        'formula': f'F_{n1}/F_{n2} = {F1}/{F2}',
                        'value': approx,
                        'deviation_pct': deviation * 100
                    })

        return patterns


class BaseConverter:
    """Convert numbers to different bases"""

    @staticmethod
    def to_base(num, base, precision=10):
        """Convert number to given base"""
        if num == 0:
            return "0"

        integer_part = int(num)
        fractional_part = num - integer_part

        # Convert integer part
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        if integer_part == 0:
            result = "0"
        else:
            while integer_part > 0:
                result = digits[integer_part % base] + result
                integer_part //= base

        # Convert fractional part
        if fractional_part > 0:
            result += "."
            for _ in range(precision):
                fractional_part *= base
                digit = int(fractional_part)
                result += digits[digit]
                fractional_part -= digit
                if fractional_part == 0:
                    break

        return result

    @staticmethod
    def analyze_base_patterns(value):
        """Analyze patterns in different bases"""
        bases = {
            2: 'binary',
            3: 'ternary',
            8: 'octal',
            12: 'duodecimal',
            16: 'hexadecimal'
        }

        representations = {}
        for base, name in bases.items():
            representations[name] = BaseConverter.to_base(value, base)

        return representations


class PeriodDoublingAnalyzer:
    """Analyzes period doubling and scaling relationships"""

    @staticmethod
    def test_period_doubling(obs_dict):
        """Test if observable × 2^n matches other observables"""
        matches = []

        for name1, val1 in obs_dict.items():
            for n in range(-10, 11):
                scaled = val1 * (2**n)

                for name2, val2 in obs_dict.items():
                    if name1 == name2:
                        continue

                    deviation = abs(scaled - val2) / val2 if val2 != 0 else abs(scaled - val2)
                    if deviation < 0.01:
                        matches.append({
                            'observable1': name1,
                            'observable2': name2,
                            'formula': f'{name1} × 2^{n}',
                            'scaled_value': scaled,
                            'target_value': val2,
                            'deviation_pct': deviation * 100
                        })

        return matches


def analyze_all_observables():
    """Comprehensive binary and modular analysis of all observables"""

    binary_analyzer = BinaryAnalyzer()
    modular_analyzer = ModularAnalyzer()
    base_converter = BaseConverter()
    period_analyzer = PeriodDoublingAnalyzer()

    all_patterns = []

    print("Analyzing 37 GIFT Framework Observables for Binary and Modular Patterns\n")
    print("=" * 80)

    for obs_name, obs_value in observables.items():
        print(f"\nAnalyzing: {obs_name} = {obs_value}")

        # 1. Binary representation
        binary_repr = binary_analyzer.to_binary_fraction(obs_value)

        # 2. Dyadic rational test
        dyadic = binary_analyzer.is_dyadic_rational(obs_value)
        if dyadic and dyadic[3] < 1.0:  # Less than 1% deviation
            all_patterns.append({
                'observable': obs_name,
                'formula': f'{dyadic[0]}/2^{dyadic[1]}',
                'experimental': obs_value,
                'theoretical': dyadic[2],
                'deviation_pct': dyadic[3],
                'binary_repr': binary_repr,
                'category': 'dyadic_rational'
            })
            print(f"  Dyadic: {dyadic[0]}/2^{dyadic[1]} = {dyadic[2]:.6f} (dev: {dyadic[3]:.4f}%)")

        # 3. Power of 2 patterns
        pow2_patterns = binary_analyzer.test_power_of_2_pattern(obs_value)
        for pattern in pow2_patterns:
            all_patterns.append({
                'observable': obs_name,
                'formula': pattern['formula'],
                'experimental': obs_value,
                'theoretical': pattern['value'],
                'deviation_pct': pattern['deviation_pct'],
                'binary_repr': binary_repr,
                'category': 'power_of_2'
            })
            print(f"  Power of 2: {pattern['formula']} (dev: {pattern['deviation_pct']:.4f}%)")

        # 4. Power of 2 with constants
        pow2_const = binary_analyzer.test_power_of_2_with_constants(obs_value)
        for pattern in pow2_const:
            all_patterns.append({
                'observable': obs_name,
                'formula': pattern['formula'],
                'experimental': obs_value,
                'theoretical': pattern['value'],
                'deviation_pct': pattern['deviation_pct'],
                'binary_repr': binary_repr,
                'category': 'power_of_2_with_constant'
            })
            print(f"  Power of 2 + const: {pattern['formula']} (dev: {pattern['deviation_pct']:.4f}%)")

        # 5. Mersenne patterns
        mersenne_patterns = modular_analyzer.test_mersenne_patterns(obs_value)
        for pattern in mersenne_patterns:
            all_patterns.append({
                'observable': obs_name,
                'formula': pattern['formula'],
                'experimental': obs_value,
                'theoretical': pattern['value'],
                'deviation_pct': pattern['deviation_pct'],
                'binary_repr': binary_repr,
                'category': 'mersenne_modular'
            })
            print(f"  Mersenne: {pattern['formula']} (dev: {pattern['deviation_pct']:.4f}%)")

        # 6. Fermat patterns
        fermat_patterns = modular_analyzer.test_fermat_patterns(obs_value)
        for pattern in fermat_patterns:
            all_patterns.append({
                'observable': obs_name,
                'formula': pattern['formula'],
                'experimental': obs_value,
                'theoretical': pattern['value'],
                'deviation_pct': pattern['deviation_pct'],
                'binary_repr': binary_repr,
                'category': 'fermat_modular'
            })
            print(f"  Fermat: {pattern['formula']} (dev: {pattern['deviation_pct']:.4f}%)")

        # 7. Rational approximation
        num, denom, rational_val = modular_analyzer.find_rational_approximation(obs_value)
        deviation = abs(obs_value - rational_val) / obs_value * 100 if obs_value != 0 else 0
        if deviation < 1.0:
            # Check if denominator is power of 2
            if denom & (denom - 1) == 0 and denom != 0:  # Check if power of 2
                category = 'binary_rational'
            else:
                category = 'rational_approximation'

            all_patterns.append({
                'observable': obs_name,
                'formula': f'{num}/{denom}',
                'experimental': obs_value,
                'theoretical': rational_val,
                'deviation_pct': deviation,
                'binary_repr': binary_repr,
                'category': category
            })
            print(f"  Rational: {num}/{denom} (dev: {deviation:.4f}%)")

        # 8. Base conversions (stored for reference)
        base_reprs = base_converter.analyze_base_patterns(obs_value)

    # 9. Period doubling analysis
    print("\n" + "=" * 80)
    print("\nPeriod Doubling Analysis (2^n scaling between observables):")
    period_matches = period_analyzer.test_period_doubling(observables)
    for match in period_matches:
        all_patterns.append({
            'observable': match['observable1'] + ' → ' + match['observable2'],
            'formula': match['formula'],
            'experimental': match['target_value'],
            'theoretical': match['scaled_value'],
            'deviation_pct': match['deviation_pct'],
            'binary_repr': '',
            'category': 'period_doubling'
        })
        print(f"  {match['formula']} ≈ {match['observable2']}")
        print(f"    {match['scaled_value']:.6f} ≈ {match['target_value']:.6f} (dev: {match['deviation_pct']:.4f}%)")

    return all_patterns


def categorize_and_rank_patterns(patterns):
    """Categorize patterns by type and rank by precision"""

    categories = {}
    for pattern in patterns:
        category = pattern['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(pattern)

    # Sort each category by deviation
    for category in categories:
        categories[category].sort(key=lambda x: x['deviation_pct'])

    return categories


def generate_summary_statistics(patterns, categorized):
    """Generate summary statistics"""

    total_patterns = len(patterns)

    # Count by deviation thresholds
    exact_matches = sum(1 for p in patterns if p['deviation_pct'] < 0.001)
    sub_01_pct = sum(1 for p in patterns if p['deviation_pct'] < 0.1)
    sub_05_pct = sum(1 for p in patterns if p['deviation_pct'] < 0.5)
    sub_1_pct = sum(1 for p in patterns if p['deviation_pct'] < 1.0)

    stats = {
        'total_patterns': total_patterns,
        'exact_matches': exact_matches,
        'sub_01_pct': sub_01_pct,
        'sub_05_pct': sub_05_pct,
        'sub_1_pct': sub_1_pct,
        'categories': {cat: len(pats) for cat, pats in categorized.items()}
    }

    return stats


def main():
    """Main execution function"""

    # Run comprehensive analysis
    all_patterns = analyze_all_observables()

    # Categorize and rank
    categorized = categorize_and_rank_patterns(all_patterns)

    # Generate statistics
    stats = generate_summary_statistics(all_patterns, categorized)

    # Save to CSV
    csv_path = '/home/user/GIFT/binary_modular_patterns.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['observable', 'formula', 'experimental', 'theoretical',
                      'deviation_pct', 'binary_repr', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by deviation before writing
        sorted_patterns = sorted(all_patterns, key=lambda x: x['deviation_pct'])
        for pattern in sorted_patterns:
            writer.writerow(pattern)

    print(f"\n\nResults saved to {csv_path}")
    print(f"\nTotal patterns found: {stats['total_patterns']}")
    print(f"  Exact matches (<0.001%): {stats['exact_matches']}")
    print(f"  Sub-0.1% deviation: {stats['sub_01_pct']}")
    print(f"  Sub-0.5% deviation: {stats['sub_05_pct']}")
    print(f"  Sub-1% deviation: {stats['sub_1_pct']}")
    print(f"\nPatterns by category:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")

    return all_patterns, categorized, stats


if __name__ == '__main__':
    patterns, categorized, statistics = main()
