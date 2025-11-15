#!/usr/bin/env python3
"""
GIFT Framework: Higher-Order Pattern Discovery - Phase 5
Systematic search for complex zeta function patterns matching physical observables
"""

import numpy as np
import itertools
from typing import List, Tuple, Dict
import csv
from pathlib import Path

# Mathematical constants
ZETA_VALUES = {
    3: 1.2020569,
    5: 1.0369278,
    7: 1.0083493,
    9: 1.0020083,
    11: 1.0004942,
    13: 1.0001227
}

DELTA_F = 4.669201609  # Feigenbaum constant
ALPHA_F = 2.502907875  # Feigenbaum alpha
PERFECT_NUMBERS = {1: 6, 2: 28, 3: 496}

# Physical observables from GIFT framework (37 total)
OBSERVABLES = {
    # Gauge sector
    'alpha_inv': 137.036,
    'alpha_s': 0.1179,
    'sin2_theta_w': 0.23122,

    # Neutrino sector
    'theta_12': 33.44,  # degrees
    'theta_13': 8.57,   # degrees
    'theta_23': 49.2,   # degrees
    'delta_cp': 197.0,  # degrees (large uncertainty)

    # Lepton sector
    'Q_koide': 0.6667,
    'm_mu_over_m_e': 206.768,
    'm_tau_over_m_e': 3477.15,

    # Quark masses (MeV)
    'm_u': 2.16,
    'm_d': 4.67,
    'm_s': 93.4,
    'm_c': 1270.0,
    'm_b': 4180.0,
    'm_t': 172500.0,

    # Quark mass ratios
    'm_s_over_m_d': 20.0,
    'm_b_over_m_u': 1935.19,
    'm_c_over_m_d': 271.94,
    'm_d_over_m_u': 2.162,
    'm_c_over_m_s': 13.6,
    'm_t_over_m_c': 135.83,
    'm_b_over_m_d': 895.07,
    'm_b_over_m_c': 3.29,
    'm_t_over_m_s': 1846.89,
    'm_b_over_m_s': 44.76,

    # CKM matrix
    'theta_C': 13.04,  # degrees

    # Higgs sector
    'lambda_H': 0.1286,
    'v_EW': 246.22,  # GeV
    'm_H': 125.25,   # GeV

    # Cosmological observables
    'Omega_DE': 0.6847,
    'Omega_DM': 0.120,
    'n_s': 0.9649,
    'H_0': 73.04,  # km/s/Mpc

    # Dark matter (predictions)
    'm_chi_1': 90.5,   # GeV
    'm_chi_2': 352.7,  # GeV

    # Temporal structure
    'D_H': 0.856220
}

class PatternSearcher:
    """Systematic search for higher-order zeta patterns"""

    def __init__(self, tolerance_pct=1.0):
        self.tolerance = tolerance_pct / 100.0
        self.patterns_found = []

    def deviation_pct(self, experimental, theoretical):
        """Calculate percentage deviation"""
        if experimental == 0:
            return float('inf')
        return abs(experimental - theoretical) / abs(experimental) * 100.0

    def complexity_score(self, formula_str):
        """Estimate formula complexity (lower is simpler)"""
        score = 0
        score += formula_str.count('ζ') * 2  # Each zeta term
        score += formula_str.count('*') * 1  # Each multiplication
        score += formula_str.count('/') * 1  # Each division
        score += formula_str.count('^') * 2  # Each power
        score += formula_str.count('P_') * 3  # Perfect numbers
        score += formula_str.count('δ_F') * 4  # Feigenbaum constant
        score += formula_str.count('α_F') * 4
        return score

    def test_pattern(self, obs_name, obs_value, theoretical, formula, pattern_type):
        """Test if pattern matches within tolerance"""
        dev = self.deviation_pct(obs_value, theoretical)
        if dev <= self.tolerance * 100:
            complexity = self.complexity_score(formula)
            self.patterns_found.append({
                'observable': obs_name,
                'formula': formula,
                'experimental': obs_value,
                'theoretical': theoretical,
                'deviation_pct': dev,
                'complexity': complexity,
                'pattern_type': pattern_type
            })
            return True
        return False

    def search_triple_zeta_ratios(self):
        """Search for ζ(a)*ζ(b)/ζ(c) patterns"""
        print("Searching triple zeta ratios...")
        count = 0

        zeta_indices = list(ZETA_VALUES.keys())

        # Test all combinations
        for a, b, c in itertools.product(zeta_indices, repeat=3):
            if a == b == c:
                continue

            za, zb, zc = ZETA_VALUES[a], ZETA_VALUES[b], ZETA_VALUES[c]
            base_value = (za * zb) / zc

            # Test with scaling factors
            for scale in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 50, 100]:
                theoretical = base_value * scale
                formula = f"{scale}*ζ({a})*ζ({b})/ζ({c})" if scale > 1 else f"ζ({a})*ζ({b})/ζ({c})"

                for obs_name, obs_value in OBSERVABLES.items():
                    if self.test_pattern(obs_name, obs_value, theoretical, formula, 'triple_zeta'):
                        count += 1

            # Test with fractional scaling
            for num, den in [(1,2), (1,3), (1,4), (1,5), (2,3), (3,4), (3,5), (4,5)]:
                theoretical = base_value * num / den
                formula = f"({num}/{den})*ζ({a})*ζ({b})/ζ({c})"

                for obs_name, obs_value in OBSERVABLES.items():
                    if self.test_pattern(obs_name, obs_value, theoretical, formula, 'triple_zeta'):
                        count += 1

        print(f"  Found {count} triple zeta patterns")
        return count

    def search_product_patterns(self):
        """Search for ζ(a)^m * ζ(b)^n patterns"""
        print("Searching product patterns...")
        count = 0

        zeta_indices = list(ZETA_VALUES.keys())

        for a, b in itertools.product(zeta_indices, repeat=2):
            za, zb = ZETA_VALUES[a], ZETA_VALUES[b]

            for m, n in itertools.product([1, 2, 3], repeat=2):
                if m == 1 and n == 1 and a == b:
                    continue

                base_value = (za ** m) * (zb ** n)

                # Test with scaling
                for scale in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]:
                    theoretical = base_value * scale

                    if m == n == 1:
                        formula = f"{scale}*ζ({a})*ζ({b})" if scale > 1 else f"ζ({a})*ζ({b})"
                    elif n == 1:
                        formula = f"{scale}*ζ({a})^{m}*ζ({b})" if scale > 1 else f"ζ({a})^{m}*ζ({b})"
                    elif m == 1:
                        formula = f"{scale}*ζ({a})*ζ({b})^{n}" if scale > 1 else f"ζ({a})*ζ({b})^{n}"
                    else:
                        formula = f"{scale}*ζ({a})^{m}*ζ({b})^{n}" if scale > 1 else f"ζ({a})^{m}*ζ({b})^{n}"

                    for obs_name, obs_value in OBSERVABLES.items():
                        if self.test_pattern(obs_name, obs_value, theoretical, formula, 'product'):
                            count += 1

        print(f"  Found {count} product patterns")
        return count

    def search_mixed_patterns(self):
        """Search for mixed patterns with perfect numbers, Feigenbaum constants"""
        print("Searching mixed patterns...")
        count = 0

        zeta_indices = list(ZETA_VALUES.keys())

        # ζ(a)/ζ(b) * P_n patterns
        for a, b in itertools.product(zeta_indices, repeat=2):
            if a == b:
                continue
            za, zb = ZETA_VALUES[a], ZETA_VALUES[b]
            ratio = za / zb

            for p_idx, p_val in PERFECT_NUMBERS.items():
                theoretical = ratio * p_val
                formula = f"ζ({a})/ζ({b})*P_{p_idx}"

                for obs_name, obs_value in OBSERVABLES.items():
                    if self.test_pattern(obs_name, obs_value, theoretical, formula, 'mixed_perfect'):
                        count += 1

        # ζ(a) * δ_F^n patterns
        for a in zeta_indices:
            za = ZETA_VALUES[a]
            for n in [1, 2, 3]:
                for scale in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]:
                    theoretical = za * (DELTA_F ** n) * scale
                    formula = f"{scale}*ζ({a})*δ_F^{n}" if scale != 1 else f"ζ({a})*δ_F^{n}"

                    for obs_name, obs_value in OBSERVABLES.items():
                        if self.test_pattern(obs_name, obs_value, theoretical, formula, 'mixed_feigenbaum'):
                            count += 1

        # ζ(a) * α_F^n patterns
        for a in zeta_indices:
            za = ZETA_VALUES[a]
            for n in [1, 2, 3]:
                for scale in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
                    theoretical = za * (ALPHA_F ** n) * scale
                    formula = f"{scale}*ζ({a})*α_F^{n}" if scale != 1 else f"ζ({a})*α_F^{n}"

                    for obs_name, obs_value in OBSERVABLES.items():
                        if self.test_pattern(obs_name, obs_value, theoretical, formula, 'mixed_alpha_F'):
                            count += 1

        # ζ(a)^2 + ζ(b)^2 patterns
        for a, b in itertools.product(zeta_indices, repeat=2):
            if a >= b:
                continue
            za, zb = ZETA_VALUES[a], ZETA_VALUES[b]

            for op in ['+', '-']:
                if op == '+':
                    base_value = za**2 + zb**2
                    op_str = '+'
                else:
                    base_value = abs(za**2 - zb**2)
                    op_str = '-'

                for scale in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500]:
                    theoretical = base_value * scale
                    formula = f"{scale}*(ζ({a})^2{op_str}ζ({b})^2)" if scale != 1 else f"ζ({a})^2{op_str}ζ({b})^2"

                    for obs_name, obs_value in OBSERVABLES.items():
                        if self.test_pattern(obs_name, obs_value, theoretical, formula, 'mixed_quadratic'):
                            count += 1

        print(f"  Found {count} mixed patterns")
        return count

    def search_nested_ratios(self):
        """Search for [ζ(a)/ζ(b)] / [ζ(c)/ζ(d)] patterns"""
        print("Searching nested ratios...")
        count = 0

        zeta_indices = list(ZETA_VALUES.keys())

        for a, b, c, d in itertools.combinations(zeta_indices, 4):
            za, zb, zc, zd = ZETA_VALUES[a], ZETA_VALUES[b], ZETA_VALUES[c], ZETA_VALUES[d]

            # Try all permutations of which goes in numerator/denominator
            for ratio1_num, ratio1_den, ratio2_num, ratio2_den in [
                (a, b, c, d), (a, c, b, d), (a, d, b, c),
                (b, a, c, d), (b, c, a, d), (b, d, a, c)
            ]:
                zr1n, zr1d = ZETA_VALUES[ratio1_num], ZETA_VALUES[ratio1_den]
                zr2n, zr2d = ZETA_VALUES[ratio2_num], ZETA_VALUES[ratio2_den]

                base_value = (zr1n / zr1d) / (zr2n / zr2d)

                for scale in [1, 2, 5, 10, 20, 50, 100]:
                    theoretical = base_value * scale
                    formula = f"{scale}*[ζ({ratio1_num})/ζ({ratio1_den})]/[ζ({ratio2_num})/ζ({ratio2_den})]" if scale > 1 else f"[ζ({ratio1_num})/ζ({ratio1_den})]/[ζ({ratio2_num})/ζ({ratio2_den})]"

                    for obs_name, obs_value in OBSERVABLES.items():
                        if self.test_pattern(obs_name, obs_value, theoretical, formula, 'nested_ratio'):
                            count += 1

        print(f"  Found {count} nested ratio patterns")
        return count

    def search_all_patterns(self):
        """Run all pattern searches"""
        print("\n" + "="*60)
        print("GIFT Framework: Higher-Order Pattern Discovery (Phase 5)")
        print("="*60 + "\n")

        print(f"Testing {len(OBSERVABLES)} observables")
        print(f"Tolerance: {self.tolerance * 100}%\n")

        self.search_triple_zeta_ratios()
        self.search_product_patterns()
        self.search_mixed_patterns()
        self.search_nested_ratios()

        print(f"\n{'='*60}")
        print(f"Total patterns found: {len(self.patterns_found)}")
        print(f"{'='*60}\n")

        return self.patterns_found

    def rank_patterns(self):
        """Rank patterns by quality metric: precision × significance / complexity"""
        for p in self.patterns_found:
            precision_score = 100 - p['deviation_pct']  # Higher is better
            significance = np.log10(abs(p['experimental']) + 1)  # Physical scale
            complexity = p['complexity']

            # Quality metric (higher is better)
            p['quality'] = (precision_score * significance) / (complexity + 1)

        # Sort by quality (descending)
        self.patterns_found.sort(key=lambda x: x['quality'], reverse=True)

    def save_results(self, csv_path):
        """Save results to CSV"""
        if not self.patterns_found:
            print("No patterns to save")
            return

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'observable', 'formula', 'experimental', 'theoretical',
                'deviation_pct', 'complexity', 'pattern_type', 'quality'
            ])
            writer.writeheader()
            writer.writerows(self.patterns_found)

        print(f"\nResults saved to: {csv_path}")

    def print_summary(self, top_n=10):
        """Print summary of top patterns"""
        if not self.patterns_found:
            print("No patterns found")
            return

        print("\n" + "="*80)
        print(f"TOP {top_n} HIGHER-ORDER PATTERNS (by quality metric)")
        print("="*80 + "\n")

        for i, p in enumerate(self.patterns_found[:top_n], 1):
            print(f"{i}. {p['observable']}")
            print(f"   Formula: {p['formula']}")
            print(f"   Experimental: {p['experimental']:.6f}")
            print(f"   Theoretical: {p['theoretical']:.6f}")
            print(f"   Deviation: {p['deviation_pct']:.4f}%")
            print(f"   Complexity: {p['complexity']}")
            print(f"   Type: {p['pattern_type']}")
            print(f"   Quality Score: {p['quality']:.2f}")
            print()

        # Count by pattern type
        print("\n" + "="*80)
        print("PATTERNS BY TYPE")
        print("="*80 + "\n")

        type_counts = {}
        for p in self.patterns_found:
            ptype = p['pattern_type']
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{ptype:20s}: {count:4d} patterns")

        # Precision distribution
        print("\n" + "="*80)
        print("PRECISION DISTRIBUTION")
        print("="*80 + "\n")

        precision_ranges = [
            (0.01, "< 0.01%"),
            (0.1, "0.01% - 0.1%"),
            (0.5, "0.1% - 0.5%"),
            (1.0, "0.5% - 1.0%")
        ]

        for threshold, label in precision_ranges:
            count = sum(1 for p in self.patterns_found if p['deviation_pct'] < threshold * 100)
            print(f"{label:15s}: {count:4d} patterns")


def main():
    """Main execution"""
    # Search with 1% tolerance
    searcher = PatternSearcher(tolerance_pct=1.0)
    patterns = searcher.search_all_patterns()

    # Rank patterns
    searcher.rank_patterns()

    # Save to CSV
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    searcher.save_results(data_dir / 'higher_order_patterns.csv')

    # Print summary
    searcher.print_summary(top_n=20)

    return searcher


if __name__ == '__main__':
    searcher = main()
