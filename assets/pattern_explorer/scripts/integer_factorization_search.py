#!/usr/bin/env python3
"""
Integer Factorization Pattern Discovery for GIFT Framework
Tests pure integer factorization patterns for 37 physical observables
"""

import numpy as np
import pandas as pd
from itertools import product, combinations
from typing import List, Tuple, Dict
import math

# GIFT Framework Observables
OBSERVABLES = {
    # Gauge sector
    'alpha_inv': 137.036,
    'alpha_s': 0.1179,
    'sin2theta_W': 0.23122,

    # Neutrino sector (angles in degrees)
    'theta_12': 33.44,
    'theta_13': 8.57,
    'theta_23': 49.2,
    'delta_CP': 197.0,

    # Lepton sector
    'Q_Koide': 0.6667,
    'm_mu_m_e': 206.768,
    'm_tau_m_e': 3477.15,

    # Quark masses (MeV)
    'm_u': 2.16,
    'm_d': 4.67,
    'm_s': 93.4,
    'm_c': 1270.0,
    'm_b': 4180.0,
    'm_t': 172500.0,

    # Quark mass ratios
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

    # CKM matrix
    'theta_C': 13.04,

    # Higgs sector
    'lambda_H': 0.1286,
    'v_EW': 246.22,
    'm_H': 125.25,

    # Cosmological
    'Omega_DE': 0.6847,
    'Omega_DM': 0.120,
    'n_s': 0.9649,
    'H_0': 73.04,

    # Dark matter (predictions)
    'm_chi1': 90.5,
    'm_chi2': 352.7,

    # Temporal
    'D_H': 0.856220
}

# Topological integers from GIFT framework
TOPOLOGICAL = {
    'dim_E8': 248,        # 2^3 × 31
    'rank_E8': 8,         # 2^3
    'dim_E8xE8': 496,     # 2^4 × 31
    'dim_K7': 7,          # Mersenne M_3
    'dim_G2': 14,         # 2 × 7
    'b2': 21,             # 3 × 7
    'b3': 77,             # 7 × 11
    'H_star': 99,         # 3^2 × 11
    'Weyl': 5,            # Fermat F_1
    'p2': 2,              # Fermat F_0
    'N_gen': 3,           # Mersenne M_2
}

# Prime numbers <= 31
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

# Mersenne primes: M_p = 2^p - 1
MERSENNE = {
    'M_2': 3,
    'M_3': 7,
    'M_5': 31,
    'M_7': 127,
    'M_13': 8191
}

# Fermat primes: F_n = 2^(2^n) + 1
FERMAT = {
    'F_0': 3,
    'F_1': 5,
    'F_2': 17,
    'F_3': 257
}

class IntegerFactorizationSearch:
    """Search for pure integer factorization patterns"""

    def __init__(self):
        self.results = []

    def factorize(self, n: int) -> Dict[int, int]:
        """Return prime factorization as dict {prime: exponent}"""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def format_prime_factors(self, factors: Dict[int, int]) -> str:
        """Format prime factorization as string"""
        if not factors:
            return "1"
        terms = []
        for p in sorted(factors.keys()):
            if factors[p] == 1:
                terms.append(str(p))
            else:
                terms.append(f"{p}^{factors[p]}")
        return " × ".join(terms)

    def deviation_pct(self, exp: float, theo: float) -> float:
        """Calculate percentage deviation"""
        if exp == 0:
            return 999.9
        return abs((theo - exp) / exp) * 100

    def test_prime_ratio(self, obs_name: str, obs_val: float):
        """Test p^a × q^b / (r^c × s^d) patterns"""
        # Test simple ratios with 1-2 primes
        for p1, p2 in combinations(PRIMES[:8], 2):  # Use smaller primes
            for e1, e2 in product(range(1, 5), repeat=2):
                # Numerator only
                theo = (p1 ** e1) * (p2 ** e2)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{p1}^{e1} × {p2}^{e2}"
                    factors = self.factorize(int(theo))
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': self.format_prime_factors(factors),
                        'category': 'prime_ratio'
                    })

                # Test with denominator
                for p3, p4 in combinations(PRIMES[:6], 2):
                    for e3, e4 in product(range(1, 4), repeat=2):
                        num = (p1 ** e1) * (p2 ** e2)
                        den = (p3 ** e3) * (p4 ** e4)
                        theo = num / den
                        dev = self.deviation_pct(obs_val, theo)
                        if dev < 1.0:
                            formula = f"({p1}^{e1} × {p2}^{e2}) / ({p3}^{e3} × {p4}^{e4})"
                            self.results.append({
                                'observable': obs_name,
                                'formula': formula,
                                'experimental': obs_val,
                                'theoretical': theo,
                                'deviation_pct': dev,
                                'prime_factors': f"num: {self.format_prime_factors(self.factorize(num))}, den: {self.format_prime_factors(self.factorize(den))}",
                                'category': 'prime_ratio'
                            })

    def test_mersenne_patterns(self, obs_name: str, obs_val: float):
        """Test Mersenne prime patterns"""
        mersenne_vals = list(MERSENNE.values())
        mersenne_names = list(MERSENNE.keys())

        # Single Mersenne
        for i, m in enumerate(mersenne_vals):
            for exp in range(1, 5):
                theo = m ** exp
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{mersenne_names[i]}^{exp}" if exp > 1 else mersenne_names[i]
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': self.format_prime_factors(self.factorize(theo)),
                        'category': 'mersenne'
                    })

        # Mersenne ratios
        for i, j in combinations(range(len(mersenne_vals)), 2):
            for e1, e2 in product(range(1, 4), repeat=2):
                theo = (mersenne_vals[i] ** e1) / (mersenne_vals[j] ** e2)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{mersenne_names[i]}^{e1} / {mersenne_names[j]}^{e2}"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': f"{mersenne_vals[i]}^{e1} / {mersenne_vals[j]}^{e2}",
                        'category': 'mersenne'
                    })

    def test_fermat_patterns(self, obs_name: str, obs_val: float):
        """Test Fermat prime patterns"""
        fermat_vals = list(FERMAT.values())
        fermat_names = list(FERMAT.keys())

        # Products and ratios
        for i, j in product(range(len(fermat_vals)), repeat=2):
            if i == j:
                continue
            for e1, e2 in product(range(1, 4), repeat=2):
                # Products
                theo = (fermat_vals[i] ** e1) * (fermat_vals[j] ** e2)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{fermat_names[i]}^{e1} × {fermat_names[j]}^{e2}"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': self.format_prime_factors(self.factorize(theo)),
                        'category': 'fermat'
                    })

                # Ratios
                theo = (fermat_vals[i] ** e1) / (fermat_vals[j] ** e2)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{fermat_names[i]}^{e1} / {fermat_names[j]}^{e2}"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': f"{fermat_vals[i]}^{e1} / {fermat_vals[j]}^{e2}",
                        'category': 'fermat'
                    })

    def test_topological_patterns(self, obs_name: str, obs_val: float):
        """Test combinations of topological integers"""
        topo_vals = list(TOPOLOGICAL.values())
        topo_names = list(TOPOLOGICAL.keys())

        # Single topological integers
        for i, val in enumerate(topo_vals):
            for exp in range(1, 4):
                theo = val ** exp
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{topo_names[i]}^{exp}" if exp > 1 else topo_names[i]
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': self.format_prime_factors(self.factorize(theo)),
                        'category': 'topological'
                    })

        # Pairs of topological integers
        for i, j in combinations(range(len(topo_vals)), 2):
            # Products
            theo = topo_vals[i] * topo_vals[j]
            dev = self.deviation_pct(obs_val, theo)
            if dev < 1.0:
                formula = f"{topo_names[i]} × {topo_names[j]}"
                self.results.append({
                    'observable': obs_name,
                    'formula': formula,
                    'experimental': obs_val,
                    'theoretical': theo,
                    'deviation_pct': dev,
                    'prime_factors': self.format_prime_factors(self.factorize(theo)),
                    'category': 'topological'
                })

            # Ratios
            if topo_vals[j] != 0:
                theo = topo_vals[i] / topo_vals[j]
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{topo_names[i]} / {topo_names[j]}"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': f"{self.format_prime_factors(self.factorize(topo_vals[i]))} / {self.format_prime_factors(self.factorize(topo_vals[j]))}",
                        'category': 'topological'
                    })

    def test_factorial_patterns(self, obs_name: str, obs_val: float):
        """Test n! / m! patterns"""
        for n in range(2, 15):
            for m in range(1, n):
                theo = math.factorial(n) / math.factorial(m)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"{n}! / {m}!"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': f"{n}! / {m}!",
                        'category': 'factorial'
                    })

    def test_binomial_patterns(self, obs_name: str, obs_val: float):
        """Test binomial coefficient patterns"""
        for n in range(2, 35):
            for k in range(1, n):
                theo = math.comb(n, k)
                dev = self.deviation_pct(obs_val, theo)
                if dev < 1.0:
                    formula = f"C({n},{k})"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': self.format_prime_factors(self.factorize(theo)),
                        'category': 'binomial'
                    })

    def test_simple_integer_patterns(self, obs_name: str, obs_val: float):
        """Test simple integer patterns"""
        # Test if observable is close to a simple integer
        for n in range(1, 10000):
            dev = self.deviation_pct(obs_val, n)
            if dev < 0.001:  # Exact match threshold
                formula = str(n)
                self.results.append({
                    'observable': obs_name,
                    'formula': formula,
                    'experimental': obs_val,
                    'theoretical': n,
                    'deviation_pct': dev,
                    'prime_factors': self.format_prime_factors(self.factorize(n)),
                    'category': 'exact_integer'
                })

        # Test simple ratios a/b
        for a in range(1, 200):
            for b in range(1, 200):
                if a == b:
                    continue
                theo = a / b
                dev = self.deviation_pct(obs_val, theo)
                if dev < 0.1:  # Very close matches
                    formula = f"{a}/{b}"
                    from math import gcd
                    g = gcd(a, b)
                    if g > 1:
                        formula = f"{a//g}/{b//g}"
                    self.results.append({
                        'observable': obs_name,
                        'formula': formula,
                        'experimental': obs_val,
                        'theoretical': theo,
                        'deviation_pct': dev,
                        'prime_factors': f"{a}/{b}",
                        'category': 'simple_ratio'
                    })

    def run_all_tests(self):
        """Run all pattern tests on all observables"""
        print("Starting integer factorization pattern search...")

        for obs_name, obs_val in OBSERVABLES.items():
            print(f"Testing {obs_name} = {obs_val}...")

            # Run all test categories
            self.test_simple_integer_patterns(obs_name, obs_val)
            self.test_topological_patterns(obs_name, obs_val)
            self.test_mersenne_patterns(obs_name, obs_val)
            self.test_fermat_patterns(obs_name, obs_val)
            self.test_factorial_patterns(obs_name, obs_val)
            self.test_binomial_patterns(obs_name, obs_val)
            self.test_prime_ratio(obs_name, obs_val)

        print(f"\nFound {len(self.results)} total patterns")

    def get_results_df(self) -> pd.DataFrame:
        """Return results as DataFrame"""
        df = pd.DataFrame(self.results)
        if len(df) > 0:
            df = df.sort_values(['observable', 'deviation_pct'])
        return df

    def get_best_patterns(self, n: int = 100) -> pd.DataFrame:
        """Get n best patterns (lowest deviation)"""
        df = self.get_results_df()
        if len(df) > 0:
            return df.nsmallest(n, 'deviation_pct')
        return df

    def get_exact_matches(self, threshold: float = 0.001) -> pd.DataFrame:
        """Get patterns with deviation < threshold"""
        df = self.get_results_df()
        if len(df) > 0:
            return df[df['deviation_pct'] < threshold]
        return df

    def analyze_by_category(self) -> Dict:
        """Analyze results by category"""
        df = self.get_results_df()
        if len(df) == 0:
            return {}

        analysis = {}
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            analysis[category] = {
                'count': len(cat_df),
                'mean_deviation': cat_df['deviation_pct'].mean(),
                'best_deviation': cat_df['deviation_pct'].min(),
                'observables_covered': cat_df['observable'].nunique()
            }
        return analysis


def main():
    """Main execution"""
    search = IntegerFactorizationSearch()
    search.run_all_tests()

    # Get results
    all_results = search.get_results_df()
    best_patterns = search.get_best_patterns(100)
    exact_matches = search.get_exact_matches()
    category_analysis = search.analyze_by_category()

    # Save to CSV
    output_file = '/home/user/GIFT/integer_factorization_patterns.csv'
    all_results.to_csv(output_file, index=False)
    print(f"\nSaved all patterns to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("INTEGER FACTORIZATION PATTERN DISCOVERY - SUMMARY")
    print("="*80)

    print(f"\nTotal patterns found: {len(all_results)}")
    print(f"Observables tested: {len(OBSERVABLES)}")
    print(f"Exact matches (< 0.001% deviation): {len(exact_matches)}")

    print("\n" + "-"*80)
    print("PATTERNS BY CATEGORY")
    print("-"*80)
    for category, stats in sorted(category_analysis.items()):
        print(f"\n{category.upper()}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean deviation: {stats['mean_deviation']:.4f}%")
        print(f"  Best deviation: {stats['best_deviation']:.6f}%")
        print(f"  Observables covered: {stats['observables_covered']}")

    if len(exact_matches) > 0:
        print("\n" + "-"*80)
        print("EXACT MATCHES (< 0.001% deviation)")
        print("-"*80)
        for _, row in exact_matches.iterrows():
            print(f"\n{row['observable']}: {row['formula']}")
            print(f"  Experimental: {row['experimental']}")
            print(f"  Theoretical: {row['theoretical']}")
            print(f"  Deviation: {row['deviation_pct']:.6f}%")
            print(f"  Prime factors: {row['prime_factors']}")

    print("\n" + "-"*80)
    print("TOP 10 SIMPLEST PATTERNS (by deviation)")
    print("-"*80)
    for i, (_, row) in enumerate(best_patterns.head(10).iterrows(), 1):
        print(f"\n{i}. {row['observable']} ≈ {row['formula']}")
        print(f"   Experimental: {row['experimental']}")
        print(f"   Theoretical: {row['theoretical']}")
        print(f"   Deviation: {row['deviation_pct']:.4f}%")
        print(f"   Category: {row['category']}")

    print("\n" + "="*80)

    return search, all_results, exact_matches, category_analysis


if __name__ == "__main__":
    search, results, exact, analysis = main()
