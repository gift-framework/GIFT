#!/usr/bin/env python3
"""
Systematic search for odd zeta values ζ(9), ζ(13), ζ(15), ζ(17), ζ(19), ζ(21)
in GIFT framework observables.

Based on validated patterns:
- ζ(3) in sin²θ_W (0.027% deviation)
- ζ(5) in n_s via 1/ζ(5) (0.053% deviation)
- ζ(11)/ζ(5) in n_s (0.0066% deviation) - NOTABLE RESULT
- ζ(7)/τ in Ω_DM (0.474% deviation)
"""

import math
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Odd zeta values (high precision)
ZETA_VALUES = {
    3: 1.2020569031595942,   # Apéry's constant
    5: 1.0369277551433699,
    7: 1.0083492773819228,
    9: 1.0020083928260822,
    11: 1.0004941886041194,
    13: 1.0001224140313004,
    15: 1.0000122713347577,
    17: 1.0000061275061856,
    19: 1.0000030588236307,
    21: 1.0000015282259408,
}

# Framework topological parameters
FRAMEWORK_PARAMS = {
    'dim_K7': 7,
    'dim_G2': 14,
    'b2': 21,
    'b3': 77,
    'H_star': 99,
    'dim_E8': 248,
    'rank_E8': 8,
    'dim_E8xE8': 496,
    'p2': 2,
    'Weyl': 5,
    'N_gen': 3,
    'M2': 3,
    'M3': 7,
    'M5': 31,
    'M13': 8191,
    'F0': 3,
    'F1': 5,
    'F2': 17,
    'pi': math.pi,
    'e': math.e,
    'phi': (1 + math.sqrt(5)) / 2,  # Golden ratio
    'gamma': 0.5772156649,  # Euler-Mascheroni
    'ln2': math.log(2),
    'tau': 10416 / 2673,  # Hierarchical parameter
    'xi': 5 * math.pi / 16,
    'delta': 2 * math.pi / 25,
    'gamma_GIFT': 511 / 884,
    'beta0': math.pi / 8,
    'delta_F': 4.669201609,  # Feigenbaum constant
}

# All 37 GIFT observables from FRAMEWORK_STATUS_SUMMARY.md
OBSERVABLES = {
    # Gauge sector
    'alpha_inv': {'exp': 137.036, 'unc': 0.0, 'desc': 'Fine structure constant inverse'},
    'alpha_s': {'exp': 0.1179, 'unc': 0.0001, 'desc': 'Strong coupling'},
    'sin2_theta_W': {'exp': 0.23122, 'unc': 0.00002, 'desc': 'Weak mixing angle'},

    # Neutrino mixing angles (in degrees, convert to radians for sin²)
    'theta_12': {'exp': 33.44, 'unc': 0.5, 'desc': 'Solar neutrino angle (degrees)'},
    'theta_13': {'exp': 8.57, 'unc': 0.03, 'desc': 'Reactor angle (degrees)'},
    'theta_23': {'exp': 49.2, 'unc': 0.5, 'desc': 'Atmospheric angle (degrees)'},
    'delta_CP': {'exp': 197, 'unc': 24, 'desc': 'CP phase (degrees)'},

    # Neutrino sin² values
    'sin2_theta_12': {'exp': math.sin(math.radians(33.44))**2, 'unc': 0.005, 'desc': 'sin²θ₁₂'},
    'sin2_theta_13': {'exp': math.sin(math.radians(8.57))**2, 'unc': 0.0003, 'desc': 'sin²θ₁₃'},
    'sin2_theta_23': {'exp': math.sin(math.radians(49.2))**2, 'unc': 0.01, 'desc': 'sin²θ₂₃'},

    # Lepton sector
    'Q_Koide': {'exp': 0.6667, 'unc': 0.0001, 'desc': 'Koide formula'},
    'm_mu_over_m_e': {'exp': 206.768, 'unc': 0.05, 'desc': 'Muon/electron mass ratio'},
    'm_tau_over_m_e': {'exp': 3477.15, 'unc': 0.2, 'desc': 'Tau/electron mass ratio'},

    # Quark masses (MeV)
    'm_u': {'exp': 2.16, 'unc': 0.49, 'desc': 'Up quark mass (MeV)'},
    'm_d': {'exp': 4.67, 'unc': 0.48, 'desc': 'Down quark mass (MeV)'},
    'm_s': {'exp': 93.4, 'unc': 8.6, 'desc': 'Strange quark mass (MeV)'},
    'm_c': {'exp': 1270, 'unc': 20, 'desc': 'Charm quark mass (MeV)'},
    'm_b': {'exp': 4180, 'unc': 30, 'desc': 'Bottom quark mass (MeV)'},
    'm_t': {'exp': 172500, 'unc': 700, 'desc': 'Top quark mass (MeV)'},

    # Quark mass ratios
    'm_s_over_m_d': {'exp': 20.0, 'unc': 1.0, 'desc': 'ms/md'},
    'm_d_over_m_u': {'exp': 2.162, 'unc': 0.005, 'desc': 'md/mu'},
    'm_b_over_m_u': {'exp': 1935.19, 'unc': 50, 'desc': 'mb/mu'},
    'm_c_over_m_d': {'exp': 271.94, 'unc': 5, 'desc': 'mc/md'},
    'm_c_over_m_s': {'exp': 13.6, 'unc': 0.3, 'desc': 'mc/ms'},
    'm_t_over_m_c': {'exp': 135.83, 'unc': 2, 'desc': 'mt/mc'},
    'm_b_over_m_d': {'exp': 895.07, 'unc': 20, 'desc': 'mb/md'},
    'm_b_over_m_c': {'exp': 3.29, 'unc': 0.05, 'desc': 'mb/mc'},
    'm_t_over_m_s': {'exp': 1846.89, 'unc': 40, 'desc': 'mt/ms'},
    'm_b_over_m_s': {'exp': 44.76, 'unc': 1, 'desc': 'mb/ms'},

    # CKM matrix
    'theta_C': {'exp': 13.04, 'unc': 0.05, 'desc': 'Cabibbo angle (degrees)'},

    # Higgs sector
    'lambda_H': {'exp': 0.1286, 'unc': 0.0007, 'desc': 'Higgs quartic coupling'},
    'v_EW': {'exp': 246.22, 'unc': 0.1, 'desc': 'EW VEV (GeV)'},
    'm_H': {'exp': 125.25, 'unc': 0.17, 'desc': 'Higgs mass (GeV)'},

    # Cosmology
    'Omega_DE': {'exp': 0.6847, 'unc': 0.0073, 'desc': 'Dark energy density'},
    'Omega_DM': {'exp': 0.260, 'unc': 0.012, 'desc': 'Dark matter density (Ω_c h²)'},
    'n_s': {'exp': 0.9648, 'unc': 0.0042, 'desc': 'Scalar spectral index'},
    'H_0': {'exp': 73.04, 'unc': 1.04, 'desc': 'Hubble constant (km/s/Mpc)'},

    # Baryon density
    'Omega_b': {'exp': 0.0486, 'unc': 0.001, 'desc': 'Baryon density'},

    # Matter power spectrum
    'sigma_8': {'exp': 0.811, 'unc': 0.006, 'desc': 'Matter fluctuation amplitude'},
}


class ZetaPatternFinder:
    """Systematic search for zeta patterns in observables."""

    def __init__(self, tolerance: float = 0.01):
        """
        Args:
            tolerance: Maximum allowed deviation (default 1%)
        """
        self.tolerance = tolerance
        self.discoveries = []

    def deviation(self, predicted: float, experimental: float) -> float:
        """Calculate percentage deviation."""
        if experimental == 0:
            return float('inf')
        return abs(predicted - experimental) / experimental * 100

    def test_direct(self, obs_name: str, obs_value: float, zeta_n: int) -> List[Dict]:
        """Test obs ≈ ζ(n) × constant."""
        results = []
        zeta = ZETA_VALUES[zeta_n]

        # Test with various framework parameters
        for param_name, param_value in FRAMEWORK_PARAMS.items():
            if param_value == 0:
                continue

            # obs ≈ ζ(n) × param
            predicted = zeta * param_value
            dev = self.deviation(predicted, obs_value)
            if dev < self.tolerance:
                results.append({
                    'observable': obs_name,
                    'formula': f'ζ({zeta_n}) × {param_name}',
                    'predicted': predicted,
                    'experimental': obs_value,
                    'deviation_%': dev,
                    'type': 'direct_product',
                    'zetas': [zeta_n],
                })

            # obs ≈ ζ(n) / param
            if param_value != 0:
                predicted = zeta / param_value
                dev = self.deviation(predicted, obs_value)
                if dev < self.tolerance:
                    results.append({
                        'observable': obs_name,
                        'formula': f'ζ({zeta_n}) / {param_name}',
                        'predicted': predicted,
                        'experimental': obs_value,
                        'deviation_%': dev,
                        'type': 'direct_ratio',
                        'zetas': [zeta_n],
                    })

            # obs ≈ param / ζ(n)
            predicted = param_value / zeta
            dev = self.deviation(predicted, obs_value)
            if dev < self.tolerance:
                results.append({
                    'observable': obs_name,
                    'formula': f'{param_name} / ζ({zeta_n})',
                    'predicted': predicted,
                    'experimental': obs_value,
                    'deviation_%': dev,
                    'type': 'inverse',
                    'zetas': [zeta_n],
                })

        return results

    def test_zeta_ratios(self, obs_name: str, obs_value: float,
                        zeta_m: int, zeta_n: int) -> List[Dict]:
        """Test obs ≈ ζ(m)/ζ(n)."""
        results = []

        # Direct ratio
        ratio = ZETA_VALUES[zeta_m] / ZETA_VALUES[zeta_n]
        dev = self.deviation(ratio, obs_value)
        if dev < self.tolerance:
            results.append({
                'observable': obs_name,
                'formula': f'ζ({zeta_m})/ζ({zeta_n})',
                'predicted': ratio,
                'experimental': obs_value,
                'deviation_%': dev,
                'type': 'zeta_ratio',
                'zetas': [zeta_m, zeta_n],
            })

        # Inverse ratio
        inv_ratio = ZETA_VALUES[zeta_n] / ZETA_VALUES[zeta_m]
        dev = self.deviation(inv_ratio, obs_value)
        if dev < self.tolerance:
            results.append({
                'observable': obs_name,
                'formula': f'ζ({zeta_n})/ζ({zeta_m})',
                'predicted': inv_ratio,
                'experimental': obs_value,
                'deviation_%': dev,
                'type': 'zeta_ratio',
                'zetas': [zeta_n, zeta_m],
            })

        # Ratios with framework parameters
        for param_name, param_value in FRAMEWORK_PARAMS.items():
            if param_value == 0:
                continue

            # ratio × param
            predicted = ratio * param_value
            dev = self.deviation(predicted, obs_value)
            if dev < self.tolerance:
                results.append({
                    'observable': obs_name,
                    'formula': f'(ζ({zeta_m})/ζ({zeta_n})) × {param_name}',
                    'predicted': predicted,
                    'experimental': obs_value,
                    'deviation_%': dev,
                    'type': 'zeta_ratio_scaled',
                    'zetas': [zeta_m, zeta_n],
                })

            # ratio / param
            if param_value != 0:
                predicted = ratio / param_value
                dev = self.deviation(predicted, obs_value)
                if dev < self.tolerance:
                    results.append({
                        'observable': obs_name,
                        'formula': f'(ζ({zeta_m})/ζ({zeta_n})) / {param_name}',
                        'predicted': predicted,
                        'experimental': obs_value,
                        'deviation_%': dev,
                        'type': 'zeta_ratio_scaled',
                        'zetas': [zeta_m, zeta_n],
                    })

        return results

    def test_products(self, obs_name: str, obs_value: float,
                     zeta_m: int, zeta_n: int) -> List[Dict]:
        """Test obs ≈ ζ(m) × ζ(n) × constant."""
        results = []
        product = ZETA_VALUES[zeta_m] * ZETA_VALUES[zeta_n]

        # Test with framework parameters
        for param_name, param_value in FRAMEWORK_PARAMS.items():
            if param_value == 0:
                continue

            # product × param
            predicted = product * param_value
            dev = self.deviation(predicted, obs_value)
            if dev < self.tolerance:
                results.append({
                    'observable': obs_name,
                    'formula': f'ζ({zeta_m}) × ζ({zeta_n}) × {param_name}',
                    'predicted': predicted,
                    'experimental': obs_value,
                    'deviation_%': dev,
                    'type': 'zeta_product',
                    'zetas': [zeta_m, zeta_n],
                })

            # product / param
            if param_value != 0:
                predicted = product / param_value
                dev = self.deviation(predicted, obs_value)
                if dev < self.tolerance:
                    results.append({
                        'observable': obs_name,
                        'formula': f'ζ({zeta_m}) × ζ({zeta_n}) / {param_name}',
                        'predicted': predicted,
                        'experimental': obs_value,
                        'deviation_%': dev,
                        'type': 'zeta_product',
                        'zetas': [zeta_m, zeta_n],
                    })

        return results

    def test_linear_combinations(self, obs_name: str, obs_value: float,
                                zeta_m: int, zeta_n: int) -> List[Dict]:
        """Test simple linear combinations."""
        results = []
        zm = ZETA_VALUES[zeta_m]
        zn = ZETA_VALUES[zeta_n]

        # Test common patterns
        patterns = [
            (zm + zn, f'ζ({zeta_m}) + ζ({zeta_n})'),
            (zm - zn, f'ζ({zeta_m}) - ζ({zeta_n})'),
            (zn - zm, f'ζ({zeta_n}) - ζ({zeta_m})'),
        ]

        for value, formula in patterns:
            dev = self.deviation(value, obs_value)
            if dev < self.tolerance:
                results.append({
                    'observable': obs_name,
                    'formula': formula,
                    'predicted': value,
                    'experimental': obs_value,
                    'deviation_%': dev,
                    'type': 'linear_combination',
                    'zetas': [zeta_m, zeta_n],
                })

            # Scale by framework parameters
            for param_name, param_value in list(FRAMEWORK_PARAMS.items())[:10]:  # Limit combinations
                if param_value == 0:
                    continue

                predicted = value * param_value
                dev = self.deviation(predicted, obs_value)
                if dev < self.tolerance:
                    results.append({
                        'observable': obs_name,
                        'formula': f'({formula}) × {param_name}',
                        'predicted': predicted,
                        'experimental': obs_value,
                        'deviation_%': dev,
                        'type': 'linear_combination_scaled',
                        'zetas': [zeta_m, zeta_n],
                    })

        return results

    def search_all_patterns(self):
        """Systematic search through all observables and zeta values."""
        print("=" * 80)
        print("SYSTEMATIC SEARCH FOR ODD ZETA VALUES IN GIFT OBSERVABLES")
        print("=" * 80)
        print(f"\nTolerance: {self.tolerance * 100}%")
        print(f"Testing {len(OBSERVABLES)} observables against ζ(9,13,15,17,19,21)")
        print(f"Using {len(FRAMEWORK_PARAMS)} framework parameters")
        print()

        # New zetas to search for
        new_zetas = [9, 13, 15, 17, 19, 21]
        # All zetas for ratio tests
        all_zetas = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

        total_tests = 0

        for obs_name, obs_data in OBSERVABLES.items():
            obs_value = obs_data['exp']

            # Test direct patterns with new zetas
            for zeta_n in new_zetas:
                results = self.test_direct(obs_name, obs_value, zeta_n)
                self.discoveries.extend(results)
                total_tests += 1

            # Test ratios between all zetas (including new ones)
            for zeta_m, zeta_n in combinations(all_zetas, 2):
                if zeta_m == zeta_n:
                    continue
                results = self.test_zeta_ratios(obs_name, obs_value, zeta_m, zeta_n)
                self.discoveries.extend(results)
                total_tests += 1

            # Test products with new zetas
            for zeta_m, zeta_n in combinations(new_zetas, 2):
                results = self.test_products(obs_name, obs_value, zeta_m, zeta_n)
                self.discoveries.extend(results)
                total_tests += 1

            # Test linear combinations with new zetas
            for zeta_m, zeta_n in combinations(new_zetas, 2):
                results = self.test_linear_combinations(obs_name, obs_value, zeta_m, zeta_n)
                self.discoveries.extend(results)
                total_tests += 1

        print(f"Completed {total_tests} pattern tests")
        print(f"Found {len(self.discoveries)} patterns with deviation < {self.tolerance * 100}%\n")

        # Sort by deviation
        self.discoveries.sort(key=lambda x: x['deviation_%'])

        return self.discoveries

    def print_report(self, max_results: int = 50):
        """Print formatted report of discoveries."""
        if not self.discoveries:
            print("No patterns found within tolerance.")
            return

        print("\n" + "=" * 80)
        print("TOP DISCOVERIES (sorted by precision)")
        print("=" * 80)

        # Group by observable
        by_observable = {}
        for disc in self.discoveries[:max_results]:
            obs = disc['observable']
            if obs not in by_observable:
                by_observable[obs] = []
            by_observable[obs].append(disc)

        for i, disc in enumerate(self.discoveries[:max_results], 1):
            obs_desc = OBSERVABLES[disc['observable']]['desc']
            obs_unc = OBSERVABLES[disc['observable']]['unc']

            print(f"\n{i}. {disc['observable']} ({obs_desc})")
            print(f"   Formula: {disc['formula']}")
            print(f"   Predicted: {disc['predicted']:.10f}")
            print(f"   Experimental: {disc['experimental']:.10f} ± {obs_unc:.6f}")
            print(f"   Deviation: {disc['deviation_%']:.6f}%")

            # Calculate statistical significance
            if obs_unc > 0:
                sigma = abs(disc['predicted'] - disc['experimental']) / obs_unc
                print(f"   Significance: {sigma:.2f}σ")

            print(f"   Type: {disc['type']}")
            print(f"   Zetas used: {disc['zetas']}")

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # By zeta value
        zeta_counts = {}
        for disc in self.discoveries:
            for z in disc['zetas']:
                zeta_counts[z] = zeta_counts.get(z, 0) + 1

        print("\nPatterns by zeta value:")
        for z in sorted(zeta_counts.keys()):
            print(f"  ζ({z}): {zeta_counts[z]} patterns")

        # By pattern type
        type_counts = {}
        for disc in self.discoveries:
            t = disc['type']
            type_counts[t] = type_counts.get(t, 0) + 1

        print("\nPatterns by type:")
        for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {count}")

        # Best for each new zeta
        print("\n" + "=" * 80)
        print("BEST MATCH FOR EACH NEW ZETA VALUE")
        print("=" * 80)

        for zeta_n in [9, 13, 15, 17, 19, 21]:
            zeta_patterns = [d for d in self.discoveries if zeta_n in d['zetas']]
            if zeta_patterns:
                best = zeta_patterns[0]
                obs_desc = OBSERVABLES[best['observable']]['desc']
                print(f"\nζ({zeta_n}): {best['observable']} ({obs_desc})")
                print(f"  Formula: {best['formula']}")
                print(f"  Deviation: {best['deviation_%']:.6f}%")
            else:
                print(f"\nζ({zeta_n}): No patterns found")


def main():
    """Run systematic search."""

    # Search with 1% tolerance (as requested)
    print("\n" + "=" * 80)
    print("PHASE 1: HIGH-PRECISION SEARCH (tolerance < 1%)")
    print("=" * 80)

    finder = ZetaPatternFinder(tolerance=0.01)
    finder.search_all_patterns()
    finder.print_report(max_results=100)

    # Save results to CSV
    import csv
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / 'odd_zeta_discoveries.csv'
    with open(output_file, 'w', newline='') as f:
        if finder.discoveries:
            fieldnames = finder.discoveries[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for disc in finder.discoveries:
                # Convert list fields to string for CSV
                row = disc.copy()
                row['zetas'] = str(row['zetas'])
                writer.writerow(row)

    print(f"\n\nResults saved to: {output_file}")

    # Also search with 5% tolerance for completeness
    print("\n\n" + "=" * 80)
    print("PHASE 2: EXTENDED SEARCH (tolerance < 5%)")
    print("=" * 80)

    finder2 = ZetaPatternFinder(tolerance=0.05)
    finder2.search_all_patterns()

    # Print only new patterns (not in Phase 1)
    phase1_formulas = {d['formula'] + d['observable'] for d in finder.discoveries}
    new_patterns = [d for d in finder2.discoveries
                   if d['formula'] + d['observable'] not in phase1_formulas]

    print(f"\nAdditional patterns found in extended search: {len(new_patterns)}")

    # Print top 20 new patterns
    if new_patterns:
        print("\nTop 20 additional patterns:")
        for i, disc in enumerate(new_patterns[:20], 1):
            obs_desc = OBSERVABLES[disc['observable']]['desc']
            print(f"\n{i}. {disc['observable']} ({obs_desc})")
            print(f"   Formula: {disc['formula']}")
            print(f"   Deviation: {disc['deviation_%']:.6f}%")

    # Save extended results
    output_file2 = data_dir / 'odd_zeta_discoveries_extended.csv'
    with open(output_file2, 'w', newline='') as f:
        if finder2.discoveries:
            fieldnames = finder2.discoveries[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for disc in finder2.discoveries:
                row = disc.copy()
                row['zetas'] = str(row['zetas'])
                writer.writerow(row)

    print(f"\nExtended results saved to: {output_file2}")

    return finder, finder2


if __name__ == '__main__':
    finder1, finder2 = main()
