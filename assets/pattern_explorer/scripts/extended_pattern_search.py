#!/usr/bin/env python3
"""
GIFT Framework - Extended Pattern Search

Searches for elusive patterns like ζ(7), ζ(9), ζ(11) with:
- Wider tolerance (up to 5%)
- More complex formulas (4-5 operations)
- Logarithmic and power transformations
- Integer factorizations

Author: GIFT Framework Team
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExtendedPattern:
    """Extended pattern discovery result"""
    observable: str
    formula: str
    formula_latex: str
    gift_value: float
    target_value: float
    deviation_pct: float
    deviation_abs: float
    complexity: int
    category: str
    interpretation: str
    confidence: str  # HIGH <0.1%, MEDIUM <1%, LOW <5%


class ExtendedPatternSearcher:
    """
    Extended pattern search with wider tolerances and transformations

    Goals:
    1. Find where ζ(7), ζ(9), ζ(11) appear
    2. Test logarithmic transformations
    3. Explore power relations
    4. Check integer factorizations
    """

    def __init__(self):
        self._initialize_constants()
        self._initialize_experimental_values()
        self.discoveries = []

    def _initialize_constants(self):
        """Initialize all mathematical constants"""

        # Framework parameters
        self.tau = 10416 / 2673
        self.Weyl = 5.0
        self.p2 = 2.0
        self.rank = 8
        self.b2 = 21
        self.b3 = 77
        self.dim_G2 = 14

        # Mathematical constants
        self.pi = np.pi
        self.e = np.e
        self.phi = (1 + np.sqrt(5)) / 2
        self.gamma = 0.5772156649015329
        self.ln2 = np.log(2)
        self.ln3 = np.log(3)

        # Odd zeta values
        self.zeta3 = 1.2020569031595942
        self.zeta5 = 1.0369277551433699
        self.zeta7 = 1.0083492773819228  # TARGET!
        self.zeta9 = 1.0020083928260822
        self.zeta11 = 1.0004941886041195

        # Chaos theory
        self.feigenbaum_delta = 4.669201609102990
        self.feigenbaum_alpha = 2.502907875095893

        # Mersenne primes
        self.M2 = 3
        self.M3 = 7
        self.M5 = 31
        self.M7 = 127

    def _initialize_experimental_values(self):
        """Initialize experimental observables"""

        self.observables = {
            # Gauge couplings
            'alpha_inv_MZ': 127.955,
            'alpha_s': 0.1179,
            'sin2_theta_W': 0.23121,

            # Koide
            'Q_Koide': 0.66670,

            # Cosmology
            'n_s': 0.9648,
            'Omega_DM': 0.26,
            'Omega_DE': 0.6889,
            'H0': 73.04,

            # CKM
            'V_us': 0.2248,
            'V_cb': 0.0410,
            'V_ub': 0.00409,

            # PMNS
            'sin2_theta_12': 0.310,
            'sin2_theta_23': 0.558,
            'sin2_theta_13': 0.02241,

            # Mass ratios
            'm_s_m_d': 20.0,
            'm_mu_m_e': 206.768,

            # Weinberg angle at different scales
            'sin2_theta_W_GUT': 0.375,  # GUT scale prediction

            # Fine structure constant
            'alpha_em': 1/137.036,

            # Other
            'delta_CP': 1.36,  # CP violation phase (radians)
        }

    def search_zeta7_systematic(self, max_tolerance_pct: float = 5.0) -> List[ExtendedPattern]:
        """
        Systematic search for ζ(7) with wider tolerance

        Tests:
        1. Direct: ζ(7) × constant
        2. Inverse: 1/ζ(7)
        3. Logarithmic: ln(ζ(7)), 1/ln(ζ(7))
        4. Powers: ζ(7)², ζ(7)³, √ζ(7)
        5. Combinations: ζ(7) × Mersenne / topology
        """

        discoveries = []

        print(f"Searching for ζ(7) = {self.zeta7:.10f} patterns...")
        print(f"  Tolerance: ≤ {max_tolerance_pct}%")

        # Build test formulas
        test_cases = []

        # Category 1: Simple multiplications
        for const_name, const_value in [
            ('γ', self.gamma),
            ('φ', self.phi),
            ('ln(2)', self.ln2),
            ('π', self.pi),
            ('τ', self.tau),
            ('e', self.e),
        ]:
            test_cases.append({
                'value': self.zeta7 * const_value,
                'formula': f"ζ(7)×{const_name}",
                'latex': f"\\zeta(7) \\times {const_name}",
                'complexity': 2,
                'category': 'zeta7_multiplicative'
            })

            # With division
            if const_value != 0:
                test_cases.append({
                    'value': self.zeta7 / const_value,
                    'formula': f"ζ(7)/{const_name}",
                    'latex': f"\\zeta(7) / {const_name}",
                    'complexity': 2,
                    'category': 'zeta7_division'
                })

        # Category 2: Inverse and logarithmic
        test_cases.extend([
            {
                'value': 1 / self.zeta7,
                'formula': "1/ζ(7)",
                'latex': "1/\\zeta(7)",
                'complexity': 1,
                'category': 'zeta7_inverse'
            },
            {
                'value': np.log(self.zeta7),
                'formula': "ln(ζ(7))",
                'latex': "\\ln(\\zeta(7))",
                'complexity': 2,
                'category': 'zeta7_logarithmic'
            },
            {
                'value': 1 / np.log(self.zeta7),
                'formula': "1/ln(ζ(7))",
                'latex': "1/\\ln(\\zeta(7))",
                'complexity': 2,
                'category': 'zeta7_logarithmic'
            },
        ])

        # Category 3: Powers
        for power, power_name in [(2, '²'), (3, '³'), (1/2, '^(1/2)'), (1/3, '^(1/3)')]:
            test_cases.append({
                'value': self.zeta7 ** power,
                'formula': f"ζ(7){power_name}",
                'latex': f"\\zeta(7)^{{{power}}}",
                'complexity': 2,
                'category': 'zeta7_power'
            })

        # Category 4: With Mersenne primes
        for M_name, M_val in [('M₂', self.M2), ('M₃', self.M3), ('M₅', self.M5)]:
            test_cases.extend([
                {
                    'value': self.zeta7 * M_val,
                    'formula': f"ζ(7)×{M_name}",
                    'latex': f"\\zeta(7) \\times {M_name}",
                    'complexity': 2,
                    'category': 'zeta7_mersenne'
                },
                {
                    'value': self.zeta7 / M_val,
                    'formula': f"ζ(7)/{M_name}",
                    'latex': f"\\zeta(7) / {M_name}",
                    'complexity': 2,
                    'category': 'zeta7_mersenne'
                },
            ])

        # Category 5: Complex formulas with topology
        for topo_name, topo_val in [('b₂', self.b2), ('rank', self.rank), ('Weyl', self.Weyl)]:
            for const_name, const_val in [('π', self.pi), ('γ', self.gamma), ('φ', self.phi)]:
                if topo_val != 0:
                    test_cases.append({
                        'value': (self.zeta7 * const_val) / topo_val,
                        'formula': f"(ζ(7)×{const_name})/{topo_name}",
                        'latex': f"(\\zeta(7) \\times {const_name}) / {topo_name}",
                        'complexity': 3,
                        'category': 'zeta7_composite'
                    })

        # Test all formulas against all observables
        for obs_name, obs_value in self.observables.items():
            for test in test_cases:
                predicted = test['value']

                if obs_value == 0:
                    continue

                deviation_abs = abs(predicted - obs_value)
                deviation_pct = deviation_abs / abs(obs_value) * 100

                if deviation_pct <= max_tolerance_pct:
                    # Assign confidence
                    if deviation_pct < 0.1:
                        confidence = "HIGH"
                    elif deviation_pct < 1.0:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"

                    interpretation = f"ζ(7) appears in {obs_name} via {test['formula']}"

                    discoveries.append(ExtendedPattern(
                        observable=obs_name,
                        formula=test['formula'],
                        formula_latex=test['latex'],
                        gift_value=predicted,
                        target_value=obs_value,
                        deviation_pct=deviation_pct,
                        deviation_abs=deviation_abs,
                        complexity=test['complexity'],
                        category=test['category'],
                        interpretation=interpretation,
                        confidence=confidence
                    ))

        # Sort by deviation
        discoveries.sort(key=lambda x: x.deviation_pct)

        print(f"  Found {len(discoveries)} ζ(7) patterns within {max_tolerance_pct}% tolerance")

        return discoveries

    def search_zeta_ratios(self, max_tolerance_pct: float = 2.0) -> List[ExtendedPattern]:
        """
        Search for ratios between zeta values

        Tests:
        - ζ(3)/ζ(5), ζ(3)/ζ(7), ζ(5)/ζ(7)
        - ζ(7)/ζ(9), ζ(9)/ζ(11)
        - Products: ζ(3)×ζ(5), ζ(5)×ζ(7)
        """

        discoveries = []

        print("Searching for zeta value ratios...")

        zeta_values = {
            'ζ(3)': self.zeta3,
            'ζ(5)': self.zeta5,
            'ζ(7)': self.zeta7,
            'ζ(9)': self.zeta9,
            'ζ(11)': self.zeta11,
        }

        # Test all pairs
        for name1, val1 in zeta_values.items():
            for name2, val2 in zeta_values.items():
                if name1 == name2:
                    continue

                # Ratio
                ratio = val1 / val2
                formula = f"{name1}/{name2}"
                latex = f"{name1}/{name2}"

                # Product
                product = val1 * val2
                formula_prod = f"{name1}×{name2}"
                latex_prod = f"{name1} \\times {name2}"

                # Test against observables
                for obs_name, obs_value in self.observables.items():
                    if obs_value == 0:
                        continue

                    # Check ratio
                    dev_ratio = abs(ratio - obs_value) / abs(obs_value) * 100
                    if dev_ratio <= max_tolerance_pct:
                        discoveries.append(ExtendedPattern(
                            observable=obs_name,
                            formula=formula,
                            formula_latex=latex,
                            gift_value=ratio,
                            target_value=obs_value,
                            deviation_pct=dev_ratio,
                            deviation_abs=abs(ratio - obs_value),
                            complexity=2,
                            category='zeta_ratio',
                            interpretation=f"Zeta ratio {formula} matches {obs_name}",
                            confidence="HIGH" if dev_ratio < 0.1 else "MEDIUM"
                        ))

                    # Check product
                    dev_prod = abs(product - obs_value) / abs(obs_value) * 100
                    if dev_prod <= max_tolerance_pct:
                        discoveries.append(ExtendedPattern(
                            observable=obs_name,
                            formula=formula_prod,
                            formula_latex=latex_prod,
                            gift_value=product,
                            target_value=obs_value,
                            deviation_pct=dev_prod,
                            deviation_abs=abs(product - obs_value),
                            complexity=2,
                            category='zeta_product',
                            interpretation=f"Zeta product {formula_prod} matches {obs_name}",
                            confidence="HIGH" if dev_prod < 0.1 else "MEDIUM"
                        ))

        discoveries.sort(key=lambda x: x.deviation_pct)
        print(f"  Found {len(discoveries)} zeta ratio/product patterns")

        return discoveries

    def search_integer_near_patterns(self, max_tolerance_pct: float = 1.0) -> List[ExtendedPattern]:
        """
        Search for patterns that produce near-integer or simple rational results

        Example: If observable ≈ 20, test if it equals exactly 20 from formula
        """

        discoveries = []

        print("Searching for near-integer patterns...")

        # Test if any observable is near a simple expression
        for obs_name, obs_value in self.observables.items():

            # Check if near integer
            nearest_int = round(obs_value)
            if nearest_int > 0:
                deviation = abs(obs_value - nearest_int) / obs_value * 100
                if deviation <= max_tolerance_pct:
                    discoveries.append(ExtendedPattern(
                        observable=obs_name,
                        formula=str(nearest_int),
                        formula_latex=str(nearest_int),
                        gift_value=float(nearest_int),
                        target_value=obs_value,
                        deviation_pct=deviation,
                        deviation_abs=abs(obs_value - nearest_int),
                        complexity=0,
                        category='exact_integer',
                        interpretation=f"{obs_name} ≈ {nearest_int} (integer)",
                        confidence="HIGH" if deviation < 0.01 else "MEDIUM"
                    ))

            # Check if near simple rational
            for denom in range(2, 21):
                for numer in range(1, 100):
                    rational = numer / denom
                    deviation = abs(obs_value - rational) / obs_value * 100
                    if deviation <= max_tolerance_pct:
                        discoveries.append(ExtendedPattern(
                            observable=obs_name,
                            formula=f"{numer}/{denom}",
                            formula_latex=f"\\frac{{{numer}}}{{{denom}}}",
                            gift_value=rational,
                            target_value=obs_value,
                            deviation_pct=deviation,
                            deviation_abs=abs(obs_value - rational),
                            complexity=1,
                            category='simple_rational',
                            interpretation=f"{obs_name} ≈ {numer}/{denom}",
                            confidence="HIGH" if deviation < 0.01 else "MEDIUM"
                        ))

        discoveries.sort(key=lambda x: x.deviation_pct)
        print(f"  Found {len(discoveries[:50])} near-integer/rational patterns")

        return discoveries[:50]  # Top 50

    def run_extended_search(self, output_dir: str = 'extended_search_results') -> None:
        """
        Run complete extended pattern search

        Args:
            output_dir: Directory for output files
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 80)
        print("GIFT FRAMEWORK - EXTENDED PATTERN SEARCH")
        print("=" * 80)
        print()

        all_discoveries = []

        # Search 1: ζ(7) systematic (5% tolerance)
        print("[1/3] ζ(7) Systematic Search...")
        zeta7_patterns = self.search_zeta7_systematic(max_tolerance_pct=5.0)
        all_discoveries.extend(zeta7_patterns)

        # Search 2: Zeta ratios (2% tolerance)
        print("[2/3] Zeta Ratio Search...")
        zeta_ratio_patterns = self.search_zeta_ratios(max_tolerance_pct=2.0)
        all_discoveries.extend(zeta_ratio_patterns)

        # Search 3: Near-integers (1% tolerance)
        print("[3/3] Near-Integer Search...")
        integer_patterns = self.search_integer_near_patterns(max_tolerance_pct=1.0)
        all_discoveries.extend(integer_patterns)

        self.discoveries = all_discoveries

        print()
        print(f"Total discoveries: {len(all_discoveries)}")
        print()

        # Generate reports
        self.generate_report(output_path)
        self.generate_csv(output_path)

        print("=" * 80)
        print("EXTENDED SEARCH COMPLETE")
        print("=" * 80)
        print()
        print(f"Results saved to: {output_path.absolute()}")
        print()

    def generate_report(self, output_dir: Path) -> None:
        """Generate markdown report"""

        output_file = output_dir / 'extended_patterns.md'

        with open(output_file, 'w') as f:
            f.write("# GIFT Framework - Extended Pattern Search Results\n\n")
            f.write(f"**Total Discoveries**: {len(self.discoveries)}\n\n")
            f.write("---\n\n")

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)

            for disc in self.discoveries:
                by_category[disc.category].append(disc)

            # Sort categories by number of discoveries
            categories = sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True)

            for category, patterns in categories:
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                f.write(f"**Count**: {len(patterns)}\n\n")

                # Top 10 per category
                patterns.sort(key=lambda x: x.deviation_pct)
                for i, p in enumerate(patterns[:10], 1):
                    status = {
                        'HIGH': '🟢',
                        'MEDIUM': '🟡',
                        'LOW': '🟠'
                    }.get(p.confidence, '⚪')

                    f.write(f"### {i}. {p.observable} {status}\n\n")
                    f.write(f"**Formula**: `{p.formula}`\n\n")
                    f.write(f"**LaTeX**: ${p.formula_latex}$\n\n")
                    f.write(f"- **GIFT Value**: {p.gift_value:.10f}\n")
                    f.write(f"- **Target**: {p.target_value:.10f}\n")
                    f.write(f"- **Deviation**: {p.deviation_pct:.4f}%\n")
                    f.write(f"- **Confidence**: {p.confidence}\n")
                    f.write(f"- **Complexity**: {p.complexity}\n\n")
                    f.write(f"{p.interpretation}\n\n")
                    f.write("---\n\n")

        print(f"  ✓ Report: {output_file}")

    def generate_csv(self, output_dir: Path) -> None:
        """Generate CSV summary"""

        output_file = output_dir / 'extended_patterns.csv'

        data = []
        for disc in self.discoveries:
            data.append({
                'Observable': disc.observable,
                'Formula': disc.formula,
                'LaTeX': disc.formula_latex,
                'GIFT_Value': disc.gift_value,
                'Target_Value': disc.target_value,
                'Deviation_%': disc.deviation_pct,
                'Deviation_Abs': disc.deviation_abs,
                'Complexity': disc.complexity,
                'Category': disc.category,
                'Confidence': disc.confidence,
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Deviation_%')
        df.to_csv(output_file, index=False)

        print(f"  ✓ CSV: {output_file}")


def main():
    """Main entry point"""

    searcher = ExtendedPatternSearcher()
    searcher.run_extended_search(output_dir='../extended_search_results')


if __name__ == '__main__':
    main()
