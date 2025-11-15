#!/usr/bin/env python3
"""
Golden Ratio Pattern Analysis for GIFT Framework
Systematically tests golden ratio patterns across 37 observables
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# Golden ratio and related constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_INV = 1 / PHI  # 0.618033988749895 = φ - 1
PHI_SQUARED = PHI ** 2  # 2.618033988749895
SQRT_PHI = np.sqrt(PHI)  # 1.272019649514069

# Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

# Lucas sequence
LUCAS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]

# Tribonacci sequence
TRIBONACCI = [1, 1, 2, 4, 7, 13, 24, 44, 81, 149, 274, 504]

# Zeta function values
def zeta(n):
    """Riemann zeta function for odd integers"""
    zeta_values = {
        3: 1.2020569031595942,
        5: 1.0369277551433699,
        7: 1.0083492773819228,
        11: 1.0004941886041194
    }
    return zeta_values.get(n, None)

# Feigenbaum constant
DELTA_F = 4.669201609102990

# Observable data from GIFT framework
OBSERVABLES = {
    # Gauge Sector
    'alpha_inv': {'value': 137.036, 'gift': 128, 'name': 'α⁻¹'},
    'alpha_s': {'value': 0.1179, 'gift': 0.1178, 'name': 'α_s'},
    'sin2_theta_W': {'value': 0.23122, 'gift': 0.23128, 'name': 'sin²θ_W'},

    # Neutrino Sector (angles in degrees)
    'theta_12': {'value': 33.44, 'gift': 33.63, 'name': 'θ₁₂'},
    'theta_13': {'value': 8.57, 'gift': 8.571, 'name': 'θ₁₃'},
    'theta_23': {'value': 49.2, 'gift': 49.13, 'name': 'θ₂₃'},
    'delta_CP': {'value': 197, 'gift': 216, 'name': 'δ_CP'},

    # Lepton Sector
    'Q_Koide': {'value': 0.6667, 'gift': 2/3, 'name': 'Q_Koide'},
    'm_mu_m_e': {'value': 206.768, 'gift': 207.012, 'name': 'm_μ/m_e'},
    'm_tau_m_e': {'value': 3477.15, 'gift': 3477, 'name': 'm_τ/m_e'},

    # Quark Masses (MeV)
    'm_u': {'value': 2.16, 'gift': 2.160, 'name': 'm_u'},
    'm_d': {'value': 4.67, 'gift': 4.673, 'name': 'm_d'},
    'm_s': {'value': 93.4, 'gift': 93.52, 'name': 'm_s'},
    'm_c': {'value': 1270, 'gift': 1280, 'name': 'm_c'},
    'm_b': {'value': 4180, 'gift': 4158, 'name': 'm_b'},
    'm_t': {'value': 172500, 'gift': 172225, 'name': 'm_t'},

    # Quark Mass Ratios
    'm_s_m_d': {'value': 20.0, 'gift': 20.0, 'name': 'm_s/m_d'},
    'm_b_m_u': {'value': 1935.19, 'gift': 1935.15, 'name': 'm_b/m_u'},
    'm_c_m_d': {'value': 271.94, 'gift': 272.0, 'name': 'm_c/m_d'},
    'm_d_m_u': {'value': 2.162, 'gift': 2.16135, 'name': 'm_d/m_u'},
    'm_c_m_s': {'value': 13.6, 'gift': 13.5914, 'name': 'm_c/m_s'},
    'm_t_m_c': {'value': 135.83, 'gift': 135.923, 'name': 'm_t/m_c'},
    'm_b_m_d': {'value': 895.07, 'gift': 896.0, 'name': 'm_b/m_d'},
    'm_b_m_c': {'value': 3.29, 'gift': 3.28648, 'name': 'm_b/m_c'},
    'm_t_m_s': {'value': 1846.89, 'gift': 1849.0, 'name': 'm_t/m_s'},
    'm_b_m_s': {'value': 44.76, 'gift': 44.6826, 'name': 'm_b/m_s'},

    # CKM Matrix
    'theta_C': {'value': 13.04, 'gift': 13.093, 'name': 'θ_C'},

    # Higgs Sector
    'lambda_H': {'value': 0.1286, 'gift': 0.12885, 'name': 'λ_H'},
    'v_EW': {'value': 246.22, 'gift': 246.87, 'name': 'v_EW'},
    'm_H': {'value': 125.25, 'gift': 124.88, 'name': 'm_H'},

    # Cosmological Observables
    'Omega_DE': {'value': 0.6847, 'gift': 0.6861, 'name': 'Ω_DE'},
    'Omega_DM': {'value': 0.120, 'gift': 0.11996, 'name': 'Ω_DM'},
    'n_s': {'value': 0.9649, 'gift': 0.9649, 'name': 'n_s'},
    'H_0': {'value': 73.04, 'gift': 72.93, 'name': 'H₀'},

    # Dark Matter Sector (GeV)
    'm_chi_1': {'value': 90.5, 'gift': 90.5, 'name': 'm_χ₁'},
    'm_chi_2': {'value': 352.7, 'gift': 352.7, 'name': 'm_χ₂'},

    # Temporal Structure
    'D_H': {'value': 0.856220, 'gift': 0.859761, 'name': 'D_H'},
}


class GoldenRatioPatternFinder:
    """Systematic search for golden ratio patterns in GIFT observables"""

    def __init__(self, tolerance_pct=1.0):
        self.tolerance = tolerance_pct / 100.0
        self.patterns = []

    def test_pattern(self, obs_key: str, formula: str, theoretical: float) -> bool:
        """Test if a pattern matches within tolerance"""
        obs = OBSERVABLES[obs_key]
        experimental = obs['value']

        deviation_pct = abs(theoretical - experimental) / experimental * 100

        if deviation_pct <= self.tolerance * 100:
            self.patterns.append({
                'observable': obs['name'],
                'obs_key': obs_key,
                'formula': formula,
                'experimental': experimental,
                'theoretical': theoretical,
                'deviation_pct': deviation_pct
            })
            return True
        return False

    def test_phi_powers(self):
        """Test φ^n for n ∈ {-5, ..., 5} with integer scaling"""
        print("Testing powers of φ...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            for n in range(-5, 6):
                phi_power = PHI ** n

                # Test direct power
                formula = f"φ^{n}"
                if self.test_pattern(obs_key, formula, phi_power):
                    count += 1
                    self.patterns[-1]['phi_power'] = n
                    self.patterns[-1]['scaling'] = 1

                # Test with integer scaling k × φ^n
                for k in range(1, 51):
                    scaled = k * phi_power
                    formula = f"{k}×φ^{n}"
                    if self.test_pattern(obs_key, formula, scaled):
                        count += 1
                        self.patterns[-1]['phi_power'] = n
                        self.patterns[-1]['scaling'] = k

        print(f"  Found {count} φ^n patterns")
        return count

    def test_fibonacci_ratios(self):
        """Test Fibonacci ratios F_n/F_m"""
        print("Testing Fibonacci ratios...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            for i, F_n in enumerate(FIBONACCI):
                for j, F_m in enumerate(FIBONACCI):
                    if F_m == 0:
                        continue
                    ratio = F_n / F_m
                    formula = f"F_{i+1}/F_{j+1}"
                    if self.test_pattern(obs_key, formula, ratio):
                        count += 1
                        self.patterns[-1]['phi_power'] = None

        print(f"  Found {count} Fibonacci ratio patterns")
        return count

    def test_lucas_ratios(self):
        """Test Lucas ratios L_n/L_m"""
        print("Testing Lucas ratios...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            for i, L_n in enumerate(LUCAS):
                for j, L_m in enumerate(LUCAS):
                    if L_m == 0:
                        continue
                    ratio = L_n / L_m
                    formula = f"L_{i+1}/L_{j+1}"
                    if self.test_pattern(obs_key, formula, ratio):
                        count += 1
                        self.patterns[-1]['phi_power'] = None

        print(f"  Found {count} Lucas ratio patterns")
        return count

    def test_golden_angle_patterns(self):
        """Test (φ-1)^n = (1/φ)^n patterns"""
        print("Testing golden angle patterns...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            for n in range(-5, 6):
                value = PHI_INV ** n
                formula = f"(φ-1)^{n}"
                if self.test_pattern(obs_key, formula, value):
                    count += 1
                    self.patterns[-1]['phi_power'] = -n

                # With scaling
                for k in range(1, 51):
                    scaled = k * value
                    formula = f"{k}×(φ-1)^{n}"
                    if self.test_pattern(obs_key, formula, scaled):
                        count += 1
                        self.patterns[-1]['phi_power'] = -n

        print(f"  Found {count} golden angle patterns")
        return count

    def test_mixed_phi_patterns(self):
        """Test mixed patterns combining φ with other constants"""
        print("Testing mixed φ patterns...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            # φ^a / ζ(b)
            for a in range(-3, 4):
                for b in [3, 5, 7, 11]:
                    zeta_b = zeta(b)
                    if zeta_b:
                        value = (PHI ** a) / zeta_b
                        formula = f"φ^{a}/ζ({b})"
                        if self.test_pattern(obs_key, formula, value):
                            count += 1
                            self.patterns[-1]['phi_power'] = a

            # φ^a × δ_F^b
            for a in range(-2, 3):
                for b in range(-2, 3):
                    if b == 0:
                        continue
                    value = (PHI ** a) * (DELTA_F ** b)
                    formula = f"φ^{a}×δ_F^{b}"
                    if self.test_pattern(obs_key, formula, value):
                        count += 1
                        self.patterns[-1]['phi_power'] = a

            # Special values
            special = [
                (SQRT_PHI, "√φ", 0.5),
                (PHI_SQUARED, "φ²", 2),
                (1/PHI_SQUARED, "1/φ²", -2),
                (np.sqrt(1/PHI), "1/√φ", -0.5),
                (PHI/2, "φ/2", 1),
                (PHI/np.pi, "φ/π", 1),
                (PHI * np.e, "φ×e", 1),
            ]

            for value, formula, power in special:
                if self.test_pattern(obs_key, formula, value):
                    count += 1
                    self.patterns[-1]['phi_power'] = power

        print(f"  Found {count} mixed φ patterns")
        return count

    def test_tribonacci_ratios(self):
        """Test Tribonacci sequence ratios"""
        print("Testing Tribonacci ratios...")
        count = 0

        for obs_key in OBSERVABLES.keys():
            for i, T_n in enumerate(TRIBONACCI):
                for j, T_m in enumerate(TRIBONACCI):
                    if T_m == 0:
                        continue
                    ratio = T_n / T_m
                    formula = f"T_{i+1}/T_{j+1}"
                    if self.test_pattern(obs_key, formula, ratio):
                        count += 1
                        self.patterns[-1]['phi_power'] = None

        print(f"  Found {count} Tribonacci ratio patterns")
        return count

    def run_all_tests(self):
        """Execute all golden ratio pattern tests"""
        print("\n" + "="*70)
        print("Golden Ratio Pattern Analysis for GIFT Framework")
        print("="*70)
        print(f"Testing {len(OBSERVABLES)} observables")
        print(f"Tolerance: {self.tolerance*100:.1f}%")
        print("="*70 + "\n")

        total = 0
        total += self.test_phi_powers()
        total += self.test_fibonacci_ratios()
        total += self.test_lucas_ratios()
        total += self.test_golden_angle_patterns()
        total += self.test_mixed_phi_patterns()
        total += self.test_tribonacci_ratios()

        print(f"\nTotal patterns found: {total}")
        return total

    def get_results_df(self) -> pd.DataFrame:
        """Return results as pandas DataFrame"""
        if not self.patterns:
            return pd.DataFrame()

        df = pd.DataFrame(self.patterns)
        df = df.sort_values('deviation_pct')
        return df

    def analyze_by_observable(self) -> Dict:
        """Analyze which observables have most φ connections"""
        obs_counts = {}
        for pattern in self.patterns:
            obs_key = pattern['obs_key']
            if obs_key not in obs_counts:
                obs_counts[obs_key] = {
                    'name': pattern['observable'],
                    'count': 0,
                    'best_deviation': float('inf'),
                    'best_formula': None
                }
            obs_counts[obs_key]['count'] += 1
            if pattern['deviation_pct'] < obs_counts[obs_key]['best_deviation']:
                obs_counts[obs_key]['best_deviation'] = pattern['deviation_pct']
                obs_counts[obs_key]['best_formula'] = pattern['formula']

        return dict(sorted(obs_counts.items(),
                          key=lambda x: x[1]['count'],
                          reverse=True))

    def analyze_by_type(self) -> Dict:
        """Analyze patterns by type"""
        type_counts = {
            'phi_powers': 0,
            'fibonacci': 0,
            'lucas': 0,
            'golden_angle': 0,
            'mixed': 0,
            'tribonacci': 0
        }

        for pattern in self.patterns:
            formula = pattern['formula']
            if 'F_' in formula and '/' in formula:
                type_counts['fibonacci'] += 1
            elif 'L_' in formula and '/' in formula:
                type_counts['lucas'] += 1
            elif 'T_' in formula and '/' in formula:
                type_counts['tribonacci'] += 1
            elif '(φ-1)' in formula or '1/φ' in formula:
                type_counts['golden_angle'] += 1
            elif 'ζ' in formula or 'δ_F' in formula or '√φ' in formula or 'φ/' in formula:
                type_counts['mixed'] += 1
            else:
                type_counts['phi_powers'] += 1

        return type_counts

    def generate_visualizations(self, output_dir: Path):
        """Generate visualization plots"""
        if not self.patterns:
            print("No patterns to visualize")
            return

        df = self.get_results_df()

        # Figure 1: Top patterns by deviation
        fig, ax = plt.subplots(figsize=(12, 8))
        top_20 = df.head(20)
        y_pos = np.arange(len(top_20))

        ax.barh(y_pos, top_20['deviation_pct'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['observable']}: {row['formula']}"
                            for _, row in top_20.iterrows()], fontsize=9)
        ax.set_xlabel('Deviation (%)', fontsize=11)
        ax.set_title('Top 20 Golden Ratio Patterns by Precision', fontsize=13)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'golden_ratio_top_patterns.png', dpi=300)
        plt.close()

        # Figure 2: Pattern distribution by observable
        obs_analysis = self.analyze_by_observable()
        top_obs = list(obs_analysis.items())[:15]

        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_obs))
        counts = [v['count'] for _, v in top_obs]
        names = [v['name'] for _, v in top_obs]

        ax.barh(y_pos, counts, alpha=0.7, color='goldenrod')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Number of φ Patterns', fontsize=11)
        ax.set_title('Observables with Most Golden Ratio Connections', fontsize=13)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'golden_ratio_by_observable.png', dpi=300)
        plt.close()

        # Figure 3: Pattern type distribution
        type_counts = self.analyze_by_type()

        fig, ax = plt.subplots(figsize=(10, 8))
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = plt.cm.Set3(range(len(labels)))

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Golden Ratio Pattern Distribution by Type', fontsize=13)
        plt.tight_layout()
        plt.savefig(output_dir / 'golden_ratio_pattern_types.png', dpi=300)
        plt.close()

        print(f"\nVisualization files saved to {output_dir}")


def main():
    """Main execution function"""
    output_dir = Path('/home/user/GIFT')

    # Run pattern search
    finder = GoldenRatioPatternFinder(tolerance_pct=1.0)
    total_patterns = finder.run_all_tests()

    # Get results
    df = finder.get_results_df()

    # Save CSV
    csv_path = output_dir / 'golden_ratio_patterns.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate visualizations
    finder.generate_visualizations(output_dir)

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Pattern type breakdown
    type_counts = finder.analyze_by_type()
    print("\nPatterns by Type:")
    for ptype, count in type_counts.items():
        print(f"  {ptype:15s}: {count:4d}")

    # Top observables
    obs_analysis = finder.analyze_by_observable()
    print("\nTop 10 Observables with Most φ Connections:")
    for i, (obs_key, data) in enumerate(list(obs_analysis.items())[:10], 1):
        print(f"  {i:2d}. {data['name']:10s}: {data['count']:3d} patterns, "
              f"best: {data['best_formula']} ({data['best_deviation']:.4f}%)")

    # Top patterns
    print("\nTop 10 Most Precise Patterns:")
    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['observable']:10s} = {row['formula']:20s} "
              f"(dev: {row['deviation_pct']:.4f}%)")

    # Precision statistics
    high_precision = df[df['deviation_pct'] < 0.5]
    medium_precision = df[df['deviation_pct'] < 1.0]

    print(f"\nPrecision Summary:")
    print(f"  Patterns < 0.5% deviation: {len(high_precision)}")
    print(f"  Patterns < 1.0% deviation: {len(medium_precision)}")
    print(f"  Total patterns found:      {total_patterns}")

    return finder


if __name__ == "__main__":
    finder = main()
