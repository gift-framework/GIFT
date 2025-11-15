#!/usr/bin/env python3
"""
GIFT Pattern Explorer - Quick Discovery Run (No Dependencies)
Rapid exploration using only Python standard library
"""

import math
import json
from datetime import datetime
from pathlib import Path

class QuickExplorer:
    """Lightweight exploration without external dependencies"""

    def __init__(self):
        # Framework parameters
        self.params = {
            'p2': 2.0,
            'Weyl': 5.0,
            'tau': 10416 / 2673,
            'beta0': math.pi / 8,
            'xi': 5 * math.pi / 16,
            'delta': 2 * math.pi / 25,
            'gamma_GIFT': 511 / 884,
            'rank_E8': 8,
            'dim_E8': 248,
            'dim_E8xE8': 496,
            'dim_G2': 14,
            'dim_K7': 7,
            'dim_J3O': 27,
            'b2': 21,
            'b3': 77,
            'H_star': 99,
            'N_gen': 3,
            'M5': 31
        }

        # Mersenne primes
        self.mersenne = {'M2': 3, 'M3': 7, 'M5': 31, 'M7': 127, 'M13': 8191}

        # Fermat primes
        self.fermat = {'F0': 3, 'F1': 5, 'F2': 17, 'F3': 257}

        # Constants
        self.phi = (1 + math.sqrt(5)) / 2
        self.zeta3 = 1.2020569031595942
        self.gamma = 0.5772156649015329

        # Observables
        self.obs = {
            'alpha_inv_MZ': 127.955,
            'sin2thetaW': 0.23122,
            'alpha_s_MZ': 0.1179,
            'theta12': 33.44,
            'theta13': 8.61,
            'theta23': 49.2,
            'delta_CP': 197.0,
            'Q_Koide': 0.6667,
            'm_mu_m_e': 206.768,
            'm_tau_m_e': 3477.0,
            'm_s_m_d': 20.0,
            'lambda_H': 0.129,
            'Omega_DE': 0.6847,
            'Omega_DM': 0.120,
            'n_s': 0.9649,
            'H0': 73.04
        }

        self.discoveries = []

    def deviation(self, pred, exp):
        """Calculate percentage deviation"""
        if exp == 0:
            return float('inf')
        return abs((pred - exp) / exp) * 100

    def test_formula(self, obs_name, obs_val, formula_str, pred_val):
        """Test a formula and record if deviation < 5%"""
        dev = self.deviation(pred_val, obs_val)

        if dev < 5.0 and not math.isnan(pred_val) and not math.isinf(pred_val):
            confidence = 'B' if dev < 0.1 else ('C' if dev < 1.0 else 'D')

            self.discoveries.append({
                'observable': obs_name,
                'formula': formula_str,
                'gift_value': pred_val,
                'experimental': obs_val,
                'deviation_pct': dev,
                'confidence': confidence
            })

            if dev < 1.0:
                print(f"  ✓ {obs_name} = {formula_str}")
                print(f"    GIFT: {pred_val:.6f}, Exp: {obs_val:.6f}, Dev: {dev:.3f}%")

    def explore_mersenne_combos(self):
        """Explore Mersenne prime combinations"""
        print("\n=== Mersenne Prime Combinations ===")

        for m_name, m_val in self.mersenne.items():
            # Test simple combinations with constants
            combos = [
                (f'phi/{m_name}', self.phi / m_val),
                (f'(pi+gamma)/{m_name}', (math.pi + self.gamma) / m_val),
                (f'ln(2)/{m_name}', math.log(2) / m_val),
                (f'zeta(3)*gamma/{m_name}', self.zeta3 * self.gamma / m_val),
                (f'sqrt({m_name})', math.sqrt(m_val)),
            ]

            for formula, value in combos:
                for obs_name, obs_val in self.obs.items():
                    self.test_formula(obs_name, obs_val, formula, value)

    def explore_fermat_combos(self):
        """Explore Fermat prime combinations"""
        print("\n=== Fermat Prime Combinations ===")

        for f_name, f_val in self.fermat.items():
            combos = [
                (f'sqrt({f_name})', math.sqrt(f_val)),
                (f'phi*{f_name}', self.phi * f_val),
                (f'ln(2)*{f_name}', math.log(2) * f_val),
            ]

            for formula, value in combos:
                for obs_name, obs_val in self.obs.items():
                    self.test_formula(obs_name, obs_val, formula, value)

    def explore_param_ratios(self):
        """Explore parameter ratios"""
        print("\n=== Parameter Ratios (Sample) ===")

        # Key ratios
        ratios = [
            ('b2/b3', self.params['b2'] / self.params['b3']),
            ('H_star/b3', self.params['H_star'] / self.params['b3']),
            ('dim_E8/dim_G2', self.params['dim_E8'] / self.params['dim_G2']),
            ('(rank_E8+b3)/H_star', (self.params['rank_E8'] + self.params['b3']) / self.params['H_star']),
            ('xi^2', self.params['xi'] ** 2),
            ('tau*sqrt(M13)', self.params['tau'] * math.sqrt(8191)),
        ]

        for formula, value in ratios:
            for obs_name, obs_val in self.obs.items():
                self.test_formula(obs_name, obs_val, formula, value)

    def explore_triple_sums(self):
        """Explore triple combinations"""
        print("\n=== Triple Combinations (Key Ones) ===")

        triples = [
            ('(p2+Weyl)', self.params['p2'] + self.params['Weyl']),
            ('(Weyl+rank_E8)', self.params['Weyl'] + self.params['rank_E8']),
            ('(p2+Weyl+rank_E8)', self.params['p2'] + self.params['Weyl'] + self.params['rank_E8']),
            ('(rank_E8-Weyl)', self.params['rank_E8'] - self.params['Weyl']),
            ('p2^2*Weyl', self.params['p2']**2 * self.params['Weyl']),
            ('dim_G2/b2', self.params['dim_G2'] / self.params['b2']),
        ]

        for formula, value in triples:
            for obs_name, obs_val in self.obs.items():
                self.test_formula(obs_name, obs_val, formula, value)

    def generate_report(self):
        """Generate markdown report"""
        # Sort by confidence and deviation
        sorted_disc = sorted(self.discoveries,
                           key=lambda d: (d['confidence'], d['deviation_pct']))

        # Count by confidence
        conf_counts = {'B': 0, 'C': 0, 'D': 0}
        for d in sorted_disc:
            conf_counts[d['confidence']] += 1

        # Generate report
        report = f"""# GIFT Quick Exploration Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Discoveries**: {len(self.discoveries)}

## Summary by Confidence

- **High (B)**: {conf_counts['B']} discoveries (dev < 0.1%)
- **Moderate (C)**: {conf_counts['C']} discoveries (0.1% < dev < 1%)
- **Interesting (D)**: {conf_counts['D']} discoveries (1% < dev < 5%)

## High Confidence Discoveries (B)

"""

        for d in sorted_disc:
            if d['confidence'] == 'B':
                report += f"""### {d['observable']} = {d['formula']}

- **GIFT Value**: {d['gift_value']:.8f}
- **Experimental**: {d['experimental']:.8f}
- **Deviation**: {d['deviation_pct']:.4f}%

"""

        report += "\n## Moderate Confidence Discoveries (C)\n\n"
        for d in sorted_disc:
            if d['confidence'] == 'C':
                report += f"- `{d['observable']} = {d['formula']}` = {d['gift_value']:.6f} (dev: {d['deviation_pct']:.3f}%)\n"

        report += "\n## Interesting Patterns (D)\n\n"
        for d in sorted_disc:
            if d['confidence'] == 'D':
                report += f"- `{d['observable']} = {d['formula']}` = {d['gift_value']:.6f} (dev: {d['deviation_pct']:.3f}%)\n"

        # Save report
        report_dir = Path(__file__).parent.parent / 'logs' / 'daily_reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"quick_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.write_text(report)

        print(f"\n{'='*60}")
        print(f"Report saved: {report_file}")
        print(f"  High confidence (B): {conf_counts['B']}")
        print(f"  Moderate (C): {conf_counts['C']}")
        print(f"  Interesting (D): {conf_counts['D']}")
        print(f"{'='*60}\n")

        return report_file

    def run(self):
        """Run quick exploration"""
        print("="*60)
        print("GIFT Quick Pattern Explorer")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        self.explore_mersenne_combos()
        self.explore_fermat_combos()
        self.explore_param_ratios()
        self.explore_triple_sums()

        report_file = self.generate_report()

        print("\n✓ Quick exploration complete!")
        print(f"Total discoveries: {len(self.discoveries)}")
        print(f"Report: {report_file}")


if __name__ == '__main__':
    explorer = QuickExplorer()
    explorer.run()
