#!/usr/bin/env python3
"""
GIFT Pattern Explorer - Systematic Exploration Tool
Explores mathematical relations between framework parameters and exotic constants
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, asdict
import sqlite3

@dataclass
class Discovery:
    """Data class for pattern discoveries"""
    id: int
    date: str
    category: str
    confidence: str  # A, B, C, D, E
    observable: str
    formula: str
    gift_value: float
    experimental_value: float
    deviation_pct: float
    interpretation: str
    cross_checks: List[str]
    status: str  # Confirmed, Under Review, Falsified

class GIFTPatternExplorer:
    """
    Systematic exploration tool for GIFT framework patterns
    """

    def __init__(self):
        # === Framework Parameters ===
        self.params = {
            # Fundamental (3)
            'p2': 2.0,
            'Weyl': 5.0,
            'tau': 10416 / 2673,

            # Derived (4)
            'beta0': np.pi / 8,
            'xi': 5 * np.pi / 16,
            'delta': 2 * np.pi / 25,
            'gamma_GIFT': 511 / 884,

            # Topological integers (11)
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

        # === Mersenne Primes ===
        self.mersenne = {
            'M2': 3,
            'M3': 7,
            'M5': 31,
            'M7': 127,
            'M13': 8191,
            'M17': 131071,
            'M19': 524287
        }

        # === Fermat Primes ===
        self.fermat = {
            'F0': 3,
            'F1': 5,
            'F2': 17,
            'F3': 257,
            'F4': 65537
        }

        # === Mathematical Constants ===
        self.constants = {
            'pi': np.pi,
            'e': np.e,
            'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio
            'sqrt2': np.sqrt(2),
            'sqrt3': np.sqrt(3),
            'sqrt5': np.sqrt(5),
            'sqrt17': np.sqrt(17),
            'ln2': np.log(2),
            'ln3': np.log(3),
            'ln10': np.log(10),
            'zeta2': np.pi**2 / 6,
            'zeta3': 1.2020569031595942,  # Apéry
            'gamma': 0.5772156649015329,  # Euler-Mascheroni
            'catalan': 0.915965594177219,  # Catalan's constant
            'glaisher': 1.282427129,  # Glaisher-Kinkelin
            'khinchin': 2.685452001,
            'mertens': 0.2614972128,
            'feigenbaum_delta': 4.669201609,
            'feigenbaum_alpha': 2.502907875,
            'levy': 1.186569110,
            'erdos_borwein': 1.606695152,
            'silver_ratio': 1 + np.sqrt(2),
            'plastic': 1.324717957
        }

        # === Experimental Observables ===
        self.observables = {
            # Gauge sector
            'alpha_inv_MZ': 127.955,
            'sin2thetaW': 0.23122,
            'alpha_s_MZ': 0.1179,

            # Neutrino sector
            'theta12': 33.44,  # degrees
            'theta13': 8.61,
            'theta23': 49.2,
            'delta_CP': 197.0,

            # Lepton sector
            'Q_Koide': 0.6667,
            'm_mu_m_e': 206.768,
            'm_tau_m_e': 3477.0,

            # Quark ratios
            'm_s_m_d': 20.0,

            # Higgs & cosmology
            'lambda_H': 0.129,
            'Omega_DE': 0.6847,
            'Omega_DM': 0.120,
            'n_s': 0.9649,
            'H0': 73.04
        }

        # === Discovery Database ===
        self.db_path = Path(__file__).parent.parent / 'data' / 'discovery_database.sqlite'
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # === Operations ===
        self.operations = {
            'identity': lambda x, y: x,
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'multiply': lambda x, y: x * y,
            'divide': lambda x, y: x / y if y != 0 else np.nan,
            'power': lambda x, y: x ** y if (x > 0 or y == int(y)) else np.nan,
            'sqrt': lambda x, y: np.sqrt(x) if x >= 0 else np.nan,
            'log': lambda x, y: np.log(x) if x > 0 else np.nan,
            'exp': lambda x, y: np.exp(x) if abs(x) < 100 else np.nan,
            'sin': lambda x, y: np.sin(x),
            'cos': lambda x, y: np.cos(x),
        }

        self.discovery_count = 0

    def _init_database(self):
        """Initialize SQLite database for discoveries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY,
                date TEXT,
                category TEXT,
                confidence TEXT,
                observable TEXT,
                formula TEXT,
                gift_value REAL,
                experimental_value REAL,
                deviation_pct REAL,
                interpretation TEXT,
                status TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def compute_deviation(self, predicted: float, experimental: float) -> float:
        """Compute percentage deviation"""
        if experimental == 0:
            return np.nan
        return abs((predicted - experimental) / experimental) * 100

    def classify_confidence(self, deviation_pct: float) -> str:
        """Classify discovery by confidence level"""
        if deviation_pct < 0.1:
            return 'B'  # HIGH CONFIDENCE
        elif deviation_pct < 1.0:
            return 'C'  # MODERATE
        elif deviation_pct < 5.0:
            return 'D'  # INTERESTING
        else:
            return 'E'  # NOISE

    def test_relation(self, observable_name: str, observable_value: float,
                     formula_str: str, predicted_value: float) -> Discovery:
        """Test a single relation and return Discovery object"""
        deviation = self.compute_deviation(predicted_value, observable_value)
        confidence = self.classify_confidence(deviation)

        discovery = Discovery(
            id=self.discovery_count,
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            category='systematic',
            confidence=confidence,
            observable=observable_name,
            formula=formula_str,
            gift_value=predicted_value,
            experimental_value=observable_value,
            deviation_pct=deviation,
            interpretation='',
            cross_checks=[],
            status='Under Review'
        )

        self.discovery_count += 1
        return discovery

    def explore_pairwise_ratios(self, threshold: float = 5.0) -> List[Discovery]:
        """Explore all pairwise ratios of framework parameters"""
        discoveries = []

        param_names = list(self.params.keys())

        print(f"\\n=== Exploring Pairwise Ratios ===")
        print(f"Total combinations: {len(param_names) * (len(param_names) - 1) // 2}")

        for i, name1 in enumerate(param_names):
            for name2 in param_names[i+1:]:
                val1 = self.params[name1]
                val2 = self.params[name2]

                # Test ratio val1/val2
                ratio = val1 / val2 if val2 != 0 else np.nan

                # Check against all observables
                for obs_name, obs_value in self.observables.items():
                    dev = self.compute_deviation(ratio, obs_value)

                    if dev < threshold:
                        formula_str = f"{obs_name} = {name1}/{name2}"
                        discovery = self.test_relation(obs_name, obs_value, formula_str, ratio)
                        discoveries.append(discovery)

                        if dev < 1.0:
                            print(f"  ✓ Found: {formula_str} = {ratio:.6f} (dev: {dev:.3f}%)")

        return discoveries

    def explore_mersenne_combinations(self, threshold: float = 5.0) -> List[Discovery]:
        """Explore combinations involving Mersenne primes"""
        discoveries = []

        print(f"\\n=== Exploring Mersenne Prime Combinations ===")

        for m_name, m_val in self.mersenne.items():
            for const_name, const_val in self.constants.items():
                # Test simple combinations
                combinations = {
                    f'{const_name}/{m_name}': const_val / m_val,
                    f'{const_name}*{m_name}': const_val * m_val,
                    f'{const_name}+{m_name}': const_val + m_val,
                    f'{const_name}-{m_name}': const_val - m_val,
                    f'sqrt({m_name})*{const_name}': np.sqrt(m_val) * const_val,
                    f'{const_name}^(1/{m_name})': const_val ** (1/m_val) if const_val > 0 else np.nan,
                }

                for formula, value in combinations.items():
                    if np.isnan(value) or np.isinf(value):
                        continue

                    for obs_name, obs_value in self.observables.items():
                        dev = self.compute_deviation(value, obs_value)

                        if dev < threshold:
                            formula_str = f"{obs_name} = {formula}"
                            discovery = self.test_relation(obs_name, obs_value, formula_str, value)
                            discoveries.append(discovery)

                            if dev < 1.0:
                                print(f"  ✓ Found: {formula_str} = {value:.6f} (dev: {dev:.3f}%)")

        return discoveries

    def explore_triple_combinations(self, param_subset: List[str] = None,
                                   threshold: float = 5.0) -> List[Discovery]:
        """Explore triple combinations of parameters"""
        discoveries = []

        if param_subset is None:
            # Focus on key topological parameters
            param_subset = ['b2', 'b3', 'H_star', 'rank_E8', 'dim_G2', 'dim_K7',
                          'N_gen', 'Weyl', 'p2']

        print(f"\\n=== Exploring Triple Combinations ===")
        print(f"Parameters: {param_subset}")

        tested = 0
        for i, name1 in enumerate(param_subset):
            for j, name2 in enumerate(param_subset[i+1:], start=i+1):
                for k, name3 in enumerate(param_subset[j+1:], start=j+1):
                    val1 = self.params[name1]
                    val2 = self.params[name2]
                    val3 = self.params[name3]

                    # Test various triple combinations
                    combos = {
                        f'({name1}+{name2})/{name3}': (val1 + val2) / val3 if val3 != 0 else np.nan,
                        f'({name1}-{name2})*{name3}': (val1 - val2) * val3,
                        f'{name1}/({name2}+{name3})': val1 / (val2 + val3) if (val2 + val3) != 0 else np.nan,
                        f'{name1}*{name2}/{name3}': val1 * val2 / val3 if val3 != 0 else np.nan,
                        f'sqrt({name1}*{name2})/{name3}': np.sqrt(abs(val1 * val2)) / val3 if val3 != 0 else np.nan,
                    }

                    for formula, value in combos.items():
                        if np.isnan(value) or np.isinf(value):
                            continue

                        for obs_name, obs_value in self.observables.items():
                            dev = self.compute_deviation(value, obs_value)

                            if dev < threshold:
                                formula_str = f"{obs_name} = {formula}"
                                discovery = self.test_relation(obs_name, obs_value, formula_str, value)
                                discoveries.append(discovery)

                                if dev < 1.0:
                                    print(f"  ✓ Found: {formula_str} = {value:.6f} (dev: {dev:.3f}%)")

                    tested += 1
                    if tested % 50 == 0:
                        print(f"  Progress: {tested} triple combinations tested...")

        return discoveries

    def save_discoveries(self, discoveries: List[Discovery], report_name: str = None):
        """Save discoveries to database and generate report"""
        if not discoveries:
            print("No discoveries to save.")
            return

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for disc in discoveries:
            cursor.execute('''
                INSERT INTO discoveries
                (id, date, category, confidence, observable, formula,
                 gift_value, experimental_value, deviation_pct, interpretation, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                disc.id, disc.date, disc.category, disc.confidence, disc.observable,
                disc.formula, disc.gift_value, disc.experimental_value,
                disc.deviation_pct, disc.interpretation, disc.status
            ))

        conn.commit()
        conn.close()

        # Generate report
        report_path = Path(__file__).parent.parent / 'logs' / 'daily_reports'
        report_path.mkdir(parents=True, exist_ok=True)

        if report_name is None:
            report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        report_file = report_path / report_name

        # Sort by confidence and deviation
        sorted_discoveries = sorted(discoveries,
                                   key=lambda d: (d.confidence, d.deviation_pct))

        # Count by confidence
        confidence_counts = {'B': 0, 'C': 0, 'D': 0, 'E': 0}
        for disc in sorted_discoveries:
            confidence_counts[disc.confidence] += 1

        with open(report_file, 'w') as f:
            f.write(f"# GIFT Pattern Explorer - Discovery Report\\n\\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Total Discoveries**: {len(discoveries)}\\n\\n")

            f.write(f"## Summary by Confidence\\n\\n")
            f.write(f"- **High (B)**: {confidence_counts['B']} (dev < 0.1%)\\n")
            f.write(f"- **Moderate (C)**: {confidence_counts['C']} (0.1% < dev < 1%)\\n")
            f.write(f"- **Interesting (D)**: {confidence_counts['D']} (1% < dev < 5%)\\n")
            f.write(f"- **Noise (E)**: {confidence_counts['E']} (dev > 5%)\\n\\n")

            f.write(f"## High Confidence Discoveries (B)\\n\\n")
            for disc in sorted_discoveries:
                if disc.confidence == 'B':
                    f.write(f"### Discovery #{disc.id:04d}\\n\\n")
                    f.write(f"**Formula**: `{disc.formula}`\\n")
                    f.write(f"**GIFT Value**: {disc.gift_value:.8f}\\n")
                    f.write(f"**Experimental**: {disc.experimental_value:.8f}\\n")
                    f.write(f"**Deviation**: {disc.deviation_pct:.4f}%\\n\\n")

            f.write(f"## Moderate Confidence Discoveries (C)\\n\\n")
            for disc in sorted_discoveries:
                if disc.confidence == 'C':
                    f.write(f"- `{disc.formula}` = {disc.gift_value:.6f} (dev: {disc.deviation_pct:.3f}%)\\n")

            f.write(f"\\n## Interesting Patterns (D)\\n\\n")
            for disc in sorted_discoveries:
                if disc.confidence == 'D':
                    f.write(f"- `{disc.formula}` = {disc.gift_value:.6f} (dev: {disc.deviation_pct:.3f}%)\\n")

        print(f"\\n✓ Report saved: {report_file}")
        print(f"  High confidence: {confidence_counts['B']}")
        print(f"  Moderate: {confidence_counts['C']}")
        print(f"  Interesting: {confidence_counts['D']}")

    def run_daily_exploration(self, quick: bool = False):
        """Run daily exploration routine"""
        print(f"\\n{'='*60}")
        print(f"GIFT Pattern Explorer - Daily Run")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        all_discoveries = []

        # 1. Pairwise ratios (quick sample)
        if not quick:
            discoveries = self.explore_pairwise_ratios(threshold=3.0)
            all_discoveries.extend(discoveries)
            print(f"\\n→ Pairwise: {len(discoveries)} discoveries")

        # 2. Mersenne combinations
        discoveries = self.explore_mersenne_combinations(threshold=3.0)
        all_discoveries.extend(discoveries)
        print(f"→ Mersenne: {len(discoveries)} discoveries")

        # 3. Triple combinations (focused)
        discoveries = self.explore_triple_combinations(threshold=3.0)
        all_discoveries.extend(discoveries)
        print(f"→ Triples: {len(discoveries)} discoveries")

        # 4. Save and report
        self.save_discoveries(all_discoveries)

        print(f"\\n{'='*60}")
        print(f"Daily exploration complete!")
        print(f"Total discoveries: {len(all_discoveries)}")
        print(f"{'='*60}\\n")


def main():
    """Main entry point"""
    explorer = GIFTPatternExplorer()

    # Run daily exploration
    explorer.run_daily_exploration(quick=False)


if __name__ == '__main__':
    main()
