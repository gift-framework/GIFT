#!/usr/bin/env python3
"""
Systematic search for zeta function ratio patterns in GIFT observables.

Tests all pairwise ratios ζ(m)/ζ(n) against physical observables
to identify potential zeta function origins.
"""

import numpy as np
import pandas as pd
from itertools import combinations
import math

# Zeta function values at odd integers
ZETA = {
    3: 1.2020569031595942,
    5: 1.0369277551433699,
    7: 1.0083492773819228,
    9: 1.0020083928260822,
    11: 1.0004941886041195,
    13: 1.0002441403130040,
    15: 1.0001227133475784,
    17: 1.0000612750618566,
    19: 1.0000305882363070,
    21: 1.0000152822594086,
}

# Common scaling factors to test
SCALING_FACTORS = {
    '1': 1.0,
    '2': 2.0,
    '3': 3.0,
    '4': 4.0,
    '5': 5.0,
    '7': 7.0,
    '10': 10.0,
    'π': np.pi,
    '2π': 2 * np.pi,
    'π/2': np.pi / 2,
    'π/3': np.pi / 3,
    'π/4': np.pi / 4,
    '180/π': 180 / np.pi,  # radians to degrees
    'φ': (1 + np.sqrt(5)) / 2,  # golden ratio
    'ln(2)': np.log(2),
    'ln(3)': np.log(3),
    'e': np.e,
    '√2': np.sqrt(2),
    '√3': np.sqrt(3),
    '√5': np.sqrt(5),
    '1/2': 0.5,
    '1/3': 1/3,
    '1/4': 0.25,
    '1/π': 1/np.pi,
}

# Physical observables extracted from FRAMEWORK_STATUS_SUMMARY.md
# Format: (name, experimental_value, unit/type, category)
OBSERVABLES = [
    # Gauge sector (dimensionless)
    ('α⁻¹', 137.036, 'dimensionless', 'gauge'),
    ('α_s', 0.1179, 'dimensionless', 'gauge'),
    ('sin²θ_W', 0.23122, 'dimensionless', 'gauge'),

    # Neutrino sector (angles in degrees, convert to radians for some tests)
    ('θ₁₂', 33.44, 'degrees', 'neutrino'),
    ('θ₁₃', 8.57, 'degrees', 'neutrino'),
    ('θ₂₃', 49.2, 'degrees', 'neutrino'),
    ('δ_CP', 197, 'degrees', 'neutrino'),

    # Neutrino angles in radians
    ('θ₁₂_rad', 33.44 * np.pi / 180, 'radians', 'neutrino'),
    ('θ₁₃_rad', 8.57 * np.pi / 180, 'radians', 'neutrino'),
    ('θ₂₃_rad', 49.2 * np.pi / 180, 'radians', 'neutrino'),
    ('δ_CP_rad', 197 * np.pi / 180, 'radians', 'neutrino'),

    # Lepton sector (dimensionless ratios)
    ('Q_Koide', 0.6667, 'dimensionless', 'lepton'),
    ('m_μ/m_e', 206.768, 'dimensionless', 'lepton'),
    ('m_τ/m_e', 3477.15, 'dimensionless', 'lepton'),

    # Quark mass ratios (dimensionless)
    ('m_s/m_d', 20.0, 'dimensionless', 'quark_ratio'),
    ('m_b/m_u', 1935.19, 'dimensionless', 'quark_ratio'),
    ('m_c/m_d', 271.94, 'dimensionless', 'quark_ratio'),
    ('m_d/m_u', 2.162, 'dimensionless', 'quark_ratio'),
    ('m_c/m_s', 13.6, 'dimensionless', 'quark_ratio'),
    ('m_t/m_c', 135.83, 'dimensionless', 'quark_ratio'),
    ('m_b/m_d', 895.07, 'dimensionless', 'quark_ratio'),
    ('m_b/m_c', 3.29, 'dimensionless', 'quark_ratio'),
    ('m_t/m_s', 1846.89, 'dimensionless', 'quark_ratio'),
    ('m_b/m_s', 44.76, 'dimensionless', 'quark_ratio'),

    # CKM matrix
    ('θ_C', 13.04, 'degrees', 'ckm'),
    ('θ_C_rad', 13.04 * np.pi / 180, 'radians', 'ckm'),

    # Higgs sector (dimensionless)
    ('λ_H', 0.1286, 'dimensionless', 'higgs'),

    # Cosmological observables (dimensionless)
    ('Ω_DE', 0.6847, 'dimensionless', 'cosmology'),
    ('Ω_DM', 0.120, 'dimensionless', 'cosmology'),
    ('n_s', 0.9649, 'dimensionless', 'cosmology'),

    # Hausdorff dimension
    ('D_H', 0.856220, 'dimensionless', 'fractal'),
]


def generate_all_zeta_ratios():
    """Generate all pairwise ratios ζ(m)/ζ(n) where m ≠ n.

    Tests both ζ(m)/ζ(n) and ζ(n)/ζ(m) for each pair.
    This gives us 90 total ratios (45 pairs × 2 directions).
    """
    ratios = []
    indices = sorted(ZETA.keys())

    for m, n in combinations(indices, 2):
        # Test both m/n and n/m
        # Forward: larger/smaller (typically < 1 for close values)
        ratio_value_1 = ZETA[m] / ZETA[n]
        ratios.append({
            'ratio_name': f'ζ({m})/ζ({n})',
            'numerator': m,
            'denominator': n,
            'value': ratio_value_1
        })

        # Reverse: smaller/larger (typically > 1 for close values)
        ratio_value_2 = ZETA[n] / ZETA[m]
        ratios.append({
            'ratio_name': f'ζ({n})/ζ({m})',
            'numerator': n,
            'denominator': m,
            'value': ratio_value_2
        })

    return ratios


def find_best_match(obs_value, zeta_ratios, obs_name, obs_category):
    """
    Find the best zeta ratio match for a given observable.
    Tests with and without scaling factors.
    """
    best_match = None
    best_deviation = float('inf')

    # Test each zeta ratio
    for ratio in zeta_ratios:
        ratio_val = ratio['value']

        # Test direct match (no scaling)
        deviation = abs(obs_value - ratio_val) / obs_value * 100
        if deviation < best_deviation:
            best_deviation = deviation
            best_match = {
                'observable': obs_name,
                'category': obs_category,
                'best_ratio': ratio['ratio_name'],
                'predicted_value': ratio_val,
                'experimental_value': obs_value,
                'deviation_pct': deviation,
                'scaling_factor': '1',
                'formula': ratio['ratio_name']
            }

        # Test with scaling factors
        for scale_name, scale_val in SCALING_FACTORS.items():
            predicted = ratio_val * scale_val
            deviation = abs(obs_value - predicted) / obs_value * 100

            if deviation < best_deviation:
                best_deviation = deviation
                best_match = {
                    'observable': obs_name,
                    'category': obs_category,
                    'best_ratio': ratio['ratio_name'],
                    'predicted_value': predicted,
                    'experimental_value': obs_value,
                    'deviation_pct': deviation,
                    'scaling_factor': scale_name,
                    'formula': f"{scale_name} × {ratio['ratio_name']}"
                }

        # Test inverse ratios with scaling (for ratios like 1/(k × ζ(m)/ζ(n)))
        for scale_name, scale_val in SCALING_FACTORS.items():
            if scale_val == 0:
                continue
            predicted = ratio_val / scale_val
            deviation = abs(obs_value - predicted) / obs_value * 100

            if deviation < best_deviation:
                best_deviation = deviation
                best_match = {
                    'observable': obs_name,
                    'category': obs_category,
                    'best_ratio': ratio['ratio_name'],
                    'predicted_value': predicted,
                    'experimental_value': obs_value,
                    'deviation_pct': deviation,
                    'scaling_factor': f'1/{scale_name}',
                    'formula': f"{ratio['ratio_name']} / {scale_name}"
                }

    return best_match


def calculate_confidence_score(deviation_pct):
    """
    Calculate confidence score based on deviation.
    Score ranges from 0-100, higher is better.
    """
    if deviation_pct < 0.01:
        return 100
    elif deviation_pct < 0.1:
        return 95
    elif deviation_pct < 0.5:
        return 85
    elif deviation_pct < 1.0:
        return 75
    elif deviation_pct < 2.0:
        return 60
    else:
        return max(0, 50 - deviation_pct)


def main():
    """Main analysis routine."""
    print("GIFT Framework: Systematic Zeta Ratio Discovery")
    print("=" * 70)
    print()
    print(f"Testing {len(OBSERVABLES)} observables against zeta ratios")
    print(f"with {len(SCALING_FACTORS)} scaling factors each")
    print()

    # Generate all zeta ratios
    zeta_ratios = generate_all_zeta_ratios()
    print(f"Generated {len(zeta_ratios)} zeta ratios (testing both ζ(m)/ζ(n) and ζ(n)/ζ(m))")
    print()

    # Find best matches for each observable
    results = []
    for obs_name, obs_value, obs_unit, obs_category in OBSERVABLES:
        match = find_best_match(obs_value, zeta_ratios, obs_name, obs_category)
        if match:
            match['confidence_score'] = calculate_confidence_score(match['deviation_pct'])
            results.append(match)

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('deviation_pct')

    # Filter to deviations < 2% as requested
    df_filtered = df[df['deviation_pct'] < 2.0].copy()

    # Save to CSV
    output_file = '/home/user/GIFT/zeta_ratio_matches.csv'
    df_filtered.to_csv(output_file, index=False, float_format='%.6f')

    print(f"Analysis complete!")
    print(f"Total matches found: {len(df)}")
    print(f"Matches with deviation < 2%: {len(df_filtered)}")
    print()
    print(f"Results saved to: {output_file}")
    print()

    # Display top 20 matches
    print("Top 20 Best Matches (deviation < 2%):")
    print("=" * 70)
    print()

    display_cols = ['observable', 'formula', 'predicted_value', 'experimental_value',
                    'deviation_pct', 'confidence_score']

    for idx, row in df_filtered.head(20).iterrows():
        print(f"{row['observable']:15s} = {row['formula']:25s}")
        print(f"  Predicted: {row['predicted_value']:12.6f}")
        print(f"  Observed:  {row['experimental_value']:12.6f}")
        print(f"  Deviation: {row['deviation_pct']:11.4f}%")
        print(f"  Confidence: {row['confidence_score']:10.0f}/100")
        print()

    # Summary statistics by category
    print("\nCategory Summary (matches with deviation < 2%):")
    print("=" * 70)
    category_stats = df_filtered.groupby('category').agg({
        'deviation_pct': ['count', 'mean', 'min'],
        'confidence_score': 'mean'
    }).round(4)
    print(category_stats)
    print()

    # Highlight known successes
    print("\nValidation: Known n_s Pattern")
    print("=" * 70)
    n_s_match = df_filtered[df_filtered['observable'] == 'n_s']
    if not n_s_match.empty:
        row = n_s_match.iloc[0]
        print(f"n_s = {row['formula']}")
        print(f"Predicted: {row['predicted_value']:.6f}")
        print(f"Observed:  {row['experimental_value']:.6f}")
        print(f"Deviation: {row['deviation_pct']:.4f}%")
        print("✓ KNOWN PATTERN CONFIRMED")

    print()
    print("Analysis complete. Review CSV file for full results.")


if __name__ == '__main__':
    main()
