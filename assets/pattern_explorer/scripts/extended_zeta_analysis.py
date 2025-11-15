#!/usr/bin/env python3
"""
Extended zeta function pattern analysis.
Tests products, sums, and more complex combinations.
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations
from pathlib import Path

# Zeta function values
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

# Observables (focusing on those not well-matched by simple ratios)
OBSERVABLES_EXTENDED = [
    ('α⁻¹', 137.036, 'dimensionless', 'gauge'),
    ('m_μ/m_e', 206.768, 'dimensionless', 'lepton'),
    ('m_τ/m_e', 3477.15, 'dimensionless', 'lepton'),
    ('m_s/m_d', 20.0, 'dimensionless', 'quark_ratio'),
    ('m_b/m_u', 1935.19, 'dimensionless', 'quark_ratio'),
    ('m_c/m_d', 271.94, 'dimensionless', 'quark_ratio'),
    ('m_t/m_c', 135.83, 'dimensionless', 'quark_ratio'),
    ('m_b/m_d', 895.07, 'dimensionless', 'quark_ratio'),
    ('m_t/m_s', 1846.89, 'dimensionless', 'quark_ratio'),
    ('m_b/m_s', 44.76, 'dimensionless', 'quark_ratio'),
    ('λ_H', 0.1286, 'dimensionless', 'higgs'),
    ('H₀_ratio', 73.04 / 67.36, 'dimensionless', 'cosmology'),  # Hubble tension ratio
]

# Extended test patterns
def test_extended_patterns(obs_value, obs_name):
    """Test more complex zeta combinations."""
    best_match = None
    best_dev = float('inf')
    indices = sorted(ZETA.keys())

    # Test 1: Products ζ(m) × ζ(n)
    for m, n in combinations(indices, 2):
        product = ZETA[m] * ZETA[n]
        for scale in [1, 2, 3, 5, 10, 100, 1000, np.pi, np.e, np.sqrt(2), np.sqrt(5)]:
            pred = product * scale
            dev = abs(obs_value - pred) / obs_value * 100
            if dev < best_dev:
                best_dev = dev
                best_match = (f"{scale:.4g} × ζ({m})×ζ({n})", pred, dev, 'product')

    # Test 2: Sums ζ(m) + ζ(n)
    for m, n in combinations(indices, 2):
        sum_val = ZETA[m] + ZETA[n]
        for scale in [1, 2, 5, 10, 100, np.pi, np.e]:
            pred = sum_val * scale
            dev = abs(obs_value - pred) / obs_value * 100
            if dev < best_dev:
                best_dev = dev
                best_match = (f"{scale:.4g} × [ζ({m})+ζ({n})]", pred, dev, 'sum')

    # Test 3: Differences ζ(m) - ζ(n)
    for m in indices:
        for n in indices:
            if m != n:
                diff = ZETA[m] - ZETA[n]
                if abs(diff) < 1e-10:  # Skip near-zero differences
                    continue
                for scale in [1, 10, 100, 1000, 10000]:
                    pred = diff * scale
                    dev = abs(obs_value - pred) / obs_value * 100
                    if dev < best_dev:
                        best_dev = dev
                        best_match = (f"{scale:.4g} × [ζ({m})-ζ({n})]", pred, dev, 'difference')

    # Test 4: Inverse sums 1/ζ(m) + 1/ζ(n)
    for m, n in combinations(indices, 2):
        inv_sum = 1/ZETA[m] + 1/ZETA[n]
        for scale in [1, 2, 10, 100]:
            pred = inv_sum * scale
            dev = abs(obs_value - pred) / obs_value * 100
            if dev < best_dev:
                best_dev = dev
                best_match = (f"{scale:.4g} × [1/ζ({m})+1/ζ({n})]", pred, dev, 'inv_sum')

    # Test 5: Power ratios ζ(m)^a / ζ(n)^b
    for m, n in combinations(indices, 2):
        for a in [1, 2, 3]:
            for b in [1, 2, 3]:
                if a == 1 and b == 1:  # Skip simple ratio (already tested)
                    continue
                power_ratio = (ZETA[m]**a) / (ZETA[n]**b)
                for scale in [1, 10, 100]:
                    pred = power_ratio * scale
                    dev = abs(obs_value - pred) / obs_value * 100
                    if dev < best_dev:
                        best_dev = dev
                        best_match = (f"{scale:.4g} × ζ({m})^{a}/ζ({n})^{b}", pred, dev, 'power_ratio')

    # Test 6: Three-zeta combinations ζ(m)/[ζ(n)×ζ(p)]
    for m in indices[:7]:  # Limit to avoid explosion
        for n, p in combinations([i for i in indices if i != m][:6], 2):
            combo = ZETA[m] / (ZETA[n] * ZETA[p])
            for scale in [1, 10, 100, 1000]:
                pred = combo * scale
                dev = abs(obs_value - pred) / obs_value * 100
                if dev < best_dev:
                    best_dev = dev
                    best_match = (f"{scale:.4g} × ζ({m})/[ζ({n})×ζ({p})]", pred, dev, 'triple')

    return best_match


def main():
    """Run extended pattern analysis."""
    print("Extended Zeta Pattern Analysis")
    print("=" * 80)
    print()
    print("Testing products, sums, differences, and higher-order combinations...")
    print()

    results = []

    for obs_name, obs_value, obs_unit, obs_category in OBSERVABLES_EXTENDED:
        print(f"Analyzing {obs_name:15s} = {obs_value:12.4f}...", end=' ')
        match = test_extended_patterns(obs_value, obs_name)

        if match:
            formula, pred, dev, pattern_type = match
            print(f"Best: {dev:6.3f}% ({pattern_type})")

            results.append({
                'observable': obs_name,
                'category': obs_category,
                'experimental_value': obs_value,
                'predicted_value': pred,
                'deviation_pct': dev,
                'formula': formula,
                'pattern_type': pattern_type
            })
        else:
            print("No match found")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values('deviation_pct')

    # Filter to < 5% deviation for extended patterns (more lenient)
    df_good = df[df['deviation_pct'] < 5.0]

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / 'extended_zeta_patterns.csv'
    df_good.to_csv(output_file, index=False, float_format='%.6f')

    print()
    print("=" * 80)
    print(f"Found {len(df_good)} patterns with < 5% deviation")
    print()
    print("Top 15 Extended Patterns:")
    print("-" * 80)

    for idx, row in df_good.head(15).iterrows():
        print(f"\n{row['observable']:15s} = {row['formula']}")
        print(f"  Predicted: {row['predicted_value']:12.6f}")
        print(f"  Observed:  {row['experimental_value']:12.6f}")
        print(f"  Deviation: {row['deviation_pct']:11.4f}%")
        print(f"  Type: {row['pattern_type']}")

    print()
    print(f"\nResults saved to: {output_file}")

    # Pattern type summary
    print("\nPattern Type Summary:")
    print("-" * 80)
    type_summary = df_good.groupby('pattern_type').agg({
        'deviation_pct': ['count', 'mean', 'min']
    }).round(3)
    print(type_summary)


if __name__ == '__main__':
    main()
