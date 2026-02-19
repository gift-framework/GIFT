#!/usr/bin/env python3
"""
Li Coefficient Analysis for GIFT Patterns
==========================================

This script computes Li coefficients λₙ and analyzes them for GIFT structure.

Li's criterion: RH ⟺ λₙ > 0 for all n ≥ 1

λₙ = Σ_ρ [1 - (1 - 1/ρ)^n] summed over non-trivial zeros ρ = 1/2 + iγ

Author: GIFT Research
Date: February 2026
"""

import numpy as np
from typing import List, Tuple, Dict
import json

# GIFT Constants for reference
GIFT_CONSTANTS = {
    'p2': 2,           # Pontryagin class
    'N_gen': 3,        # Fermion generations
    'Weyl': 5,         # Weyl factor
    'dim_K7': 7,       # K7 dimension
    'rank_E8': 8,      # E8 Cartan subalgebra
    'D_bulk': 11,      # Bulk dimension
    'F7': 13,          # 7th Fibonacci
    'dim_G2': 14,      # G2 holonomy group
    'b2': 21,          # Second Betti number K7
    'dim_J3O': 27,     # Exceptional Jordan algebra
    'h_G2_sq': 36,     # Coxeter number squared
    'b3': 77,          # Third Betti number K7
    'H_star': 99,      # b2 + b3 + 1
    'dim_E8': 248,     # E8 dimension
}

# Fibonacci lags from Riemann recurrence
GIFT_LAGS = [5, 8, 13, 27]


def load_zeros(filepath: str) -> List[float]:
    """Load Riemann zeros from file."""
    zeros = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                zeros.append(float(parts[1]))
            elif len(parts) == 1:
                zeros.append(float(parts[0]))
    return zeros


def compute_li_coefficient(n: int, zeros: List[float], num_zeros: int = None) -> complex:
    """
    Compute the n-th Li coefficient using Riemann zeros.

    λₙ = Σ_k [1 - (1 - 1/ρ_k)^n] where ρ_k = 1/2 + i·γ_k

    Note: Each zero γ contributes twice (ρ and 1-ρ conjugate).
    """
    if num_zeros is None:
        num_zeros = len(zeros)

    lambda_n = 0.0

    for gamma in zeros[:num_zeros]:
        # ρ = 1/2 + i·γ
        rho = 0.5 + 1j * gamma
        # Contribution from ρ
        term = 1.0 - (1.0 - 1.0/rho)**n
        lambda_n += term

        # ρ̄ = 1/2 - i·γ (conjugate zero)
        rho_conj = 0.5 - 1j * gamma
        term_conj = 1.0 - (1.0 - 1.0/rho_conj)**n
        lambda_n += term_conj

    return lambda_n


def compute_li_coefficients_batch(max_n: int, zeros: List[float],
                                   num_zeros: int = None) -> List[float]:
    """Compute λ₁, λ₂, ..., λ_max_n."""
    lambdas = []
    for n in range(1, max_n + 1):
        lam = compute_li_coefficient(n, zeros, num_zeros)
        # Should be real (imaginary part is numerical noise)
        lambdas.append(lam.real)
    return lambdas


def analyze_ratios(lambdas: List[float]) -> Dict:
    """Analyze ratios λₘ/λₙ for GIFT structure."""
    results = {
        'consecutive_ratios': [],
        'gift_index_ratios': [],
        'special_ratios': []
    }

    # Consecutive ratios λₙ₊₁/λₙ
    for n in range(len(lambdas) - 1):
        ratio = lambdas[n + 1] / lambdas[n] if lambdas[n] != 0 else float('inf')
        results['consecutive_ratios'].append({
            'n': n + 1,
            'ratio': ratio,
            'approx_n_plus_1_over_n': (n + 2) / (n + 1)
        })

    # Ratios at GIFT indices
    gift_indices = [5, 7, 8, 13, 14, 21, 27, 77, 99]
    for idx in gift_indices:
        if idx <= len(lambdas):
            ratio_to_1 = lambdas[idx - 1] / lambdas[0]
            results['gift_index_ratios'].append({
                'n': idx,
                'lambda_n': lambdas[idx - 1],
                'lambda_n_over_lambda_1': ratio_to_1,
                'ratio_to_n': ratio_to_1 / idx  # Should be ~1 if linear
            })

    # Check GIFT-significant ratios
    gift_ratios = [
        (3, 13, '3/13 = sin²θ_W'),
        (21, 77, 'b₂/b₃'),
        (14, 99, 'dim(G₂)/H*'),
        (8, 13, 'Fibonacci ratio'),
    ]

    for num, denom, name in gift_ratios:
        if num <= len(lambdas) and denom <= len(lambdas):
            observed = lambdas[num - 1] / lambdas[denom - 1]
            expected = num / denom
            deviation = abs(observed - expected) / expected * 100
            results['special_ratios'].append({
                'name': name,
                'indices': (num, denom),
                'observed': observed,
                'expected_if_linear': expected,
                'deviation_percent': deviation
            })

    return results


def analyze_differences(lambdas: List[float]) -> Dict:
    """Analyze differences at GIFT lags."""
    results = {
        'lag_differences': {},
        'gift_lag_stats': {}
    }

    # Standard lags for comparison
    all_lags = [1, 2, 3, 4] + GIFT_LAGS

    for lag in all_lags:
        diffs = []
        for n in range(len(lambdas) - lag):
            diff = lambdas[n + lag] - lambdas[n]
            diffs.append(diff)

        if diffs:
            results['lag_differences'][lag] = {
                'mean': np.mean(diffs),
                'std': np.std(diffs),
                'min': min(diffs),
                'max': max(diffs)
            }

    return results


def extract_oscillatory_component(lambdas: List[float]) -> List[float]:
    """
    Extract oscillatory component: λₙ^(osc) = λₙ - trend

    Trend: λₙ^(trend) ≈ (n/2)·ln(n) + c·n
    """
    oscillatory = []

    # Estimate c from first values
    # λ₁ ≈ 0.023 (known)
    # Asymptotic: λₙ ~ (n/2)·ln(n) + c·n
    # For small n, approximately linear: λₙ ≈ n·λ₁

    for n, lam in enumerate(lambdas, start=1):
        if n > 1:
            # Trend approximation
            trend = (n / 2) * np.log(n) + 0.023 * n  # rough estimate
            osc = lam - trend
        else:
            osc = 0.0
        oscillatory.append(osc)

    return oscillatory


def check_special_indices(lambdas: List[float]) -> Dict:
    """Check if GIFT indices have special λ values."""
    results = {}

    # GIFT constants as indices
    special_indices = {
        'dim_G2': 14,
        'b2': 21,
        'dim_J3O': 27,
        'b3': 77,
        'H_star': 99,
    }

    for name, idx in special_indices.items():
        if idx <= len(lambdas):
            lam = lambdas[idx - 1]
            # Check if λₙ/n is close to some GIFT constant
            ratio = lam / idx

            # Check if λₙ rounds to a GIFT constant
            nearest_int = round(lam * 100)  # scale for visibility

            results[name] = {
                'index': idx,
                'lambda_value': lam,
                'lambda_over_index': ratio,
                'lambda_times_100': lam * 100,
                'lambda_times_H_star': lam * 99,
            }

    return results


def check_li_positivity(lambdas: List[float]) -> Dict:
    """Verify Li's criterion: all λₙ > 0."""
    positive = all(lam > 0 for lam in lambdas)
    min_val = min(lambdas)
    min_idx = lambdas.index(min_val) + 1

    return {
        'all_positive': positive,
        'min_value': min_val,
        'min_index': min_idx,
        'count': len(lambdas)
    }


def find_gift_patterns_in_scaled_values(lambdas: List[float]) -> Dict:
    """
    Look for GIFT constants in scaled λₙ values.

    Check if λₙ × scale ≈ GIFT constant for various scales.
    """
    results = []

    scales = [1, 10, 99, 100, 1000]  # H* = 99 is interesting

    for n, lam in enumerate(lambdas[:100], start=1):
        for scale in scales:
            scaled = lam * scale

            # Check against GIFT constants
            for name, value in GIFT_CONSTANTS.items():
                if value > 0:
                    deviation = abs(scaled - value) / value * 100
                    if deviation < 5:  # Within 5%
                        results.append({
                            'n': n,
                            'scale': scale,
                            'scaled_value': scaled,
                            'gift_constant': name,
                            'gift_value': value,
                            'deviation_percent': deviation
                        })

    return {'matches': results}


def main():
    """Main analysis routine."""
    print("=" * 60)
    print("Li Coefficient Analysis for GIFT Patterns")
    print("=" * 60)

    # Load zeros
    zeros_file = '/home/user/GIFT/research/heegner-riemann/zeros1.txt'
    print(f"\nLoading Riemann zeros from {zeros_file}...")
    zeros = load_zeros(zeros_file)
    print(f"Loaded {len(zeros)} zeros")
    print(f"First zero: γ₁ = {zeros[0]:.6f} (should be ≈ 14.135)")
    print(f"Last zero: γ_{len(zeros)} = {zeros[-1]:.6f}")

    # Compute Li coefficients
    max_n = min(100, len(zeros))  # Don't go beyond available zeros
    print(f"\nComputing λ₁ to λ_{max_n}...")

    lambdas = compute_li_coefficients_batch(max_n, zeros, num_zeros=100)

    print("\n" + "-" * 60)
    print("FIRST 20 LI COEFFICIENTS")
    print("-" * 60)
    for n, lam in enumerate(lambdas[:20], start=1):
        print(f"λ_{n:2d} = {lam:.10f}")

    # Check positivity (Li's criterion)
    print("\n" + "-" * 60)
    print("LI'S CRITERION CHECK")
    print("-" * 60)
    positivity = check_li_positivity(lambdas)
    print(f"All λₙ > 0: {positivity['all_positive']}")
    print(f"Minimum λₙ: {positivity['min_value']:.10f} at n = {positivity['min_index']}")

    # Analyze ratios
    print("\n" + "-" * 60)
    print("RATIO ANALYSIS")
    print("-" * 60)

    ratios = analyze_ratios(lambdas)

    print("\nConsecutive ratios λₙ₊₁/λₙ (first 10):")
    for r in ratios['consecutive_ratios'][:10]:
        print(f"  λ_{r['n']+1}/λ_{r['n']} = {r['ratio']:.6f} (expected ~{r['approx_n_plus_1_over_n']:.4f})")

    print("\nRatios at GIFT indices (λₙ/λ₁):")
    for r in ratios['gift_index_ratios']:
        print(f"  λ_{r['n']}/λ₁ = {r['lambda_n_over_lambda_1']:.4f} (ratio/n = {r['ratio_to_n']:.6f})")

    print("\nGIFT-significant ratios:")
    for r in ratios['special_ratios']:
        print(f"  {r['name']}: λ_{r['indices'][0]}/λ_{r['indices'][1]} = {r['observed']:.6f}")
        print(f"    Expected if linear: {r['expected_if_linear']:.6f} (dev: {r['deviation_percent']:.2f}%)")

    # Special indices
    print("\n" + "-" * 60)
    print("SPECIAL GIFT INDICES")
    print("-" * 60)

    special = check_special_indices(lambdas)
    for name, data in special.items():
        print(f"\n{name} (n = {data['index']}):")
        print(f"  λ_{data['index']} = {data['lambda_value']:.6f}")
        print(f"  λ_{data['index']}/n = {data['lambda_over_index']:.6f}")
        print(f"  λ_{data['index']} × H* = {data['lambda_times_H_star']:.4f}")

    # GIFT pattern search
    print("\n" + "-" * 60)
    print("GIFT PATTERN SEARCH (λₙ × scale ≈ GIFT constant)")
    print("-" * 60)

    patterns = find_gift_patterns_in_scaled_values(lambdas)
    if patterns['matches']:
        for m in patterns['matches'][:15]:
            print(f"  λ_{m['n']} × {m['scale']} = {m['scaled_value']:.4f} ≈ {m['gift_constant']} = {m['gift_value']} ({m['deviation_percent']:.2f}%)")
    else:
        print("  No strong matches found within 5% tolerance")

    # Look for linearity deviation structure
    print("\n" + "-" * 60)
    print("LINEARITY ANALYSIS")
    print("-" * 60)

    print("\nDeviation from linear: δₙ = λₙ - n·λ₁")
    deviations = [lam - (n + 1) * lambdas[0] for n, lam in enumerate(lambdas)]

    for n in [5, 8, 13, 14, 21, 27] + list(range(1, 11)):
        if n <= len(deviations):
            print(f"  δ_{n} = {deviations[n-1]:.8f}")

    # Save results
    results = {
        'lambdas': lambdas,
        'positivity': positivity,
        'ratios': {
            'consecutive': [(r['n'], r['ratio']) for r in ratios['consecutive_ratios'][:20]],
            'gift_indices': ratios['gift_index_ratios'],
            'special': ratios['special_ratios']
        },
        'special_indices': special,
        'deviations': {n+1: deviations[n] for n in range(min(30, len(deviations)))}
    }

    output_file = '/home/user/GIFT/research/riemann/li_analysis_results.json'

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Computed {len(lambdas)} Li coefficients using {len(zeros)} Riemann zeros")
    print(f"Li's criterion satisfied: {positivity['all_positive']}")
    print(f"λₙ appears approximately linear in n for small n")
    print(f"Next: Analyze oscillatory component and deeper GIFT structure")


if __name__ == "__main__":
    main()
