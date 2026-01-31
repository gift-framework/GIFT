#!/usr/bin/env python3
"""
GIFT Phase 2.10b: Multi-conductor L-function Analysis
======================================================

Compare GIFT structure across multiple Dirichlet L-functions.

GIFT-pertinent conductors:
- q=5   (Weyl orbits)
- q=7   (dim K₇)
- q=21  (b₂)
- q=77  (b₃)
- q=248 (dim E₈)
"""

import numpy as np
import json
import os
import glob
from typing import List, Tuple, Dict


def load_lfunction_zeros(filepath: str) -> Tuple[np.ndarray, str]:
    """Load L-function zeros from LMFDB format."""
    zeros = []
    label = os.path.basename(filepath)

    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()

        # Skip comment lines and find JSON
        lines = content.split('\n')
        json_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('{') or stripped.startswith('['):
                json_start = i
                break

        json_content = '\n'.join(lines[json_start:])
        if json_content.strip().startswith('{') or json_content.strip().startswith('['):
            data = json.loads(json_content)

            if isinstance(data, dict) and 'positive_zeros' in data:
                zeros = [float(z) for z in data['positive_zeros']]
            elif isinstance(data, list):
                zeros = [float(z) for z in data]
        else:
            # Line by line format
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        zeros.append(float(line.split()[0]))
                    except:
                        continue

    except Exception as e:
        print(f"Error loading {filepath}: {e}")

    return np.array(sorted(zeros)), label


def fit_recurrence(gamma: np.ndarray, lags: List[int]) -> Tuple[np.ndarray, float]:
    """Fit linear recurrence."""
    max_lag = max(lags)
    start = max_lag + 5
    end = len(gamma)

    if end - start < 50:
        return None, float('inf')

    n_points = end - start
    n_params = len(lags) + 1

    X = np.zeros((n_points, n_params))
    for i, lag in enumerate(lags):
        X[:, i] = gamma[start - lag:end - lag]
    X[:, -1] = 1.0

    y = gamma[start:end]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    errors = np.abs(y_pred - y)

    return coeffs, float(np.mean(errors))


def analyze_lfunction(zeros: np.ndarray, label: str) -> Dict:
    """Analyze single L-function."""

    # Extract conductor from label (e.g., "1-77-77.76-r0-0-0" -> 77)
    parts = label.replace('.zeros.txt', '').replace('.json', '').split('-')
    try:
        conductor = int(parts[1]) if len(parts) > 1 else 0
    except:
        conductor = 0

    result = {
        'label': label,
        'conductor': conductor,
        'n_zeros': len(zeros),
        'gamma_max': float(zeros[-1]) if len(zeros) > 0 else 0
    }

    if len(zeros) < 50:
        result['status'] = 'insufficient_data'
        return result

    # GIFT analysis
    gift_lags = [5, 8, 13, 27]
    usable_gift = [l for l in gift_lags if l < len(zeros) // 2]

    if len(usable_gift) >= 2:
        coeffs, error = fit_recurrence(zeros, usable_gift)
        if coeffs is not None:
            products = {l: l * coeffs[i] for i, l in enumerate(usable_gift)}
            result['gift'] = {
                'lags': usable_gift,
                'error': error,
                'products': {str(l): float(products[l]) for l in usable_gift}
            }

            # Ratio 8×a₈ / 13×a₁₃
            if 8 in usable_gift and 13 in usable_gift:
                p8 = products[8]
                p13 = products[13]
                if p13 != 0:
                    ratio = p8 / p13
                    result['gift']['ratio_8_13'] = float(ratio)
                    result['gift']['deviation'] = float(abs(ratio - 1))

    # Standard analysis
    std_lags = [1, 2, 3, 4]
    coeffs_std, error_std = fit_recurrence(zeros, std_lags)
    if coeffs_std is not None:
        result['standard'] = {'error': error_std}

        # Compare
        if 'gift' in result:
            gift_better = result['gift']['error'] < error_std
            improvement = (error_std - result['gift']['error']) / error_std * 100
            result['comparison'] = {
                'gift_better': bool(gift_better),
                'improvement_pct': float(improvement)
            }

    result['status'] = 'success'
    return result


def main():
    print("=" * 70)
    print("GIFT Phase 2.10b: MULTI-CONDUCTOR L-FUNCTION ANALYSIS")
    print("=" * 70)

    # Find all L-function files
    search_patterns = [
        'zeta/1-*.zeros.txt',
        'zeta/1-*.json',
        'zeta/L_q*.json',
        'zeta/L_q*.txt',
    ]

    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern))

    # Remove duplicates
    files = list(set(files))

    if not files:
        print("\nNo L-function files found in zeta/ directory.")
        print("Expected filenames like: 1-77-77.76-r0-0-0.zeros.txt")
        return

    print(f"\nFound {len(files)} L-function file(s)")

    # Analyze each
    results = []
    for filepath in sorted(files):
        zeros, label = load_lfunction_zeros(filepath)
        if len(zeros) > 0:
            print(f"\n   {label}: {len(zeros)} zeros")
            result = analyze_lfunction(zeros, label)
            results.append(result)

    if not results:
        print("No valid data found")
        return

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: GIFT STRUCTURE ACROSS CONDUCTORS")
    print("=" * 70)

    print(f"\n{'Conductor':<12} {'N zeros':<10} {'Ratio 8/13':<12} {'|R-1|':<10} {'GIFT wins?':<12}")
    print("-" * 60)

    # Sort by conductor
    results.sort(key=lambda x: x.get('conductor', 0))

    for r in results:
        q = r.get('conductor', '?')
        n = r.get('n_zeros', 0)

        if 'gift' in r and 'ratio_8_13' in r['gift']:
            ratio = r['gift']['ratio_8_13']
            dev = r['gift']['deviation']
            gift_wins = r.get('comparison', {}).get('gift_better', False)
            wins_str = 'YES' if gift_wins else 'NO'
        else:
            ratio = '-'
            dev = '-'
            wins_str = '-'

        print(f"q={q:<10} {n:<10} {ratio if isinstance(ratio, str) else f'{ratio:.4f}':<12} "
              f"{dev if isinstance(dev, str) else f'{dev:.4f}':<10} {wins_str:<12}")

    # GIFT interpretation
    print("\n" + "=" * 70)
    print("GIFT INTERPRETATION")
    print("=" * 70)

    gift_conductors = {5: 'Weyl', 7: 'dim(K₇)', 14: 'dim(G₂)', 21: 'b₂', 27: 'dim(J₃𝕆)', 77: 'b₃', 248: 'dim(E₈)'}

    for r in results:
        q = r.get('conductor', 0)
        if q in gift_conductors:
            meaning = gift_conductors[q]
            if 'gift' in r and 'ratio_8_13' in r['gift']:
                ratio = r['gift']['ratio_8_13']
                dev = r['gift']['deviation']
                print(f"\n   q={q} ({meaning}):")
                print(f"      Ratio = {ratio:.4f}, deviation = {dev*100:.1f}%")

                if dev < 0.1:
                    print(f"      → Fibonacci constraint WELL SATISFIED")
                elif dev < 0.3:
                    print(f"      → Fibonacci constraint PARTIALLY satisfied")
                else:
                    print(f"      → Fibonacci constraint NOT satisfied")

    # Check if q=77 is special
    print("\n" + "=" * 70)
    print("IS q=77 (b₃) SPECIAL?")
    print("=" * 70)

    deviations = [(r.get('conductor', 0), r['gift'].get('deviation', 999))
                  for r in results if 'gift' in r and 'deviation' in r['gift']]

    if deviations:
        deviations.sort(key=lambda x: x[1])
        print(f"\n   Ranking by |ratio - 1|:")
        for i, (q, dev) in enumerate(deviations):
            marker = " ← b₃" if q == 77 else ""
            print(f"      {i+1}. q={q}: {dev*100:.1f}%{marker}")

    # Export
    output = {
        'analysis': 'multi_conductor_comparison',
        'results': results
    }

    with open('phase210b_multi_conductor.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to phase210b_multi_conductor.json")


if __name__ == "__main__":
    main()
