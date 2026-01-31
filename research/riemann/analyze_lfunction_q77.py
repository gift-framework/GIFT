#!/usr/bin/env python3
"""
GIFT Phase 2.10: L-function Analysis for q=77
==============================================

Analyser les zéros de la fonction L de Dirichlet pour le conducteur q=77.

Pourquoi q=77 ?
- 77 = b₃ (third Betti number of K₇ in GIFT)
- 77 = 7 × 11 = dim(K₇) × L₅ (Lucas)
- If GIFT structure is universal, q=77 should show special properties

Data source: LMFDB (https://www.lmfdb.org/L/)
Download: https://www.lmfdb.org/L/download_zeros/1/77.X/1/

Usage:
1. Download zeros from LMFDB
2. Place in zeta/ directory as 'L_q77_zeros.txt'
3. Run this script
"""

import numpy as np
from typing import List, Tuple, Dict
import json
import os


def load_lfunction_zeros(filepath: str) -> np.ndarray:
    """
    Load L-function zeros from various formats.

    Supports:
    - LMFDB format: one zero per line
    - Comma-separated
    - Tab-separated
    """

    zeros = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue

                # Try different formats
                for sep in [None, ',', '\t', ' ']:
                    try:
                        parts = line.split(sep) if sep else [line]
                        for part in parts:
                            part = part.strip()
                            if part:
                                zeros.append(float(part))
                        break
                    except ValueError:
                        continue

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])

    return np.array(sorted(zeros))


def fit_recurrence(gamma: np.ndarray, lags: List[int],
                   start: int = None, end: int = None) -> Tuple[np.ndarray, float]:
    """Fit linear recurrence."""
    max_lag = max(lags)
    if start is None:
        start = max_lag
    if end is None:
        end = len(gamma)

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

    return coeffs, np.mean(errors)


def analyze_lfunction(zeros: np.ndarray, name: str = "L-function") -> Dict:
    """
    Analyze L-function zeros with GIFT methodology.
    """

    results = {
        'name': name,
        'n_zeros': len(zeros),
        'gamma_range': [float(zeros[0]), float(zeros[-1])] if len(zeros) > 0 else None
    }

    if len(zeros) < 100:
        results['status'] = 'insufficient_data'
        results['message'] = f"Need at least 100 zeros, got {len(zeros)}"
        return results

    # GIFT lags
    gift_lags = [5, 8, 13, 27]

    # Only use lags we have data for
    max_usable_lag = min(len(zeros) // 3, 27)
    usable_lags = [lag for lag in gift_lags if lag <= max_usable_lag]

    if len(usable_lags) < 2:
        results['status'] = 'insufficient_data'
        results['message'] = f"Can only use lags {usable_lags}"
        return results

    # Fit with GIFT lags
    try:
        max_lag = max(usable_lags)
        start = max_lag + 10
        end = len(zeros)

        coeffs_gift, error_gift = fit_recurrence(zeros, usable_lags, start, end)

        results['gift_analysis'] = {
            'lags': usable_lags,
            'coefficients': {f'a_{lag}': float(coeffs_gift[i]) for i, lag in enumerate(usable_lags)},
            'c': float(coeffs_gift[-1]),
            'error': float(error_gift)
        }

        # Products
        products = {f'{lag}_times_a_{lag}': float(lag * coeffs_gift[i])
                   for i, lag in enumerate(usable_lags)}
        results['gift_analysis']['products'] = products

        # Check 8×a₈ = 13×a₁₃ if available
        if 8 in usable_lags and 13 in usable_lags:
            idx_8 = usable_lags.index(8)
            idx_13 = usable_lags.index(13)
            prod_8 = 8 * coeffs_gift[idx_8]
            prod_13 = 13 * coeffs_gift[idx_13]

            if prod_13 != 0:
                ratio = prod_8 / prod_13
                results['gift_analysis']['ratio_8_13'] = float(ratio)
                results['gift_analysis']['deviation_from_1'] = float(abs(ratio - 1))

        results['status'] = 'success'

    except Exception as e:
        results['status'] = 'error'
        results['message'] = str(e)

    # Compare with standard lags [1,2,3,4]
    standard_lags = [1, 2, 3, 4]

    try:
        coeffs_std, error_std = fit_recurrence(zeros, standard_lags, 10, len(zeros))

        results['standard_analysis'] = {
            'lags': standard_lags,
            'coefficients': {f'a_{lag}': float(coeffs_std[i]) for i, lag in enumerate(standard_lags)},
            'c': float(coeffs_std[-1]),
            'error': float(error_std)
        }

        # Compare errors
        if 'gift_analysis' in results and 'error' in results['gift_analysis']:
            gift_err = results['gift_analysis']['error']
            std_err = error_std
            results['comparison'] = {
                'gift_error': float(gift_err),
                'standard_error': float(std_err),
                'gift_better': bool(gift_err < std_err),
                'improvement_pct': float((std_err - gift_err) / std_err * 100) if std_err != 0 else 0
            }

    except Exception as e:
        results['standard_analysis'] = {'status': 'error', 'message': str(e)}

    return results


def main():
    print("=" * 70)
    print("GIFT Phase 2.10: L-FUNCTION ANALYSIS (q=77)")
    print("=" * 70)

    # Potential file locations
    potential_files = [
        'zeta/L_q77_zeros.txt',
        'zeta/dirichlet_77.txt',
        'zeta/L-function-1-77-1.1-c1-0-0.txt',  # LMFDB naming
        'L_q77_zeros.txt',
    ]

    zeros = None
    loaded_file = None

    for filepath in potential_files:
        if os.path.exists(filepath):
            zeros = load_lfunction_zeros(filepath)
            if len(zeros) > 0:
                loaded_file = filepath
                break

    if zeros is None or len(zeros) == 0:
        print("\n" + "=" * 70)
        print("NO L-FUNCTION DATA FOUND")
        print("=" * 70)

        print("""
To run this analysis, download L-function zeros from LMFDB:

1. Go to: https://www.lmfdb.org/L/

2. Search for conductor q=77

3. Download zeros for a primitive character

4. Save as one of:
   - zeta/L_q77_zeros.txt
   - zeta/dirichlet_77.txt

Expected format: one zero per line (imaginary parts)

Example LMFDB URLs:
   https://www.lmfdb.org/L/1/77/77.76/r0/0/0
   (Click "Download zeros")

GIFT-pertinent conductors to try:
   q = 5   (Weyl orbits)
   q = 7   (dim K₇)
   q = 8   (rank E₈)
   q = 14  (dim G₂)
   q = 21  (b₂)
   q = 27  (dim J₃(O))
   q = 77  (b₃) ← THIS ONE
   q = 248 (dim E₈)
""")
        return

    print(f"\n   Loaded {len(zeros):,} zeros from {loaded_file}")

    # Analyze
    results = analyze_lfunction(zeros, name=f"L-function q=77")

    print("\n" + "-" * 70)
    print("ANALYSIS RESULTS")
    print("-" * 70)

    if results['status'] != 'success':
        print(f"\n   Status: {results['status']}")
        print(f"   Message: {results.get('message', 'N/A')}")
        return

    # GIFT analysis
    if 'gift_analysis' in results:
        ga = results['gift_analysis']
        print(f"\n   GIFT Lags: {ga['lags']}")
        print(f"   Coefficients:")
        for lag in ga['lags']:
            print(f"      a_{lag} = {ga['coefficients'][f'a_{lag}']:.6f}")
        print(f"      c = {ga['c']:.4f}")
        print(f"   Error: {ga['error']:.6f}")

        print(f"\n   Products lag × a:")
        for lag in ga['lags']:
            key = f'{lag}_times_a_{lag}'
            if key in ga['products']:
                print(f"      {lag} × a_{lag} = {ga['products'][key]:.4f}")

        if 'ratio_8_13' in ga:
            print(f"\n   Ratio (8×a₈)/(13×a₁₃) = {ga['ratio_8_13']:.4f}")
            print(f"   Deviation from 1: {ga['deviation_from_1']:.4f} ({ga['deviation_from_1']*100:.1f}%)")

    # Comparison
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n   COMPARISON GIFT vs Standard [1,2,3,4]:")
        print(f"      GIFT error:     {comp['gift_error']:.6f}")
        print(f"      Standard error: {comp['standard_error']:.6f}")
        print(f"      GIFT better?    {'YES' if comp['gift_better'] else 'NO'}")
        if comp['gift_better']:
            print(f"      Improvement:    {comp['improvement_pct']:.1f}%")

    # Save results
    with open('phase210_lfunction_q77.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   Results saved to phase210_lfunction_q77.json")

    # ============================================================
    # UNIVERSALITY TEST
    # ============================================================

    print("\n" + "=" * 70)
    print("UNIVERSALITY TEST")
    print("=" * 70)

    # The key test: does 8×a₈ ≈ 13×a₁₃ hold for L-functions?

    if 'gift_analysis' in results and 'ratio_8_13' in results['gift_analysis']:
        ratio = results['gift_analysis']['ratio_8_13']
        dev = results['gift_analysis']['deviation_from_1']

        print(f"\n   For ζ(s):     8×β₈ = 13×β₁₃ = 36 (h_G₂²)")
        print(f"   For L(s,χ₇₇): 8×a₈/13×a₁₃ = {ratio:.4f}")

        if dev < 0.1:
            print(f"\n   VERDICT: Fibonacci constraint PRESERVED (dev < 10%)")
            print(f"            → Structure may be UNIVERSAL")
        elif dev < 0.3:
            print(f"\n   VERDICT: Fibonacci constraint PARTIAL (10% < dev < 30%)")
            print(f"            → Needs more data to confirm")
        else:
            print(f"\n   VERDICT: Fibonacci constraint NOT satisfied (dev > 30%)")
            print(f"            → Structure may be SPECIFIC to ζ(s)")
    else:
        print("\n   Cannot test: need both lags 8 and 13 available")


if __name__ == "__main__":
    main()
