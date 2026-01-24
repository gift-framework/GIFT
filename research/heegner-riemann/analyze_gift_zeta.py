#!/usr/bin/env python3
"""
GIFT-Zeta Correspondence Analysis Tool

Analyzes correspondences between Riemann zeta zeros and GIFT framework constants.
Uses Odlyzko's tables of zeta zeros.

Usage:
    python analyze_gift_zeta.py [zeros_file] [--threshold 0.5]

Author: GIFT Research Team
Date: 2026-01-24
"""

import math
import argparse
from pathlib import Path


# =============================================================================
# GIFT CONSTANTS CATALOG
# =============================================================================

GIFT_CONSTANTS = {
    # Core topological (Tier 1)
    1: ("dim(U₁)", "U(1) dimension"),
    2: ("p₂", "Pontryagin class contribution"),
    3: ("N_gen", "Number of generations"),
    7: ("dim(K₇)", "Joyce manifold dimension"),
    8: ("rank(E₈)", "E₈ Cartan subalgebra dimension"),
    14: ("dim(G₂)", "G₂ holonomy group dimension"),
    21: ("b₂", "Second Betti number of K₇"),
    27: ("dim(J₃(O))", "Exceptional Jordan algebra dimension"),
    77: ("b₃", "Third Betti number of K₇"),
    99: ("H*", "b₂ + b₃ + 1 total cohomology"),

    # Lie algebra dimensions (Tier 2)
    78: ("dim(E₆)", "E₆ Lie algebra dimension"),
    133: ("dim(E₇)", "E₇ Lie algebra dimension"),
    248: ("dim(E₈)", "E₈ Lie algebra dimension"),
    496: ("dim(E₈×E₈)", "Heterotic gauge group dimension"),

    # Root systems (Tier 2)
    72: ("|Roots(E₆)|", "E₆ root count"),
    126: ("|Roots(E₇)|", "E₇ root count"),
    240: ("|Roots(E₈)|", "E₈ root count"),
    480: ("|Roots(E₈×E₈)|", "Heterotic root count"),

    # Heegner numbers (Tier 1-2)
    11: ("D_bulk", "Bulk dimension = Heegner"),
    19: ("prime_8", "8th prime = Heegner"),
    43: ("visible_dim", "Visible dimension = Heegner"),
    67: ("b₃ - 2×Weyl", "Heegner number"),
    163: ("|Roots(E₈)| - b₃", "Maximum Heegner number"),

    # Derived constants (Tier 3)
    13: ("b₂ - rank(E₈)", "Betti minus rank"),
    61: ("b₃ - dim(G₂) - p₂", "Cohomological difference"),
    91: ("b₂ + b₃ - dim(K₇)", "Betti sum minus dimension"),
    98: ("H* - 1", "Cohomology minus 1"),
    100: ("H* + 1", "Cohomology plus 1"),
    112: ("dim(E₇) - b₂", "E₇ minus second Betti"),
    134: ("dim(E₇) + 1", "E₇ plus 1"),
    141: ("dim(E₇) + rank(E₈)", "E₇ plus rank"),
    155: ("dim(E₇) + b₂ + 1", "E₇ plus Betti plus 1"),
    162: ("Heegner_max - 1", "163 minus 1"),
    164: ("Heegner_max + 1", "163 plus 1"),
    239: ("|Roots(E₈)| - 1", "240 minus 1"),
    241: ("|Roots(E₈)| + 1", "240 plus 1"),
    247: ("dim(E₈) - 1", "248 minus 1"),
    249: ("dim(E₈) + 1", "248 plus 1"),
    255: ("dim(E₈) + dim(K₇)", "E₈ plus K₇"),
    256: ("2⁸", "Power of 2"),
    262: ("dim(E₈) + dim(G₂)", "E₈ plus G₂"),
}

# Add multiples of dim(K₇) = 7
for n in range(3, 150):
    target = 7 * n
    if target not in GIFT_CONSTANTS:
        GIFT_CONSTANTS[target] = (f"{n} × dim(K₇)", f"Multiple of 7")


def load_zeros(filepath: str) -> list:
    """Load zeta zeros from Odlyzko text file."""
    zeros = []
    with open(filepath, 'r') as f:
        for line in f:
            val = line.strip()
            if val:
                try:
                    zeros.append(float(val))
                except ValueError:
                    # Skip header lines or malformed entries
                    continue
    return zeros


def find_best_match(zeros: list, target: float) -> tuple:
    """Find the zero closest to target value."""
    best_idx = None
    best_diff = float('inf')

    for i, gamma in enumerate(zeros):
        diff = abs(gamma - target)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
        # Early exit since zeros are sorted
        if gamma > target * 1.1:
            break

    if best_idx is not None:
        gamma = zeros[best_idx]
        precision = abs(gamma - target) / target * 100
        return best_idx + 1, gamma, precision, gamma - target
    return None, None, None, None


def find_all_matches(zeros: list, threshold_pct: float = 0.5) -> list:
    """Find all GIFT constants matched by zeros within threshold."""
    matches = []

    for target, (short_name, _) in sorted(GIFT_CONSTANTS.items()):
        if target > zeros[-1] * 1.1:  # Skip targets beyond our data
            continue

        idx, gamma, precision, diff = find_best_match(zeros, target)

        if idx is not None and precision < threshold_pct:
            matches.append({
                'index': idx,
                'gamma': gamma,
                'target': target,
                'name': short_name,
                'precision': precision,
                'diff': diff
            })

    return matches


def print_report(zeros: list, matches: list, threshold: float):
    """Print formatted analysis report."""
    print("=" * 75)
    print("GIFT-ZETA CORRESPONDENCE ANALYSIS")
    print("=" * 75)
    print(f"\nData: {len(zeros)} zeros (γ₁ = {zeros[0]:.6f} to γ_{len(zeros)} = {zeros[-1]:.6f})")
    print(f"Threshold: {threshold}%")
    print(f"Matches found: {len(matches)}/{len(GIFT_CONSTANTS)} targets")

    # Sort by target value
    print(f"\n{'Index':>7} | {'γₙ':>14} | {'Target':>6} | {'GIFT Constant':25} | {'Precision':>10}")
    print("-" * 75)

    for m in sorted(matches, key=lambda x: x['target']):
        print(f"γ_{m['index']:<5} | {m['gamma']:14.9f} | {m['target']:6} | {m['name']:25} | {m['precision']:9.5f}%")

    # Ultra-precise matches
    ultra = [m for m in matches if m['precision'] < 0.05]
    if ultra:
        print(f"\n{'=' * 75}")
        print(f"ULTRA-PRECISE MATCHES (< 0.05%): {len(ultra)}")
        print("=" * 75)

        for m in sorted(ultra, key=lambda x: x['precision']):
            print(f"★ γ_{m['index']} = {m['gamma']:.9f} ≈ {m['target']} ({m['name']})")
            print(f"   Δ = {m['diff']:+.9f} | Precision: {m['precision']:.6f}%\n")

    # Key GIFT constants summary
    print("=" * 75)
    print("KEY GIFT CONSTANTS STATUS")
    print("=" * 75)

    key_targets = [14, 21, 77, 99, 133, 163, 240, 248, 496]
    for t in key_targets:
        name = GIFT_CONSTANTS.get(t, ("?", ""))[0]
        idx, gamma, prec, diff = find_best_match(zeros, t)
        if idx:
            status = "✓ MATCH" if prec < 0.5 else "~ close" if prec < 1 else "✗ no match"
            print(f"  {t:3} ({name:20}) → γ_{idx:5} = {gamma:.6f} ({prec:.4f}%) {status}")


def main():
    parser = argparse.ArgumentParser(description='Analyze GIFT-zeta correspondences')
    parser.add_argument('zeros_file', nargs='?', default='zeros1.txt',
                        help='Path to Odlyzko zeros file')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Precision threshold in percent (default: 0.5)')
    args = parser.parse_args()

    # Find zeros file
    zeros_path = Path(args.zeros_file)
    if not zeros_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        zeros_path = script_dir / args.zeros_file

    if not zeros_path.exists():
        print(f"Error: Cannot find zeros file: {args.zeros_file}")
        return 1

    print(f"Loading zeros from {zeros_path}...")
    zeros = load_zeros(zeros_path)

    if not zeros:
        print("Error: No zeros loaded from file")
        return 1

    matches = find_all_matches(zeros, args.threshold)
    print_report(zeros, matches, args.threshold)

    return 0


if __name__ == '__main__':
    exit(main())
