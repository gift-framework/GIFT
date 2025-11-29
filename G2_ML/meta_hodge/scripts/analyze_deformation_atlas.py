#!/usr/bin/env python
"""Analyze K7_GIFT deformation atlas results.

This script loads results from run_deformation_atlas.py and produces
analysis plots and summary statistics.

Key insight: The effective moduli are:
  u = sigma * s  (controls stability - must be <= 1)
  v = sigma / s  (shape parameter - less constrained)
  alpha          (asymmetry)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_results(results_dir: Path) -> List[Dict]:
    """Load results from JSON file."""
    json_path = results_dir / "deformation_results.json"
    with json_path.open() as f:
        return json.load(f)


def add_effective_moduli(results: List[Dict]) -> List[Dict]:
    """Add effective moduli u = sigma*s and v = sigma/s to results."""
    for r in results:
        r["u"] = r["sigma"] * r["s"]
        r["v"] = r["sigma"] / r["s"] if r["s"] != 0 else float("inf")
    return results


def analyze_stability_region(results: List[Dict]) -> Dict:
    """Analyze the stability region in parameter space."""
    stable = [r for r in results if r["stable"]]
    unstable = [r for r in results if not r["stable"]]

    # Extract parameter ranges
    sigmas = sorted(set(r["sigma"] for r in results))
    ss = sorted(set(r["s"] for r in results))
    alphas = sorted(set(r["alpha"] for r in results))

    # Create stability map
    stability_map = {}
    for r in results:
        key = (r["sigma"], r["s"], r["alpha"])
        stability_map[key] = r["stable"]

    # Find contiguous stable regions
    stable_sigmas = sorted(set(r["sigma"] for r in stable)) if stable else []
    stable_ss = sorted(set(r["s"] for r in stable)) if stable else []
    stable_alphas = sorted(set(r["alpha"] for r in stable)) if stable else []

    # Check symmetries
    sigma_s_symmetry = []
    for r in stable:
        sigma, s, alpha = r["sigma"], r["s"], r["alpha"]
        swapped_key = (s, sigma, alpha)
        if swapped_key in stability_map and stability_map[swapped_key]:
            sigma_s_symmetry.append((sigma, s, alpha))

    alpha_sign_symmetry = []
    for r in stable:
        sigma, s, alpha = r["sigma"], r["s"], r["alpha"]
        flipped_key = (sigma, s, -alpha)
        if flipped_key in stability_map and stability_map[flipped_key]:
            alpha_sign_symmetry.append((sigma, s, alpha))

    return {
        "total_points": len(results),
        "stable_count": len(stable),
        "stable_fraction": len(stable) / len(results) if results else 0,
        "stable_sigmas": stable_sigmas,
        "stable_ss": stable_ss,
        "stable_alphas": stable_alphas,
        "sigma_s_symmetric_pairs": len(sigma_s_symmetry) // 2,
        "alpha_sign_symmetric_pairs": len(alpha_sign_symmetry) // 2,
        "all_sigmas": sigmas,
        "all_ss": ss,
        "all_alphas": alphas,
    }


def analyze_effective_moduli(results: List[Dict]) -> Dict:
    """Analyze stability in (u, v, alpha) space where u=sigma*s, v=sigma/s."""
    results = add_effective_moduli(results)
    stable = [r for r in results if r["stable"]]

    if not stable:
        return {"stable_u": [], "stable_v": [], "u_range": (0, 0), "v_range": (0, 0)}

    stable_u = [r["u"] for r in stable]
    stable_v = [r["v"] for r in stable]
    all_u = [r["u"] for r in results]
    all_v = [r["v"] for r in results]

    # Stability rate by u value
    u_values = sorted(set(all_u))
    u_stability = {}
    for u in u_values:
        pts_at_u = [r for r in results if abs(r["u"] - u) < 0.01]
        stable_at_u = [r for r in pts_at_u if r["stable"]]
        u_stability[u] = len(stable_at_u) / len(pts_at_u) if pts_at_u else 0

    # Stability rate by v value
    v_values = sorted(set(all_v))
    v_stability = {}
    for v in v_values:
        pts_at_v = [r for r in results if abs(r["v"] - v) < 0.01]
        stable_at_v = [r for r in pts_at_v if r["stable"]]
        v_stability[v] = len(stable_at_v) / len(pts_at_v) if pts_at_v else 0

    return {
        "stable_u": stable_u,
        "stable_v": stable_v,
        "u_range": (min(stable_u), max(stable_u)),
        "v_range": (min(stable_v), max(stable_v)),
        "u_mean": np.mean(stable_u),
        "u_std": np.std(stable_u),
        "v_mean": np.mean(stable_v),
        "v_std": np.std(stable_v),
        "u_stability": u_stability,
        "v_stability": v_stability,
    }


def print_analysis(results: List[Dict], analysis: Dict):
    """Print analysis summary."""
    print("=" * 60)
    print("K7_GIFT Deformation Atlas Analysis")
    print("=" * 60)
    print()
    print(f"Total points: {analysis['total_points']}")
    print(f"Stable points: {analysis['stable_count']} ({100*analysis['stable_fraction']:.1f}%)")
    print()

    print("Parameter ranges:")
    print(f"  sigma: {analysis['all_sigmas']}")
    print(f"  s:     {analysis['all_ss']}")
    print(f"  alpha: {analysis['all_alphas']}")
    print()

    print("Stable region bounds:")
    print(f"  sigma: [{min(analysis['stable_sigmas'])}, {max(analysis['stable_sigmas'])}]" if analysis['stable_sigmas'] else "  sigma: none")
    print(f"  s:     [{min(analysis['stable_ss'])}, {max(analysis['stable_ss'])}]" if analysis['stable_ss'] else "  s:     none")
    print(f"  alpha: [{min(analysis['stable_alphas'])}, {max(analysis['stable_alphas'])}]" if analysis['stable_alphas'] else "  alpha: none")
    print()

    print("Symmetry analysis:")
    print(f"  sigma <-> s symmetric pairs: {analysis['sigma_s_symmetric_pairs']}")
    print(f"  alpha sign-flip symmetric pairs: {analysis['alpha_sign_symmetric_pairs']}")
    print()

    # Print stable points table
    stable = [r for r in results if r["stable"]]
    if stable:
        print("Stable points:")
        print("  sigma    s     alpha    kappa_T    phi_norm    sigma*s")
        print("  " + "-" * 55)
        for r in sorted(stable, key=lambda x: (x["sigma"], x["s"], x["alpha"])):
            product = r["sigma"] * r["s"]
            print(f"  {r['sigma']:.2f}    {r['s']:.2f}   {r['alpha']:+.2f}     {r['kappa_T']:.4f}    {r['phi_norm_mean']:.2f}       {product:.2f}")

    # Analyze sigma*s product for stable points
    if stable:
        products = [r["sigma"] * r["s"] for r in stable]
        print()
        print(f"sigma*s product for stable points:")
        print(f"  range: [{min(products):.2f}, {max(products):.2f}]")
        print(f"  mean:  {np.mean(products):.2f}")
        print(f"  std:   {np.std(products):.2f}")


def print_effective_moduli_analysis(results: List[Dict], eff_analysis: Dict):
    """Print effective moduli (u, v) analysis."""
    print()
    print("=" * 60)
    print("Effective Moduli Analysis: u = sigma*s, v = sigma/s")
    print("=" * 60)
    print()

    print("Stable region in (u, v) space:")
    print(f"  u range: [{eff_analysis['u_range'][0]:.2f}, {eff_analysis['u_range'][1]:.2f}]")
    print(f"  u mean:  {eff_analysis['u_mean']:.2f} +/- {eff_analysis['u_std']:.2f}")
    print(f"  v range: [{eff_analysis['v_range'][0]:.2f}, {eff_analysis['v_range'][1]:.2f}]")
    print(f"  v mean:  {eff_analysis['v_mean']:.2f} +/- {eff_analysis['v_std']:.2f}")
    print()

    print("Stability rate by u = sigma*s:")
    for u, rate in sorted(eff_analysis['u_stability'].items()):
        bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
        print(f"  u={u:.2f}: [{bar}] {100*rate:.0f}%")
    print()

    print("Stability rate by v = sigma/s:")
    for v, rate in sorted(eff_analysis['v_stability'].items()):
        bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
        print(f"  v={v:.2f}: [{bar}] {100*rate:.0f}%")
    print()

    # Interpretation
    print("Interpretation:")
    u_rates = list(eff_analysis['u_stability'].values())
    v_rates = list(eff_analysis['v_stability'].values())
    u_var = np.var(u_rates) if u_rates else 0
    v_var = np.var(v_rates) if v_rates else 0

    if u_var > v_var * 2:
        print("  -> u = sigma*s is the PRIMARY stability modulus")
        print("  -> v = sigma/s has little effect (shape invariance)")
    elif v_var > u_var * 2:
        print("  -> v = sigma/s is the PRIMARY stability modulus")
        print("  -> u = sigma*s has little effect")
    else:
        print("  -> Both u and v affect stability comparably")

    # Find stability threshold
    stable_u = eff_analysis['stable_u']
    if stable_u:
        u_max = max(stable_u)
        print(f"  -> Stability requires: u <= {u_max:.2f}")


def print_uv_map(results: List[Dict]):
    """Print ASCII map of stability in (u, v) space."""
    results = add_effective_moduli(results)

    # Get unique u and v values
    u_vals = sorted(set(r["u"] for r in results))
    v_vals = sorted(set(r["v"] for r in results), reverse=True)  # High v at top

    print()
    print("Stability map in (u, v) space:")
    print("  (aggregated over alpha; S=stable, .=unstable, o=mixed)")
    print()

    # Header
    header = "  v\\u  "
    for u in u_vals:
        header += f" {u:.2f}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for v in v_vals:
        row = f"  {v:.2f} |"
        for u in u_vals:
            # Find all points at this (u, v)
            pts = [r for r in results if abs(r["u"] - u) < 0.01 and abs(r["v"] - v) < 0.01]
            if not pts:
                row += "    "
            else:
                stable_count = sum(1 for p in pts if p["stable"])
                if stable_count == len(pts):
                    row += "  S "
                elif stable_count == 0:
                    row += "  . "
                else:
                    row += "  o "
        print(row)
    print()
    print("  Legend: S=all stable, o=mixed, .=all unstable")


def print_u_alpha_slice(results: List[Dict]):
    """Print 2D slice in (u, alpha) space, aggregated over v."""
    results = add_effective_moduli(results)

    # Get unique u and alpha values
    u_vals = sorted(set(r["u"] for r in results))
    alpha_vals = sorted(set(r["alpha"] for r in results), reverse=True)  # High alpha at top

    print()
    print("=" * 60)
    print("Stability in (u, alpha) space (aggregated over v)")
    print("=" * 60)
    print("  This is the PRIMARY stability diagram")
    print("  u = sigma * s controls geometric stability")
    print("  alpha controls left-right asymmetry")
    print()

    # Header
    header = "  alpha\\u "
    for u in u_vals:
        header += f" {u:.2f}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for alpha in alpha_vals:
        row = f"  {alpha:+.2f}  |"
        for u in u_vals:
            # Find all points at this (u, alpha) - any v
            pts = [r for r in results if abs(r["u"] - u) < 0.01 and abs(r["alpha"] - alpha) < 0.01]
            if not pts:
                row += "    "
            else:
                stable_count = sum(1 for p in pts if p["stable"])
                total = len(pts)
                if stable_count == total:
                    row += "  S "
                elif stable_count == 0:
                    row += "  . "
                else:
                    # Show ratio
                    row += f" {stable_count}/{total} "
        print(row)

    print()
    print("  Legend: S=all stable, .=all unstable, n/m=n stable out of m")

    # Compute stability rate by u (marginalized over v and alpha)
    print()
    print("Stability rate by u (critical modulus):")
    u_rates = {}
    for u in u_vals:
        pts = [r for r in results if abs(r["u"] - u) < 0.01]
        stable = sum(1 for p in pts if p["stable"])
        rate = stable / len(pts) if pts else 0
        u_rates[u] = rate

    # Find transition point
    sorted_u = sorted(u_rates.keys())
    transition_u = None
    for i, u in enumerate(sorted_u[:-1]):
        if u_rates[u] > 0 and u_rates[sorted_u[i+1]] == 0:
            transition_u = (u + sorted_u[i+1]) / 2

    for u in sorted_u:
        rate = u_rates[u]
        bar = "#" * int(rate * 20) + "." * (20 - int(rate * 20))
        marker = " <-- threshold" if transition_u and abs(u - transition_u) < 0.1 else ""
        print(f"  u={u:.2f}: [{bar}] {100*rate:5.1f}%{marker}")

    if transition_u:
        print()
        print(f"  Critical threshold: u_max ~ {transition_u:.2f}")
        print(f"  Stability requires: sigma * s <= {transition_u:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze deformation atlas results")
    parser.add_argument(
        "results_dir", type=Path, nargs="?",
        help="Directory containing deformation_results.json"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use the latest results directory"
    )

    args = parser.parse_args()

    # Find results directory
    if args.latest or args.results_dir is None:
        base_dir = Path(__file__).parent.parent / "artifacts" / "deformation_atlas"
        if base_dir.exists():
            dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
            if dirs:
                args.results_dir = dirs[-1]
                print(f"Using latest results: {args.results_dir}")
            else:
                print(f"No results found in {base_dir}")
                return 1
        else:
            print(f"Base directory not found: {base_dir}")
            return 1

    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        return 1

    results = load_results(args.results_dir)
    analysis = analyze_stability_region(results)
    print_analysis(results, analysis)

    # Effective moduli analysis
    eff_analysis = analyze_effective_moduli(results)
    print_effective_moduli_analysis(results, eff_analysis)

    # ASCII stability map in (u, v)
    print_uv_map(results)

    # Primary diagram: (u, alpha) slice
    print_u_alpha_slice(results)

    return 0


if __name__ == "__main__":
    exit(main())
