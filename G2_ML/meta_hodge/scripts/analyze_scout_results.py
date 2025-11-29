#!/usr/bin/env python
"""Analyze scout campaign results to identify stable regions and patterns."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent


def load_latest_results() -> tuple:
    """Load the most recent scout campaign results."""
    campaigns_dir = meta_hodge_dir / "artifacts" / "scout_campaigns"
    if not campaigns_dir.exists():
        return None, None

    dirs = sorted([d for d in campaigns_dir.iterdir() if d.is_dir()])
    if not dirs:
        return None, None

    latest = dirs[-1]
    probes_path = latest / "probes.json"
    summary_path = latest / "summary.json"

    probes = []
    summary = {}

    if probes_path.exists():
        with open(probes_path) as f:
            probes = json.load(f)

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return probes, summary


def analyze_stable_points(probes: List[Dict]) -> Dict:
    """Analyze stable points to find patterns."""
    stable = [p for p in probes if p["stable"]]
    unstable = [p for p in probes if not p["stable"]]

    if not stable:
        return {"error": "No stable points found"}

    # Extract features
    stable_u = np.array([p["u"] for p in stable])
    stable_alpha = np.array([p["alpha"] for p in stable])
    stable_v = np.array([p["v"] for p in stable])

    # Analyze u vs |alpha| relationship
    abs_alpha = np.abs(stable_alpha)

    # Check if they follow the ridge formula: u ~ 1 - b*|alpha|
    if len(stable) > 3:
        A = np.vstack([np.ones_like(abs_alpha), abs_alpha]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, stable_u, rcond=None)
        c_fit, neg_b_fit = coeffs
        b_fit = -neg_b_fit

        # R^2
        u_pred = c_fit - b_fit * abs_alpha
        ss_res = np.sum((stable_u - u_pred) ** 2)
        ss_tot = np.sum((stable_u - stable_u.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        c_fit, b_fit, r2 = None, None, None

    # Find clusters by alpha sign
    pos_alpha = [p for p in stable if p["alpha"] > 0.1]
    neg_alpha = [p for p in stable if p["alpha"] < -0.1]
    near_zero = [p for p in stable if abs(p["alpha"]) <= 0.1]

    # Find anomalies (stable points far from expected ridge)
    expected_ridge = lambda alpha: 1.0 - 1.13 * abs(alpha)
    anomalies = []
    for p in stable:
        expected_u = expected_ridge(p["alpha"])
        deviation = p["u"] - expected_u
        if abs(deviation) > 0.2:
            anomalies.append({
                "point": (p["sigma"], p["s"], p["alpha"]),
                "u": p["u"],
                "expected_u": expected_u,
                "deviation": deviation,
            })

    return {
        "n_stable": len(stable),
        "n_unstable": len(unstable),
        "stability_rate": len(stable) / len(probes),
        "u_range": (float(stable_u.min()), float(stable_u.max())),
        "alpha_range": (float(stable_alpha.min()), float(stable_alpha.max())),
        "v_range": (float(stable_v.min()), float(stable_v.max())),
        "ridge_fit": {
            "formula": f"u = {c_fit:.3f} - {b_fit:.3f}*|alpha|" if c_fit else None,
            "c": float(c_fit) if c_fit else None,
            "b": float(b_fit) if b_fit else None,
            "r2": float(r2) if r2 else None,
        },
        "clusters": {
            "positive_alpha": len(pos_alpha),
            "negative_alpha": len(neg_alpha),
            "near_zero": len(near_zero),
        },
        "anomalies": anomalies,
    }


def print_analysis(probes: List[Dict], summary: Dict):
    """Print detailed analysis."""
    print("=" * 70)
    print("K7 MODULI SPACE SCOUT - ANALYSIS REPORT")
    print("=" * 70)
    print()

    analysis = analyze_stable_points(probes)

    print(f"Total probes: {len(probes)}")
    print(f"Stable: {analysis['n_stable']} ({100*analysis['stability_rate']:.1f}%)")
    print()

    print("STABLE REGION BOUNDS:")
    print(f"  u (sigma*s): [{analysis['u_range'][0]:.3f}, {analysis['u_range'][1]:.3f}]")
    print(f"  alpha:       [{analysis['alpha_range'][0]:.3f}, {analysis['alpha_range'][1]:.3f}]")
    print(f"  v (sigma/s): [{analysis['v_range'][0]:.3f}, {analysis['v_range'][1]:.3f}]")
    print()

    if analysis["ridge_fit"]["formula"]:
        print("RIDGE FIT (from stable points):")
        print(f"  {analysis['ridge_fit']['formula']}")
        print(f"  R^2 = {analysis['ridge_fit']['r2']:.4f}")
        print()
        print("  Compare to original: u = 1.00 - 1.13*|alpha|")
        print()

    print("CLUSTER ANALYSIS:")
    print(f"  Positive alpha (>0.1):  {analysis['clusters']['positive_alpha']} points")
    print(f"  Negative alpha (<-0.1): {analysis['clusters']['negative_alpha']} points")
    print(f"  Near zero (|a|<=0.1):   {analysis['clusters']['near_zero']} points")
    print()

    if analysis["anomalies"]:
        print("ANOMALIES (far from expected ridge):")
        for a in analysis["anomalies"][:10]:
            p = a["point"]
            print(f"  ({p[0]:.2f}, {p[1]:.2f}, {p[2]:+.2f}): u={a['u']:.3f} vs expected {a['expected_u']:.3f} (dev={a['deviation']:+.3f})")
        print()

    # Print all stable points grouped by alpha sign
    stable = [p for p in probes if p["stable"]]

    print("STABLE POINTS BY REGION:")
    print()

    print("  HIGH POSITIVE ALPHA (alpha > 0.3):")
    high_pos = [p for p in stable if p["alpha"] > 0.3]
    if high_pos:
        for p in sorted(high_pos, key=lambda x: -x["alpha"]):
            print(f"    ({p['sigma']:.2f}, {p['s']:.2f}, {p['alpha']:+.2f}) u={p['u']:.3f} v={p['v']:.3f}")
    else:
        print("    (none)")

    print()
    print("  HIGH NEGATIVE ALPHA (alpha < -0.3):")
    high_neg = [p for p in stable if p["alpha"] < -0.3]
    if high_neg:
        for p in sorted(high_neg, key=lambda x: x["alpha"]):
            print(f"    ({p['sigma']:.2f}, {p['s']:.2f}, {p['alpha']:+.2f}) u={p['u']:.3f} v={p['v']:.3f}")
    else:
        print("    (none)")

    print()
    print("  MODERATE ALPHA (-0.3 <= alpha <= 0.3):")
    moderate = [p for p in stable if -0.3 <= p["alpha"] <= 0.3]
    if moderate:
        for p in sorted(moderate, key=lambda x: -x["u"])[:15]:
            print(f"    ({p['sigma']:.2f}, {p['s']:.2f}, {p['alpha']:+.2f}) u={p['u']:.3f} v={p['v']:.3f}")
        if len(moderate) > 15:
            print(f"    ... and {len(moderate) - 15} more")
    else:
        print("    (none)")

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()

    # Check for discoveries
    extreme_alpha = [p for p in stable if abs(p["alpha"]) > 0.4]
    low_u = [p for p in stable if p["u"] < 0.6]
    high_v = [p for p in stable if p["v"] > 2.0 or p["v"] < 0.5]

    if extreme_alpha:
        print("- FOUND stable points at EXTREME ALPHA (|alpha| > 0.4)!")
        print(f"  This extends the known stability region.")
        print()

    if low_u:
        print("- FOUND stable points at LOW U (u < 0.6)!")
        print(f"  Original ridge predicted instability here.")
        print()

    if high_v:
        print("- FOUND stable points at EXTREME V (v > 2 or v < 0.5)!")
        print(f"  Strong sigma/s asymmetry is tolerated.")
        print()

    # Sigma-s symmetry check
    print("SIGMA <-> S SYMMETRY CHECK:")
    for p in stable[:5]:
        # Look for mirror point
        mirror_found = False
        for q in stable:
            if abs(q["sigma"] - p["s"]) < 0.1 and abs(q["s"] - p["sigma"]) < 0.1 and abs(q["alpha"] - p["alpha"]) < 0.05:
                mirror_found = True
                break
        status = "has mirror" if mirror_found else "no mirror in sample"
        print(f"  ({p['sigma']:.2f}, {p['s']:.2f}, {p['alpha']:+.2f}) -> {status}")


def main():
    probes, summary = load_latest_results()

    if not probes:
        print("No scout results found!")
        return 1

    print_analysis(probes, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
