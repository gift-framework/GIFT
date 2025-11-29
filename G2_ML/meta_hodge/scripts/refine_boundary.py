#!/usr/bin/env python
"""Adaptive boundary refinement for the stability valley.

This script refines the stability boundary by:
1. Loading coarse grid results
2. Identifying boundary regions (stable points adjacent to unstable)
3. Sampling more densely near the boundary
4. Fitting a more precise constraint

Goal: Improve the constraint u + b|α| <= c from R²=0.943 to R²>0.99
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import itertools

import numpy as np
import torch

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))

from meta_hodge.deformation_explorer import (
    DeformationConfig,
    load_baseline_data,
    explore_one_point,
    BaselineData,
)


@dataclass
class BoundaryPoint:
    """A point near the stability boundary."""
    sigma: float
    s: float
    alpha: float
    u: float
    stable: bool
    distance_to_boundary: float  # Estimated


def load_coarse_results(results_dir: Path) -> List[Dict]:
    """Load results from coarse grid."""
    json_path = results_dir / "deformation_results.json"
    with json_path.open() as f:
        results = json.load(f)

    # Add effective moduli
    for r in results:
        r["u"] = r["sigma"] * r["s"]
        r["v"] = r["sigma"] / r["s"] if r["s"] != 0 else float("inf")

    return results


def identify_boundary_region(results: List[Dict]) -> List[Dict]:
    """Identify points near the stability boundary.

    A point is near the boundary if:
    - It's stable and has an unstable neighbor
    - Or it's unstable and has a stable neighbor
    """
    stable_set = {(r["sigma"], r["s"], r["alpha"]) for r in results if r["stable"]}
    all_set = {(r["sigma"], r["s"], r["alpha"]) for r in results}

    # Get grid spacing
    sigmas = sorted(set(r["sigma"] for r in results))
    ss = sorted(set(r["s"] for r in results))
    alphas = sorted(set(r["alpha"] for r in results))

    d_sigma = sigmas[1] - sigmas[0] if len(sigmas) > 1 else 0.2
    d_s = ss[1] - ss[0] if len(ss) > 1 else 0.2
    d_alpha = alphas[1] - alphas[0] if len(alphas) > 1 else 0.15

    boundary_points = []

    for r in results:
        sigma, s, alpha = r["sigma"], r["s"], r["alpha"]
        is_stable = r["stable"]

        # Check if any neighbor has different stability
        is_boundary = False
        for ds in [-d_sigma, 0, d_sigma]:
            for dd in [-d_s, 0, d_s]:
                for da in [-d_alpha, 0, d_alpha]:
                    if ds == 0 and dd == 0 and da == 0:
                        continue

                    neighbor = (sigma + ds, s + dd, alpha + da)
                    # Check if neighbor exists and has different stability
                    if neighbor in all_set:
                        neighbor_stable = neighbor in stable_set
                        if neighbor_stable != is_stable:
                            is_boundary = True
                            break
                if is_boundary:
                    break
            if is_boundary:
                break

        if is_boundary:
            boundary_points.append(r)

    return boundary_points


def generate_refinement_points(
    boundary_points: List[Dict],
    n_refinements: int = 3,
    refinement_radius: float = 0.1,
) -> List[Tuple[float, float, float]]:
    """Generate new points to sample near the boundary.

    For each boundary point, generate points in a finer grid around it.
    """
    new_points = set()

    for bp in boundary_points:
        sigma, s, alpha = bp["sigma"], bp["s"], bp["alpha"]

        # Generate points in a small cube around this point
        for i in range(n_refinements):
            for j in range(n_refinements):
                for k in range(n_refinements):
                    # Offset from center
                    ds = (i - n_refinements // 2) * refinement_radius / n_refinements
                    dd = (j - n_refinements // 2) * refinement_radius / n_refinements
                    da = (k - n_refinements // 2) * refinement_radius / n_refinements

                    new_sigma = sigma + ds
                    new_s = s + dd
                    new_alpha = alpha + da

                    # Bounds check
                    if 0.3 <= new_sigma <= 2.0 and 0.3 <= new_s <= 2.0 and -0.5 <= new_alpha <= 0.5:
                        new_points.add((round(new_sigma, 3), round(new_s, 3), round(new_alpha, 3)))

    return list(new_points)


def fit_boundary_constraint(
    stable_points: List[Dict],
    unstable_points: List[Dict],
) -> Dict:
    """Fit a more precise boundary constraint.

    Try several functional forms:
    1. Linear: u + b|α| <= c
    2. Quadratic: u + b|α| + d*α² <= c
    3. Product: u * (1 + b|α|) <= c
    """
    if not stable_points or not unstable_points:
        return {}

    # Extract features
    stable_u = np.array([r["u"] for r in stable_points])
    stable_alpha = np.array([np.abs(r["alpha"]) for r in stable_points])

    unstable_u = np.array([r["u"] for r in unstable_points])
    unstable_alpha = np.array([np.abs(r["alpha"]) for r in unstable_points])

    results = {}

    # 1. Linear fit: u + b|α| = c at boundary
    # Find the boundary by looking at max u for each |α| bin
    alpha_bins = np.linspace(0, 0.5, 11)
    boundary_points_fit = []

    for i in range(len(alpha_bins) - 1):
        a_low, a_high = alpha_bins[i], alpha_bins[i + 1]
        a_mid = (a_low + a_high) / 2

        # Stable points in this bin
        mask_s = (stable_alpha >= a_low) & (stable_alpha < a_high)
        # Unstable points in this bin
        mask_u = (unstable_alpha >= a_low) & (unstable_alpha < a_high)

        if mask_s.any() and mask_u.any():
            u_max_stable = stable_u[mask_s].max()
            u_min_unstable = unstable_u[mask_u].min()
            # Boundary is between these
            u_boundary = (u_max_stable + u_min_unstable) / 2
            boundary_points_fit.append((a_mid, u_boundary))
        elif mask_s.any():
            boundary_points_fit.append((a_mid, stable_u[mask_s].max()))

    if len(boundary_points_fit) >= 2:
        alphas_fit = np.array([p[0] for p in boundary_points_fit])
        us_fit = np.array([p[1] for p in boundary_points_fit])

        # Linear: u = c - b * |α|
        A = np.vstack([np.ones_like(alphas_fit), alphas_fit]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, us_fit, rcond=None)
        c_lin, neg_b_lin = coeffs
        b_lin = -neg_b_lin

        # R² for linear
        u_pred_lin = c_lin - b_lin * alphas_fit
        ss_res_lin = np.sum((us_fit - u_pred_lin) ** 2)
        ss_tot = np.sum((us_fit - us_fit.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0

        results["linear"] = {
            "formula": f"u + {b_lin:.3f}|α| <= {c_lin:.3f}",
            "b": b_lin,
            "c": c_lin,
            "r_squared": r2_lin,
            "boundary_points": boundary_points_fit,
        }

        # Quadratic: u = c - b|α| - d*α²
        if len(boundary_points_fit) >= 3:
            A_quad = np.vstack([np.ones_like(alphas_fit), alphas_fit, alphas_fit**2]).T
            coeffs_quad, _, _, _ = np.linalg.lstsq(A_quad, us_fit, rcond=None)
            c_quad, neg_b_quad, neg_d_quad = coeffs_quad
            b_quad, d_quad = -neg_b_quad, -neg_d_quad

            u_pred_quad = c_quad - b_quad * alphas_fit - d_quad * alphas_fit**2
            ss_res_quad = np.sum((us_fit - u_pred_quad) ** 2)
            r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

            results["quadratic"] = {
                "formula": f"u + {b_quad:.3f}|α| + {d_quad:.3f}α² <= {c_quad:.3f}",
                "b": b_quad,
                "c": c_quad,
                "d": d_quad,
                "r_squared": r2_quad,
            }

    return results


def explore_refinement_points(
    baseline: BaselineData,
    points: List[Tuple[float, float, float]],
    config: DeformationConfig,
    progress_callback=None,
) -> List[Dict]:
    """Explore the refinement points."""
    results = []
    total = len(points)

    for i, (sigma, s, alpha) in enumerate(points):
        result = explore_one_point(baseline, sigma, s, alpha, config)

        results.append({
            "sigma": sigma,
            "s": s,
            "alpha": alpha,
            "u": sigma * s,
            "v": sigma / s if s != 0 else float("inf"),
            "stable": result.stable,
            "det_mean": result.det_mean,
            "kappa_T": result.kappa_T,
            "phi_norm_mean": result.phi_norm_mean,
        })

        if progress_callback:
            progress_callback(i + 1, total, results[-1])

    return results


def main():
    parser = argparse.ArgumentParser(description="Refine stability boundary")
    parser.add_argument("--coarse-dir", type=Path, help="Coarse grid results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest coarse results")
    parser.add_argument("--n-refinements", type=int, default=3, help="Refinement level per dimension")
    parser.add_argument("--radius", type=float, default=0.08, help="Refinement radius")
    parser.add_argument("--samples", type=int, default=150, help="Coordinate samples")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be done")

    args = parser.parse_args()

    # Find coarse results
    if args.latest or args.coarse_dir is None:
        base_dir = meta_hodge_dir / "artifacts" / "deformation_atlas"
        if base_dir.exists():
            dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
            if dirs:
                args.coarse_dir = dirs[-1]
            else:
                print("No coarse results found")
                return 1

    print("=" * 60)
    print("Stability Boundary Refinement")
    print("=" * 60)
    print(f"Coarse results: {args.coarse_dir}")

    # Load coarse results
    coarse_results = load_coarse_results(args.coarse_dir)
    stable_coarse = [r for r in coarse_results if r["stable"]]
    unstable_coarse = [r for r in coarse_results if not r["stable"]]

    print(f"Coarse grid: {len(coarse_results)} points ({len(stable_coarse)} stable)")

    # Identify boundary region
    boundary_points = identify_boundary_region(coarse_results)
    print(f"Boundary points identified: {len(boundary_points)}")

    # Generate refinement points
    refinement_points = generate_refinement_points(
        boundary_points,
        n_refinements=args.n_refinements,
        refinement_radius=args.radius,
    )

    # Remove points already in coarse grid
    existing = {(r["sigma"], r["s"], r["alpha"]) for r in coarse_results}
    refinement_points = [p for p in refinement_points if p not in existing]

    print(f"New refinement points: {len(refinement_points)}")

    if args.dry_run:
        print("\nDry run - would explore these points:")
        for i, (sigma, s, alpha) in enumerate(refinement_points[:20]):
            print(f"  ({sigma:.2f}, {s:.2f}, {alpha:+.2f}) -> u={sigma*s:.2f}")
        if len(refinement_points) > 20:
            print(f"  ... and {len(refinement_points) - 20} more")
        return 0

    # Load baseline
    print("\nLoading baseline data...")
    baseline = load_baseline_data("1_6", num_samples=args.samples)
    config = DeformationConfig()

    # Explore refinement points
    print(f"\nExploring {len(refinement_points)} refinement points...")

    def progress(i, total, result):
        status = "stable" if result["stable"] else "unstable"
        bar = "=" * (30 * i // total) + "-" * (30 - 30 * i // total)
        print(f"\r[{bar}] {i}/{total} ({result['sigma']:.2f},{result['s']:.2f},{result['alpha']:+.2f}) -> {status}    ", end="")

    refined_results = explore_refinement_points(baseline, refinement_points, config, progress)
    print()

    # Combine with coarse results
    all_results = coarse_results + refined_results
    stable_all = [r for r in all_results if r["stable"]]
    unstable_all = [r for r in all_results if not r["stable"]]

    print(f"\nCombined: {len(all_results)} points ({len(stable_all)} stable)")

    # Fit improved boundary
    print("\n" + "=" * 60)
    print("Boundary Constraint Fitting")
    print("=" * 60)

    constraints = fit_boundary_constraint(stable_all, unstable_all)

    for name, data in constraints.items():
        print(f"\n{name.upper()}:")
        print(f"  Formula: {data['formula']}")
        print(f"  R²: {data['r_squared']:.4f}")

    # Save results
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = meta_hodge_dir / "artifacts" / "boundary_refinement" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "refined_results.json").open("w") as f:
        json.dump(refined_results, f, indent=2)

    with (output_dir / "all_results.json").open("w") as f:
        json.dump(all_results, f, indent=2)

    with (output_dir / "constraints.json").open("w") as f:
        # Convert numpy to Python types
        constraints_json = {}
        for name, data in constraints.items():
            constraints_json[name] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in data.items()
                if k != "boundary_points"
            }
        json.dump(constraints_json, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
