#!/usr/bin/env python
"""ML-based stability boundary classifier.

Uses the existing grid data to train a classifier that can:
1. Predict stability at any (sigma, s, alpha) point
2. Learn the exact boundary shape
3. Estimate uncertainty near the boundary

This avoids expensive PINN evaluations for most points.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# Add parent directories to path
script_dir = Path(__file__).resolve().parent
meta_hodge_dir = script_dir.parent
g2_ml_dir = meta_hodge_dir.parent
if str(g2_ml_dir) not in sys.path:
    sys.path.insert(0, str(g2_ml_dir))


@dataclass
class BoundaryModel:
    """A trained boundary classifier."""
    # Linear constraint: u + b*|alpha| <= c
    b_linear: float
    c_linear: float
    r2_linear: float

    # Quadratic constraint: u + b*|alpha| + d*alpha^2 <= c
    b_quad: float
    c_quad: float
    d_quad: float
    r2_quad: float

    # Support vectors (boundary points)
    boundary_u: np.ndarray
    boundary_alpha: np.ndarray

    def predict_linear(self, u: float, alpha: float) -> bool:
        """Predict stability using linear model."""
        return u + self.b_linear * abs(alpha) <= self.c_linear

    def predict_quadratic(self, u: float, alpha: float) -> bool:
        """Predict stability using quadratic model."""
        return u + self.b_quad * abs(alpha) + self.d_quad * alpha**2 <= self.c_quad

    def distance_to_boundary(self, u: float, alpha: float) -> float:
        """Distance to the linear boundary (positive = stable side)."""
        return self.c_linear - u - self.b_linear * abs(alpha)


def load_data(results_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load stable and unstable points from results."""
    json_path = results_dir / "deformation_results.json"
    with json_path.open() as f:
        results = json.load(f)

    for r in results:
        r["u"] = r["sigma"] * r["s"]
        r["v"] = r["sigma"] / r["s"] if r["s"] != 0 else float("inf")

    stable = [r for r in results if r["stable"]]
    unstable = [r for r in results if not r["stable"]]

    return stable, unstable


def fit_svm_boundary(stable: List[Dict], unstable: List[Dict]) -> Optional[Dict]:
    """Fit an SVM to find the optimal separating hyperplane.

    Returns coefficients for: w0 + w1*u + w2*|alpha| = 0
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("sklearn not available, skipping SVM fit")
        return None

    # Prepare data in (u, |alpha|) space
    X_stable = np.array([[r["u"], abs(r["alpha"])] for r in stable])
    X_unstable = np.array([[r["u"], abs(r["alpha"])] for r in unstable])

    X = np.vstack([X_stable, X_unstable])
    y = np.array([1] * len(stable) + [0] * len(unstable))

    # Fit SVM with linear kernel
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_scaled, y)

    # Get decision boundary coefficients
    # Decision function: w @ x + b = 0
    w_scaled = svm.coef_[0]
    b_scaled = svm.intercept_[0]

    # Transform back to original space
    w = w_scaled / scaler.scale_
    b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)

    # Normalize so coefficient of u is 1
    # w[0]*u + w[1]*|alpha| + b = 0
    # => u + (w[1]/w[0])*|alpha| = -b/w[0]
    if abs(w[0]) > 1e-6:
        b_coef = w[1] / w[0]
        c_coef = -b / w[0]
    else:
        b_coef = 0
        c_coef = 1

    # Compute accuracy
    y_pred = svm.predict(X_scaled)
    accuracy = (y_pred == y).mean()

    # Get support vectors in original space
    support_vectors = scaler.inverse_transform(svm.support_vectors_)

    return {
        "b": b_coef,
        "c": c_coef,
        "accuracy": accuracy,
        "n_support_vectors": len(svm.support_vectors_),
        "support_vectors": support_vectors.tolist(),
        "formula": f"u + {b_coef:.4f}|alpha| <= {c_coef:.4f}",
    }


def fit_analytical_boundary(stable: List[Dict], unstable: List[Dict]) -> Dict:
    """Fit boundary by finding max u for each |alpha| bin."""

    # Extract (u, |alpha|) for stable points
    stable_u = np.array([r["u"] for r in stable])
    stable_alpha = np.array([abs(r["alpha"]) for r in stable])

    unstable_u = np.array([r["u"] for r in unstable])
    unstable_alpha = np.array([abs(r["alpha"]) for r in unstable])

    # Find boundary by binning
    alpha_bins = np.linspace(0, 0.5, 11)
    boundary_points = []

    for i in range(len(alpha_bins) - 1):
        a_low, a_high = alpha_bins[i], alpha_bins[i + 1]
        a_mid = (a_low + a_high) / 2

        mask_s = (stable_alpha >= a_low) & (stable_alpha < a_high)
        mask_u = (unstable_alpha >= a_low) & (unstable_alpha < a_high)

        if mask_s.any() and mask_u.any():
            u_max_stable = stable_u[mask_s].max()
            u_min_unstable = unstable_u[mask_u].min()
            u_boundary = (u_max_stable + u_min_unstable) / 2
            boundary_points.append((a_mid, u_boundary))
        elif mask_s.any():
            boundary_points.append((a_mid, stable_u[mask_s].max()))

    if len(boundary_points) < 2:
        return {}

    alphas_fit = np.array([p[0] for p in boundary_points])
    us_fit = np.array([p[1] for p in boundary_points])

    # Linear fit: u = c - b * |alpha|
    A = np.vstack([np.ones_like(alphas_fit), alphas_fit]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, us_fit, rcond=None)
    c_lin, neg_b_lin = coeffs
    b_lin = -neg_b_lin

    u_pred = c_lin - b_lin * alphas_fit
    ss_res = np.sum((us_fit - u_pred) ** 2)
    ss_tot = np.sum((us_fit - us_fit.mean()) ** 2)
    r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Quadratic fit: u = c - b*|alpha| - d*alpha^2
    result = {
        "linear": {
            "b": b_lin,
            "c": c_lin,
            "r_squared": r2_lin,
            "formula": f"u + {b_lin:.4f}|alpha| <= {c_lin:.4f}",
            "boundary_points": boundary_points,
        }
    }

    if len(boundary_points) >= 3:
        A_quad = np.vstack([np.ones_like(alphas_fit), alphas_fit, alphas_fit**2]).T
        coeffs_quad, _, _, _ = np.linalg.lstsq(A_quad, us_fit, rcond=None)
        c_quad, neg_b_quad, neg_d_quad = coeffs_quad
        b_quad, d_quad = -neg_b_quad, -neg_d_quad

        u_pred_quad = c_quad - b_quad * alphas_fit - d_quad * alphas_fit**2
        ss_res_quad = np.sum((us_fit - u_pred_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

        result["quadratic"] = {
            "b": b_quad,
            "c": c_quad,
            "d": d_quad,
            "r_squared": r2_quad,
            "formula": f"u + {b_quad:.4f}|alpha| + {d_quad:.4f}*alpha^2 <= {c_quad:.4f}",
        }

    return result


def fit_gaussian_process(stable: List[Dict], unstable: List[Dict]) -> Optional[Dict]:
    """Fit a Gaussian Process classifier for uncertainty estimation."""
    try:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    except ImportError:
        print("sklearn not available, skipping GP fit")
        return None

    X_stable = np.array([[r["u"], abs(r["alpha"])] for r in stable])
    X_unstable = np.array([[r["u"], abs(r["alpha"])] for r in unstable])

    X = np.vstack([X_stable, X_unstable])
    y = np.array([1] * len(stable) + [0] * len(unstable))

    # Fit GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.2)
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gp.fit(X, y)

    # Evaluate on grid
    u_grid = np.linspace(0.3, 1.5, 50)
    alpha_grid = np.linspace(0, 0.5, 25)
    U, A = np.meshgrid(u_grid, alpha_grid)
    X_grid = np.column_stack([U.ravel(), A.ravel()])

    proba = gp.predict_proba(X_grid)[:, 1]  # P(stable)
    proba_grid = proba.reshape(U.shape)

    # Find 50% contour (boundary)
    boundary_u = []
    boundary_alpha = []
    for j, a in enumerate(alpha_grid):
        # Find where probability crosses 0.5
        prob_slice = proba_grid[j, :]
        crossings = np.where(np.diff(np.sign(prob_slice - 0.5)))[0]
        if len(crossings) > 0:
            # Interpolate
            i = crossings[-1]  # Take last crossing (highest u)
            u_cross = u_grid[i] + (0.5 - prob_slice[i]) / (prob_slice[i+1] - prob_slice[i]) * (u_grid[i+1] - u_grid[i])
            boundary_u.append(u_cross)
            boundary_alpha.append(a)

    # Fit linear to GP boundary
    if len(boundary_u) >= 2:
        boundary_u = np.array(boundary_u)
        boundary_alpha = np.array(boundary_alpha)

        A_fit = np.vstack([np.ones_like(boundary_alpha), boundary_alpha]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, boundary_u, rcond=None)
        c_gp, neg_b_gp = coeffs
        b_gp = -neg_b_gp

        u_pred = c_gp - b_gp * boundary_alpha
        ss_res = np.sum((boundary_u - u_pred) ** 2)
        ss_tot = np.sum((boundary_u - boundary_u.mean()) ** 2)
        r2_gp = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "b": b_gp,
            "c": c_gp,
            "r_squared": r2_gp,
            "formula": f"u + {b_gp:.4f}|alpha| <= {c_gp:.4f}",
            "boundary_points": list(zip(boundary_alpha.tolist(), boundary_u.tolist())),
            "accuracy": gp.score(X, y),
        }

    return None


def active_learning_suggest(
    model: BoundaryModel,
    n_suggestions: int = 20,
    u_range: Tuple[float, float] = (0.3, 1.5),
    alpha_range: Tuple[float, float] = (-0.5, 0.5),
) -> List[Tuple[float, float, float]]:
    """Suggest points to sample next using active learning.

    Strategy: Sample points near the predicted boundary where
    uncertainty is highest.
    """
    suggestions = []

    # Generate candidate points near boundary
    for _ in range(n_suggestions * 10):
        # Sample uniformly
        u = np.random.uniform(*u_range)
        alpha = np.random.uniform(*alpha_range)

        # Distance to boundary
        dist = model.distance_to_boundary(u, alpha)

        # Accept if close to boundary
        if abs(dist) < 0.15:  # Within 0.15 of boundary
            # Convert back to (sigma, s, alpha)
            # Use v=1 (symmetric point) for simplicity
            sigma = np.sqrt(u)
            s = np.sqrt(u)

            # Bounds check
            if 0.3 <= sigma <= 2.0 and 0.3 <= s <= 2.0:
                suggestions.append((sigma, s, alpha, abs(dist)))

    # Sort by distance to boundary (closest first)
    suggestions.sort(key=lambda x: x[3])

    return [(s[0], s[1], s[2]) for s in suggestions[:n_suggestions]]


def main():
    parser = argparse.ArgumentParser(description="ML boundary classifier")
    parser.add_argument("--results-dir", type=Path, help="Results directory")
    parser.add_argument("--latest", action="store_true", help="Use latest results")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--suggest", type=int, default=0, help="Suggest N points to sample")

    args = parser.parse_args()

    # Find results directory
    if args.latest or args.results_dir is None:
        base_dir = meta_hodge_dir / "artifacts" / "deformation_atlas"
        if base_dir.exists():
            dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
            if dirs:
                args.results_dir = dirs[-1]
            else:
                print("No results found")
                return 1

    print("=" * 60)
    print("ML Stability Boundary Classifier")
    print("=" * 60)
    print(f"Results: {args.results_dir}")

    # Load data
    stable, unstable = load_data(args.results_dir)
    print(f"Data: {len(stable)} stable, {len(unstable)} unstable")
    print()

    # Fit analytical boundary
    print("=" * 60)
    print("Analytical Fit (binned boundary)")
    print("=" * 60)
    analytical = fit_analytical_boundary(stable, unstable)

    for name, data in analytical.items():
        print(f"\n{name.upper()}:")
        print(f"  Formula: {data['formula']}")
        print(f"  R²: {data['r_squared']:.4f}")

    # Fit SVM
    print()
    print("=" * 60)
    print("SVM Fit (optimal separating hyperplane)")
    print("=" * 60)
    svm_result = fit_svm_boundary(stable, unstable)

    if svm_result:
        print(f"\nFormula: {svm_result['formula']}")
        print(f"Accuracy: {svm_result['accuracy']:.1%}")
        print(f"Support vectors: {svm_result['n_support_vectors']}")

    # Fit Gaussian Process
    print()
    print("=" * 60)
    print("Gaussian Process Fit (probabilistic boundary)")
    print("=" * 60)
    gp_result = fit_gaussian_process(stable, unstable)

    if gp_result:
        print(f"\nFormula: {gp_result['formula']}")
        print(f"R²: {gp_result['r_squared']:.4f}")
        print(f"Accuracy: {gp_result['accuracy']:.1%}")

    # Summary comparison
    print()
    print("=" * 60)
    print("SUMMARY: Boundary Constraint Comparison")
    print("=" * 60)
    print()
    print("Method           | b       | c       | R²/Acc  | Formula")
    print("-" * 70)

    if "linear" in analytical:
        lin = analytical["linear"]
        print(f"Analytical (lin) | {lin['b']:7.4f} | {lin['c']:7.4f} | {lin['r_squared']:6.4f} | {lin['formula']}")

    if "quadratic" in analytical:
        quad = analytical["quadratic"]
        print(f"Analytical (quad)| {quad['b']:7.4f} | {quad['c']:7.4f} | {quad['r_squared']:6.4f} | (quadratic)")

    if svm_result:
        print(f"SVM              | {svm_result['b']:7.4f} | {svm_result['c']:7.4f} | {svm_result['accuracy']:6.1%} | {svm_result['formula']}")

    if gp_result:
        print(f"Gaussian Process | {gp_result['b']:7.4f} | {gp_result['c']:7.4f} | {gp_result['r_squared']:6.4f} | {gp_result['formula']}")

    # Active learning suggestions
    if args.suggest > 0:
        print()
        print("=" * 60)
        print(f"Active Learning: Top {args.suggest} points to sample next")
        print("=" * 60)

        if "linear" in analytical:
            lin = analytical["linear"]
            model = BoundaryModel(
                b_linear=lin["b"], c_linear=lin["c"], r2_linear=lin["r_squared"],
                b_quad=0, c_quad=0, d_quad=0, r2_quad=0,
                boundary_u=np.array([]), boundary_alpha=np.array([]),
            )

            suggestions = active_learning_suggest(model, args.suggest)
            print()
            for i, (sigma, s, alpha) in enumerate(suggestions):
                u = sigma * s
                print(f"  {i+1}. (sigma={sigma:.3f}, s={s:.3f}, alpha={alpha:+.3f}) -> u={u:.3f}")

    # Save results
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = meta_hodge_dir / "artifacts" / "ml_boundary" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "analytical": {k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                          for kk, vv in v.items()}
                      for k, v in analytical.items()},
        "svm": svm_result,
        "gp": gp_result,
    }

    with (output_dir / "ml_boundary_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
