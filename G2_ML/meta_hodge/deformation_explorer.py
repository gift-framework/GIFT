"""Deformation explorer for K7_GIFT moduli space mapping.

This module provides tools to explore the neighborhood around the baseline
K7_GIFT G2 structure by varying deformation parameters (sigma, s, alpha)
and tracking geometric/cohomological stability.

See K7_DEFORMATION_ATLAS.md for the theoretical framework.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch

from .geometry_loader import (
    phi_to_metric_exact,
    phi_to_metric_approximate,
    phi_components_to_tensor,
    enforce_spd,
    load_version_model,
    sample_coords,
)
from .config import locate_historical_assets


__all__ = [
    "DeformationResult",
    "DeformationConfig",
    "BaselineData",
    "load_baseline_data",
    "deform_phi",
    "compute_basic_invariants",
    "explore_one_point",
    "explore_grid",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class DeformationConfig:
    """Configuration for deformation exploration."""
    # Deformation parameter ranges
    sigma_values: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    s_values: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    alpha_values: Tuple[float, ...] = (-0.3, -0.15, 0.0, 0.15, 0.3)

    # Tolerances for stability classification
    det_target: float = 65.0 / 32.0
    det_tol_frac: float = 0.05  # 5%

    # Torsion (kappa_T) target - note: compute_torsion_estimate uses ||phi||^2_g deviation
    # For calibrated baseline, we set this dynamically from baseline
    kappa_target: float = 1.0 / 61.0
    kappa_tol_frac: float = 0.50  # 50% - relaxed because estimation is approximate

    # phi_norm target - dynamically set from baseline at (1,1,0)
    phi_norm_target: float = 7.0
    phi_norm_tol_frac: float = 0.20  # 20% - allow variation from baseline

    # Computation options
    use_exact_metric: bool = False  # Use approximate for speed
    compute_yukawa: bool = False  # Skip Yukawa for initial pass

    # Dynamic calibration (set at runtime from baseline)
    baseline_phi_norm: Optional[float] = None
    baseline_kappa: Optional[float] = None

    @property
    def det_tol(self) -> float:
        return self.det_tol_frac * self.det_target

    @property
    def kappa_tol(self) -> float:
        target = self.baseline_kappa if self.baseline_kappa else self.kappa_target
        return self.kappa_tol_frac * target

    @property
    def phi_norm_tol(self) -> float:
        target = self.baseline_phi_norm if self.baseline_phi_norm else self.phi_norm_target
        return self.phi_norm_tol_frac * target


@dataclass
class DeformationResult:
    """Result from exploring a single point in deformation space."""
    # Parameters
    sigma: float
    s: float
    alpha: float

    # Metric invariants
    det_mean: float
    det_std: float
    kappa_T: float
    g_posdef: bool

    # G2 structure
    phi_norm_mean: float
    phi_norm_std: float

    # Yukawa (optional)
    y_rank: Optional[int] = None
    m2_over_m3: Optional[float] = None
    m1_over_m3: Optional[float] = None

    # Stability classification
    stable: bool = False
    notes: str = ""


@dataclass
class BaselineData:
    """Container for baseline K7_GIFT data."""
    coords: torch.Tensor  # (N, 7)
    phi_local: torch.Tensor  # (N, 7, 7, 7)
    phi_global: torch.Tensor  # (N, 7, 7, 7)
    phi_total: torch.Tensor  # (N, 7, 7, 7)
    g: torch.Tensor  # (N, 7, 7)
    det_g: torch.Tensor  # (N,)
    version: str = "1_6"
    notes: str = ""


# =============================================================================
# Data loading
# =============================================================================

def _tensor_to_7x7x7(phi_flat: torch.Tensor) -> torch.Tensor:
    """Convert flat 35 components to (N, 7, 7, 7) antisymmetric tensor."""
    return phi_components_to_tensor(phi_flat)


def _find_sample_path(version: str, registry: Dict) -> Optional[Path]:
    """Find sample_coords file for a version."""
    if version not in registry:
        return None
    info = registry[version]
    for cp in info.checkpoint_paths:
        if "sample_coords" in cp.name:
            return cp
    return None


def load_baseline_data(
    version: str = "1_6",
    num_samples: Optional[int] = None,
    registry: Optional[Dict] = None,
) -> BaselineData:
    """Load baseline K7_GIFT data for deformation exploration.

    Args:
        version: Model version to load (default: 1_6 for GIFT-calibrated)
        num_samples: Number of samples to use (None = use all available)
        registry: Pre-loaded registry (optional)

    Returns:
        BaselineData with phi_local, phi_global, coords, etc.
    """
    registry = registry or locate_historical_assets()

    # Try to load precomputed data with local/global split
    sample_path = _find_sample_path(version, registry)

    if sample_path is not None:
        data = torch.load(sample_path, map_location="cpu", weights_only=False)

        if "phi_local" in data and "phi_global" in data:
            phi_local = data["phi_local"]
            phi_global = data["phi_global"]
            phi_total = data.get("phi_total", phi_local + phi_global)
            coords = data["x"]
            g = data.get("g")
            det_g = data.get("det_g")

            # Subsample if requested
            if num_samples is not None and num_samples < coords.shape[0]:
                idx = torch.randperm(coords.shape[0])[:num_samples]
                phi_local = phi_local[idx]
                phi_global = phi_global[idx]
                phi_total = phi_total[idx]
                coords = coords[idx]
                if g is not None:
                    g = g[idx]
                if det_g is not None:
                    det_g = det_g[idx]

            # Compute metric if not available
            if g is None:
                g = phi_to_metric_approximate(phi_total)
            if det_g is None:
                det_g = torch.linalg.det(g)

            return BaselineData(
                coords=coords,
                phi_local=phi_local,
                phi_global=phi_global,
                phi_total=phi_total,
                g=g,
                det_g=det_g,
                version=version,
                notes=f"Loaded from {sample_path.name}",
            )

    # Fallback: generate from model
    bundle = load_version_model(version, registry)
    n_pts = num_samples or 1000
    coords = sample_coords(n_pts)

    if bundle.phi_fn is not None:
        phi_flat = bundle.phi_fn(coords)
        phi_total = _tensor_to_7x7x7(phi_flat)
    else:
        phi_total = torch.zeros(n_pts, 7, 7, 7)

    g = bundle.metric_fn(coords)
    det_g = torch.linalg.det(g)

    # For fallback, phi_local = phi_total, phi_global = 0
    phi_local = phi_total.clone()
    phi_global = torch.zeros_like(phi_total)

    return BaselineData(
        coords=coords,
        phi_local=phi_local,
        phi_global=phi_global,
        phi_total=phi_total,
        g=g,
        det_g=det_g,
        version=version,
        notes=f"Generated from model {version}",
    )


# =============================================================================
# Deformation functions
# =============================================================================

def deform_phi(
    phi_local: torch.Tensor,
    phi_global: torch.Tensor,
    coords: torch.Tensor,
    sigma: float,
    s: float,
    alpha: float,
) -> torch.Tensor:
    """Apply deformation to phi.

    The deformation formula is:
        phi_deformed = phi_local + sigma * s * (1 + alpha * sgn(x0)) * phi_global

    Args:
        phi_local: Local (constant) G2 form (N, 7, 7, 7)
        phi_global: Position-dependent modulation (N, 7, 7, 7)
        coords: Sample coordinates (N, 7)
        sigma: Neck scale parameter
        s: Global amplitude parameter
        alpha: Asymmetry parameter

    Returns:
        Deformed phi tensor (N, 7, 7, 7)
    """
    x0 = coords[:, 0]

    # Apply asymmetry modulation
    if abs(alpha) > 1e-10:
        # sign(x0) centered around 0.5 for [0,1] domain
        sign_x0 = torch.sign(x0 - 0.5)
        # Broadcast to match phi shape
        sign_x0 = sign_x0.view(-1, 1, 1, 1)
        phi_mod = (1.0 + alpha * sign_x0) * phi_global
    else:
        phi_mod = phi_global

    # Combine with scaling
    phi_deformed = phi_local + sigma * s * phi_mod

    return phi_deformed


def compute_basic_invariants(
    phi: torch.Tensor,
    config: DeformationConfig,
    baseline_g: torch.Tensor = None,
    baseline_det: torch.Tensor = None,
) -> Dict[str, Any]:
    """Compute basic geometric invariants from deformed phi.

    Args:
        phi: Deformed G2 3-form (N, 7, 7, 7)
        config: Deformation configuration
        baseline_g: Pre-computed baseline metric (N, 7, 7) - used for calibration
        baseline_det: Pre-computed baseline det(g) (N,) - used for calibration

    Returns:
        Dictionary with det_mean, det_std, kappa_T, g_posdef, phi_norm_mean, phi_norm_std
    """
    # Compute raw metric from phi
    if config.use_exact_metric:
        g_raw = phi_to_metric_exact(phi)
    else:
        g_raw = phi_to_metric_approximate(phi)

    # If baseline is provided, calibrate the metric to match baseline det(g)
    # This accounts for the fact that the v1.6 data was pre-calibrated
    if baseline_g is not None and baseline_det is not None:
        # Compute raw determinant
        det_raw = torch.linalg.det(g_raw)
        det_raw_mean = det_raw.mean()

        if det_raw_mean > 0:
            # Scale factor to match baseline determinant
            # det(c*g) = c^7 * det(g), so c = (det_target/det_raw)^(1/7)
            target_det = baseline_det.mean()
            scale = (target_det / det_raw_mean) ** (1.0 / 7.0)
            g = g_raw * scale
        else:
            g = g_raw
    else:
        g = g_raw

    # Check positive definiteness
    with torch.no_grad():
        try:
            evals = torch.linalg.eigvalsh(g)
            g_posdef = bool((evals > 0).all())
        except Exception:
            g_posdef = False

    if not g_posdef:
        return {
            "det_mean": float("nan"),
            "det_std": float("nan"),
            "kappa_T": float("nan"),
            "g_posdef": False,
            "phi_norm_mean": float("nan"),
            "phi_norm_std": float("nan"),
            "g": g,
        }

    # Determinant
    det_g = torch.linalg.det(g)
    det_mean = det_g.mean().item()
    det_std = det_g.std().item()

    # Phi norm squared: ||phi||^2_g = phi_ijk * phi_lmn * g^{il} * g^{jm} * g^{kn}
    # Simplified: use approximate formula ||phi||^2 ~ tr(phi @ phi^T) / scale
    # For canonical G2, this should be 7
    try:
        g_inv = torch.linalg.inv(g)
        # Contract: phi_norm_sq = phi_ijk * phi_lmn * g^{il} * g^{jm} * g^{kn}
        # Use einsum with batch index 'b'
        phi_up = torch.einsum("bijk,bil,bjm,bkn->blmn", phi, g_inv, g_inv, g_inv)
        phi_norm_sq = torch.einsum("bijk,bijk->b", phi, phi_up)
        phi_norm_mean = phi_norm_sq.mean().item()
        phi_norm_std = phi_norm_sq.std().item()
    except Exception:
        phi_norm_mean = float("nan")
        phi_norm_std = float("nan")

    # Torsion estimate (simplified)
    # kappa_T = ||d(phi)|| / ||phi|| approximately
    # For now, use the deviation from canonical structure as proxy
    kappa_T = compute_torsion_estimate(phi, g)

    return {
        "det_mean": det_mean,
        "det_std": det_std,
        "kappa_T": kappa_T,
        "g_posdef": g_posdef,
        "phi_norm_mean": phi_norm_mean,
        "phi_norm_std": phi_norm_std,
        "g": g,
    }


def compute_torsion_estimate(phi: torch.Tensor, g: torch.Tensor) -> float:
    """Estimate torsion magnitude from phi and g.

    Uses the deviation from the G2 identity as a proxy for torsion.
    For exact G2 holonomy, ||phi||^2_g = 7 exactly.
    """
    try:
        g_inv = torch.linalg.inv(g)
        phi_up = torch.einsum("bijk,bil,bjm,bkn->blmn", phi, g_inv, g_inv, g_inv)
        phi_norm_sq = torch.einsum("bijk,bijk->b", phi, phi_up)

        # Deviation from 7
        deviation = (phi_norm_sq - 7.0).abs()
        kappa_T = deviation.mean().item() / 7.0  # Relative deviation

        return kappa_T
    except Exception:
        return float("nan")


# =============================================================================
# Main exploration functions
# =============================================================================

def explore_one_point(
    baseline: BaselineData,
    sigma: float,
    s: float,
    alpha: float,
    config: DeformationConfig,
) -> DeformationResult:
    """Explore a single point in deformation space.

    Args:
        baseline: Baseline K7_GIFT data
        sigma: Neck scale parameter
        s: Global amplitude parameter
        alpha: Asymmetry parameter
        config: Deformation configuration

    Returns:
        DeformationResult with all computed invariants
    """
    # Apply deformation
    phi_def = deform_phi(
        baseline.phi_local,
        baseline.phi_global,
        baseline.coords,
        sigma,
        s,
        alpha,
    )

    # Compute invariants with baseline calibration
    inv = compute_basic_invariants(
        phi_def, config,
        baseline_g=baseline.g,
        baseline_det=baseline.det_g,
    )

    # Early exit if metric is not positive definite
    if not inv["g_posdef"]:
        return DeformationResult(
            sigma=sigma,
            s=s,
            alpha=alpha,
            det_mean=inv["det_mean"],
            det_std=inv["det_std"],
            kappa_T=inv["kappa_T"],
            g_posdef=False,
            phi_norm_mean=inv["phi_norm_mean"],
            phi_norm_std=inv["phi_norm_std"],
            y_rank=None,
            m2_over_m3=None,
            m1_over_m3=None,
            stable=False,
            notes="g not positive definite",
        )

    # Yukawa computation (optional, expensive)
    y_rank = None
    m2_over_m3 = None
    m1_over_m3 = None

    if config.compute_yukawa:
        # TODO: Implement full Yukawa computation
        # For now, skip
        pass

    # Stability classification
    det_ok = abs(inv["det_mean"] - config.det_target) <= config.det_tol

    # Use calibrated baselines if available
    kappa_target = config.baseline_kappa if config.baseline_kappa else config.kappa_target
    phi_norm_target = config.baseline_phi_norm if config.baseline_phi_norm else config.phi_norm_target

    kappa_ok = not (inv["kappa_T"] != inv["kappa_T"])  # Check not NaN
    if kappa_ok:
        kappa_ok = abs(inv["kappa_T"] - kappa_target) <= config.kappa_tol
    phi_norm_ok = not (inv["phi_norm_mean"] != inv["phi_norm_mean"])
    if phi_norm_ok:
        phi_norm_ok = abs(inv["phi_norm_mean"] - phi_norm_target) <= config.phi_norm_tol

    stable = inv["g_posdef"] and det_ok and kappa_ok and phi_norm_ok

    notes_parts = []
    if not det_ok:
        notes_parts.append(f"det={inv['det_mean']:.4f}")
    if not kappa_ok:
        notes_parts.append(f"kappa={inv['kappa_T']:.4f}")
    if not phi_norm_ok:
        notes_parts.append(f"norm={inv['phi_norm_mean']:.4f}")

    return DeformationResult(
        sigma=sigma,
        s=s,
        alpha=alpha,
        det_mean=inv["det_mean"],
        det_std=inv["det_std"],
        kappa_T=inv["kappa_T"],
        g_posdef=inv["g_posdef"],
        phi_norm_mean=inv["phi_norm_mean"],
        phi_norm_std=inv["phi_norm_std"],
        y_rank=y_rank,
        m2_over_m3=m2_over_m3,
        m1_over_m3=m1_over_m3,
        stable=stable,
        notes="; ".join(notes_parts) if notes_parts else "stable",
    )


def explore_grid(
    baseline: BaselineData,
    config: DeformationConfig,
    progress_callback=None,
) -> list[DeformationResult]:
    """Explore a grid of deformation parameters.

    Args:
        baseline: Baseline K7_GIFT data
        config: Deformation configuration with parameter grids
        progress_callback: Optional callback(i, total, result) for progress

    Returns:
        List of DeformationResult for each grid point
    """
    import itertools

    # Calibrate config targets from baseline at (1,1,0)
    if config.baseline_phi_norm is None or config.baseline_kappa is None:
        baseline_result = explore_one_point(baseline, 1.0, 1.0, 0.0, config)
        if not (baseline_result.phi_norm_mean != baseline_result.phi_norm_mean):  # not NaN
            config.baseline_phi_norm = baseline_result.phi_norm_mean
            config.phi_norm_target = baseline_result.phi_norm_mean
        if not (baseline_result.kappa_T != baseline_result.kappa_T):  # not NaN
            config.baseline_kappa = baseline_result.kappa_T
            config.kappa_target = baseline_result.kappa_T

    grid_points = list(itertools.product(
        config.sigma_values,
        config.s_values,
        config.alpha_values,
    ))
    total = len(grid_points)

    results = []
    for i, (sigma, s, alpha) in enumerate(grid_points):
        result = explore_one_point(baseline, sigma, s, alpha, config)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

    return results


def save_results(
    results: list[DeformationResult],
    output_dir: Path,
    format: str = "both",
) -> Dict[str, Path]:
    """Save exploration results to files.

    Args:
        results: List of DeformationResult
        output_dir: Output directory
        format: "csv", "json", or "both"

    Returns:
        Dictionary with paths to saved files
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    if format in ("csv", "both"):
        csv_path = output_dir / "deformation_results.csv"
        with csv_path.open("w", newline="") as f:
            if results:
                fieldnames = list(asdict(results[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
        paths["csv"] = csv_path

    if format in ("json", "both"):
        json_path = output_dir / "deformation_results.json"
        with json_path.open("w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        paths["json"] = json_path

    return paths


def summarize_results(results: list[DeformationResult]) -> str:
    """Generate a summary of exploration results."""
    if not results:
        return "No results"

    total = len(results)
    stable = sum(1 for r in results if r.stable)
    posdef = sum(1 for r in results if r.g_posdef)

    det_values = [r.det_mean for r in results if r.g_posdef]
    kappa_values = [r.kappa_T for r in results if r.g_posdef and r.kappa_T == r.kappa_T]

    lines = [
        f"Deformation Atlas Summary",
        f"=" * 40,
        f"Total points: {total}",
        f"Positive definite: {posdef} ({100*posdef/total:.1f}%)",
        f"Stable: {stable} ({100*stable/total:.1f}%)",
    ]

    if det_values:
        det_min, det_max = min(det_values), max(det_values)
        lines.append(f"det(g) range: [{det_min:.4f}, {det_max:.4f}] (target: {65/32:.4f})")

    if kappa_values:
        kappa_min, kappa_max = min(kappa_values), max(kappa_values)
        lines.append(f"kappa_T range: [{kappa_min:.4f}, {kappa_max:.4f}] (target: {1/61:.4f})")

    # Find stability boundaries
    stable_sigma = [r.sigma for r in results if r.stable]
    stable_s = [r.s for r in results if r.stable]
    stable_alpha = [r.alpha for r in results if r.stable]

    if stable_sigma:
        lines.append(f"Stable sigma range: [{min(stable_sigma)}, {max(stable_sigma)}]")
    if stable_s:
        lines.append(f"Stable s range: [{min(stable_s)}, {max(stable_s)}]")
    if stable_alpha:
        lines.append(f"Stable alpha range: [{min(stable_alpha)}, {max(stable_alpha)}]")

    return "\n".join(lines)
