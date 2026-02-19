#!/usr/bin/env python3
"""
Calibrated Spectral Analysis for G₂ Manifolds
==============================================

This module implements proper unit calibration for graph Laplacian eigenvalues.

Key insight: Graph Laplacian eigenvalues μ₁ are NOT directly comparable to
continuous Laplace-Beltrami eigenvalues λ₁. We need:

    λ̂₁ = μ₁ / σ²  (ε-rescaling)

Then calibrate against known manifolds (S³, S⁷, T⁷).

Author: GIFT Project
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import json
from pathlib import Path
from datetime import datetime


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SpectralConfig:
    """Configuration for spectral computation."""
    N: int = 5000                    # Number of sample points
    k: int = 50                      # k-NN neighbors
    seed: int = 42                   # Random seed
    sigma_method: str = "median_knn" # "median_knn", "mean_knn", "sqrt_dim_k"
    sigma_factor: float = 1.0        # Multiplier for sigma
    laplacian_type: str = "symmetric"  # "symmetric", "random_walk", "unnormalized"
    use_geodesic: bool = True        # Use geodesic distances where applicable
    n_eigenvalues: int = 10          # Number of eigenvalues to compute


@dataclass
class CalibratedResult:
    """
    Complete result with calibration information.

    This is the GOLD STANDARD for reporting spectral results.
    Every run should produce this structure.
    """
    # Identification
    manifold: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Kernel information (CRITICAL for calibration)
    sigma: float = 0.0               # Actual σ used
    sigma_method: str = ""           # How σ was computed
    kernel_type: str = "gaussian"    # Kernel function

    # Raw graph eigenvalues
    mu1_graph: float = 0.0           # First non-zero eigenvalue (raw)
    eigenvalues_raw: List[float] = field(default_factory=list)

    # Calibrated eigenvalue (ε-rescaling)
    lambda1_hat: float = 0.0         # := μ₁ / σ²

    # Topology
    H_star: int = 99                 # b₂ + b₃ + 1

    # Products
    product_raw: float = 0.0         # μ₁ × H*
    product_calibrated: float = 0.0  # λ̂₁ × H*

    # Calibration factor (if known reference)
    calibration_factor: Optional[float] = None
    lambda1_physical: Optional[float] = None  # λ̂₁ / C

    # Diagnostics
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def sample_sphere(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample uniformly from S^{dim} (embedded in R^{dim+1}).

    Uses the standard method: normalize Gaussian vectors.
    """
    points = rng.standard_normal((n, dim + 1)).astype(np.float32)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def sample_S3(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from S³ (unit quaternions)."""
    return sample_sphere(n, 3, rng)


def sample_S7(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from S⁷ (unit octonions)."""
    return sample_sphere(n, 7, rng)


def sample_torus(n: int, dim: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from T^dim with given period."""
    return rng.uniform(0, period, size=(n, dim)).astype(np.float32)


def sample_T7(n: int, rng: np.random.Generator, period: float = 2*np.pi) -> np.ndarray:
    """Sample uniformly from T⁷ with period 2π."""
    return sample_torus(n, 7, period, rng)


def sample_TCS(
    n: int,
    rng: np.random.Generator,
    ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from TCS = S¹ × S³ × S³ (Twisted Connected Sum).

    Returns:
        theta: S¹ coordinates (N,)
        q1: S³₁ coordinates (N, 4)
        q2: S³₂ coordinates (N, 4)
        points: combined for distance computation
    """
    theta = rng.uniform(0, 2*np.pi, n).astype(np.float32)
    q1 = sample_S3(n, rng)
    q2 = sample_S3(n, rng)

    # For distance computation, we'll handle each factor separately
    return theta, q1, q2, None


# =============================================================================
# DISTANCE FUNCTIONS
# =============================================================================

def geodesic_sphere(points: np.ndarray) -> np.ndarray:
    """
    Geodesic distance matrix on S^n.

    For unit sphere: d(p,q) = arccos(p·q)
    We use chord → geodesic conversion for numerical stability.
    """
    dot = np.clip(points @ points.T, -1, 1)
    return np.arccos(dot)


def geodesic_S3(Q: np.ndarray) -> np.ndarray:
    """Geodesic distance on S³."""
    dot = np.clip(np.abs(Q @ Q.T), 0, 1)  # |p·q| for quaternions
    return 2 * np.arccos(dot)


def geodesic_S1(theta: np.ndarray) -> np.ndarray:
    """Geodesic distance on S¹."""
    diff = np.abs(theta[:, None] - theta[None, :])
    return np.minimum(diff, 2*np.pi - diff)


def periodic_torus(points: np.ndarray, period: float) -> np.ndarray:
    """Geodesic distance on flat torus."""
    n = len(points)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        diff = np.abs(points[i] - points)
        diff = np.minimum(diff, period - diff)
        D[i] = np.sqrt(np.sum(diff**2, axis=1))

    return D


def tcs_distance_matrix(
    theta: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    ratio: float = 1.0,
    alpha_s1: float = 1.0,
    det_g: float = 65/32
) -> np.ndarray:
    """
    Compute TCS distance matrix with configurable metric.

    d² = α·d_S1² + d_S3₁² + r²·d_S3₂²

    where α = det_g / r³ × alpha_s1 (from G₂ metric ansatz).
    """
    d_s1 = geodesic_S1(theta)
    d_s3_1 = geodesic_S3(q1)
    d_s3_2 = geodesic_S3(q2)

    alpha = det_g / (ratio**3) * alpha_s1
    d_sq = alpha * d_s1**2 + d_s3_1**2 + (ratio**2) * d_s3_2**2

    return np.sqrt(np.maximum(d_sq, 0))


# =============================================================================
# GRAPH LAPLACIAN WITH σ LOGGING
# =============================================================================

def compute_sigma(
    D: np.ndarray,
    k: int,
    method: str = "median_knn",
    factor: float = 1.0
) -> float:
    """
    Compute kernel bandwidth σ with explicit method logging.

    Methods:
    - "median_knn": median of k-NN distances (robust)
    - "mean_knn": mean of k-NN distances
    - "sqrt_dim_k": sqrt(dim/k) heuristic

    Returns σ (will be logged in results).
    """
    n = D.shape[0]
    k_actual = min(k, n - 1)

    # Get k-NN distances (excluding self)
    knn_dists = np.partition(D, k_actual, axis=1)[:, 1:k_actual+1]

    if method == "median_knn":
        sigma = np.median(knn_dists)
    elif method == "mean_knn":
        sigma = np.mean(knn_dists)
    elif method == "sqrt_dim_k":
        # Assumes 7D manifold
        sigma = np.sqrt(7.0 / k_actual)
    else:
        raise ValueError(f"Unknown sigma method: {method}")

    sigma = max(sigma * factor, 1e-10)
    return float(sigma)


def build_graph_laplacian(
    D: np.ndarray,
    k: int,
    sigma: float,
    laplacian_type: str = "symmetric"
) -> csr_matrix:
    """
    Build graph Laplacian with explicit σ.

    Args:
        D: Distance matrix (N × N)
        k: Number of neighbors for sparsification
        sigma: Kernel bandwidth (PRE-COMPUTED, not adaptive)
        laplacian_type: "symmetric", "random_walk", or "unnormalized"

    Returns:
        Sparse Laplacian matrix
    """
    n = D.shape[0]
    k_actual = min(k, n - 1)

    # Gaussian weights
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # k-NN sparsification
    for i in range(n):
        threshold = np.partition(W[i], -k_actual)[-k_actual]
        W[i, W[i] < threshold] = 0

    # Symmetrize
    W = (W + W.T) / 2

    # Degree
    d = np.maximum(W.sum(axis=1), 1e-10)

    if laplacian_type == "unnormalized":
        L = diags(d) - csr_matrix(W)

    elif laplacian_type == "random_walk":
        d_inv = 1.0 / d
        L = eye(n) - diags(d_inv) @ csr_matrix(W)

    elif laplacian_type == "symmetric":
        d_inv_sqrt = 1.0 / np.sqrt(d)
        L = eye(n) - diags(d_inv_sqrt) @ csr_matrix(W) @ diags(d_inv_sqrt)

    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    return L.tocsr()


def compute_eigenvalues(
    L: csr_matrix,
    n_eig: int = 10
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute smallest eigenvalues and eigenvectors.

    Returns:
        mu1: First non-zero eigenvalue
        eigenvalues: All computed eigenvalues
        eigenvectors: Corresponding eigenvectors (for mode analysis)
    """
    eigenvalues, eigenvectors = eigsh(L, k=n_eig, which='SM', tol=1e-8)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find first non-zero
    mu1 = 0.0
    for ev in eigenvalues:
        if ev > 1e-8:
            mu1 = float(ev)
            break

    if mu1 == 0.0 and len(eigenvalues) > 1:
        mu1 = float(eigenvalues[1])

    return mu1, eigenvalues, eigenvectors


# =============================================================================
# MAIN CALIBRATED COMPUTATION
# =============================================================================

def compute_spectrum_calibrated(
    manifold: str,
    config: SpectralConfig,
    H_star: int = 99,
    ratio: float = None,
    calibration_factor: float = None
) -> CalibratedResult:
    """
    Compute spectrum with full calibration logging.

    This is the PRIMARY FUNCTION for all spectral computations.

    Args:
        manifold: "S3", "S7", "T7", "K7", "TCS"
        config: SpectralConfig with all parameters
        H_star: Topological invariant (b₂ + b₃ + 1)
        ratio: TCS anisotropy (default H*/84 for K7)
        calibration_factor: If known, apply to get physical λ₁

    Returns:
        CalibratedResult with all logged quantities
    """
    import time
    t0 = time.time()

    rng = np.random.default_rng(config.seed)
    result = CalibratedResult(
        manifold=manifold,
        config=asdict(config),
        H_star=H_star,
        sigma_method=config.sigma_method
    )

    try:
        # === STEP 1: Sample manifold ===
        if manifold == "S3":
            points = sample_S3(config.N, rng)
            D = geodesic_S3(points)

        elif manifold == "S7":
            points = sample_S7(config.N, rng)
            D = geodesic_sphere(points)

        elif manifold == "T7":
            points = sample_T7(config.N, rng)
            D = periodic_torus(points, 2*np.pi)

        elif manifold in ["K7", "TCS"]:
            if ratio is None:
                ratio = H_star / 84.0
            theta, q1, q2, _ = sample_TCS(config.N, rng, ratio)
            D = tcs_distance_matrix(theta, q1, q2, ratio)

        else:
            raise ValueError(f"Unknown manifold: {manifold}")

        # === STEP 2: Compute σ (LOGGED) ===
        sigma = compute_sigma(D, config.k, config.sigma_method, config.sigma_factor)
        result.sigma = sigma

        # === STEP 3: Build Laplacian ===
        L = build_graph_laplacian(D, config.k, sigma, config.laplacian_type)

        # === STEP 4: Compute eigenvalues ===
        mu1, eigenvalues, eigenvectors = compute_eigenvalues(L, config.n_eigenvalues)

        result.mu1_graph = mu1
        result.eigenvalues_raw = eigenvalues.tolist()

        # === STEP 5: ε-rescaling (CRITICAL) ===
        # λ̂₁ = μ₁ / σ²
        lambda1_hat = mu1 / (sigma ** 2)
        result.lambda1_hat = lambda1_hat

        # === STEP 6: Products ===
        result.product_raw = mu1 * H_star
        result.product_calibrated = lambda1_hat * H_star

        # === STEP 7: Physical calibration (if available) ===
        if calibration_factor is not None:
            result.calibration_factor = calibration_factor
            result.lambda1_physical = lambda1_hat / calibration_factor

    except Exception as e:
        result.error = str(e)

    result.elapsed_seconds = time.time() - t0
    return result


# =============================================================================
# CALIBRATION ON KNOWN MANIFOLDS
# =============================================================================

# Known exact first eigenvalues (Laplace-Beltrami)
KNOWN_LAMBDA1 = {
    "S3": 3.0,   # λ₁(S³) = n(n+2)/R² = 1×3 = 3 for R=1
    "S7": 7.0,   # λ₁(S⁷) = 7 for R=1
    "T7": 1.0,   # λ₁(T⁷) = 1 for period 2π
}


def calibrate_on_reference(
    manifold: str,
    config: SpectralConfig,
    n_trials: int = 5
) -> Dict[str, Any]:
    """
    Run calibration on a reference manifold with known spectrum.

    Returns calibration factor C such that:
        λ₁(physical) = λ̂₁(graph) / C
    """
    if manifold not in KNOWN_LAMBDA1:
        raise ValueError(f"No known λ₁ for {manifold}")

    lambda1_exact = KNOWN_LAMBDA1[manifold]
    results = []

    for trial in range(n_trials):
        cfg = SpectralConfig(
            N=config.N,
            k=config.k,
            seed=config.seed + trial,
            sigma_method=config.sigma_method,
            sigma_factor=config.sigma_factor,
            laplacian_type=config.laplacian_type
        )

        result = compute_spectrum_calibrated(manifold, cfg, H_star=1)
        if result.error is None:
            results.append(result)

    if not results:
        return {"error": "All trials failed", "manifold": manifold}

    # Compute calibration factors
    lambda1_hats = [r.lambda1_hat for r in results]
    calibration_factors = [lh / lambda1_exact for lh in lambda1_hats]

    return {
        "manifold": manifold,
        "lambda1_exact": lambda1_exact,
        "lambda1_hat_mean": float(np.mean(lambda1_hats)),
        "lambda1_hat_std": float(np.std(lambda1_hats)),
        "calibration_factor_mean": float(np.mean(calibration_factors)),
        "calibration_factor_std": float(np.std(calibration_factors)),
        "n_trials": len(results),
        "config": asdict(config)
    }


def run_full_calibration(
    config: SpectralConfig,
    n_trials: int = 5
) -> Dict[str, Any]:
    """
    Run calibration on all reference manifolds and compute unified factor.
    """
    print("=" * 60)
    print("FULL CALIBRATION SUITE")
    print("=" * 60)

    calibrations = {}

    for manifold in ["S3", "S7", "T7"]:
        print(f"\nCalibrating on {manifold}...")
        cal = calibrate_on_reference(manifold, config, n_trials)
        calibrations[manifold] = cal

        if "error" not in cal:
            print(f"  λ₁(exact) = {cal['lambda1_exact']}")
            print(f"  λ̂₁(graph) = {cal['lambda1_hat_mean']:.4f} ± {cal['lambda1_hat_std']:.4f}")
            print(f"  C = {cal['calibration_factor_mean']:.4f} ± {cal['calibration_factor_std']:.4f}")
        else:
            print(f"  ERROR: {cal['error']}")

    # Unified calibration factor (weighted average)
    valid_cals = [c for c in calibrations.values() if "error" not in c]
    if valid_cals:
        C_values = [c["calibration_factor_mean"] for c in valid_cals]
        C_unified = float(np.mean(C_values))
        C_std = float(np.std(C_values))
    else:
        C_unified = None
        C_std = None

    print("\n" + "=" * 60)
    print("UNIFIED CALIBRATION FACTOR")
    print("=" * 60)
    if C_unified is not None:
        print(f"C = {C_unified:.4f} ± {C_std:.4f}")
        print(f"\nTo convert graph eigenvalue to physical:")
        print(f"  λ₁(physical) = λ̂₁(graph) / {C_unified:.4f}")
    else:
        print("CALIBRATION FAILED")

    return {
        "calibrations": calibrations,
        "C_unified": C_unified,
        "C_std": C_std,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_K7_calibrated(
    N: int = 5000,
    k: int = 50,
    H_star: int = 99,
    ratio: float = None,
    calibration_factor: float = None,
    seed: int = 42
) -> CalibratedResult:
    """
    Analyze K₇ with full calibration.

    This is the go-to function for K₇ spectral analysis.
    """
    config = SpectralConfig(N=N, k=k, seed=seed)

    if ratio is None:
        ratio = H_star / 84.0

    return compute_spectrum_calibrated(
        "K7", config, H_star=H_star,
        ratio=ratio, calibration_factor=calibration_factor
    )


def save_result(result: CalibratedResult, path: Path):
    """Save result to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def load_result(path: Path) -> CalibratedResult:
    """Load result from JSON."""
    with open(path) as f:
        data = json.load(f)
    return CalibratedResult(**data)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CALIBRATED SPECTRAL ANALYSIS - Move #1 Implementation")
    print("=" * 70 + "\n")

    # Default configuration
    config = SpectralConfig(
        N=2000,  # Smaller for quick test
        k=30,
        seed=42,
        sigma_method="median_knn",
        laplacian_type="symmetric"
    )

    # Run full calibration
    cal_results = run_full_calibration(config, n_trials=3)

    # Save calibration
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(cal_results, f, indent=2, default=str)

    print(f"\nCalibration saved to: {output_dir / 'calibration_results.json'}")

    # Test on K₇
    if cal_results["C_unified"] is not None:
        print("\n" + "=" * 60)
        print("TEST ON K₇")
        print("=" * 60)

        result = analyze_K7_calibrated(
            N=2000, k=30, H_star=99,
            calibration_factor=cal_results["C_unified"],
            seed=42
        )

        print(f"\nK₇ Results (N={result.config['N']}, k={result.config['k']}):")
        print(f"  σ = {result.sigma:.6f}")
        print(f"  μ₁ (raw graph) = {result.mu1_graph:.6f}")
        print(f"  λ̂₁ (ε-rescaled) = {result.lambda1_hat:.4f}")
        print(f"  λ₁ (physical) = {result.lambda1_physical:.4f}")
        print(f"  μ₁ × H* = {result.product_raw:.4f}")
        print(f"  λ̂₁ × H* = {result.product_calibrated:.4f}")

        # Save K₇ result
        save_result(result, output_dir / "K7_calibrated_result.json")
        print(f"\nK₇ result saved to: {output_dir / 'K7_calibrated_result.json'}")
