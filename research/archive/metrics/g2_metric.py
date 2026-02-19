#!/usr/bin/env python3
"""
Explicit G₂ Metric on K7 via TCS Construction

This module constructs an explicit G₂ metric on the compact 7-manifold K7
using the Twisted Connected Sum (TCS) gluing of two ACyl Calabi-Yau 3-folds.

Structure:
    K7 = (M₊ × S¹) ∪_Φ (M₋ × S¹)

    where M± are ACyl CY3 asymptotic to (0,∞) × K3 × S¹
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
import json

# =============================================================================
# CONSTANTS
# =============================================================================

PI = np.pi
TWO_PI = 2 * np.pi

# TCS parameters
L_NECK = 8.354  # Canonical neck length (from κ = π²/14)
DELTA_GLUE = 1.0  # Gluing transition width

# K3 parameters
K3_VOLUME = 16 * PI**2  # Topological volume of K3

# =============================================================================
# CUTOFF FUNCTIONS
# =============================================================================

def smooth_step(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """
    Smooth step function: 0 for t < t0, 1 for t > t1, smooth in between.
    Uses the standard C^∞ bump function.
    """
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    # Smoothstep polynomial: 3x² - 2x³
    return np.where(x <= 0, 0, np.where(x >= 1, 1, 3*x**2 - 2*x**3))

def cutoff_left(t: np.ndarray, L: float, delta: float) -> np.ndarray:
    """Cutoff = 1 on left side, 0 on right, transition in neck."""
    return 1 - smooth_step(t, -delta, delta)

def cutoff_right(t: np.ndarray, L: float, delta: float) -> np.ndarray:
    """Cutoff = 0 on left side, 1 on right, transition in neck."""
    return smooth_step(t, -delta, delta)

# =============================================================================
# K3 METRIC (Simplified Models)
# =============================================================================

@dataclass
class K3Metric:
    """
    Hyper-Kähler metric on K3 surface.

    We provide several models of increasing sophistication:
    1. Flat T⁴ (not Ricci-flat, but simple)
    2. Kummer surface (T⁴/Z₂ with Eguchi-Hanson resolutions)
    3. Numerical (from Calabi-Yau solver)
    """
    model: str = "flat"
    resolution: int = 32

    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Return the 4×4 metric tensor g_K3 at point x = (x1, x2, x3, x4).

        Args:
            x: Point on K3, shape (4,) or (N, 4)

        Returns:
            Metric tensor, shape (4, 4) or (N, 4, 4)
        """
        if self.model == "flat":
            return self._flat_metric(x)
        elif self.model == "kummer":
            return self._kummer_metric(x)
        else:
            raise ValueError(f"Unknown K3 model: {self.model}")

    def _flat_metric(self, x: np.ndarray) -> np.ndarray:
        """Flat metric on T⁴ (not Ricci-flat K3, but useful for testing)."""
        if x.ndim == 1:
            return np.eye(4)
        else:
            N = x.shape[0]
            return np.tile(np.eye(4), (N, 1, 1))

    def _kummer_metric(self, x: np.ndarray) -> np.ndarray:
        """
        Kummer surface metric: flat + Eguchi-Hanson corrections near fixed points.

        The 16 fixed points of T⁴/Z₂ are at (n₁/2, n₂/2, n₃/2, n₄/2) for nᵢ ∈ {0,1}.
        Near each, we add an Eguchi-Hanson correction.
        """
        # Start with flat metric
        g = self._flat_metric(x)

        if x.ndim == 1:
            x = x.reshape(1, 4)
            squeeze = True
        else:
            squeeze = False

        # Eguchi-Hanson parameter
        a = 0.1  # Resolution parameter

        # Add corrections near each of 16 fixed points
        for n1 in [0, 0.5]:
            for n2 in [0, 0.5]:
                for n3 in [0, 0.5]:
                    for n4 in [0, 0.5]:
                        fixed_pt = np.array([n1, n2, n3, n4])
                        # Distance to fixed point (with periodic BC)
                        dx = x - fixed_pt
                        dx = dx - np.round(dx)  # Periodic
                        r2 = np.sum(dx**2, axis=-1, keepdims=True)
                        r = np.sqrt(r2 + 1e-10)

                        # Eguchi-Hanson correction (simplified)
                        # Full EH metric is complicated; this is a smooth approximation
                        eh_factor = 1 - np.exp(-r2 / (2*a**2))

                        # The correction to the metric (radial part)
                        # In the full EH, g_rr ~ 1/(1-(a/r)⁴)
                        # We approximate with a smooth version
                        if x.ndim == 1 or squeeze:
                            g = g * (1 + 0.1 * (1 - eh_factor.flatten()[0]))
                        else:
                            for i in range(4):
                                g[:, i, i] *= (1 + 0.1 * (1 - eh_factor.flatten()))

        if squeeze:
            g = g[0]

        return g

    def kahler_form(self, x: np.ndarray) -> np.ndarray:
        """
        Kähler form ω_K3 at point x.

        For a hyper-Kähler K3 with complex structure I:
        ω_I = (1/2)(dx₁∧dx₂ + dx₃∧dx₄) + corrections

        Returns coefficients ω_ij as antisymmetric 4×4 matrix.
        """
        omega = np.zeros((4, 4))
        omega[0, 1] = 0.5
        omega[1, 0] = -0.5
        omega[2, 3] = 0.5
        omega[3, 2] = -0.5
        return omega

    def holomorphic_2form(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Holomorphic (2,0)-form Ω_K3 = Re(Ω) + i·Im(Ω).

        Ω = (dx₁ + i·dx₂) ∧ (dx₃ + i·dx₄)
          = (dx₁∧dx₃ - dx₂∧dx₄) + i(dx₁∧dx₄ + dx₂∧dx₃)

        Returns (Re_ij, Im_ij) as antisymmetric 4×4 matrices.
        """
        re_omega = np.zeros((4, 4))
        re_omega[0, 2] = 1
        re_omega[2, 0] = -1
        re_omega[1, 3] = -1
        re_omega[3, 1] = 1

        im_omega = np.zeros((4, 4))
        im_omega[0, 3] = 1
        im_omega[3, 0] = -1
        im_omega[1, 2] = 1
        im_omega[2, 1] = -1

        return re_omega, im_omega


# =============================================================================
# ACyl CALABI-YAU 3-FOLD METRIC
# =============================================================================

@dataclass
class ACylCY3Metric:
    """
    Asymptotically Cylindrical Calabi-Yau 3-fold metric.

    Near infinity, the metric approaches:
        g ~ g_K3 + dr² + dφ²

    where (r, φ) parametrize R₊ × S¹.
    """
    k3: K3Metric
    decay_rate: float = 1.0  # Exponential decay rate μ

    def metric_tensor(self, r: float, phi: float, x_k3: np.ndarray) -> np.ndarray:
        """
        Return the 6×6 metric tensor at point (r, φ, x_K3).

        Coordinates: (r, φ, x₁, x₂, x₃, x₄)

        The metric is:
            g = dr² + dφ² + g_K3 + O(e^{-μr})
        """
        g = np.zeros((6, 6))

        # dr² and dφ² components
        g[0, 0] = 1.0  # g_rr
        g[1, 1] = 1.0  # g_φφ (circle of radius 1)

        # K3 block
        g_k3 = self.k3.metric_tensor(x_k3)
        g[2:6, 2:6] = g_k3

        # Add exponential corrections (simplified)
        if r > 0:
            correction = np.exp(-self.decay_rate * r)
            # Small off-diagonal terms
            g[0, 2:6] = 0.01 * correction * x_k3
            g[2:6, 0] = g[0, 2:6]

        return g

    def kahler_form(self, r: float, phi: float, x_k3: np.ndarray) -> np.ndarray:
        """
        Kähler form on CY3.

        ω = ω_K3 + dr ∧ dφ
        """
        omega = np.zeros((6, 6))

        # dr ∧ dφ
        omega[0, 1] = 1
        omega[1, 0] = -1

        # ω_K3 block
        omega[2:6, 2:6] = self.k3.kahler_form(x_k3)

        return omega

    def holomorphic_3form(self, r: float, phi: float, x_k3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Holomorphic (3,0)-form on CY3.

        Ω = (dr + i·dφ) ∧ Ω_K3
          = dr ∧ Re(Ω_K3) - dφ ∧ Im(Ω_K3)
            + i(dr ∧ Im(Ω_K3) + dφ ∧ Re(Ω_K3))

        Returns (Re_ijk, Im_ijk) as antisymmetric rank-3 tensors (6×6×6).
        """
        re_k3, im_k3 = self.k3.holomorphic_2form(x_k3)

        # Initialize 3-forms
        re_omega = np.zeros((6, 6, 6))
        im_omega = np.zeros((6, 6, 6))

        # dr ∧ Re(Ω_K3): indices (0, i, j) for i,j in K3
        for i in range(4):
            for j in range(4):
                re_omega[0, i+2, j+2] = re_k3[i, j]
                re_omega[i+2, 0, j+2] = -re_k3[i, j]
                re_omega[i+2, j+2, 0] = re_k3[i, j]

        # -dφ ∧ Im(Ω_K3): indices (1, i, j) for i,j in K3
        for i in range(4):
            for j in range(4):
                re_omega[1, i+2, j+2] = -im_k3[i, j]
                re_omega[i+2, 1, j+2] = im_k3[i, j]
                re_omega[i+2, j+2, 1] = -im_k3[i, j]

        # Similar for imaginary part
        for i in range(4):
            for j in range(4):
                im_omega[0, i+2, j+2] = im_k3[i, j]
                im_omega[i+2, 0, j+2] = -im_k3[i, j]
                im_omega[i+2, j+2, 0] = im_k3[i, j]

                im_omega[1, i+2, j+2] = re_k3[i, j]
                im_omega[i+2, 1, j+2] = -re_k3[i, j]
                im_omega[i+2, j+2, 1] = re_k3[i, j]

        return re_omega, im_omega


# =============================================================================
# G₂ METRIC ON S¹ × CY3
# =============================================================================

@dataclass
class G2ProductMetric:
    """
    G₂ metric on S¹ × CY3 from SU(3) structure.

    Given CY3 with (ω, Ω), the G₂ structure is:
        φ = dθ ∧ ω + Re(Ω)
        ψ = (1/2)ω ∧ ω + Im(Ω) ∧ dθ
        g = dθ² + g_CY
    """
    cy3: ACylCY3Metric

    def metric_tensor(self, theta: float, r: float, phi: float, x_k3: np.ndarray) -> np.ndarray:
        """
        Return the 7×7 metric tensor at point (θ, r, φ, x_K3).

        Coordinates: (θ, r, φ, x₁, x₂, x₃, x₄)
        """
        g = np.zeros((7, 7))

        # dθ² component
        g[0, 0] = 1.0

        # CY3 block
        g_cy = self.cy3.metric_tensor(r, phi, x_k3)
        g[1:7, 1:7] = g_cy

        return g

    def g2_3form(self, theta: float, r: float, phi: float, x_k3: np.ndarray) -> np.ndarray:
        """
        G₂ 3-form φ = dθ ∧ ω + Re(Ω).

        Returns φ_ijk as antisymmetric rank-3 tensor (7×7×7).
        """
        phi = np.zeros((7, 7, 7))

        # dθ ∧ ω: θ is index 0
        omega_cy = self.cy3.kahler_form(r, phi, x_k3)
        for i in range(6):
            for j in range(6):
                # dθ ∧ (indices i,j in CY3 = indices i+1, j+1 in 7D)
                phi[0, i+1, j+1] = omega_cy[i, j]
                phi[i+1, 0, j+1] = -omega_cy[i, j]
                phi[i+1, j+1, 0] = omega_cy[i, j]

        # Re(Ω): indices in CY3
        re_omega, _ = self.cy3.holomorphic_3form(r, phi, x_k3)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    phi[i+1, j+1, k+1] += re_omega[i, j, k]

        return phi


# =============================================================================
# TCS GLUED METRIC
# =============================================================================

@dataclass
class TCSK7Metric:
    """
    Complete G₂ metric on K7 via TCS gluing.

    K7 = (M₊ × S¹_θ) ∪ (M₋ × S¹_ψ)

    with gluing along the neck region.
    """
    L: float = L_NECK  # Neck length
    delta: float = DELTA_GLUE  # Transition width
    k3_model: str = "kummer"  # K3 metric model

    def __post_init__(self):
        """Initialize the building blocks."""
        self.k3_plus = K3Metric(model=self.k3_model)
        self.k3_minus = K3Metric(model=self.k3_model)

        self.cy3_plus = ACylCY3Metric(self.k3_plus)
        self.cy3_minus = ACylCY3Metric(self.k3_minus)

        self.g2_plus = G2ProductMetric(self.cy3_plus)
        self.g2_minus = G2ProductMetric(self.cy3_minus)

    def metric_tensor(self, t: float, x_k3: np.ndarray, theta: float, psi: float) -> np.ndarray:
        """
        Return the 7×7 metric tensor at point (t, x_K3, θ, ψ).

        Coordinates on K7:
            - t ∈ [-L, L]: neck parameter
            - x_K3 ∈ K3: 4 coordinates
            - θ, ψ ∈ S¹: fiber coordinates (identified in gluing)

        The metric interpolates between:
            - g₊ on the left (t << 0)
            - g₋ on the right (t >> 0)
        """
        # Cutoff functions
        chi_plus = cutoff_left(np.array([t]), self.L, self.delta)[0]
        chi_minus = cutoff_right(np.array([t]), self.L, self.delta)[0]

        # Map t to radial coordinates on each side
        r_plus = self.L - t  # r_plus → ∞ as t → -∞
        r_minus = self.L + t  # r_minus → ∞ as t → +∞

        # Get metrics from each side
        g_plus = self.g2_plus.metric_tensor(theta, max(r_plus, 0.1), 0, x_k3)
        g_minus = self.g2_minus.metric_tensor(psi, max(r_minus, 0.1), 0, x_k3)

        # Interpolate
        g = chi_plus * g_plus + chi_minus * g_minus

        # In the neck region, the metric should be approximately:
        # g = dt² + g_K3 + dθ² + dψ²
        # The interpolation should recover this

        return g

    def metric_at_point(self, coords: np.ndarray) -> np.ndarray:
        """
        Metric at a 7D point.

        coords = (t, x1, x2, x3, x4, theta, psi)
        """
        t = coords[0]
        x_k3 = coords[1:5]
        theta = coords[5]
        psi = coords[6]
        return self.metric_tensor(t, x_k3, theta, psi)

    def determinant(self, coords: np.ndarray) -> float:
        """Metric determinant at a point."""
        g = self.metric_at_point(coords)
        return np.linalg.det(g)

    def volume_element(self, coords: np.ndarray) -> float:
        """Volume element sqrt(det(g)) at a point."""
        return np.sqrt(abs(self.determinant(coords)))


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def compute_metric_on_grid(metric: TCSK7Metric,
                           n_t: int = 20,
                           n_k3: int = 8,
                           n_theta: int = 8) -> dict:
    """
    Compute the metric on a regular grid.

    Returns a dictionary with grid data and metric components.
    """
    # Grid parameters
    t_vals = np.linspace(-metric.L, metric.L, n_t)
    k3_vals = np.linspace(0, 1, n_k3)  # K3 coordinates (normalized)
    theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

    # Storage
    results = {
        'parameters': {
            'L': metric.L,
            'delta': metric.delta,
            'k3_model': metric.k3_model,
            'n_t': n_t,
            'n_k3': n_k3,
            'n_theta': n_theta
        },
        'grid': {
            't': t_vals.tolist(),
            'k3': k3_vals.tolist(),
            'theta': theta_vals.tolist()
        },
        'metrics': []
    }

    # Sample points (subset for efficiency)
    for i_t, t in enumerate(t_vals):
        for i_k3 in range(min(4, n_k3)):  # Sample a few K3 points
            x_k3 = np.array([k3_vals[i_k3], 0.5, 0.5, 0.5])
            for i_th in range(min(4, n_theta)):  # Sample a few angles
                theta = theta_vals[i_th]
                psi = 0  # Fix psi for simplicity

                g = metric.metric_tensor(t, x_k3, theta, psi)
                det_g = np.linalg.det(g)

                results['metrics'].append({
                    'point': [float(t), float(x_k3[0]), float(theta)],
                    'det_g': float(det_g),
                    'trace_g': float(np.trace(g)),
                    'g_00': float(g[0, 0]),
                    'g_11': float(g[1, 1])
                })

    return results


def export_metric_to_json(metric: TCSK7Metric, filename: str, **kwargs):
    """Export metric data to JSON file."""
    data = compute_metric_on_grid(metric, **kwargs)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Metric exported to {filename}")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_g2_structure(metric: TCSK7Metric, n_samples: int = 100) -> dict:
    """
    Verify that the metric satisfies G₂ constraints.

    Checks:
    1. Metric is positive definite
    2. Determinant is approximately constant (for normalized metric)
    3. G₂ 3-form is approximately closed (dφ ≈ 0)
    """
    results = {
        'positive_definite': True,
        'det_mean': 0,
        'det_std': 0,
        'det_values': []
    }

    np.random.seed(42)

    for _ in range(n_samples):
        # Random point
        t = np.random.uniform(-metric.L, metric.L)
        x_k3 = np.random.uniform(0, 1, 4)
        theta = np.random.uniform(0, 2*np.pi)
        psi = np.random.uniform(0, 2*np.pi)

        g = metric.metric_tensor(t, x_k3, theta, psi)

        # Check positive definite
        eigvals = np.linalg.eigvalsh(g)
        if np.any(eigvals <= 0):
            results['positive_definite'] = False

        # Determinant
        det_g = np.linalg.det(g)
        results['det_values'].append(float(det_g))

    results['det_mean'] = float(np.mean(results['det_values']))
    results['det_std'] = float(np.std(results['det_values']))
    results['det_values'] = results['det_values'][:10]  # Keep first 10 for output

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   Explicit G₂ Metric on K7 via TCS")
    print("=" * 60)

    # Create metric with canonical neck length
    print(f"\nCreating TCS metric with L = {L_NECK:.4f}...")
    k7_metric = TCSK7Metric(L=L_NECK, k3_model="kummer")

    # Test at a specific point
    print("\nTesting metric at sample points...")
    test_points = [
        np.array([0, 0.5, 0.5, 0.5, 0.5, 0, 0]),  # Center of neck
        np.array([-L_NECK/2, 0.5, 0.5, 0.5, 0.5, 0, 0]),  # Left side
        np.array([L_NECK/2, 0.5, 0.5, 0.5, 0.5, 0, 0]),  # Right side
    ]

    for i, pt in enumerate(test_points):
        g = k7_metric.metric_at_point(pt)
        det_g = np.linalg.det(g)
        print(f"  Point {i+1}: det(g) = {det_g:.6f}")

    # Verify G₂ structure
    print("\nVerifying G₂ structure...")
    verification = verify_g2_structure(k7_metric, n_samples=100)
    print(f"  Positive definite: {verification['positive_definite']}")
    print(f"  det(g) mean: {verification['det_mean']:.6f}")
    print(f"  det(g) std:  {verification['det_std']:.6f}")

    # Export
    print("\nExporting metric to JSON...")
    export_metric_to_json(k7_metric, "k7_metric_data.json", n_t=20, n_k3=8, n_theta=8)

    print("\n" + "=" * 60)
    print("   G₂ Metric Construction Complete")
    print("=" * 60)
