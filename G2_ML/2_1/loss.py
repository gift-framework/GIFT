"""Variational loss functions for GIFT v2.2 G2 metric extraction.

The primary variational problem:
    Minimize F[phi] = ||d phi||^2 + ||d* phi||^2

Subject to GIFT constraints (handled by constraints.py).

This module implements the torsion minimization objective and combines
it with constraint penalties for the full loss function.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable

from .config import GIFTConfig, default_config
from .constraints import GIFTConstraints
from .g2_geometry import MetricFromPhi


# =============================================================================
# Torsion functional: ||d phi||^2 + ||d* phi||^2
# =============================================================================

class TorsionFunctional(nn.Module):
    """Torsion functional F[phi] = ||d phi||^2 + ||d* phi||^2.

    For a G2 structure phi, minimizing this functional drives toward
    torsion-free (holonomy exactly G2).

    GIFT target: we don't want torsion-free, but kappa_T = 1/61.
    So we minimize (||T|| - 1/61)^2 instead of ||T||^2.

    The functional is computed via finite differences on phi.
    """

    def __init__(self, config: GIFTConfig = None):
        super().__init__()
        self.config = config or default_config
        self.metric_extractor = MetricFromPhi()

    def exterior_derivative_norm(self, phi: torch.Tensor, x: torch.Tensor,
                                  phi_fn: Callable) -> torch.Tensor:
        """Compute ||d phi||^2 via finite differences.

        For a 3-form phi, d phi is a 4-form with C(7,4) = 35 components.

        Args:
            phi: 3-form values (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi at new points

        Returns:
            norm_sq: ||d phi||^2 (batch,)
        """
        batch = phi.shape[0]
        device = phi.device
        eps = 1e-4

        # d phi has components (d phi)_{ijkl} = partial_i phi_{jkl} - ...
        # We compute a simplified proxy: sum of |grad phi|^2
        d_phi_sq = torch.zeros(batch, device=device)

        for i in range(7):
            x_plus = x.clone()
            x_plus[:, i] += eps
            x_minus = x.clone()
            x_minus[:, i] -= eps

            phi_plus = phi_fn(x_plus)
            phi_minus = phi_fn(x_minus)

            grad_i = (phi_plus - phi_minus) / (2 * eps)
            d_phi_sq += (grad_i ** 2).sum(dim=-1)

        return d_phi_sq

    def codifferential_norm(self, phi: torch.Tensor, x: torch.Tensor,
                            phi_fn: Callable, metric: torch.Tensor) -> torch.Tensor:
        """Compute ||d* phi||^2 (codifferential of 3-form).

        The codifferential d* = *d* maps 3-forms to 2-forms.
        For G2, d* phi = 0 is equivalent to coclosed.

        Simplified: we use divergence-like proxy.

        Args:
            phi: 3-form values (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            metric: metric tensor (batch, 7, 7)

        Returns:
            norm_sq: ||d* phi||^2 (batch,)
        """
        batch = phi.shape[0]
        device = phi.device
        eps = 1e-4

        # Simplified: divergence proxy
        g_inv = torch.linalg.inv(metric)
        det_g = torch.det(metric)
        sqrt_det_g = torch.sqrt(det_g.abs())

        d_star_sq = torch.zeros(batch, device=device)

        for i in range(7):
            x_plus = x.clone()
            x_plus[:, i] += eps
            x_minus = x.clone()
            x_minus[:, i] -= eps

            phi_plus = phi_fn(x_plus)
            phi_minus = phi_fn(x_minus)

            # Metric-weighted divergence
            grad_i = (phi_plus - phi_minus) / (2 * eps)
            weighted = grad_i * sqrt_det_g.unsqueeze(-1)
            d_star_sq += (weighted ** 2).sum(dim=-1)

        return d_star_sq / (sqrt_det_g ** 2 + 1e-10)

    def forward(self, phi: torch.Tensor, x: torch.Tensor,
                phi_fn: Callable, metric: torch.Tensor) -> torch.Tensor:
        """Compute torsion functional F[phi].

        Args:
            phi: 3-form (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            metric: metric tensor (batch, 7, 7)

        Returns:
            F: functional value (batch,)
        """
        d_phi_sq = self.exterior_derivative_norm(phi, x, phi_fn)
        d_star_phi_sq = self.codifferential_norm(phi, x, phi_fn, metric)

        return d_phi_sq + d_star_phi_sq

    def torsion_norm(self, phi: torch.Tensor, x: torch.Tensor,
                     phi_fn: Callable, metric: torch.Tensor) -> torch.Tensor:
        """Compute torsion magnitude ||T||.

        Args:
            phi: 3-form (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            metric: metric tensor (batch, 7, 7)

        Returns:
            torsion: ||T|| (batch,)
        """
        F = self.forward(phi, x, phi_fn, metric)
        return torch.sqrt(F / 7)  # Normalize by dimension


# =============================================================================
# Combined variational loss
# =============================================================================

class VariationalLoss(nn.Module):
    """Combined loss for GIFT variational problem.

    L = L_torsion + lambda_1 * L_det + lambda_2 * L_positivity + lambda_3 * L_cohomology

    where:
    - L_torsion: (||T|| - kappa_T)^2, targeting kappa_T = 1/61
    - L_det: (det(g) - 65/32)^2
    - L_positivity: sum of negative eigenvalues of g
    - L_cohomology: ||G - I||^2 for Gram matrices
    """

    def __init__(self, config: GIFTConfig = None):
        super().__init__()
        self.config = config or default_config

        self.torsion_fn = TorsionFunctional(config)
        self.constraints = GIFTConstraints(config)
        self.metric_extractor = MetricFromPhi()

    def forward(self, phi: torch.Tensor, x: torch.Tensor,
                phi_fn: Callable,
                weights: Dict[str, float],
                omega: Optional[torch.Tensor] = None,
                Phi: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total variational loss.

        Args:
            phi: 3-form (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            weights: loss component weights
            omega: optional H2 forms (batch, 21, 21)
            Phi: optional H3 forms (batch, 77, 35)

        Returns:
            total_loss, loss_dict
        """
        device = phi.device
        metric = self.metric_extractor(phi)

        losses = {}
        total = torch.tensor(0.0, device=device)

        # 1. Torsion loss: (||T|| - kappa_T)^2
        torsion = self.torsion_fn.torsion_norm(phi, x, phi_fn, metric)
        target_torsion = self.config.kappa_T
        torsion_loss = ((torsion - target_torsion) ** 2).mean()
        losses['torsion'] = torsion_loss.item()
        losses['torsion_value'] = torsion.mean().item()

        if 'torsion' in weights:
            total = total + weights['torsion'] * torsion_loss

        # 2. Determinant constraint
        det_g = torch.det(metric)
        det_loss = ((det_g - self.config.det_g_target) ** 2).mean()
        losses['det'] = det_loss.item()
        losses['det_value'] = det_g.mean().item()

        if 'det' in weights:
            total = total + weights['det'] * det_loss

        # 3. Positivity constraint
        eigenvalues = torch.linalg.eigvalsh(metric)
        min_eig = eigenvalues.min(dim=-1).values
        positivity_loss = torch.relu(-min_eig + self.config.positivity_eps).mean()
        losses['positivity'] = positivity_loss.item()
        losses['min_eigenvalue'] = min_eig.mean().item()

        if 'positivity' in weights:
            total = total + weights['positivity'] * positivity_loss

        # 4. Cohomology constraints (if forms provided)
        if omega is not None:
            h2_loss = self._gram_loss(omega, metric)
            losses['h2'] = h2_loss.item()
            if 'cohomology' in weights:
                total = total + weights['cohomology'] * h2_loss

        if Phi is not None:
            h3_loss = self._gram_loss(Phi, metric)
            losses['h3'] = h3_loss.item()
            if 'cohomology' in weights:
                total = total + weights['cohomology'] * h3_loss

        losses['total'] = total.item()

        return total, losses

    def _gram_loss(self, forms: torch.Tensor,
                   metric: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix orthonormality loss.

        Args:
            forms: (batch, n_modes, n_components)
            metric: (batch, 7, 7)

        Returns:
            loss: ||G - I||^2 / n^2
        """
        batch, n_modes, n_comp = forms.shape

        det_g = torch.det(metric)
        vol = torch.sqrt(det_g.abs())

        weighted = forms * vol.unsqueeze(-1).unsqueeze(-1)
        G = torch.einsum('bic,bjc->ij', weighted, forms) / batch

        I = torch.eye(n_modes, device=forms.device)
        return ((G - I) ** 2).mean()


# =============================================================================
# Phase-aware loss manager
# =============================================================================

class PhasedLossManager:
    """Manages loss weights across training phases.

    Each phase focuses on different aspects:
    1. Initialization: establish G2 structure
    2. Constraint satisfaction: det(g) = 65/32
    3. Torsion targeting: kappa_T = 1/61
    4. Cohomology refinement: (b2, b3) = (21, 77)
    """

    def __init__(self, config: GIFTConfig = None):
        self.config = config or default_config
        self.current_phase = 0
        self.epoch_in_phase = 0

    def get_weights(self) -> Dict[str, float]:
        """Get loss weights for current phase."""
        if self.current_phase < len(self.config.phases):
            return self.config.phases[self.current_phase]["weights"]
        return self.config.phases[-1]["weights"]

    def get_phase_name(self) -> str:
        """Get current phase name."""
        if self.current_phase < len(self.config.phases):
            return self.config.phases[self.current_phase]["name"]
        return "final"

    def get_phase_epochs(self) -> int:
        """Get epochs for current phase."""
        if self.current_phase < len(self.config.phases):
            return self.config.phases[self.current_phase]["epochs"]
        return 1000

    def step(self) -> bool:
        """Advance one epoch, return True if phase changed."""
        self.epoch_in_phase += 1

        if self.epoch_in_phase >= self.get_phase_epochs():
            self.current_phase += 1
            self.epoch_in_phase = 0
            return True

        return False

    def reset(self):
        """Reset to initial state."""
        self.current_phase = 0
        self.epoch_in_phase = 0

    @property
    def total_epochs(self) -> int:
        """Total epochs across all phases."""
        return sum(p["epochs"] for p in self.config.phases)


# =============================================================================
# Loss logging utilities
# =============================================================================

def format_loss_dict(losses: Dict[str, float], precision: int = 6) -> str:
    """Format loss dictionary for logging.

    Args:
        losses: loss dictionary
        precision: decimal places

    Returns:
        formatted string
    """
    parts = []
    for key, value in losses.items():
        if key == 'total':
            parts.insert(0, f"L={value:.{precision}f}")
        elif 'value' in key:
            parts.append(f"{key.replace('_value', '')}={value:.{precision}f}")
        else:
            parts.append(f"L_{key}={value:.{precision}f}")

    return " | ".join(parts)


def log_constraints(losses: Dict[str, float], config: GIFTConfig = None) -> str:
    """Log constraint satisfaction status.

    Args:
        losses: loss dictionary with _value entries
        config: GIFT configuration

    Returns:
        status string
    """
    config = config or default_config

    lines = []

    if 'det_value' in losses:
        det = losses['det_value']
        target = config.det_g_target
        error = 100 * abs(det - target) / target
        status = "OK" if error < 0.1 else "FAIL"
        lines.append(f"det(g): {det:.6f} (target: {target:.6f}, error: {error:.2f}%) [{status}]")

    if 'torsion_value' in losses:
        torsion = losses['torsion_value']
        target = config.kappa_T
        error = 100 * abs(torsion - target) / target
        status = "OK" if error < 5 else "FAIL"
        lines.append(f"kappa_T: {torsion:.6f} (target: {target:.6f}, error: {error:.1f}%) [{status}]")

    if 'min_eigenvalue' in losses:
        min_eig = losses['min_eigenvalue']
        status = "OK" if min_eig > 0 else "FAIL"
        lines.append(f"min(eigenvalue): {min_eig:.6f} [{status}]")

    return "\n".join(lines)
