"""Constraint functions for GIFT v2.2 variational problem.

Implements the constraint enforcement for:
- det(g) = 65/32 (metric determinant)
- kappa_T = 1/61 (torsion magnitude)
- phi in Lambda^3_+ (G2 positivity)
- (b2, b3) = (21, 77) (cohomology)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from itertools import combinations

from .config import GIFTConfig, default_config
from .g2_geometry import MetricFromPhi, G2Positivity


# =============================================================================
# Determinant constraint: det(g) = 65/32
# =============================================================================

class DeterminantConstraint(nn.Module):
    """Constraint for metric determinant det(g(phi)) = 65/32.

    The metric g is induced from phi via the G2 contraction formula.
    This constraint fixes the "volume scale" of the geometry.

    GIFT derivation: det(g) = 65/32 comes from h* = 99 structure.
    """

    def __init__(self, target: float = 65/32, config: GIFTConfig = None):
        super().__init__()
        self.target = target
        self.config = config or default_config
        self.metric_extractor = MetricFromPhi()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute determinant of induced metric.

        Args:
            phi: 3-form (batch, 35)

        Returns:
            det_g: (batch,)
        """
        g = self.metric_extractor(phi)
        return torch.det(g)

    def loss(self, phi: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute constraint violation loss.

        Args:
            phi: 3-form (batch, 35)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: constraint violation
        """
        det_g = self.forward(phi)
        violation = (det_g - self.target) ** 2

        if reduction == 'mean':
            return violation.mean()
        elif reduction == 'sum':
            return violation.sum()
        else:
            return violation

    def relative_error(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute relative error from target.

        Args:
            phi: 3-form (batch, 35)

        Returns:
            error: |det(g) - target| / target
        """
        det_g = self.forward(phi)
        return (det_g - self.target).abs() / self.target


# =============================================================================
# Torsion constraint: kappa_T = 1/61
# =============================================================================

class TorsionConstraint(nn.Module):
    """Constraint for torsion magnitude kappa_T = 1/61.

    For a G2 structure phi:
    - Torsion-free: d(phi) = 0 and d(*phi) = 0
    - GIFT target: non-zero torsion with ||T|| = 1/61

    The torsion measures how far from exact G2 holonomy.
    GIFT predicts a specific non-zero value from topology.
    """

    def __init__(self, target: float = 1/61, config: GIFTConfig = None):
        super().__init__()
        self.target = target
        self.config = config or default_config

    def compute_torsion_norm(self, phi: torch.Tensor, x: torch.Tensor,
                              phi_fn: callable) -> torch.Tensor:
        """Estimate torsion magnitude via finite differences.

        Computes ||d phi|| as proxy for torsion.

        Args:
            phi: 3-form values (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi at new points

        Returns:
            torsion_norm: (batch,)
        """
        batch = phi.shape[0]
        device = phi.device
        eps = 1e-4

        # Estimate |d phi|^2 via finite differences
        d_phi_sq = torch.zeros(batch, device=device)

        for i in range(7):
            x_plus = x.clone()
            x_plus[:, i] += eps
            x_minus = x.clone()
            x_minus[:, i] -= eps

            phi_plus = phi_fn(x_plus)
            phi_minus = phi_fn(x_minus)

            dphi_di = (phi_plus - phi_minus) / (2 * eps)
            d_phi_sq += (dphi_di ** 2).sum(dim=-1)

        return torch.sqrt(d_phi_sq / 7)  # Normalize by dimension

    def loss(self, torsion_norm: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
        """Compute loss for achieving target torsion.

        Args:
            torsion_norm: computed torsion (batch,)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: (torsion - target)^2
        """
        violation = (torsion_norm - self.target) ** 2

        if reduction == 'mean':
            return violation.mean()
        elif reduction == 'sum':
            return violation.sum()
        else:
            return violation

    def relative_error(self, torsion_norm: torch.Tensor) -> torch.Tensor:
        """Compute relative error from target.

        Args:
            torsion_norm: computed torsion (batch,)

        Returns:
            error: |torsion - target| / target
        """
        return (torsion_norm.mean() - self.target).abs() / self.target


# =============================================================================
# Positivity constraint: phi in Lambda^3_+
# =============================================================================

class PositivityConstraint(nn.Module):
    """Constraint for G2 positivity: phi in Lambda^3_+.

    A 3-form phi defines a valid G2 structure iff:
    - The induced metric g(phi) is positive definite
    - phi lies in the open GL(7)-orbit Lambda^3_+

    Violation is measured by sum of negative eigenvalues of g.
    """

    def __init__(self, eps: float = 1e-6, config: GIFTConfig = None):
        super().__init__()
        self.eps = eps
        self.config = config or default_config
        self.metric_extractor = MetricFromPhi()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute positivity violation.

        Args:
            phi: 3-form (batch, 35)

        Returns:
            violation: (batch,) sum of negative eigenvalues
        """
        g = self.metric_extractor(phi)
        eigenvalues = torch.linalg.eigvalsh(g)
        negative_part = torch.relu(-eigenvalues + self.eps)
        return negative_part.sum(dim=-1)

    def loss(self, phi: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute positivity violation loss.

        Args:
            phi: 3-form (batch, 35)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: positivity violation
        """
        violation = self.forward(phi)

        if reduction == 'mean':
            return violation.mean()
        elif reduction == 'sum':
            return violation.sum()
        else:
            return violation

    def is_satisfied(self, phi: torch.Tensor) -> torch.Tensor:
        """Check if positivity constraint is satisfied.

        Args:
            phi: 3-form (batch, 35)

        Returns:
            satisfied: (batch,) boolean
        """
        return self.forward(phi) < self.eps


# =============================================================================
# Cohomology constraint: (b2, b3) = (21, 77)
# =============================================================================

class CohomologyConstraint(nn.Module):
    """Constraint for Betti numbers (b2, b3) = (21, 77).

    This is enforced via the harmonic forms network:
    - Extract Gram matrices for H2 and H3 forms
    - Gram matrix should be approximately identity (orthonormal forms)
    - Effective rank should match target Betti numbers

    Unlike other constraints, this is verified rather than directly enforced.
    """

    def __init__(self, b2_target: int = 21, b3_target: int = 77,
                 config: GIFTConfig = None):
        super().__init__()
        self.b2_target = b2_target
        self.b3_target = b3_target
        self.config = config or default_config

    def orthonormality_loss(self, G: torch.Tensor) -> torch.Tensor:
        """Loss for Gram matrix deviation from identity.

        Args:
            G: Gram matrix (n, n)

        Returns:
            loss: ||G - I||^2 / n^2
        """
        n = G.shape[0]
        I = torch.eye(n, device=G.device)
        return ((G - I) ** 2).mean()

    def effective_rank(self, G: torch.Tensor, threshold: float = 0.01) -> int:
        """Compute effective rank of Gram matrix.

        Args:
            G: Gram matrix (n, n)
            threshold: eigenvalue threshold for counting

        Returns:
            rank: number of eigenvalues above threshold
        """
        eigenvalues = torch.linalg.eigvalsh(G)
        max_eig = eigenvalues.max()
        return (eigenvalues > threshold * max_eig).sum().item()

    def loss_h2(self, omega: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Compute H2 cohomology loss.

        Args:
            omega: 2-forms (batch, 21, 21)
            metric: metric tensor (batch, 7, 7)

        Returns:
            loss: orthonormality deviation
        """
        # Compute Gram matrix
        det_g = torch.det(metric)
        vol = torch.sqrt(det_g.abs())

        weighted = omega * vol.unsqueeze(-1).unsqueeze(-1)
        G = torch.einsum('bic,bjc->ij', weighted, omega) / omega.shape[0]

        return self.orthonormality_loss(G)

    def loss_h3(self, Phi: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Compute H3 cohomology loss.

        Args:
            Phi: 3-forms (batch, 77, 35)
            metric: metric tensor (batch, 7, 7)

        Returns:
            loss: orthonormality deviation
        """
        det_g = torch.det(metric)
        vol = torch.sqrt(det_g.abs())

        weighted = Phi * vol.unsqueeze(-1).unsqueeze(-1)
        G = torch.einsum('bic,bjc->ij', weighted, Phi) / Phi.shape[0]

        return self.orthonormality_loss(G)


# =============================================================================
# Combined constraint handler
# =============================================================================

class GIFTConstraints(nn.Module):
    """Combined constraint handler for all GIFT v2.2 constraints.

    Manages:
    - Determinant: det(g) = 65/32
    - Torsion: kappa_T = 1/61
    - Positivity: phi in Lambda^3_+
    - Cohomology: (b2, b3) = (21, 77)
    """

    def __init__(self, config: GIFTConfig = None):
        super().__init__()
        self.config = config or default_config

        self.det_constraint = DeterminantConstraint(
            target=self.config.det_g_target, config=config)
        self.torsion_constraint = TorsionConstraint(
            target=self.config.kappa_T, config=config)
        self.positivity_constraint = PositivityConstraint(config=config)
        self.cohomology_constraint = CohomologyConstraint(
            b2_target=self.config.b2_K7,
            b3_target=self.config.b3_K7,
            config=config)

    def compute_all(self, phi: torch.Tensor, x: torch.Tensor,
                    phi_fn: callable,
                    omega: Optional[torch.Tensor] = None,
                    Phi: Optional[torch.Tensor] = None,
                    metric: Optional[torch.Tensor] = None
                    ) -> Dict[str, torch.Tensor]:
        """Compute all constraint values.

        Args:
            phi: 3-form (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            omega: optional H2 forms (batch, 21, 21)
            Phi: optional H3 forms (batch, 77, 35)
            metric: optional metric (batch, 7, 7)

        Returns:
            dict with all constraint values and losses
        """
        results = {}

        # Determinant
        det_g = self.det_constraint(phi)
        results['det_g'] = det_g.mean()
        results['det_loss'] = self.det_constraint.loss(phi)
        results['det_error'] = self.det_constraint.relative_error(phi).mean()

        # Torsion
        torsion = self.torsion_constraint.compute_torsion_norm(phi, x, phi_fn)
        results['torsion'] = torsion.mean()
        results['torsion_loss'] = self.torsion_constraint.loss(torsion)
        results['torsion_error'] = self.torsion_constraint.relative_error(torsion)

        # Positivity
        pos_violation = self.positivity_constraint(phi)
        results['positivity_violation'] = pos_violation.mean()
        results['positivity_loss'] = self.positivity_constraint.loss(phi)
        results['is_positive'] = self.positivity_constraint.is_satisfied(phi).float().mean()

        # Cohomology (if forms provided)
        if omega is not None and metric is not None:
            results['h2_loss'] = self.cohomology_constraint.loss_h2(omega, metric)
        if Phi is not None and metric is not None:
            results['h3_loss'] = self.cohomology_constraint.loss_h3(Phi, metric)

        return results

    def total_loss(self, phi: torch.Tensor, x: torch.Tensor,
                   phi_fn: callable,
                   weights: Dict[str, float],
                   omega: Optional[torch.Tensor] = None,
                   Phi: Optional[torch.Tensor] = None,
                   metric: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted total constraint loss.

        Args:
            phi: 3-form (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi
            weights: loss weights dict
            omega: optional H2 forms
            Phi: optional H3 forms
            metric: optional metric

        Returns:
            total_loss, loss_dict
        """
        results = self.compute_all(phi, x, phi_fn, omega, Phi, metric)

        total = torch.tensor(0.0, device=phi.device)

        # Core losses
        if 'det' in weights:
            total = total + weights['det'] * results['det_loss']
        if 'torsion' in weights:
            total = total + weights['torsion'] * results['torsion_loss']
        if 'positivity' in weights:
            total = total + weights['positivity'] * results['positivity_loss']

        # Cohomology losses
        if 'cohomology' in weights:
            if 'h2_loss' in results:
                total = total + weights['cohomology'] * results['h2_loss']
            if 'h3_loss' in results:
                total = total + weights['cohomology'] * results['h3_loss']

        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in results.items()}
        loss_dict['total'] = total.item()

        return total, loss_dict
