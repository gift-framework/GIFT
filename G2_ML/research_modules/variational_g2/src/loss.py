"""
Variational Loss Functions for G2 Problem

This module implements the composite loss function for the GIFT v2.2
variational problem:

    L = L_torsion + lambda_1 * L_det + lambda_2 * L_cohomology + lambda_3 * L_positivity

where:
    - L_torsion = ||d*phi||^2 + ||d*phi||^2  (PRIMARY - what we minimize)
    - L_det = |det(g) - 65/32|^2              (CONSTRAINT)
    - L_cohomology = penalty for wrong (b2, b3) (CONSTRAINT)
    - L_positivity = penalty if phi not in G2 cone (CONSTRAINT)

The loss hierarchy matters:
1. Primary objective: minimize torsion (toward kappa_T = 1/61)
2. Hard constraints: det(g) = 65/32, positivity
3. Soft constraints: cohomology (b2, b3) via harmonic extraction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .constraints import (
    metric_from_phi,
    det_constraint_loss,
    g2_positivity_check,
    phi_norm_squared,
    expand_to_antisymmetric,
)


@dataclass
class LossWeights:
    """Weights for different loss components."""
    torsion: float = 1.0
    det: float = 1.0
    positivity: float = 1.0
    cohomology: float = 0.0
    phi_norm: float = 0.1  # ||phi||^2 = 7 regularization

    def to_dict(self) -> Dict[str, float]:
        return {
            'torsion': self.torsion,
            'det': self.det,
            'positivity': self.positivity,
            'cohomology': self.cohomology,
            'phi_norm': self.phi_norm,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'LossWeights':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LossOutput:
    """Output from loss computation."""
    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    metrics: Dict[str, torch.Tensor]


class TorsionLoss(nn.Module):
    """
    Torsion loss: ||T|| targeting kappa_T = 1/61.

    The torsion of a G2 structure measures deviation from being torsion-free.
    For GIFT v2.2, we target a specific non-zero value kappa_T = 1/61.

    We use a simplified torsion computation based on the gradient of phi.
    """

    def __init__(
        self,
        target_kappa: float = 1.0 / 61.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.target_kappa = target_kappa
        self.reduction = reduction

    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        d_phi: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute torsion loss.

        Args:
            phi: 3-form of shape (batch, 7, 7, 7) or (batch, 35)
            x: Coordinates of shape (batch, 7), requires_grad=True
            d_phi: Optional precomputed derivative

        Returns:
            loss: Scalar loss value
            torsion_norm: Actual torsion magnitude
        """
        if phi.shape[-1] == 35:
            phi = expand_to_antisymmetric(phi)

        batch_size = x.shape[0]

        if d_phi is None:
            # Compute derivative of phi w.r.t. x
            # d_phi[b,l,i,j,k] = partial_l phi_{ijk}
            d_phi = self._compute_derivative(phi, x)

        # Torsion from d*phi (simplified)
        # ||d*phi||^2 ~ sum over indices of (partial_l phi_{ijk})^2
        torsion_sq = torch.sum(d_phi ** 2, dim=(-1, -2, -3, -4))
        torsion_norm = torch.sqrt(torsion_sq + 1e-10)

        # Loss: (||T|| - kappa_T)^2
        loss = (torsion_norm - self.target_kappa) ** 2

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss, torsion_norm.mean()

    def _compute_derivative(
        self,
        phi: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of phi w.r.t. x."""
        batch_size = x.shape[0]
        d_phi = torch.zeros(batch_size, 7, 7, 7, 7, device=x.device, dtype=x.dtype)

        # Only compute for independent components
        for i in range(7):
            for j in range(i + 1, 7):
                for k in range(j + 1, 7):
                    grad = torch.autograd.grad(
                        phi[:, i, j, k].sum(),
                        x,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]

                    if grad is not None:
                        # Fill antisymmetric positions
                        d_phi[:, :, i, j, k] = grad
                        d_phi[:, :, i, k, j] = -grad
                        d_phi[:, :, j, i, k] = -grad
                        d_phi[:, :, j, k, i] = grad
                        d_phi[:, :, k, i, j] = grad
                        d_phi[:, :, k, j, i] = -grad

        return d_phi


class DeterminantLoss(nn.Module):
    """
    Determinant constraint loss: det(g) = 65/32.

    The metric determinant is a topological invariant derived from
    the cohomological dimension h* = 99.
    """

    def __init__(
        self,
        target_det: float = 65.0 / 32.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.target_det = target_det
        self.reduction = reduction

    def forward(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute determinant constraint loss.

        Args:
            phi: 3-form tensor

        Returns:
            loss: Scalar loss
            det_g: Actual determinant value
        """
        if phi.shape[-1] == 35:
            phi = expand_to_antisymmetric(phi)

        g = metric_from_phi(phi)
        det_g = torch.det(g)

        # Squared error
        loss = (det_g - self.target_det) ** 2

        if self.reduction == 'mean':
            loss = loss.mean()
            det_g_mean = det_g.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            det_g_mean = det_g.mean()
        else:
            det_g_mean = det_g

        return loss, det_g_mean


class PositivityLoss(nn.Module):
    """
    Positivity loss: ensure phi is in G2 cone.

    A 3-form is positive (defines a G2 structure) iff the induced
    metric is positive definite.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positivity violation loss.

        Args:
            phi: 3-form tensor

        Returns:
            loss: Scalar loss (0 if positive)
            min_eigenvalue: Minimum eigenvalue of metric
        """
        if phi.shape[-1] == 35:
            phi = expand_to_antisymmetric(phi)

        g = metric_from_phi(phi)
        eigenvalues = torch.linalg.eigvalsh(g)

        # Penalty for negative eigenvalues
        violation = torch.relu(-eigenvalues)
        loss = violation.sum(dim=-1)

        min_eig = eigenvalues.min(dim=-1)[0]

        if self.reduction == 'mean':
            loss = loss.mean()
            min_eig = min_eig.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            min_eig = min_eig.mean()

        return loss, min_eig


class PhiNormLoss(nn.Module):
    """
    Phi norm regularization: ||phi||^2_g = 7.

    This is the G2 identity that holds for valid G2 structures.
    """

    def __init__(self, target_norm_sq: float = 7.0, reduction: str = 'mean'):
        super().__init__()
        self.target_norm_sq = target_norm_sq
        self.reduction = reduction

    def forward(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phi norm loss.

        Args:
            phi: 3-form tensor

        Returns:
            loss: Scalar loss
            norm_sq: Actual ||phi||^2_g
        """
        if phi.shape[-1] == 35:
            phi = expand_to_antisymmetric(phi)

        norm_sq = phi_norm_squared(phi)

        loss = (norm_sq - self.target_norm_sq) ** 2

        if self.reduction == 'mean':
            loss = loss.mean()
            norm_sq = norm_sq.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            norm_sq = norm_sq.mean()

        return loss, norm_sq


class CohomologyLoss(nn.Module):
    """
    Cohomology loss: (b2, b3) = (21, 77).

    This is computed via harmonic form extraction and is expensive.
    Only used in later training phases.
    """

    def __init__(
        self,
        target_b2: int = 21,
        target_b3: int = 77,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.target_b2 = target_b2
        self.target_b3 = target_b3
        self.reduction = reduction

    def forward(
        self,
        b2_effective: torch.Tensor,
        b3_effective: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute cohomology loss from effective Betti numbers.

        Args:
            b2_effective: Effective second Betti number
            b3_effective: Effective third Betti number

        Returns:
            loss: Scalar loss
            (b2, b3): Actual Betti numbers
        """
        loss_b2 = (b2_effective - self.target_b2) ** 2
        loss_b3 = (b3_effective - self.target_b3) ** 2

        loss = loss_b2 + loss_b3

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss, (b2_effective, b3_effective)


class VariationalLoss(nn.Module):
    """
    Complete variational loss for G2 problem.

    Combines all loss components with configurable weights:
        L = w_torsion * L_torsion
          + w_det * L_det
          + w_positivity * L_positivity
          + w_cohomology * L_cohomology
          + w_phi_norm * L_phi_norm
    """

    def __init__(
        self,
        weights: Optional[LossWeights] = None,
        target_det: float = 65.0 / 32.0,
        target_kappa: float = 1.0 / 61.0,
        target_b2: int = 21,
        target_b3: int = 77,
    ):
        super().__init__()

        self.weights = weights or LossWeights()

        # Individual loss components
        self.torsion_loss = TorsionLoss(target_kappa=target_kappa)
        self.det_loss = DeterminantLoss(target_det=target_det)
        self.positivity_loss = PositivityLoss()
        self.phi_norm_loss = PhiNormLoss()
        self.cohomology_loss = CohomologyLoss(target_b2=target_b2, target_b3=target_b3)

        # Targets for logging
        self.target_det = target_det
        self.target_kappa = target_kappa

    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        d_phi: Optional[torch.Tensor] = None,
        b2_effective: Optional[torch.Tensor] = None,
        b3_effective: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute total loss with all components.

        Args:
            phi: 3-form tensor of shape (batch, 35) or (batch, 7, 7, 7)
            x: Coordinates of shape (batch, 7), requires grad
            d_phi: Optional precomputed derivative
            b2_effective: Optional effective b2 (expensive to compute)
            b3_effective: Optional effective b3 (expensive to compute)

        Returns:
            LossOutput with total loss, components, and metrics
        """
        components = {}
        metrics = {}

        # Expand phi if needed
        if phi.shape[-1] == 35:
            phi_full = expand_to_antisymmetric(phi)
        else:
            phi_full = phi

        # Torsion loss (primary objective)
        if self.weights.torsion > 0:
            loss_torsion, torsion_norm = self.torsion_loss(phi_full, x, d_phi)
            components['torsion'] = loss_torsion * self.weights.torsion
            metrics['torsion_norm'] = torsion_norm
            metrics['torsion_error'] = torch.abs(torsion_norm - self.target_kappa)
        else:
            components['torsion'] = torch.tensor(0.0, device=phi.device)

        # Determinant loss
        if self.weights.det > 0:
            loss_det, det_g = self.det_loss(phi_full)
            components['det'] = loss_det * self.weights.det
            metrics['det_g'] = det_g
            metrics['det_error'] = torch.abs(det_g - self.target_det)
        else:
            components['det'] = torch.tensor(0.0, device=phi.device)

        # Positivity loss
        if self.weights.positivity > 0:
            loss_pos, min_eig = self.positivity_loss(phi_full)
            components['positivity'] = loss_pos * self.weights.positivity
            metrics['min_eigenvalue'] = min_eig
        else:
            components['positivity'] = torch.tensor(0.0, device=phi.device)

        # Phi norm loss
        if self.weights.phi_norm > 0:
            loss_norm, norm_sq = self.phi_norm_loss(phi_full)
            components['phi_norm'] = loss_norm * self.weights.phi_norm
            metrics['phi_norm_sq'] = norm_sq
        else:
            components['phi_norm'] = torch.tensor(0.0, device=phi.device)

        # Cohomology loss (only if Betti numbers provided)
        if self.weights.cohomology > 0 and b2_effective is not None:
            loss_coh, (b2, b3) = self.cohomology_loss(b2_effective, b3_effective)
            components['cohomology'] = loss_coh * self.weights.cohomology
            metrics['b2_effective'] = b2
            metrics['b3_effective'] = b3
        else:
            components['cohomology'] = torch.tensor(0.0, device=phi.device)

        # Total loss
        total = sum(components.values())

        return LossOutput(
            total=total,
            components=components,
            metrics=metrics
        )

    def update_weights(self, new_weights: LossWeights):
        """Update loss weights (for phased training)."""
        self.weights = new_weights

    def get_weights(self) -> LossWeights:
        """Get current loss weights."""
        return self.weights


def create_loss(config: Dict[str, Any]) -> VariationalLoss:
    """
    Create loss function from configuration.

    Args:
        config: Configuration dictionary with physics and training sections

    Returns:
        Configured VariationalLoss instance
    """
    physics = config.get('physics', {})

    return VariationalLoss(
        target_det=physics.get('det_g', 65.0 / 32.0),
        target_kappa=physics.get('kappa_T', 1.0 / 61.0),
        target_b2=physics.get('b2', 21),
        target_b3=physics.get('b3', 77),
    )
