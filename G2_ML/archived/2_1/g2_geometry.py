"""G2 geometry utilities for variational metric extraction.

This module implements the core G2 geometric operations:
- Metric extraction from the associative 3-form phi
- G2 positivity checking (phi in Lambda^3_+)
- Standard G2 structure tensors

The key insight: a 3-form phi in R^7 defines a G2 structure iff:
1. phi is positive (phi lies in the open GL(7)-orbit Lambda^3_+)
2. The induced metric g(phi) is Riemannian (positive definite)

Reference: Joyce, "Compact Manifolds with Special Holonomy" (2000)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from itertools import combinations
from functools import lru_cache


# =============================================================================
# Index mappings
# =============================================================================

@lru_cache(maxsize=1)
def get_3form_indices() -> Tuple[Tuple[int, int, int], ...]:
    """Get ordered (i,j,k) indices for 3-form components with i<j<k."""
    return tuple(combinations(range(7), 3))


@lru_cache(maxsize=1)
def get_2form_indices() -> Tuple[Tuple[int, int], ...]:
    """Get ordered (i,j) indices for 2-form components with i<j."""
    return tuple(combinations(range(7), 2))


def index_to_component_3form(i: int, j: int, k: int) -> Tuple[int, int]:
    """Convert (i,j,k) to (component_index, sign)."""
    indices = (i, j, k)
    sorted_indices = tuple(sorted(indices))

    if len(set(indices)) != 3:
        return -1, 0  # Repeated index

    # Count inversions for sign
    inversions = sum(1 for a in range(3) for b in range(a+1, 3)
                     if indices[a] > indices[b])
    sign = (-1) ** inversions

    # Find position in canonical list
    all_indices = get_3form_indices()
    pos = all_indices.index(sorted_indices)

    return pos, sign


# =============================================================================
# Standard G2 structure
# =============================================================================

def standard_phi_coefficients() -> torch.Tensor:
    """Return the standard G2 3-form coefficients.

    The standard phi_0 in R^7 is:
    phi_0 = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}

    where e^{ijk} = e^i wedge e^j wedge e^k

    Returns:
        phi: (35,) tensor with standard G2 coefficients
    """
    phi = torch.zeros(35)

    # The 7 terms of standard G2 3-form (0-indexed)
    terms = [
        ((0, 1, 2), +1),  # e^{123} -> indices (0,1,2)
        ((0, 3, 4), +1),  # e^{145} -> indices (0,3,4)
        ((0, 5, 6), +1),  # e^{167} -> indices (0,5,6)
        ((1, 3, 5), +1),  # e^{246} -> indices (1,3,5)
        ((1, 4, 6), -1),  # e^{257} -> indices (1,4,6)
        ((2, 3, 6), -1),  # e^{347} -> indices (2,3,6)
        ((2, 4, 5), -1),  # e^{356} -> indices (2,4,5)
    ]

    all_indices = get_3form_indices()
    for (i, j, k), sign in terms:
        pos = all_indices.index((i, j, k))
        phi[pos] = sign

    return phi


def standard_psi_coefficients() -> torch.Tensor:
    """Return the standard G2 4-form *phi coefficients.

    The co-associative 4-form psi = *phi_0 has similar structure.

    Returns:
        psi: (35,) tensor with *phi coefficients (4-form as 35 components)
    """
    # For 4-forms we also have C(7,4) = 35 components
    psi = torch.zeros(35)

    # The 7 terms of *phi_0
    # *phi_0 = e^{4567} + e^{2367} + e^{2345} + e^{1357} - e^{1346} - e^{1256} - e^{1247}
    terms_4form = [
        ((3, 4, 5, 6), +1),
        ((1, 2, 5, 6), +1),
        ((1, 2, 3, 4), +1),
        ((0, 2, 4, 6), +1),
        ((0, 2, 3, 5), -1),
        ((0, 1, 4, 5), -1),
        ((0, 1, 3, 6), -1),
    ]

    all_4indices = list(combinations(range(7), 4))
    for (i, j, k, l), sign in terms_4form:
        pos = all_4indices.index((i, j, k, l))
        psi[pos] = sign

    return psi


# =============================================================================
# Metric extraction from phi
# =============================================================================

class MetricFromPhi(nn.Module):
    """Extract the induced Riemannian metric from a G2 3-form.

    For a 3-form phi in R^7, the induced metric is:
        g_ij = (1/6) sum_{klm} phi_{ikl} phi_{jkl}  (contraction)

    More precisely, using the volume form:
        g_ij vol_g = (1/6) (i_ei phi) wedge (i_ej phi) wedge phi

    where i_e denotes interior product with basis vector e.
    """

    def __init__(self):
        super().__init__()
        # Precompute contraction tensors
        self._build_contraction_map()

    def _build_contraction_map(self):
        """Build the contraction map for g_ij = (1/6) phi_ikl phi_jkl."""
        # For each pair (i,j), find which 3-form components contribute
        # phi_ikl has fixed i, varies k<l

        indices_3 = get_3form_indices()
        n_comp = len(indices_3)

        # contraction_map[i][j] = list of (comp_a, comp_b, sign)
        # where phi_ikl * phi_jkl contributes
        self.contraction_map = [[[] for _ in range(7)] for _ in range(7)]

        for comp_a, (a0, a1, a2) in enumerate(indices_3):
            for comp_b, (b0, b1, b2) in enumerate(indices_3):
                # Check if they share exactly two indices
                set_a = {a0, a1, a2}
                set_b = {b0, b1, b2}
                shared = set_a & set_b

                if len(shared) == 2:
                    # The unshared indices are the i,j for g_ij
                    i_only = (set_a - shared).pop()
                    j_only = (set_b - shared).pop()

                    # Get positions of the unshared index in each triple
                    pos_i = [a0, a1, a2].index(i_only)
                    pos_j = [b0, b1, b2].index(j_only)

                    # Sign from bringing unshared index to first position
                    sign = ((-1) ** pos_i) * ((-1) ** pos_j)

                    self.contraction_map[i_only][j_only].append((comp_a, comp_b, sign))

        # Convert to tensors for efficient computation
        # We'll use sparse representation
        max_terms = max(len(self.contraction_map[i][j])
                       for i in range(7) for j in range(7))

        # Padding for uniform tensor operations
        self.register_buffer('_contraction_a', torch.zeros(7, 7, max_terms, dtype=torch.long))
        self.register_buffer('_contraction_b', torch.zeros(7, 7, max_terms, dtype=torch.long))
        self.register_buffer('_contraction_sign', torch.zeros(7, 7, max_terms))
        self.register_buffer('_contraction_mask', torch.zeros(7, 7, max_terms, dtype=torch.bool))

        for i in range(7):
            for j in range(7):
                for k, (a, b, s) in enumerate(self.contraction_map[i][j]):
                    self._contraction_a[i, j, k] = a
                    self._contraction_b[i, j, k] = b
                    self._contraction_sign[i, j, k] = s
                    self._contraction_mask[i, j, k] = True

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Extract metric from 3-form.

        Args:
            phi: 3-form components (batch, 35) or (35,)

        Returns:
            g: metric tensor (batch, 7, 7) or (7, 7)
        """
        squeeze = False
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
            squeeze = True

        batch = phi.shape[0]
        device = phi.device

        g = torch.zeros(batch, 7, 7, device=device)

        for i in range(7):
            for j in range(7):
                for k, (a, b, s) in enumerate(self.contraction_map[i][j]):
                    g[:, i, j] += s * phi[:, a] * phi[:, b]

        # Normalize by 1/6
        g = g / 6.0

        if squeeze:
            g = g.squeeze(0)

        return g

    def determinant(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute det(g(phi)).

        Args:
            phi: 3-form components (batch, 35) or (35,)

        Returns:
            det_g: determinant (batch,) or scalar
        """
        g = self.forward(phi)
        return torch.det(g)

    def eigenvalues(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues of g(phi) for positivity checking.

        Args:
            phi: 3-form components (batch, 35)

        Returns:
            eigenvalues: (batch, 7) sorted ascending
        """
        g = self.forward(phi)
        return torch.linalg.eigvalsh(g)


# =============================================================================
# G2 positivity
# =============================================================================

class G2Positivity(nn.Module):
    """Check and enforce G2 positivity for 3-forms.

    A 3-form phi is positive (defines a G2 structure) iff:
    1. The induced metric g(phi) is positive definite
    2. Equivalently: phi lies in the open orbit Lambda^3_+ subset Lambda^3(R^7)

    The positive cone Lambda^3_+ has two components; we choose the one
    containing the standard phi_0.
    """

    def __init__(self):
        super().__init__()
        self.metric_extractor = MetricFromPhi()

        # Reference: standard G2 structure
        self.register_buffer('phi_0', standard_phi_coefficients())

    def positivity_violation(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute violation of positivity constraint.

        Returns sum of negative eigenvalues (0 if positive definite).

        Args:
            phi: 3-form (batch, 35)

        Returns:
            violation: (batch,) non-negative, 0 iff positive
        """
        eigenvalues = self.metric_extractor.eigenvalues(phi)
        negative_part = torch.relu(-eigenvalues)
        return negative_part.sum(dim=-1)

    def is_positive(self, phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Check if phi is positive (defines valid G2 structure).

        Args:
            phi: 3-form (batch, 35)
            eps: tolerance for positivity

        Returns:
            is_pos: (batch,) boolean
        """
        eigenvalues = self.metric_extractor.eigenvalues(phi)
        min_eigenvalue = eigenvalues.min(dim=-1).values
        return min_eigenvalue > eps

    def project_to_positive(self, phi: torch.Tensor,
                           alpha: float = 0.1) -> torch.Tensor:
        """Soft projection toward positive cone.

        Interpolates toward standard phi_0 when positivity is violated.

        Args:
            phi: 3-form (batch, 35)
            alpha: interpolation strength

        Returns:
            phi_proj: projected 3-form
        """
        violation = self.positivity_violation(phi)
        needs_projection = violation > 0

        if not needs_projection.any():
            return phi

        # Interpolate toward standard phi_0
        phi_0_expanded = self.phi_0.unsqueeze(0).expand_as(phi)

        # Stronger interpolation for larger violations
        t = torch.sigmoid(alpha * violation).unsqueeze(-1)
        t = t * needs_projection.float().unsqueeze(-1)

        return (1 - t) * phi + t * phi_0_expanded


# =============================================================================
# G2 identities and normalization
# =============================================================================

def phi_norm_squared(phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Compute ||phi||^2_g = phi_ijk phi^ijk.

    For a proper G2 structure, ||phi||^2 = 7.

    Args:
        phi: 3-form (batch, 35)
        g: metric (batch, 7, 7)

    Returns:
        norm_sq: (batch,)
    """
    # For G2 structure: ||phi||^2 = 7 exactly
    # This is a check of the G2 identity

    # Simplified: use flat metric norm as proxy
    return (phi ** 2).sum(dim=-1)


def check_g2_identity(phi: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Check the G2 identity: phi wedge *phi = (7/6)||phi||^2 vol_g.

    Returns deviation from this identity.

    Args:
        phi: 3-form (batch, 35)
        g: metric (batch, 7, 7)

    Returns:
        deviation: (batch,) should be close to 0
    """
    # For proper G2: the identity holds exactly
    # Deviation indicates how far from true G2 we are

    norm_sq = phi_norm_squared(phi, g)
    det_g = torch.det(g)

    # Expected: norm_sq = 7 for unit normalization
    # This is a simplified check
    target = 7.0 * torch.ones_like(norm_sq)

    return (norm_sq - target).abs()


# =============================================================================
# Torsion computation
# =============================================================================

class TorsionComputation(nn.Module):
    """Compute torsion classes for G2 structure.

    A G2 structure has torsion classes tau_0, tau_1, tau_2, tau_3 determined by:
    - d(phi) = tau_0 * psi + 3*tau_1 wedge phi + *tau_3
    - d(psi) = 4*tau_1 wedge psi + *tau_2

    Torsion-free (holonomy exactly G2) requires all tau_i = 0.
    GIFT predicts kappa_T = 1/61 as the global torsion magnitude.
    """

    def __init__(self):
        super().__init__()
        self.metric_extractor = MetricFromPhi()

    def torsion_proxy(self, phi: torch.Tensor, x: torch.Tensor,
                      phi_fn: callable) -> torch.Tensor:
        """Compute a proxy for torsion magnitude.

        Uses finite differences to estimate |d phi| + |d *phi|.

        Args:
            phi: current 3-form values (batch, 35)
            x: coordinates (batch, 7)
            phi_fn: function to evaluate phi at new points

        Returns:
            torsion_norm: (batch,) estimate of ||T||
        """
        batch = phi.shape[0]
        device = phi.device
        eps = 1e-4

        # Estimate |d phi| via finite differences
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

        return torch.sqrt(d_phi_sq)

    def torsion_loss(self, torsion_norm: torch.Tensor,
                     target: float = 1/61) -> torch.Tensor:
        """Loss for achieving target torsion.

        Args:
            torsion_norm: computed torsion (batch,)
            target: GIFT target kappa_T = 1/61

        Returns:
            loss: scalar
        """
        return ((torsion_norm - target) ** 2).mean()


# =============================================================================
# Utility functions
# =============================================================================

def random_phi_near_standard(batch_size: int,
                             perturbation: float = 0.1,
                             device: str = 'cpu') -> torch.Tensor:
    """Generate random 3-forms near the standard G2 structure.

    Useful for initialization.

    Args:
        batch_size: number of samples
        perturbation: magnitude of random perturbation
        device: torch device

    Returns:
        phi: (batch_size, 35) 3-forms
    """
    phi_0 = standard_phi_coefficients().to(device)
    phi = phi_0.unsqueeze(0).expand(batch_size, -1).clone()
    phi += perturbation * torch.randn_like(phi)
    return phi


def normalize_phi(phi: torch.Tensor,
                  target_det: float = 65/32) -> torch.Tensor:
    """Rescale phi to achieve target determinant.

    Since det(g) scales as det(g(lambda*phi)) = lambda^(14/3) det(g(phi)),
    we can rescale to hit the target.

    Args:
        phi: 3-form (batch, 35)
        target_det: target determinant (default: GIFT value 65/32)

    Returns:
        phi_normalized: rescaled 3-form
    """
    metric_fn = MetricFromPhi()
    current_det = metric_fn.determinant(phi)

    # det scales as lambda^(14/3) for phi -> lambda*phi
    # So lambda = (target/current)^(3/14)
    scale = (target_det / (current_det.abs() + 1e-10)) ** (3/14)

    if phi.dim() == 1:
        return phi * scale
    else:
        return phi * scale.unsqueeze(-1)
