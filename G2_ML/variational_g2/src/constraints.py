"""
G2 Constraint Functions for Variational Problem

This module implements all constraint functions for the GIFT v2.2 variational problem:
    1. Metric extraction from 3-form (g_ij from phi)
    2. Determinant constraint (det(g) = 65/32)
    3. Torsion computation and target (kappa_T = 1/61)
    4. G2 positivity check (phi in Lambda^3_+)
    5. Exterior derivative and codifferential

The constraints are PRIMARY INPUTS - we solve for the geometry that satisfies them.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


# G2 structure constants: the 3-form phi on R7 in standard basis
# phi = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}
# where e^{ijk} = e^i wedge e^j wedge e^k

# Indices for the 7 terms of the standard G2 3-form (0-indexed)
G2_STANDARD_INDICES = [
    (0, 1, 2),  # e^{123}
    (0, 3, 4),  # e^{145}
    (0, 5, 6),  # e^{167}
    (1, 3, 5),  # e^{246}
    (1, 4, 6),  # e^{257} with sign -1
    (2, 3, 6),  # e^{347} with sign -1
    (2, 4, 5),  # e^{356} with sign -1
]

G2_STANDARD_SIGNS = [1, 1, 1, 1, -1, -1, -1]


def generate_3form_indices() -> torch.Tensor:
    """
    Generate all 35 independent indices for a 3-form on R7.

    Returns:
        Tensor of shape (35, 3) with indices (i, j, k) where i < j < k
    """
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append([i, j, k])
    return torch.tensor(indices, dtype=torch.long)


def expand_to_antisymmetric(phi_components: torch.Tensor) -> torch.Tensor:
    """
    Expand 35 independent components to full antisymmetric 7x7x7 tensor.

    Args:
        phi_components: Tensor of shape (..., 35) with independent components

    Returns:
        Tensor of shape (..., 7, 7, 7) fully antisymmetric
    """
    batch_shape = phi_components.shape[:-1]
    device = phi_components.device
    dtype = phi_components.dtype

    # Initialize full tensor
    phi_full = torch.zeros(*batch_shape, 7, 7, 7, device=device, dtype=dtype)

    # Get indices
    indices = generate_3form_indices().to(device)

    # Fill in values with antisymmetry
    for idx, (i, j, k) in enumerate(indices):
        val = phi_components[..., idx]
        # All 6 permutations with appropriate signs
        phi_full[..., i, j, k] = val
        phi_full[..., i, k, j] = -val
        phi_full[..., j, i, k] = -val
        phi_full[..., j, k, i] = val
        phi_full[..., k, i, j] = val
        phi_full[..., k, j, i] = -val

    return phi_full


def metric_from_phi(phi: torch.Tensor) -> torch.Tensor:
    """
    Extract the induced metric from a G2 3-form.

    The metric is defined by the contraction formula:
        g_ij = (1/6) * sum_{k,l} phi_{ikl} * phi_{jkl}

    This formula ensures g is symmetric and, for phi in the G2 cone,
    positive definite.

    Args:
        phi: 3-form tensor of shape (..., 7, 7, 7) or (..., 35) for components

    Returns:
        Metric tensor of shape (..., 7, 7)
    """
    # Expand if given as components
    if phi.shape[-1] == 35 and len(phi.shape) >= 1:
        if len(phi.shape) == 1 or phi.shape[-2] != 7:
            phi = expand_to_antisymmetric(phi)

    # Contract: g_ij = (1/6) * phi_{ikl} * phi_{jkl}
    # Using einsum for clarity
    g = torch.einsum('...ikl,...jkl->...ij', phi, phi) / 6.0

    return g


def det_constraint_loss(
    phi: torch.Tensor,
    target_det: float = 65.0 / 32.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute loss for determinant constraint: det(g) = 65/32.

    Args:
        phi: 3-form tensor
        target_det: Target determinant value (default: 65/32 = 2.03125)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value (scalar if reduction != 'none')
    """
    g = metric_from_phi(phi)
    det_g = torch.det(g)

    # Squared error from target
    loss = (det_g - target_det) ** 2

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def exterior_derivative(
    phi: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute exterior derivative d(phi) of the 3-form.

    For a 3-form phi = sum phi_{ijk} dx^i ^ dx^j ^ dx^k,
    d(phi) is a 4-form:
        d(phi)_{ijkl} = partial_i phi_{jkl} - partial_j phi_{ikl}
                       + partial_k phi_{ijl} - partial_l phi_{ijk}

    Args:
        phi: 3-form tensor of shape (batch, 7, 7, 7)
        x: Coordinate tensor of shape (batch, 7), requires_grad=True
        create_graph: Whether to create graph for higher derivatives

    Returns:
        4-form tensor of shape (batch, 7, 7, 7, 7)
    """
    batch_size = x.shape[0]
    device = x.device
    dtype = phi.dtype

    # Ensure phi has batch dimension
    if phi.dim() == 3:
        phi = phi.unsqueeze(0).expand(batch_size, -1, -1, -1)

    # Initialize 4-form
    d_phi = torch.zeros(batch_size, 7, 7, 7, 7, device=device, dtype=dtype)

    # Compute partial derivatives
    # d(phi)_{ijkl} involves derivatives of phi_{jkl}, phi_{ikl}, phi_{ijl}, phi_{ijk}
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if j != k:  # phi_{ijk} is antisymmetric
                    # Gradient of phi_{ijk} with respect to all coordinates
                    grad = torch.autograd.grad(
                        phi[:, i, j, k].sum(),
                        x,
                        create_graph=create_graph,
                        retain_graph=True,
                        allow_unused=True
                    )[0]

                    if grad is not None:
                        for l in range(7):
                            # Antisymmetrize
                            d_phi[:, l, i, j, k] += grad[:, l]

    # Antisymmetrize the result
    d_phi = antisymmetrize_4form(d_phi)

    return d_phi


def antisymmetrize_4form(tensor: torch.Tensor) -> torch.Tensor:
    """
    Antisymmetrize a 4-form tensor.

    Args:
        tensor: Tensor of shape (..., 7, 7, 7, 7)

    Returns:
        Antisymmetrized tensor
    """
    # Full antisymmetrization over all 24 permutations
    result = torch.zeros_like(tensor)

    # Generate permutations with signs
    from itertools import permutations
    indices = [0, 1, 2, 3]

    for perm in permutations(indices):
        sign = permutation_sign(list(perm))
        result += sign * tensor.permute(
            *range(tensor.dim() - 4),
            tensor.dim() - 4 + perm[0],
            tensor.dim() - 4 + perm[1],
            tensor.dim() - 4 + perm[2],
            tensor.dim() - 4 + perm[3]
        )

    return result / 24.0


def permutation_sign(perm: list) -> int:
    """Compute sign of a permutation."""
    n = len(perm)
    sign = 1
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                sign *= -1
    return sign


def codifferential(
    phi: torch.Tensor,
    x: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    create_graph: bool = True
) -> torch.Tensor:
    """
    Compute codifferential d*(phi) = *d*(phi) of the 3-form.

    The codifferential maps 3-forms to 2-forms:
        d*(phi) = (-1)^{3(7-3)+1} * d * phi = *d*phi

    For torsion-free G2: d*phi = 0

    Args:
        phi: 3-form tensor of shape (batch, 7, 7, 7)
        x: Coordinate tensor of shape (batch, 7)
        g: Optional metric tensor (computed from phi if not given)
        create_graph: Whether to create graph for higher derivatives

    Returns:
        2-form tensor of shape (batch, 7, 7)
    """
    batch_size = x.shape[0]
    device = x.device
    dtype = phi.dtype

    if g is None:
        g = metric_from_phi(phi)

    # Compute inverse metric
    g_inv = torch.inverse(g)

    # Compute sqrt(det(g)) for volume form
    det_g = torch.det(g)
    sqrt_det_g = torch.sqrt(torch.abs(det_g) + 1e-10)

    # d*phi is a 2-form
    # (d*phi)_{ij} = g^{kl} nabla_k phi_{lij}
    d_star_phi = torch.zeros(batch_size, 7, 7, device=device, dtype=dtype)

    # Simplified computation: divergence of phi
    for i in range(7):
        for j in range(7):
            if i != j:
                for k in range(7):
                    # Derivative of phi_{kij} with respect to x^k
                    grad = torch.autograd.grad(
                        phi[:, k, i, j].sum(),
                        x,
                        create_graph=create_graph,
                        retain_graph=True,
                        allow_unused=True
                    )[0]

                    if grad is not None:
                        # Contract with inverse metric
                        d_star_phi[:, i, j] += torch.einsum(
                            '...l,...l->...',
                            g_inv[:, k, :],
                            grad
                        )

    # Antisymmetrize
    d_star_phi = 0.5 * (d_star_phi - d_star_phi.transpose(-1, -2))

    return d_star_phi


def torsion_norm(
    phi: torch.Tensor,
    x: torch.Tensor,
    compute_components: bool = False
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Compute the torsion norm ||T|| = sqrt(||d*phi||^2 + ||d*phi||^2).

    For a G2 structure, torsion measures deviation from being torsion-free.
    GIFT v2.2 targets kappa_T = 1/61.

    Args:
        phi: 3-form tensor
        x: Coordinate tensor
        compute_components: If True, also return (d_phi, d_star_phi)

    Returns:
        Torsion norm (scalar per batch)
        Optionally: (d_phi, d_star_phi) tensors
    """
    d_phi = exterior_derivative(phi, x)
    d_star_phi = codifferential(phi, x)

    # L2 norms
    d_phi_norm_sq = torch.sum(d_phi ** 2, dim=(-1, -2, -3, -4))
    d_star_phi_norm_sq = torch.sum(d_star_phi ** 2, dim=(-1, -2))

    torsion = torch.sqrt(d_phi_norm_sq + d_star_phi_norm_sq + 1e-10)

    if compute_components:
        return torsion, (d_phi, d_star_phi)
    return torsion, None


def torsion_loss(
    phi: torch.Tensor,
    x: torch.Tensor,
    target_kappa: float = 1.0 / 61.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute loss for torsion target: ||T|| = kappa_T = 1/61.

    Note: We target a specific non-zero torsion value, not torsion-free.

    Args:
        phi: 3-form tensor
        x: Coordinate tensor (requires grad)
        target_kappa: Target torsion magnitude (default: 1/61)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss value
    """
    torsion, _ = torsion_norm(phi, x)

    loss = (torsion - target_kappa) ** 2

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def g2_positivity_check(phi: torch.Tensor, return_eigenvalues: bool = False) -> torch.Tensor:
    """
    Check if phi is in the G2 cone (positive 3-form).

    A 3-form phi is positive (defines a G2 structure) iff:
    1. The induced metric g(phi) is positive definite
    2. The volume form is positive

    We check (1) via eigenvalue positivity.

    Args:
        phi: 3-form tensor
        return_eigenvalues: If True, also return eigenvalues

    Returns:
        Positivity violation (0 if satisfied, positive if violated)
        Optionally: eigenvalues of the metric
    """
    g = metric_from_phi(phi)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(g)

    # Violation: sum of negative eigenvalues (relu of -eigenvalues)
    violation = torch.relu(-eigenvalues).sum(dim=-1)

    if return_eigenvalues:
        return violation, eigenvalues
    return violation


def project_to_g2_cone(phi_components: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Project phi onto the G2 cone by ensuring metric positivity.

    Method:
    1. Compute metric from phi
    2. Eigendecompose metric
    3. Clamp negative eigenvalues to eps
    4. Reconstruct phi (approximate)

    This is an approximate projection - exact projection is complex.

    Args:
        phi_components: 3-form components of shape (..., 35)
        eps: Minimum eigenvalue after projection

    Returns:
        Projected phi components
    """
    phi_full = expand_to_antisymmetric(phi_components)
    g = metric_from_phi(phi_full)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(g)

    # Clamp eigenvalues
    eigenvalues_clamped = torch.clamp(eigenvalues, min=eps)

    # Check if any clamping occurred
    needs_projection = (eigenvalues < eps).any(dim=-1)

    if not needs_projection.any():
        return phi_components

    # Reconstruct metric
    g_projected = torch.einsum(
        '...ij,...j,...kj->...ik',
        eigenvectors,
        eigenvalues_clamped,
        eigenvectors
    )

    # Scale phi to match new metric determinant
    det_original = torch.det(g)
    det_projected = torch.det(g_projected)

    scale = (det_projected / (det_original + 1e-10)) ** (1.0 / 6.0)
    scale = torch.where(
        needs_projection.unsqueeze(-1),
        scale.unsqueeze(-1),
        torch.ones_like(scale.unsqueeze(-1))
    )

    return phi_components * scale


def phi_norm_squared(phi: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute ||phi||^2 with respect to the induced metric.

    For a valid G2 structure, ||phi||^2_g = 7 (the G2 identity).

    Args:
        phi: 3-form tensor (7,7,7) or (batch,7,7,7) or components (35,) or (batch,35)
        g: Optional metric (computed from phi if not given)

    Returns:
        ||phi||^2_g (should equal 7 for G2 structure)
    """
    if phi.shape[-1] == 35:
        phi = expand_to_antisymmetric(phi)

    if g is None:
        g = metric_from_phi(phi)

    g_inv = torch.inverse(g)

    # ||phi||^2_g = g^{ia} g^{jb} g^{kc} phi_{ijk} phi_{abc}
    # Simplified: contract phi with phi using inverse metric
    norm_sq = torch.einsum(
        '...ia,...jb,...kc,...ijk,...abc->...',
        g_inv, g_inv, g_inv, phi, phi
    )

    return norm_sq


def standard_g2_phi(device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return the standard G2 3-form on R7.

    phi_0 = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}

    This is a reference G2 structure with:
        - det(g) = 1
        - kappa_T = 0 (torsion-free)
        - phi in G2 cone

    Returns:
        3-form components of shape (35,)
    """
    if device is None:
        device = torch.device('cpu')

    # Initialize to zero
    phi = torch.zeros(35, device=device, dtype=dtype)

    # Map (i,j,k) to linear index
    def to_index(i, j, k):
        # For i < j < k in 0..6
        count = 0
        for a in range(7):
            for b in range(a + 1, 7):
                for c in range(b + 1, 7):
                    if a == i and b == j and c == k:
                        return count
                    count += 1
        return -1

    # Set standard G2 values
    for indices, sign in zip(G2_STANDARD_INDICES, G2_STANDARD_SIGNS):
        idx = to_index(*indices)
        if idx >= 0:
            phi[idx] = float(sign)

    return phi
