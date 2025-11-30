"""Exterior calculus operators for G2 manifolds.

Implements the differential operators needed for harmonic form training:
- Exterior derivative d
- Hodge star *
- Codifferential δ = (-1)^k * d * (on k-forms in 7D)
- Wedge product ∧

For a 7-dimensional manifold with G2 holonomy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from itertools import combinations, permutations
from functools import lru_cache


# =============================================================================
# Index mappings for differential forms
# =============================================================================

@lru_cache(maxsize=10)
def form_indices(dim: int, degree: int) -> List[Tuple[int, ...]]:
    """Get ordered multi-indices for k-forms in dim dimensions.

    Returns list of tuples (i1, i2, ..., ik) with i1 < i2 < ... < ik.
    """
    return list(combinations(range(dim), degree))


def index_to_position(indices: Tuple[int, ...], dim: int, degree: int) -> int:
    """Convert multi-index to position in component array."""
    all_indices = form_indices(dim, degree)
    return all_indices.index(indices)


def position_to_index(pos: int, dim: int, degree: int) -> Tuple[int, ...]:
    """Convert position in component array to multi-index."""
    return form_indices(dim, degree)[pos]


# =============================================================================
# Levi-Civita symbol and sign computations
# =============================================================================

def levi_civita_sign(perm: Tuple[int, ...]) -> int:
    """Compute sign of permutation (+1 even, -1 odd, 0 if repeated)."""
    n = len(perm)
    if len(set(perm)) != n:
        return 0

    # Count inversions
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1

    return 1 if inversions % 2 == 0 else -1


def wedge_sign(indices1: Tuple[int, ...], indices2: Tuple[int, ...]) -> int:
    """Compute sign when wedging two forms with given indices."""
    combined = indices1 + indices2
    if len(set(combined)) != len(combined):
        return 0  # Repeated index -> zero

    # Sort and count transpositions
    sorted_combined = tuple(sorted(combined))
    return levi_civita_sign(tuple(sorted(range(len(combined)),
                                         key=lambda i: combined[i])))


# =============================================================================
# Exterior derivative
# =============================================================================

class ExteriorDerivative(nn.Module):
    """Exterior derivative operator d.

    For a k-form ω with components ω_{i1...ik}:
    (dω)_{j i1...ik} = ∂ω_{i1...ik}/∂x_j - ∂ω_{j i2...ik}/∂x_{i1} + ...

    Uses automatic differentiation for gradients.
    """

    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim = dim

        # Precompute index mappings
        self._build_derivative_maps()

    def _build_derivative_maps(self):
        """Build maps for exterior derivative computation."""
        # For 1-form -> 2-form
        self.d1_map = []  # List of (output_idx, input_idx, deriv_coord, sign)
        indices_1 = form_indices(self.dim, 1)
        indices_2 = form_indices(self.dim, 2)

        for out_pos, (i, j) in enumerate(indices_2):
            # d(ω) = ∂_i ω_j - ∂_j ω_i
            in_pos_j = index_to_position((j,), self.dim, 1)
            in_pos_i = index_to_position((i,), self.dim, 1)
            self.d1_map.append((out_pos, in_pos_j, i, +1))  # ∂_i ω_j
            self.d1_map.append((out_pos, in_pos_i, j, -1))  # -∂_j ω_i

        # For 2-form -> 3-form
        self.d2_map = []
        indices_3 = form_indices(self.dim, 3)

        for out_pos, (i, j, k) in enumerate(indices_3):
            # (dω)_ijk = ∂_i ω_jk - ∂_j ω_ik + ∂_k ω_ij
            for perm_idx, (a, b, c) in enumerate([(i,j,k), (j,i,k), (k,i,j)]):
                try:
                    bc_sorted = tuple(sorted([b, c]))
                    in_pos = index_to_position(bc_sorted, self.dim, 2)
                    sign = (-1) ** perm_idx
                    if (b, c) != bc_sorted:
                        sign *= -1  # Antisymmetry
                    self.d2_map.append((out_pos, in_pos, a, sign))
                except ValueError:
                    pass

    def apply_1form(self, omega: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply d to a 1-form.

        Args:
            omega: 1-form components (batch, 7)
            x: coordinates (batch, 7)

        Returns:
            d_omega: 2-form components (batch, 21)
        """
        batch = omega.shape[0]
        device = omega.device

        # Compute gradients via finite differences (more stable than autograd for batched)
        eps = 1e-4
        d_omega = torch.zeros(batch, 21, device=device)

        for out_pos, in_pos, deriv_coord, sign in self.d1_map:
            # Finite difference for ∂ω[in_pos]/∂x[deriv_coord]
            x_plus = x.clone()
            x_plus[:, deriv_coord] += eps
            x_minus = x.clone()
            x_minus[:, deriv_coord] -= eps

            # This requires evaluating omega at x_plus and x_minus
            # For neural network forms, we'd need to re-evaluate
            # Here we use the gradient of the current omega values
            grad = torch.autograd.grad(
                omega[:, in_pos].sum(), x,
                create_graph=True, retain_graph=True
            )[0][:, deriv_coord]

            d_omega[:, out_pos] += sign * grad

        return d_omega

    def apply_2form_fd(
        self,
        omega: torch.Tensor,
        x: torch.Tensor,
        omega_fn: callable
    ) -> torch.Tensor:
        """Apply d to a 2-form using finite differences.

        Args:
            omega: 2-form components (batch, 21)
            x: coordinates (batch, 7)
            omega_fn: Function that computes omega from x

        Returns:
            d_omega: 3-form components (batch, 35)
        """
        batch = omega.shape[0]
        device = omega.device
        eps = 1e-4

        d_omega = torch.zeros(batch, 35, device=device)

        for out_pos, in_pos, deriv_coord, sign in self.d2_map:
            x_plus = x.clone()
            x_plus[:, deriv_coord] += eps
            x_minus = x.clone()
            x_minus[:, deriv_coord] -= eps

            omega_plus = omega_fn(x_plus)
            omega_minus = omega_fn(x_minus)

            grad = (omega_plus[:, in_pos] - omega_minus[:, in_pos]) / (2 * eps)
            d_omega[:, out_pos] += sign * grad

        return d_omega


# =============================================================================
# Hodge star operator
# =============================================================================

class HodgeStar(nn.Module):
    """Hodge star operator for G2 manifolds.

    Maps k-forms to (7-k)-forms using the metric:
    (*ω)_{j1...j(7-k)} = (1/k!) ε^{i1...ik}_{j1...j(7-k)} ω_{i1...ik} sqrt(|g|)

    For G2 holonomy, the metric is constrained by the associative 3-form φ.
    """

    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim = dim
        self._build_star_maps()

    def _build_star_maps(self):
        """Build contraction maps for Hodge star."""
        # For 2-form -> 5-form (we'll need this for d*)
        # *ω has components (*ω)_{j1j2j3j4j5} = ε_{i1i2 j1j2j3j4j5} ω^{i1i2}

        # For 3-form -> 4-form
        # *Φ has components (*Φ)_{j1j2j3j4} = ε_{i1i2i3 j1j2j3j4} Φ^{i1i2i3}
        pass

    def apply_metric(
        self,
        omega: torch.Tensor,
        metric: torch.Tensor,
        degree: int
    ) -> torch.Tensor:
        """Apply Hodge star with given metric.

        Args:
            omega: k-form components (batch, C(7,k))
            metric: metric tensor (batch, 7, 7)
            degree: k

        Returns:
            star_omega: (7-k)-form components (batch, C(7,7-k))
        """
        batch = omega.shape[0]
        device = omega.device

        # Compute metric determinant and inverse
        det_g = torch.det(metric)
        sqrt_det_g = torch.sqrt(det_g.abs())
        g_inv = torch.linalg.inv(metric)

        # For a 2-form, *ω is a 5-form
        # For a 3-form, *ω is a 4-form
        out_degree = self.dim - degree
        n_out = len(form_indices(self.dim, out_degree))

        star_omega = torch.zeros(batch, n_out, device=device)

        in_indices = form_indices(self.dim, degree)
        out_indices = form_indices(self.dim, out_degree)

        # Full Levi-Civita contraction
        for out_pos, out_idx in enumerate(out_indices):
            for in_pos, in_idx in enumerate(in_indices):
                # Check if indices are complementary
                combined = in_idx + out_idx
                if len(set(combined)) != self.dim:
                    continue

                # Get permutation sign
                sign = levi_civita_sign(combined)
                if sign == 0:
                    continue

                # Raise indices using metric inverse
                # ω^{i1i2} = g^{i1 j1} g^{i2 j2} ω_{j1 j2}
                metric_factor = 1.0
                for i, idx in enumerate(in_idx):
                    metric_factor *= g_inv[:, idx, idx]  # Simplified: diagonal approx

                star_omega[:, out_pos] += sign * metric_factor * omega[:, in_pos] * sqrt_det_g

        # Normalize by k!
        from math import factorial
        star_omega /= factorial(degree)

        return star_omega


# =============================================================================
# Codifferential (adjoint of d)
# =============================================================================

class Codifferential(nn.Module):
    """Codifferential operator δ = *d*.

    For a k-form ω on a 7-manifold:
    δω = (-1)^{7(k+1)+1} * d * ω = (-1)^{k} * d * ω

    This is the formal adjoint of d with respect to the L2 inner product.
    """

    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim = dim
        self.d = ExteriorDerivative(dim)
        self.star = HodgeStar(dim)

    def apply(
        self,
        omega: torch.Tensor,
        x: torch.Tensor,
        metric: torch.Tensor,
        degree: int,
        omega_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """Apply codifferential to a k-form.

        δω = (-1)^k * d * ω

        Args:
            omega: k-form components
            x: coordinates
            metric: metric tensor
            degree: k
            omega_fn: Function to evaluate omega (for finite differences)

        Returns:
            delta_omega: (k-1)-form components
        """
        # Apply first Hodge star: k-form -> (7-k)-form
        star1 = self.star.apply_metric(omega, metric, degree)

        # Apply exterior derivative: (7-k)-form -> (8-k)-form
        # Note: 8-k = 7-(k-1), so this gives a (7-(k-1))-form
        if omega_fn is not None:
            def star_omega_fn(x_):
                return self.star.apply_metric(omega_fn(x_), metric, degree)
            d_star = self.d.apply_2form_fd(star1, x, star_omega_fn)
        else:
            # Simplified: use current values
            d_star = torch.zeros_like(star1[:, :len(form_indices(self.dim, self.dim - degree + 1))])

        # Apply second Hodge star: (8-k)-form -> (k-1)-form
        star2 = self.star.apply_metric(d_star, metric, self.dim - degree + 1)

        # Sign factor
        sign = (-1) ** degree

        return sign * star2


# =============================================================================
# Wedge product
# =============================================================================

class WedgeProduct(nn.Module):
    """Wedge product of differential forms.

    For a p-form α and q-form β:
    (α ∧ β)_{i1...ip j1...jq} = (p+q)!/(p!q!) Alt(α_{i1...ip} β_{j1...jq})
    """

    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim = dim

    def apply(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        p: int,
        q: int
    ) -> torch.Tensor:
        """Compute wedge product α ∧ β.

        Args:
            alpha: p-form components (batch, C(dim,p))
            beta: q-form components (batch, C(dim,q))
            p: degree of alpha
            q: degree of beta

        Returns:
            wedge: (p+q)-form components (batch, C(dim,p+q))
        """
        batch = alpha.shape[0]
        device = alpha.device

        if p + q > self.dim:
            return torch.zeros(batch, 1, device=device)

        out_degree = p + q
        n_out = len(form_indices(self.dim, out_degree))
        wedge = torch.zeros(batch, n_out, device=device)

        alpha_indices = form_indices(self.dim, p)
        beta_indices = form_indices(self.dim, q)
        out_indices = form_indices(self.dim, out_degree)

        for out_pos, out_idx in enumerate(out_indices):
            for a_pos, a_idx in enumerate(alpha_indices):
                for b_pos, b_idx in enumerate(beta_indices):
                    # Check if a_idx and b_idx partition out_idx
                    combined = a_idx + b_idx
                    if sorted(combined) != list(out_idx):
                        continue

                    # Compute sign from permutation
                    sign = wedge_sign(a_idx, b_idx)

                    wedge[:, out_pos] += sign * alpha[:, a_pos] * beta[:, b_pos]

        return wedge

    def triple_wedge(
        self,
        omega1: torch.Tensor,  # 2-form
        omega2: torch.Tensor,  # 2-form
        Phi: torch.Tensor,     # 3-form
    ) -> torch.Tensor:
        """Compute ω1 ∧ ω2 ∧ Φ (gives a 7-form = volume form).

        Args:
            omega1: 2-form (batch, 21)
            omega2: 2-form (batch, 21)
            Phi: 3-form (batch, 35)

        Returns:
            result: 7-form coefficient (batch,) - scalar since 7-form is volume form
        """
        batch = omega1.shape[0]
        device = omega1.device

        # First compute ω1 ∧ ω2 -> 4-form
        omega12 = self.apply(omega1, omega2, 2, 2)  # (batch, C(7,4)) = (batch, 35)

        # Then wedge with Φ -> 7-form
        result = self.apply(omega12, Phi, 4, 3)  # (batch, 1)

        return result.squeeze(-1)


# =============================================================================
# Laplacian (Hodge Laplacian)
# =============================================================================

class HodgeLaplacian(nn.Module):
    """Hodge Laplacian Δ = dδ + δd.

    Harmonic forms satisfy Δω = 0, which is equivalent to:
    - dω = 0 (closed)
    - δω = 0 (co-closed)
    """

    def __init__(self, dim: int = 7):
        super().__init__()
        self.dim = dim
        self.d = ExteriorDerivative(dim)
        self.delta = Codifferential(dim)

    def apply(
        self,
        omega: torch.Tensor,
        x: torch.Tensor,
        metric: torch.Tensor,
        degree: int,
        omega_fn: Optional[callable] = None
    ) -> torch.Tensor:
        """Apply Hodge Laplacian to a k-form."""
        # dδω
        delta_omega = self.delta.apply(omega, x, metric, degree, omega_fn)
        d_delta_omega = self.d.apply_2form_fd(delta_omega, x, lambda x_: delta_omega)

        # δdω
        if degree == 2:
            d_omega = self.d.apply_2form_fd(omega, x, omega_fn)
        else:
            d_omega = torch.zeros_like(omega)
        delta_d_omega = self.delta.apply(d_omega, x, metric, degree + 1)

        return d_delta_omega + delta_d_omega


# =============================================================================
# Harmonic form loss functions
# =============================================================================

def closedness_loss(
    omega: torch.Tensor,
    x: torch.Tensor,
    omega_fn: callable,
    degree: int = 2
) -> torch.Tensor:
    """Loss for dω = 0 (closed form).

    Uses finite differences to compute exterior derivative.
    """
    d = ExteriorDerivative(dim=7)

    if degree == 2:
        d_omega = d.apply_2form_fd(omega, x, omega_fn)
    else:
        d_omega = torch.zeros(omega.shape[0], 1, device=omega.device)

    return torch.mean(d_omega ** 2)


def coclosedness_loss(
    omega: torch.Tensor,
    x: torch.Tensor,
    metric: torch.Tensor,
    omega_fn: callable,
    degree: int = 2
) -> torch.Tensor:
    """Loss for δω = 0 (co-closed form)."""
    delta = Codifferential(dim=7)
    delta_omega = delta.apply(omega, x, metric, degree, omega_fn)
    return torch.mean(delta_omega ** 2)


def orthonormality_loss(
    forms: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """Loss for Gram matrix = Identity.

    Args:
        forms: (batch, n_modes, n_components)
        metric: (batch, 7, 7)

    Returns:
        loss: scalar
    """
    batch, n_modes, n_comp = forms.shape

    # Volume element
    det_g = torch.det(metric)
    vol = torch.sqrt(det_g.abs())  # (batch,)

    # Weighted inner products
    # G_ij = (1/batch) sum_x vol(x) * <form_i(x), form_j(x)>
    weighted_forms = forms * vol.unsqueeze(-1).unsqueeze(-1)
    G = torch.einsum('bic,bjc->ij', weighted_forms, forms) / batch

    # Target: identity
    I = torch.eye(n_modes, device=forms.device)

    return torch.mean((G - I) ** 2)
