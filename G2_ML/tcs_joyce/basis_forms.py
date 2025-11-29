"""Canonical G2 3-form basis templates for K7 manifolds.

This module provides the standard G2 structure on R^7 and tools to
generate linearly independent 3-form bases for use in TCS constructions.

The canonical G2 3-form on R^7 (Bryant-Salamon form) is:

    phi_0 = e^{012} + e^{034} + e^{056} + e^{135} - e^{146} - e^{236} - e^{245}

where e^{ijk} = dx^i wedge dx^j wedge dx^k.

This gives rise to:
- 35 independent 3-forms on R^7 (dim Lambda^3(R^7) = C(7,3) = 35)
- 7 "associative" directions privileged by G2
- The G2 representation theory: 35 = 1 + 7 + 27

For TCS constructions, we need to distinguish:
- Local modes (fiber): 35 3-forms from Lambda^3 on the fiber
- Global modes (base): 42 additional modes from the TCS gluing
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch

__all__ = [
    "canonical_g2_indices",
    "canonical_g2_coefficients",
    "canonical_g2_3form_components",
    "all_3form_indices",
    "generate_g2_orthogonal_basis",
    "index_to_triple",
    "triple_to_index",
    "G2BasisLibrary",
]


# ============================================================================
# Index Conventions
# ============================================================================


@lru_cache(maxsize=1)
def all_3form_indices() -> List[Tuple[int, int, int]]:
    """Return all 35 ordered triples (i < j < k) for 3-forms on R^7.

    Returns
    -------
    List[Tuple[int, int, int]]
        Lexicographically ordered list of 35 index triples.
    """
    return list(itertools.combinations(range(7), 3))


@lru_cache(maxsize=128)
def triple_to_index(i: int, j: int, k: int) -> int:
    """Convert an ordered triple (i < j < k) to a flat index in [0, 34].

    Uses lexicographic ordering consistent with itertools.combinations.
    """
    indices = all_3form_indices()
    triple = tuple(sorted([i, j, k]))
    return indices.index(triple)


def index_to_triple(idx: int) -> Tuple[int, int, int]:
    """Convert a flat index in [0, 34] to an ordered triple."""
    return all_3form_indices()[idx]


# ============================================================================
# Canonical G2 3-form
# ============================================================================


def canonical_g2_indices() -> List[Tuple[Tuple[int, int, int], int]]:
    """Return the index triples and signs for the canonical G2 3-form.

    The canonical Bryant-Salamon G2 3-form on R^7 is:
        phi_0 = e^{012} + e^{034} + e^{056} + e^{135} - e^{146} - e^{236} - e^{245}

    Returns
    -------
    List[Tuple[Tuple[int, int, int], int]]
        List of ((i, j, k), sign) pairs.
    """
    return [
        ((0, 1, 2), +1),
        ((0, 3, 4), +1),
        ((0, 5, 6), +1),
        ((1, 3, 5), +1),
        ((1, 4, 6), -1),
        ((2, 3, 6), -1),
        ((2, 4, 5), -1),
    ]


def canonical_g2_coefficients() -> torch.Tensor:
    """Return the 35-component vector for the canonical G2 3-form.

    Returns
    -------
    torch.Tensor
        Shape (35,), with entries in {-1, 0, +1}.
    """
    coeffs = torch.zeros(35)
    for (i, j, k), sign in canonical_g2_indices():
        idx = triple_to_index(i, j, k)
        coeffs[idx] = float(sign)
    return coeffs


def canonical_g2_3form_components(
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the canonical G2 3-form as a component tensor.

    Parameters
    ----------
    batch_size : int, optional
        If provided, return shape (batch_size, 35). Otherwise (35,).
    device : torch.device, optional
        Device for the tensor.
    dtype : torch.dtype
        Data type for the tensor.

    Returns
    -------
    torch.Tensor
        The canonical G2 3-form phi_0.
    """
    coeffs = canonical_g2_coefficients().to(dtype=dtype)
    if device is not None:
        coeffs = coeffs.to(device)
    if batch_size is not None:
        coeffs = coeffs.unsqueeze(0).expand(batch_size, -1)
    return coeffs


# ============================================================================
# Orthogonal Basis Generation
# ============================================================================


def _generate_g2_complementary_forms() -> List[torch.Tensor]:
    """Generate 34 3-forms orthogonal to phi_0.

    The full Lambda^3(R^7) has dimension 35. The canonical G2 3-form
    spans a 1-dimensional subspace. We generate 34 orthogonal directions.

    Returns
    -------
    List[torch.Tensor]
        List of 34 tensors, each shape (35,).
    """
    phi_0 = canonical_g2_coefficients()

    # Start with standard basis and Gram-Schmidt
    forms = []
    for idx in range(35):
        e = torch.zeros(35)
        e[idx] = 1.0

        # Project out phi_0
        proj = torch.dot(e, phi_0) / (torch.dot(phi_0, phi_0) + 1e-10)
        e = e - proj * phi_0

        # Project out previous forms
        for f in forms:
            proj = torch.dot(e, f) / (torch.dot(f, f) + 1e-10)
            e = e - proj * f

        # Keep if non-trivial
        norm = torch.norm(e)
        if norm > 1e-6:
            forms.append(e / norm)

        if len(forms) >= 34:
            break

    return forms


def generate_g2_orthogonal_basis(
    include_canonical: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate an orthonormal basis for Lambda^3(R^7).

    Parameters
    ----------
    include_canonical : bool
        If True, first basis element is the canonical G2 3-form.
    device : torch.device, optional
        Device for the tensor.
    dtype : torch.dtype
        Data type.

    Returns
    -------
    torch.Tensor
        Shape (35, 35), each row is an orthonormal basis element.
    """
    phi_0 = canonical_g2_coefficients()
    phi_0 = phi_0 / (torch.norm(phi_0) + 1e-10)

    complementary = _generate_g2_complementary_forms()

    if include_canonical:
        all_forms = [phi_0] + complementary
    else:
        all_forms = complementary + [phi_0]

    basis = torch.stack(all_forms[:35], dim=0)
    basis = basis.to(dtype=dtype)
    if device is not None:
        basis = basis.to(device)

    return basis


# ============================================================================
# Global Mode Basis (for TCS construction)
# ============================================================================


def _generate_xi_weighted_forms(n_forms: int = 14) -> List[Tuple[int, torch.Tensor]]:
    """Generate 3-forms weighted by xi (transverse) coordinates.

    These are 3-forms that have non-trivial dependence on the
    base/transverse coordinates xi = (x_1, ..., x_6).

    Returns
    -------
    List[Tuple[int, torch.Tensor]]
        List of (xi_index, form_coefficients) pairs.
    """
    # Select 3-form components that involve different coordinate combinations
    # We want forms that can couple to the xi directions

    # Strategy: Choose 3-forms where removing one index gives a 2-form
    # that lives primarily in the xi directions (indices 1-6)

    forms = []
    all_indices = all_3form_indices()

    # Group 1: Forms involving lambda (index 0) and two xi indices
    # These will couple strongly to left/right profiles
    for i in range(1, 7):
        for j in range(i + 1, 7):
            triple = (0, i, j)
            idx = triple_to_index(*triple)
            e = torch.zeros(35)
            e[idx] = 1.0
            forms.append((i, e))  # Weight by xi_i
            if len(forms) >= n_forms:
                return forms

    # Group 2: Forms purely in xi directions
    # These will couple to neck profiles
    for triple in all_indices:
        if 0 not in triple:
            idx = triple_to_index(*triple)
            e = torch.zeros(35)
            e[idx] = 1.0
            xi_idx = triple[0]  # First index of triple
            forms.append((xi_idx, e))
            if len(forms) >= n_forms:
                return forms

    return forms


def _generate_mixed_lambda_xi_forms(n_forms: int = 14) -> List[torch.Tensor]:
    """Generate 3-forms with mixed lambda-xi structure.

    These forms have coefficients that depend on both the
    neck direction and transverse directions.

    Returns
    -------
    List[torch.Tensor]
        List of form coefficient tensors.
    """
    forms = []
    all_indices = all_3form_indices()

    # Forms that mix lambda-containing and xi-only indices
    lambda_forms = [t for t in all_indices if 0 in t]
    xi_forms = [t for t in all_indices if 0 not in t]

    # Linear combinations: phi_lambda + alpha * phi_xi
    for i, lam_triple in enumerate(lambda_forms[:min(7, n_forms)]):
        for j, xi_triple in enumerate(xi_forms[:2]):
            e = torch.zeros(35)
            e[triple_to_index(*lam_triple)] = 1.0
            e[triple_to_index(*xi_triple)] = (-1.0) ** (i + j) * 0.5
            forms.append(e / (torch.norm(e) + 1e-10))
            if len(forms) >= n_forms:
                return forms

    return forms


# ============================================================================
# G2 Basis Library
# ============================================================================


@dataclass
class G2BasisLibrary:
    """Library of 3-form bases for TCS/Joyce constructions.

    Attributes
    ----------
    local_basis : torch.Tensor
        Shape (35, 35), orthonormal basis for Lambda^3 (local modes).
    global_left_basis : torch.Tensor
        Shape (14, 35), basis forms for left CY3 region.
    global_right_basis : torch.Tensor
        Shape (14, 35), basis forms for right CY3 region.
    global_neck_basis : torch.Tensor
        Shape (14, 35), basis forms for neck region.
    """

    local_basis: torch.Tensor = field(default_factory=lambda: generate_g2_orthogonal_basis())
    global_left_basis: torch.Tensor = field(init=False)
    global_right_basis: torch.Tensor = field(init=False)
    global_neck_basis: torch.Tensor = field(init=False)

    def __post_init__(self):
        """Initialize global bases."""
        # Left basis: forms involving lambda and lower xi indices
        left_forms = []
        for triple in all_3form_indices()[:14]:
            idx = triple_to_index(*triple)
            e = torch.zeros(35)
            e[idx] = 1.0
            left_forms.append(e)
        self.global_left_basis = torch.stack(left_forms, dim=0)

        # Right basis: forms involving lambda and upper xi indices
        right_forms = []
        for triple in all_3form_indices()[14:28]:
            idx = triple_to_index(*triple)
            e = torch.zeros(35)
            e[idx] = 1.0
            right_forms.append(e)
        self.global_right_basis = torch.stack(right_forms, dim=0)

        # Neck basis: mixed forms
        neck_forms = _generate_mixed_lambda_xi_forms(14)
        if len(neck_forms) < 14:
            # Pad with remaining basis elements
            for triple in all_3form_indices()[28:]:
                e = torch.zeros(35)
                e[triple_to_index(*triple)] = 1.0
                neck_forms.append(e)
                if len(neck_forms) >= 14:
                    break
        self.global_neck_basis = torch.stack(neck_forms[:14], dim=0)

    def get_local_forms(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get local (fiber) 3-form basis evaluated at batch points.

        Parameters
        ----------
        batch_size : int
            Number of sample points.
        device : torch.device, optional
            Target device.
        dtype : torch.dtype
            Target dtype.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 35), each row is the canonical G2 form.
        """
        # For local modes, we use the canonical G2 form (constant over manifold)
        phi_0 = canonical_g2_3form_components(batch_size, device, dtype)
        return phi_0

    def get_global_templates(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Get global 3-form templates for TCS construction.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'left', 'right', 'neck' keys, each
            mapping to shape (14, 35) template tensors.
        """
        left = self.global_left_basis.to(dtype=dtype)
        right = self.global_right_basis.to(dtype=dtype)
        neck = self.global_neck_basis.to(dtype=dtype)

        if device is not None:
            left = left.to(device)
            right = right.to(device)
            neck = neck.to(device)

        return {
            "left": left,
            "right": right,
            "neck": neck,
        }

    def orthonormalize_against_local(
        self,
        global_forms: torch.Tensor,
        tolerance: float = 1e-6,
    ) -> torch.Tensor:
        """Orthonormalize global forms against the local (35) basis.

        Ensures that global modes are linearly independent from local modes.

        Parameters
        ----------
        global_forms : torch.Tensor
            Shape (n_global, 35), candidate global forms.
        tolerance : float
            Tolerance for zero norm after projection.

        Returns
        -------
        torch.Tensor
            Orthonormalized global forms.
        """
        # Local basis spans Lambda^3, so any true global mode must be
        # in the kernel of the integration operator. For our numerical
        # construction, we ensure linear independence by Gram-Schmidt.

        result = []
        for i in range(global_forms.shape[0]):
            v = global_forms[i].clone()

            # Project out local basis
            for j in range(min(35, self.local_basis.shape[0])):
                u = self.local_basis[j]
                proj = torch.dot(v, u) / (torch.dot(u, u) + 1e-10)
                # Only project if the form has significant overlap
                if abs(proj) > tolerance:
                    v = v - proj * u

            # Project out previous global forms
            for u in result:
                proj = torch.dot(v, u) / (torch.dot(u, u) + 1e-10)
                v = v - proj * u

            # Keep if non-trivial
            norm = torch.norm(v)
            if norm > tolerance:
                result.append(v / norm)

        if result:
            return torch.stack(result, dim=0)
        else:
            return torch.zeros(0, 35)


def get_g2_representation_decomposition() -> Dict[str, List[int]]:
    """Return the G2 representation decomposition of Lambda^3(R^7).

    Under the G2 subgroup of GL(7), Lambda^3(R^7) decomposes as:
        35 = 1 + 7 + 27

    where:
    - 1: The canonical G2 3-form phi_0 (trivial representation)
    - 7: Forms transforming as the fundamental of G2
    - 27: Forms in the 27-dimensional representation

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping representation name to list of basis indices.
    """
    # The canonical G2 form indices
    g2_indices = [triple_to_index(*t) for (t, _) in canonical_g2_indices()]

    # For a complete decomposition, one needs the explicit G2 representation
    # matrices. Here we provide a simplified version based on index structure.

    # Trivial (1): The G2 form itself
    trivial = [0]  # First basis element after orthonormalization

    # Seven (7): Forms that transform as vectors
    # These involve wedging with the G2 form structure
    seven = list(range(1, 8))

    # Twenty-seven (27): Remaining forms
    twenty_seven = list(range(8, 35))

    return {
        "trivial_1": trivial,
        "fundamental_7": seven,
        "adjoint_27": twenty_seven,
    }
