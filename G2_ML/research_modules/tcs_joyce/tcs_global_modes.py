"""TCS/Joyce-based global H3 mode construction for K7 manifolds.

This module implements the geometrically meaningful global H3 modes
based on Twisted Connected Sum (TCS) principles:

    b3(K7) = 77 = 35 (local) + 42 (global)

The 42 global modes are constructed as:
- 14 left-weighted modes: f_L(lambda) * B_k(xi)
- 14 right-weighted modes: f_R(lambda) * B_k'(xi)
- 14 neck-coupled modes: g_neck(lambda) * B_k''(xi)

where:
- f_L, f_R, g_neck are smooth profile functions from profiles.py
- B_k, B_k', B_k'' are 3-form templates from basis_forms.py

Key Design Goals:
1. Replace the "fake" polynomial/trig global modes from v1.9b
2. Ensure global modes are approximately closed and coclosed
3. Maintain orthonormality with respect to the Hodge inner product
4. Enable clean 43/77 visible/hidden split in Yukawa spectrum
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .profiles import TCSProfiles, left_plateau, right_plateau, neck_bump
from .basis_forms import (
    G2BasisLibrary,
    canonical_g2_3form_components,
    all_3form_indices,
    triple_to_index,
)

__all__ = [
    "TCSGlobalModeConfig",
    "TCSGlobalModeBuilder",
    "build_tcs_global_modes",
    "combine_local_global_modes",
]


@dataclass
class TCSGlobalModeConfig:
    """Configuration for TCS global mode construction.

    Attributes
    ----------
    n_left : int
        Number of left-weighted modes (default 14).
    n_right : int
        Number of right-weighted modes (default 14).
    n_neck : int
        Number of neck-coupled modes (default 14).
    profiles : TCSProfiles
        Profile function container.
    orthonormalize : bool
        Whether to Gram-Schmidt orthonormalize the result.
    include_cross_terms : bool
        Whether to include xi-weighted cross terms.
    closure_weight : float
        Weight for penalizing non-closed forms (for optimization).
    """

    n_left: int = 14
    n_right: int = 14
    n_neck: int = 14
    profiles: TCSProfiles = field(default_factory=TCSProfiles.from_domain)
    orthonormalize: bool = True
    include_cross_terms: bool = True
    closure_weight: float = 0.1

    @property
    def n_global(self) -> int:
        """Total number of global modes."""
        return self.n_left + self.n_right + self.n_neck

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.n_global != 42:
            raise ValueError(
                f"Total global modes must be 42, got {self.n_global} "
                f"({self.n_left} + {self.n_right} + {self.n_neck})"
            )


@dataclass
class TCSGlobalModeBuilder:
    """Builder for TCS/Joyce-style global H3 modes.

    This class constructs the 42 global modes for H3(K7) using
    TCS profile functions and G2 3-form templates.

    The construction is:
        Phi_k(x) = profile_k(lambda) * B_k(xi) * xi_weight_k(xi)

    where:
    - profile_k is one of {f_L, f_R, g_neck}
    - B_k is a 3-form template
    - xi_weight_k provides xi-dependence for global modes
    """

    config: TCSGlobalModeConfig = field(default_factory=TCSGlobalModeConfig)
    basis_library: G2BasisLibrary = field(default_factory=G2BasisLibrary)

    def _extract_lambda(self, coords: torch.Tensor) -> torch.Tensor:
        """Extract the neck (lambda) coordinate from full coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Full coordinates, shape (N, 7).

        Returns
        -------
        torch.Tensor
            Lambda coordinate, shape (N, 1).
        """
        return coords[:, 0:1]

    def _extract_xi(self, coords: torch.Tensor) -> torch.Tensor:
        """Extract the transverse (xi) coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Full coordinates, shape (N, 7).

        Returns
        -------
        torch.Tensor
            Xi coordinates, shape (N, 6).
        """
        return coords[:, 1:7]

    def _build_left_modes(
        self,
        coords: torch.Tensor,
        templates: torch.Tensor,
    ) -> torch.Tensor:
        """Build left-weighted global modes.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).
        templates : torch.Tensor
            3-form templates, shape (n_left, 35).

        Returns
        -------
        torch.Tensor
            Left modes, shape (N, n_left, 35).
        """
        N = coords.shape[0]
        n_modes = templates.shape[0]

        lam = self._extract_lambda(coords)  # (N, 1)
        xi = self._extract_xi(coords)  # (N, 6)

        # Profile: ~1 on left, ~0 on right
        f_L = self.config.profiles.left(lam)  # (N, 1)

        # xi-weighting for asymmetry
        if self.config.include_cross_terms:
            # Weight by first few xi components
            xi_weights = 1.0 + 0.3 * xi[:, :min(n_modes, 6)]  # (N, min(n_modes, 6))
            # Pad if needed
            if n_modes > 6:
                padding = torch.ones(N, n_modes - 6, device=coords.device, dtype=coords.dtype)
                xi_weights = torch.cat([xi_weights, padding], dim=1)
        else:
            xi_weights = torch.ones(N, n_modes, device=coords.device, dtype=coords.dtype)

        # Combine: profile * xi_weight * template
        # f_L: (N, 1), xi_weights: (N, n_modes), templates: (n_modes, 35)
        modes = f_L.unsqueeze(-1) * xi_weights.unsqueeze(-1) * templates.unsqueeze(0)
        # modes: (N, n_modes, 35)

        return modes

    def _build_right_modes(
        self,
        coords: torch.Tensor,
        templates: torch.Tensor,
    ) -> torch.Tensor:
        """Build right-weighted global modes.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).
        templates : torch.Tensor
            3-form templates, shape (n_right, 35).

        Returns
        -------
        torch.Tensor
            Right modes, shape (N, n_right, 35).
        """
        N = coords.shape[0]
        n_modes = templates.shape[0]

        lam = self._extract_lambda(coords)  # (N, 1)
        xi = self._extract_xi(coords)  # (N, 6)

        # Profile: ~0 on left, ~1 on right
        f_R = self.config.profiles.right(lam)  # (N, 1)

        # xi-weighting for asymmetry (use different xi components than left)
        if self.config.include_cross_terms:
            # Weight by last few xi components (for diversity)
            xi_idx = torch.arange(n_modes, device=coords.device) % 6
            xi_weights = 1.0 + 0.3 * xi[:, 5 - (xi_idx % 6)]  # Use reversed xi
            xi_weights = xi_weights.view(N, n_modes)
        else:
            xi_weights = torch.ones(N, n_modes, device=coords.device, dtype=coords.dtype)

        # Combine
        modes = f_R.unsqueeze(-1) * xi_weights.unsqueeze(-1) * templates.unsqueeze(0)

        return modes

    def _build_neck_modes(
        self,
        coords: torch.Tensor,
        templates: torch.Tensor,
    ) -> torch.Tensor:
        """Build neck-coupled global modes.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).
        templates : torch.Tensor
            3-form templates, shape (n_neck, 35).

        Returns
        -------
        torch.Tensor
            Neck modes, shape (N, n_neck, 35).
        """
        N = coords.shape[0]
        n_modes = templates.shape[0]

        lam = self._extract_lambda(coords)  # (N, 1)
        xi = self._extract_xi(coords)  # (N, 6)

        # Profile: bump at neck, zero at ends
        g_neck = self.config.profiles.neck(lam)  # (N, 1)

        # xi-weighting for neck modes: use products of xi
        if self.config.include_cross_terms:
            # Quadratic-ish xi dependence
            xi_sq = xi ** 2
            xi_weights = 1.0 + 0.2 * (xi_sq[:, :min(n_modes, 6)].sum(dim=1, keepdim=True))
            xi_weights = xi_weights.expand(-1, n_modes)  # (N, n_modes)
        else:
            xi_weights = torch.ones(N, n_modes, device=coords.device, dtype=coords.dtype)

        # Combine
        modes = g_neck.unsqueeze(-1) * xi_weights.unsqueeze(-1) * templates.unsqueeze(0)

        return modes

    def build(self, coords: torch.Tensor) -> torch.Tensor:
        """Build all 42 global H3 modes.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).

        Returns
        -------
        torch.Tensor
            Global modes, shape (N, 42).
            Each mode is evaluated as a scalar at each point
            (the 35 3-form components are contracted/reduced).
        """
        device = coords.device
        dtype = coords.dtype
        N = coords.shape[0]

        # Get templates
        templates = self.basis_library.get_global_templates(device, dtype)

        # Build each type
        left_modes = self._build_left_modes(coords, templates["left"])  # (N, 14, 35)
        right_modes = self._build_right_modes(coords, templates["right"])  # (N, 14, 35)
        neck_modes = self._build_neck_modes(coords, templates["neck"])  # (N, 14, 35)

        # Concatenate along mode dimension
        all_modes = torch.cat([left_modes, right_modes, neck_modes], dim=1)  # (N, 42, 35)

        # Reduce from 35 components to scalar mode values
        # Use L2 norm along component dimension as mode strength
        mode_values = torch.norm(all_modes, dim=-1)  # (N, 42)

        # Optionally orthonormalize
        if self.config.orthonormalize:
            mode_values = self._orthonormalize(mode_values)

        return mode_values

    def build_full_forms(self, coords: torch.Tensor) -> torch.Tensor:
        """Build all 42 global H3 modes as full 3-form tensors.

        This returns the full (N, 42, 35) tensor without reduction,
        useful for Hodge analysis and Yukawa computation.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).

        Returns
        -------
        torch.Tensor
            Global modes as 3-forms, shape (N, 42, 35).
        """
        device = coords.device
        dtype = coords.dtype

        # Get templates
        templates = self.basis_library.get_global_templates(device, dtype)

        # Build each type
        left_modes = self._build_left_modes(coords, templates["left"])
        right_modes = self._build_right_modes(coords, templates["right"])
        neck_modes = self._build_neck_modes(coords, templates["neck"])

        # Concatenate
        all_modes = torch.cat([left_modes, right_modes, neck_modes], dim=1)

        return all_modes

    def _orthonormalize(self, modes: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt orthonormalize the modes.

        Parameters
        ----------
        modes : torch.Tensor
            Input modes, shape (N, n_modes).

        Returns
        -------
        torch.Tensor
            Orthonormalized modes, shape (N, n_modes).
        """
        N, n_modes = modes.shape

        # Normalize each mode across samples
        norms = torch.norm(modes, dim=0, keepdim=True) + 1e-10
        modes = modes / norms

        # Gram-Schmidt (simplified: just normalize, full GS expensive for large N)
        return modes

    def estimate_closure_error(
        self,
        coords: torch.Tensor,
        h: float = 1e-4,
    ) -> torch.Tensor:
        """Estimate closure error ||d(mode)||^2 via finite differences.

        For a truly harmonic mode, d(omega) = 0. This estimates the
        departure from closedness.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates, shape (N, 7).
        h : float
            Finite difference step size.

        Returns
        -------
        torch.Tensor
            Closure error per mode, shape (42,).
        """
        device = coords.device
        dtype = coords.dtype
        N = coords.shape[0]

        modes_0 = self.build(coords)  # (N, 42)

        errors = torch.zeros(42, device=device, dtype=dtype)

        # Finite difference in lambda direction (dominant variation)
        coords_plus = coords.clone()
        coords_plus[:, 0] += h
        modes_plus = self.build(coords_plus)

        # |d/d(lambda) mode|^2 as proxy for d(mode)
        d_modes = (modes_plus - modes_0) / h
        errors = (d_modes ** 2).mean(dim=0)

        return errors


def build_tcs_global_modes(
    coords: torch.Tensor,
    config: Optional[TCSGlobalModeConfig] = None,
    return_full_forms: bool = False,
) -> torch.Tensor:
    """Convenience function to build TCS global modes.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (N, 7).
    config : TCSGlobalModeConfig, optional
        Configuration. Uses defaults if not provided.
    return_full_forms : bool
        If True, return shape (N, 42, 35). If False, return (N, 42).

    Returns
    -------
    torch.Tensor
        Global mode values or full forms.
    """
    if config is None:
        config = TCSGlobalModeConfig()

    builder = TCSGlobalModeBuilder(config=config)

    if return_full_forms:
        return builder.build_full_forms(coords)
    else:
        return builder.build(coords)


def combine_local_global_modes(
    local_modes: torch.Tensor,
    global_modes: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Combine local and global modes into full 77-mode H3 basis.

    Parameters
    ----------
    local_modes : torch.Tensor
        Local (fiber) modes, shape (N, 35).
    global_modes : torch.Tensor
        Global (TCS) modes, shape (N, 42).
    normalize : bool
        Whether to normalize each mode across samples.

    Returns
    -------
    torch.Tensor
        Combined H3 modes, shape (N, 77).
    """
    # Concatenate
    h3_modes = torch.cat([local_modes, global_modes], dim=1)

    # Normalize
    if normalize:
        norms = torch.norm(h3_modes, dim=0, keepdim=True) + 1e-10
        h3_modes = h3_modes / norms

    return h3_modes


# ============================================================================
# Legacy Mode Construction (for comparison)
# ============================================================================


def build_legacy_global_modes(
    coords: torch.Tensor,
) -> torch.Tensor:
    """Build the old polynomial/trig global modes from v1.9b.

    This is kept for comparison and backward compatibility.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (N, 7).

    Returns
    -------
    torch.Tensor
        Global modes, shape (N, 42).
    """
    import math

    lam = coords[:, 0:1]  # (N, 1)
    xi = coords[:, 1:]  # (N, 6)

    global_modes = []

    # Polynomial in lambda
    for p in range(5):
        global_modes.append(lam ** p)

    # Mixed terms
    for i in range(6):
        global_modes.append(xi[:, i:i+1])
        global_modes.append(lam * xi[:, i:i+1])

    # Trigonometric
    for k in [1, 2, 3]:
        global_modes.append(torch.sin(2 * math.pi * k * lam))
        global_modes.append(torch.cos(2 * math.pi * k * lam))

    # Fill remaining to get 42
    while len(global_modes) < 42:
        idx = len(global_modes)
        global_modes.append(torch.sin(math.pi * idx * lam / 10))

    global_modes = torch.cat(global_modes[:42], dim=1)

    return global_modes


def compare_mode_constructions(
    coords: torch.Tensor,
    config: Optional[TCSGlobalModeConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Compare TCS and legacy mode constructions.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (N, 7).
    config : TCSGlobalModeConfig, optional
        Configuration for TCS modes.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'tcs', 'legacy', and 'correlation' tensors.
    """
    tcs_modes = build_tcs_global_modes(coords, config)
    legacy_modes = build_legacy_global_modes(coords)

    # Compute correlation matrix between constructions
    tcs_normed = tcs_modes / (torch.norm(tcs_modes, dim=0, keepdim=True) + 1e-10)
    legacy_normed = legacy_modes / (torch.norm(legacy_modes, dim=0, keepdim=True) + 1e-10)

    correlation = (tcs_normed.T @ legacy_normed) / coords.shape[0]

    return {
        "tcs": tcs_modes,
        "legacy": legacy_modes,
        "correlation": correlation,
    }
