"""TCS/Joyce-inspired global mode construction for K7 manifolds.

This package implements geometrically meaningful global H3 modes based on
Twisted Connected Sum (TCS) / Joyce G2 manifold construction principles.

The key idea is to replace the "fake" polynomial/trigonometric global modes
from v1.9b with modes that reflect the actual TCS geometry:

- Two asymptotic CY3 regions (left and right)
- A neck region where the transition occurs
- Smooth profile functions encoding the TCS gluing

Module overview:
- profiles: Smooth 1D profile functions (left/right plateaus, neck bump)
- basis_forms: Canonical G2 3-form templates on R^7
- tcs_global_modes: Full TCS/Joyce-based global H3 mode construction
- config: Configuration options for mode construction

References:
- Joyce, D. "Compact Riemannian 7-manifolds with holonomy G2" (1996)
- Kovalev, A. "Twisted connected sums and special Riemannian holonomy" (2003)
- Corti et al. "G2-manifolds and associative submanifolds via semi-Fano 3-folds" (2015)
"""
from __future__ import annotations

from .profiles import (
    smooth_step,
    left_plateau,
    right_plateau,
    neck_bump,
    TCSProfiles,
)
from .basis_forms import (
    canonical_g2_indices,
    canonical_g2_3form_components,
    generate_g2_orthogonal_basis,
    G2BasisLibrary,
)
from .tcs_global_modes import (
    TCSGlobalModeConfig,
    TCSGlobalModeBuilder,
    build_tcs_global_modes,
)
from .config import (
    GlobalModeConstruction,
    TCSJoyceConfig,
    DEFAULT_TCS_CONFIG,
)

__all__ = [
    # Profiles
    "smooth_step",
    "left_plateau",
    "right_plateau",
    "neck_bump",
    "TCSProfiles",
    # Basis forms
    "canonical_g2_indices",
    "canonical_g2_3form_components",
    "generate_g2_orthogonal_basis",
    "G2BasisLibrary",
    # Global modes
    "TCSGlobalModeConfig",
    "TCSGlobalModeBuilder",
    "build_tcs_global_modes",
    # Config
    "GlobalModeConstruction",
    "TCSJoyceConfig",
    "DEFAULT_TCS_CONFIG",
]

__version__ = "2.0.0"
