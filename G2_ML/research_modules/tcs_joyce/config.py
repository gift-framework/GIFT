"""Configuration for TCS/Joyce global mode construction.

This module provides configuration options for the global mode
construction strategy used in the meta-hodge pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .profiles import TCSProfiles

__all__ = [
    "GlobalModeConstruction",
    "TCSJoyceConfig",
    "DEFAULT_TCS_CONFIG",
    "get_global_mode_builder",
]


class GlobalModeConstruction(Enum):
    """Strategy for constructing global H3 modes.

    Attributes
    ----------
    TCS_JOYCE : str
        Use TCS/Joyce-inspired geometric construction.
        Global modes built from profile functions and 3-form templates.
    PCA_FAKE : str
        Use the legacy polynomial/trigonometric construction.
        Global modes are artificial algebraic combinations (v1.9b).
    HYBRID : str
        Use a mix of TCS and legacy modes for comparison.
    """

    TCS_JOYCE = "tcs_joyce"
    PCA_FAKE = "pca_fake"
    HYBRID = "hybrid"


@dataclass
class TCSJoyceConfig:
    """Full configuration for TCS/Joyce mode construction.

    This combines profile configuration, basis form selection,
    and mode construction parameters.

    Attributes
    ----------
    construction : GlobalModeConstruction
        Strategy for global mode construction.
    domain : Tuple[float, float]
        Domain bounds for lambda coordinate.
    neck_fraction : float
        Relative position of neck within domain (0.5 = center).
    transition_width : float
        Width of profile transitions as fraction of domain.
    n_left : int
        Number of left-weighted modes.
    n_right : int
        Number of right-weighted modes.
    n_neck : int
        Number of neck-coupled modes.
    orthonormalize : bool
        Whether to Gram-Schmidt orthonormalize modes.
    include_xi_weighting : bool
        Whether to include xi-dependent modulation.
    closure_penalty : float
        Weight for closure error in optimization.
    """

    construction: GlobalModeConstruction = GlobalModeConstruction.TCS_JOYCE
    domain: Tuple[float, float] = (0.0, 1.0)
    neck_fraction: float = 0.5
    transition_width: float = 0.15
    n_left: int = 14
    n_right: int = 14
    n_neck: int = 14
    orthonormalize: bool = True
    include_xi_weighting: bool = True
    closure_penalty: float = 0.1

    def get_profiles(self) -> TCSProfiles:
        """Create TCSProfiles from this configuration."""
        return TCSProfiles.from_domain(
            domain=self.domain,
            neck_fraction=self.neck_fraction,
            transition_width=self.transition_width,
        )

    @property
    def n_global(self) -> int:
        """Total number of global modes."""
        return self.n_left + self.n_right + self.n_neck

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.n_global != 42:
            raise ValueError(
                f"Total global modes must be 42, got {self.n_global}"
            )
        if not 0 < self.neck_fraction < 1:
            raise ValueError(
                f"neck_fraction must be in (0, 1), got {self.neck_fraction}"
            )
        if self.transition_width <= 0:
            raise ValueError(
                f"transition_width must be positive, got {self.transition_width}"
            )


# Default configuration for TCS/Joyce construction
DEFAULT_TCS_CONFIG = TCSJoyceConfig()


def get_global_mode_builder(
    config: Optional[TCSJoyceConfig] = None,
):
    """Get the appropriate global mode builder based on configuration.

    Parameters
    ----------
    config : TCSJoyceConfig, optional
        Configuration. Uses defaults if not provided.

    Returns
    -------
    Callable
        Function that takes coords and returns global modes.
    """
    from .tcs_global_modes import (
        build_tcs_global_modes,
        build_legacy_global_modes,
        TCSGlobalModeConfig,
    )

    if config is None:
        config = DEFAULT_TCS_CONFIG

    if config.construction == GlobalModeConstruction.TCS_JOYCE:
        tcs_config = TCSGlobalModeConfig(
            n_left=config.n_left,
            n_right=config.n_right,
            n_neck=config.n_neck,
            profiles=config.get_profiles(),
            orthonormalize=config.orthonormalize,
            include_cross_terms=config.include_xi_weighting,
            closure_weight=config.closure_penalty,
        )
        return lambda coords: build_tcs_global_modes(coords, tcs_config)

    elif config.construction == GlobalModeConstruction.PCA_FAKE:
        return build_legacy_global_modes

    elif config.construction == GlobalModeConstruction.HYBRID:
        # Mix both constructions
        tcs_config = TCSGlobalModeConfig(
            n_left=config.n_left // 2,
            n_right=config.n_right // 2,
            n_neck=config.n_neck // 2,
            profiles=config.get_profiles(),
            orthonormalize=False,
            include_cross_terms=config.include_xi_weighting,
        )

        def hybrid_builder(coords):
            import torch
            tcs = build_tcs_global_modes(coords, tcs_config)
            legacy = build_legacy_global_modes(coords)
            # Take first 21 from each
            combined = torch.cat([tcs[:, :21], legacy[:, :21]], dim=1)
            return combined

        return hybrid_builder

    else:
        raise ValueError(f"Unknown construction: {config.construction}")


# ============================================================================
# Integration with meta_hodge
# ============================================================================


def create_h3_collector(
    config: Optional[TCSJoyceConfig] = None,
):
    """Create a complete H3 mode collector for the meta-hodge pipeline.

    This returns a function that produces all 77 H3 modes:
    - 35 local modes from the canonical G2 form
    - 42 global modes from TCS/Joyce construction

    Parameters
    ----------
    config : TCSJoyceConfig, optional
        Configuration for global mode construction.

    Returns
    -------
    Callable
        Function taking (phi, coords) and returning 77-mode tensor.
    """
    import torch
    from .tcs_global_modes import combine_local_global_modes

    global_builder = get_global_mode_builder(config)

    def collect_h3_modes(
        phi: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Collect all 77 H3 modes.

        Parameters
        ----------
        phi : torch.Tensor
            Local phi values, shape (N, 35).
        coords : torch.Tensor
            Coordinates, shape (N, 7).

        Returns
        -------
        torch.Tensor
            All H3 modes, shape (N, 77).
        """
        # Local modes: direct from phi (35 components)
        local_modes = phi

        # Global modes: from TCS construction (42 components)
        global_modes = global_builder(coords)

        # Combine
        return combine_local_global_modes(local_modes, global_modes)

    return collect_h3_modes
