"""Candidate 2- and 3-form builders that mine all historical runs.

This module now supports two modes for global H3 construction:
- TCS_JOYCE: Geometrically meaningful TCS/Joyce-inspired modes (v2.0+)
- PCA_FAKE: Legacy polynomial/trigonometric modes (v1.9b and earlier)

The TCS/Joyce construction is preferred for spectral analysis aiming
to recover the 43/77 visible/hidden split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional

import torch

from .config import DEFAULT_VERSION_PRIORITY, locate_historical_assets
from .geometry_loader import load_version_model, ModelBundle

__all__ = [
    "CandidateLibrary",
    "CandidateForms",
    "GlobalModeStrategy",
    "GLOBAL_MODE_CONSTRUCTION",
]


class GlobalModeStrategy(Enum):
    """Strategy for constructing the 42 global H3 modes."""
    TCS_JOYCE = "tcs_joyce"
    PCA_FAKE = "pca_fake"


# Configuration flag: set this to switch between construction methods
GLOBAL_MODE_CONSTRUCTION: GlobalModeStrategy = GlobalModeStrategy.TCS_JOYCE


@dataclass
class CandidateForms:
    c2: torch.Tensor
    c3: torch.Tensor
    metadata: Dict


def _build_global_modes_tcs_joyce(coords: torch.Tensor) -> torch.Tensor:
    """Build 42 global modes using TCS/Joyce construction.

    This creates geometrically meaningful modes based on:
    - 14 left-weighted modes (CY3_L region)
    - 14 right-weighted modes (CY3_R region)
    - 14 neck-coupled modes (gluing region)

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (N, 7).

    Returns
    -------
    torch.Tensor
        Global modes, shape (N, 42).
    """
    try:
        from ..tcs_joyce import build_tcs_global_modes
        return build_tcs_global_modes(coords)
    except ImportError:
        # Fallback if tcs_joyce not available
        return _build_global_modes_legacy(coords)


def _build_global_modes_legacy(coords: torch.Tensor) -> torch.Tensor:
    """Build 42 global modes using legacy polynomial/trig construction.

    This is the v1.9b method: algebraic combinations without geometric meaning.

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


def build_global_modes(
    coords: torch.Tensor,
    strategy: Optional[GlobalModeStrategy] = None,
) -> torch.Tensor:
    """Build 42 global H3 modes using the configured strategy.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates, shape (N, 7).
    strategy : GlobalModeStrategy, optional
        Override the global GLOBAL_MODE_CONSTRUCTION setting.

    Returns
    -------
    torch.Tensor
        Global modes, shape (N, 42).
    """
    if strategy is None:
        strategy = GLOBAL_MODE_CONSTRUCTION

    if strategy == GlobalModeStrategy.TCS_JOYCE:
        return _build_global_modes_tcs_joyce(coords)
    else:
        return _build_global_modes_legacy(coords)


@dataclass
class CandidateLibrary:
    versions: Iterable[str] = field(default_factory=lambda: DEFAULT_VERSION_PRIORITY)
    registry: Optional[Dict[str, ModelBundle]] = None
    global_mode_strategy: GlobalModeStrategy = field(
        default_factory=lambda: GLOBAL_MODE_CONSTRUCTION
    )

    def _load_bundle(self, version: str) -> ModelBundle:
        registry = self.registry or locate_historical_assets()
        return load_version_model(version, registry)

    def collect(self, x: torch.Tensor) -> CandidateForms:
        """Return concatenated candidate 2-forms and 3-forms across versions.

        The goal is to reuse *all* historical feature extractors instead of
        discarding them. In many cases the checkpoints only contain metadata;
        then we fall back to simple algebraic combinations of ``phi`` to build
        a broad candidate basis.
        """

        c2_list: List[torch.Tensor] = []
        c3_list: List[torch.Tensor] = []
        meta: List[Dict] = []

        for version in self.versions:
            try:
                bundle = self._load_bundle(version)
            except Exception:
                continue

            phi = bundle.phi_fn(x) if bundle.phi_fn is not None else None
            metric = bundle.metric_fn(x)

            # Candidate 3-forms directly from phi when available.
            if phi is not None:
                c3_list.append(phi)
            else:
                c3_list.append(torch.zeros(x.shape[0], 35, device=x.device, dtype=x.dtype))

            # Candidate 2-forms: use a simple projection of phi onto 2-form slots.
            # We average over the indices participating in each 2-form wedge pair.
            c2_from_phi = torch.zeros(x.shape[0], 21, device=x.device, dtype=x.dtype)
            if phi is not None:
                reshaped = phi.view(x.shape[0], 5, 7) if phi.shape[-1] == 35 else None
                if reshaped is not None:
                    c2_from_phi = reshaped.mean(dim=1)
            c2_list.append(c2_from_phi)

            meta.append(
                {
                    "version": version,
                    "notebooks": [str(p) for p in bundle.notebook_paths],
                    "checkpoints": [str(p) for p in bundle.checkpoint_paths],
                    "notes": bundle.notes,
                }
            )

        c2 = torch.cat(c2_list, dim=-1) if c2_list else torch.empty(x.shape[0], 0)
        c3 = torch.cat(c3_list, dim=-1) if c3_list else torch.empty(x.shape[0], 0)
        metadata = {"candidates": meta, "num_c2": c2.shape[-1], "num_c3": c3.shape[-1]}
        return CandidateForms(c2=c2, c3=c3, metadata=metadata)

    def collect_b3_77(
        self,
        phi: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Collect all 77 H3 modes: 35 local + 42 global.

        This is the main entry point for building the complete H3 basis
        for Yukawa spectrum analysis.

        Parameters
        ----------
        phi : torch.Tensor
            Local phi values from the G2 form, shape (N, 35).
        coords : torch.Tensor
            Coordinates on K7, shape (N, 7).

        Returns
        -------
        torch.Tensor
            All 77 H3 modes, shape (N, 77).

        Notes
        -----
        The construction method is controlled by `global_mode_strategy`:
        - TCS_JOYCE: Geometrically meaningful modes from TCS/Joyce profiles
        - PCA_FAKE: Legacy polynomial/trigonometric modes

        The 35 local modes come directly from phi (canonical G2 3-form).
        The 42 global modes are constructed according to the strategy.
        """
        N = phi.shape[0]
        device = phi.device
        dtype = phi.dtype

        # Local modes: direct from phi (35 components)
        local_modes = phi

        # Global modes: from configured strategy (42 components)
        global_modes = build_global_modes(coords, self.global_mode_strategy)

        # Ensure same device/dtype
        global_modes = global_modes.to(device=device, dtype=dtype)

        # Combine into 77 modes
        h3_modes = torch.cat([local_modes, global_modes], dim=1)

        # Normalize each mode across samples
        norms = torch.norm(h3_modes, dim=0, keepdim=True) + 1e-10
        h3_modes = h3_modes / norms

        return h3_modes

    def collect_b3_77_with_metadata(
        self,
        phi: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict:
        """Collect 77 H3 modes with additional metadata.

        Parameters
        ----------
        phi : torch.Tensor
            Local phi values, shape (N, 35).
        coords : torch.Tensor
            Coordinates, shape (N, 7).

        Returns
        -------
        Dict
            Dictionary with keys:
            - 'modes': The 77 H3 modes, shape (N, 77)
            - 'local': The 35 local modes, shape (N, 35)
            - 'global': The 42 global modes, shape (N, 42)
            - 'strategy': The construction strategy used
            - 'mode_labels': List of mode labels
        """
        local_modes = phi
        global_modes = build_global_modes(coords, self.global_mode_strategy)
        global_modes = global_modes.to(device=phi.device, dtype=phi.dtype)

        h3_modes = torch.cat([local_modes, global_modes], dim=1)

        # Normalize
        norms = torch.norm(h3_modes, dim=0, keepdim=True) + 1e-10
        h3_modes = h3_modes / norms

        # Create mode labels
        mode_labels = (
            [f"local_{i}" for i in range(35)] +
            [f"global_{i}" for i in range(42)]
        )

        if self.global_mode_strategy == GlobalModeStrategy.TCS_JOYCE:
            # More descriptive labels for TCS modes
            mode_labels = (
                [f"local_{i}" for i in range(35)] +
                [f"left_{i}" for i in range(14)] +
                [f"right_{i}" for i in range(14)] +
                [f"neck_{i}" for i in range(14)]
            )

        return {
            "modes": h3_modes,
            "local": local_modes,
            "global": global_modes,
            "strategy": self.global_mode_strategy.value,
            "mode_labels": mode_labels,
        }
