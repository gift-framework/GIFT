"""Unified accessors for historical G2/K7 PINN artifacts.

The goal is *not* to rewrite legacy notebooks but to provide a gentle shim
that allows downstream meta-analysis code to sample coordinates, query learned
metrics, and, when available, the underlying G2 3-form ``phi``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from .config import DEFAULT_SAMPLE_SIZE, locate_historical_assets, VersionInfo

__all__ = ["ModelBundle", "load_version_model", "sample_coords"]


@dataclass
class ModelBundle:
    """Container for a single version's geometry predictors.

    ``metric_fn`` should map ``x -> g_ij(x)`` with shape ``(N, 7, 7)``.
    ``phi_fn`` (optional) maps ``x -> phi(x)`` with shape ``(N, 35)`` using the
    convention of lexicographically ordered 3-form components.
    """

    version: str
    notebook_paths: list[Path]
    checkpoint_paths: list[Path]
    metric_fn: Callable[[torch.Tensor], torch.Tensor]
    phi_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    notes: str = ""


# --- Coordinate sampling ----------------------------------------------------


def sample_coords(n_points: int = DEFAULT_SAMPLE_SIZE, domain_config: Optional[Dict] = None) -> torch.Tensor:
    """Sample coordinates on the reference T^7 chart.

    Parameters
    ----------
    n_points:
        Number of samples to draw.
    domain_config:
        Optional dictionary with keys like ``periodic`` or ``range``. When not
        provided, the default is uniform ``[0, 1)`` for each coordinate.
    """

    domain_config = domain_config or {}
    coord_range = domain_config.get("range", (0.0, 1.0))
    low, high = coord_range
    x = torch.rand(n_points, 7)
    return low + (high - low) * x


# --- Model loading ---------------------------------------------------------


def _identity_metric(x: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(7, device=x.device, dtype=x.dtype)
    return eye.unsqueeze(0).expand(x.shape[0], -1, -1)


def _zero_phi(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], 35, device=x.device, dtype=x.dtype)


def _load_checkpoint_stub(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return {}


def load_version_model(version: str, registry: Optional[Dict[str, VersionInfo]] = None) -> ModelBundle:
    """Create a :class:`ModelBundle` for the requested version.

    The function attempts to locate notebooks and checkpoints. If a checkpoint
    is present but cannot be deserialized into a model, an identity metric is
    used as a placeholder so the meta-pipeline remains runnable.
    """

    registry = registry or locate_historical_assets()
    if version not in registry:
        raise ValueError(f"Version {version} not found in registry: {list(registry)}")

    info = registry[version]
    checkpoint_payload = None
    if info.checkpoint_paths:
        checkpoint_payload = _load_checkpoint_stub(info.checkpoint_paths[-1])

    # Placeholders until explicit model classes are wired in.
    metric_fn: Callable[[torch.Tensor], torch.Tensor] = _identity_metric
    phi_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = _zero_phi

    if checkpoint_payload and isinstance(checkpoint_payload, dict):
        # Keep the payload around for downstream customization.
        def metric_from_payload(x: torch.Tensor, payload=checkpoint_payload) -> torch.Tensor:
            meta = payload.get("metric_info") or payload.get("metric")
            if meta is None:
                return _identity_metric(x)
            tensor = torch.as_tensor(meta, dtype=x.dtype, device=x.device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != 7:
                return _identity_metric(x)
            return tensor.expand(x.shape[0], -1, -1)

        metric_fn = metric_from_payload

        def phi_from_payload(x: torch.Tensor, payload=checkpoint_payload) -> torch.Tensor:
            meta = payload.get("phi") or payload.get("phi_info")
            if meta is None:
                return _zero_phi(x)
            tensor = torch.as_tensor(meta, dtype=x.dtype, device=x.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != 35:
                return _zero_phi(x)
            return tensor.expand(x.shape[0], -1)

        phi_fn = phi_from_payload

    notes = info.notes
    if checkpoint_payload:
        notes += " | checkpoint payload keys: " + ",".join(checkpoint_payload.keys())

    return ModelBundle(
        version=version,
        notebook_paths=info.notebook_paths,
        checkpoint_paths=info.checkpoint_paths,
        metric_fn=metric_fn,
        phi_fn=phi_fn,
        notes=notes,
    )
