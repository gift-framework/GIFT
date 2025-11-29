"""Candidate 2- and 3-form builders that mine all historical runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

from .config import DEFAULT_VERSION_PRIORITY, locate_historical_assets
from .geometry_loader import load_version_model, ModelBundle

__all__ = ["CandidateLibrary", "CandidateForms"]


@dataclass
class CandidateForms:
    c2: torch.Tensor
    c3: torch.Tensor
    metadata: Dict


@dataclass
class CandidateLibrary:
    versions: Iterable[str] = field(default_factory=lambda: DEFAULT_VERSION_PRIORITY)
    registry: Optional[Dict[str, ModelBundle]] = None

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
