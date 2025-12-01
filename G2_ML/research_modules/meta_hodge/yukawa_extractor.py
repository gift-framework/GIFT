"""Monte Carlo Yukawa extraction for mined harmonic bases."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

__all__ = ["YukawaExtractor"]


@dataclass
class YukawaExtractor:
    sample_points: torch.Tensor
    metric: torch.Tensor

    def volume_weights(self) -> torch.Tensor:
        det = torch.linalg.det(self.metric)
        return torch.sqrt(torch.clamp(det, min=1e-12))

    def compute(self, h2_values: torch.Tensor, h3_values: torch.Tensor) -> torch.Tensor:
        """Compute approximate Yukawa couplings Y_{ijk} = ∫ ω_i ∧ ω_j ∧ Ω_k.

        Parameters
        ----------
        h2_values: torch.Tensor
            Evaluations of harmonic 2-forms at the sample points with shape
            ``(N, b2)`` where ``b2`` ~ 21.
        h3_values: torch.Tensor
            Evaluations of harmonic 3-forms at the sample points with shape
            ``(N, b3)`` where ``b3`` ~ 77.
        """

        weights = self.volume_weights().unsqueeze(-1)
        n_points = self.sample_points.shape[0]
        y = torch.einsum("ni,nj,nk,n->ijk", h2_values, h2_values, h3_values, weights)
        return y / float(n_points)

    def save(self, tensor: torch.Tensor, out_dir: Path) -> Dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = out_dir / "yukawa_tensor.pt"
        npz_path = out_dir / "yukawa_summary.npz"
        torch.save(tensor, pt_path)
        import numpy as np

        np.savez(npz_path, yukawa=tensor.cpu().numpy())
        return {"torch": pt_path, "numpy": npz_path}
