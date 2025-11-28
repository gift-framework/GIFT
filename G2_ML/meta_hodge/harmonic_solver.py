"""Eigen-problem solvers for approximate harmonic forms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .hodge_operators import HodgeOperator

__all__ = ["HarmonicSolver", "HarmonicResult"]


@dataclass
class HarmonicResult:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor

    def normalized(self) -> "HarmonicResult":
        norm = torch.linalg.norm(self.eigenvectors, dim=0, keepdim=True) + 1e-8
        return HarmonicResult(self.eigenvalues, self.eigenvectors / norm)


@dataclass
class HarmonicSolver:
    op: HodgeOperator

    def solve_subspace(self, basis: torch.Tensor, star_fn, k: int) -> HarmonicResult:
        lap = self.op.laplacian_matrix(basis, star_fn)
        evals, evecs = torch.linalg.eigh(lap)
        idx = torch.argsort(evals)[:k]
        return HarmonicResult(eigenvalues=evals[idx], eigenvectors=evecs[:, idx]).normalized()

    def solve(self, c2: torch.Tensor, c3: torch.Tensor, k2: int = 21, k3: int = 77) -> Dict[str, HarmonicResult]:
        h2 = self.solve_subspace(c2, self.op.star_2, k2)
        h3 = self.solve_subspace(c3, self.op.star_3, k3)
        return {"H2": h2, "H3": h3}
