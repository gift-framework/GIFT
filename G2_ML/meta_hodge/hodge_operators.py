"""Numerical Hodge operators on sampled metrics."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch

__all__ = [
    "HodgeOperator",
    "assemble_hodge_star_matrices",
    "pair_combinations",
]


@lru_cache(maxsize=None)
def pair_combinations(n: int, k: int) -> List[Tuple[int, ...]]:
    return list(itertools.combinations(range(n), k))


def complement(indices: Sequence[int], n: int) -> Tuple[int, ...]:
    return tuple(i for i in range(n) if i not in indices)


def permutation_parity(indices: Sequence[int]) -> int:
    swaps = 0
    seen = []
    for idx in indices:
        position = len([s for s in seen if s > idx])
        swaps += position
        seen.append(idx)
    return -1 if swaps % 2 else 1


def _components_to_tensor(components: torch.Tensor, combos: List[Tuple[int, ...]], n: int) -> torch.Tensor:
    """Map flattened antisymmetric components to a dense tensor with p indices."""
    p = len(combos[0]) if combos else 0
    target_shape = list(components.shape[:-1]) + [n] * p
    tensor = torch.zeros(*target_shape, device=components.device, dtype=components.dtype)

    for comp_idx, combo in enumerate(combos):
        # Assign to the sorted ordering with appropriate sign for permutations.
        tensor[(...,) + combo] = components[..., comp_idx]
        # Fill other permutations
        for perm in itertools.permutations(combo):
            sign = permutation_parity(perm)
            tensor[(...,) + perm] = sign * components[..., comp_idx]
    return tensor


def _raise_indices(tensor: torch.Tensor, g_inv: torch.Tensor) -> torch.Tensor:
    """Raise all indices of an antisymmetric tensor using g^{-1}."""
    result = tensor
    rank = result.dim() - g_inv.dim() + 2  # subtract batch dims
    for i in range(rank):
        result = torch.einsum("...i...,ij->...j...", result, g_inv)
    return result


def _hodge_star_pform(components: torch.Tensor, g: torch.Tensor, p: int) -> torch.Tensor:
    n = g.shape[-1]
    combos_p = pair_combinations(n, p)
    combos_q = pair_combinations(n, n - p)
    comp_tensor = _components_to_tensor(components, combos_p, n)
    g_inv = torch.linalg.inv(g)
    vol = torch.sqrt(torch.clamp(torch.linalg.det(g), min=1e-12))
    alpha_up = _raise_indices(comp_tensor, g_inv)

    # Map combo tuple to index
    combo_to_index = {combo: idx for idx, combo in enumerate(combos_q)}
    result = torch.zeros(*components.shape[:-1], len(combos_q), device=components.device, dtype=components.dtype)
    factor = 1.0 / math.factorial(p)

    for idx_I, I in enumerate(combos_p):
        C = complement(I, n)
        idx_C = combo_to_index[C]
        sign = permutation_parity(I + C)
        # Gather contravariant component
        comp_value = alpha_up[(slice(None),) * (alpha_up.dim() - p) + I]
        result[..., idx_C] += factor * vol * sign * comp_value

    return result


@dataclass
class HodgeOperator:
    g: torch.Tensor

    def star_2(self, omega: torch.Tensor) -> torch.Tensor:
        return _hodge_star_pform(omega, self.g, p=2)

    def star_3(self, Omega: torch.Tensor) -> torch.Tensor:
        return _hodge_star_pform(Omega, self.g, p=3)

    def laplacian_matrix(self, basis: torch.Tensor, star_fn) -> torch.Tensor:
        """Assemble a discrete Laplacian in the span of ``basis``.

        ``basis`` has shape ``(N, d)`` where ``N`` are sample points and ``d``
        basis elements evaluated at those points. The Laplacian is approximated
        via ``d* d`` assuming a flat exterior derivative built from gradients of
        basis functions (not explicitly modeled here). This placeholder uses a
        Gramian of Hodge-starred forms to keep the pipeline testable.
        """

        starred = star_fn(basis)
        gram = torch.einsum("nd,nd->dd", basis, basis)
        dual_gram = torch.einsum("nd,nd->dd", starred, starred)
        return gram + dual_gram


def assemble_hodge_star_matrices(g: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return reusable Hodge-star matrices for 2- and 3-forms."""
    op = HodgeOperator(g)
    # Build canonical basis for 2-forms and 3-forms.
    comb2 = pair_combinations(7, 2)
    comb3 = pair_combinations(7, 3)
    e2 = torch.zeros(len(comb2), len(comb2))
    e3 = torch.zeros(len(comb3), len(comb3))
    for i in range(len(comb2)):
        e2[i, i] = 1.0
    for i in range(len(comb3)):
        e3[i, i] = 1.0

    dummy = g.new_zeros(1, 7, 7)
    dummy_metric = dummy + g.mean(dim=0, keepdim=True)
    op = HodgeOperator(dummy_metric)
    star2 = op.star_2(e2)
    star3 = op.star_3(e3)
    return {"star2": star2, "star3": star3}
