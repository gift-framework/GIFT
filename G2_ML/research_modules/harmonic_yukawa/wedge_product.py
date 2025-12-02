"""Wedge product computations for differential forms.

The key operation for Yukawa tensors is:
    Y_ijk = integral_{K7} omega_i wedge omega_j wedge Phi_k

where omega_i, omega_j are 2-forms and Phi_k is a 3-form.
The wedge product 2 + 2 + 3 = 7, giving a 7-form (volume form on K7).
"""
from __future__ import annotations

from itertools import combinations, permutations
from functools import lru_cache
from typing import Tuple, List

import torch
import torch.nn as nn


@lru_cache(maxsize=1)
def get_2form_indices() -> Tuple[Tuple[int, int], ...]:
    """C(7,2) = 21 ordered pairs."""
    return tuple(combinations(range(7), 2))


@lru_cache(maxsize=1)
def get_3form_indices() -> Tuple[Tuple[int, int, int], ...]:
    """C(7,3) = 35 ordered triples."""
    return tuple(combinations(range(7), 3))


def permutation_sign(perm: Tuple[int, ...]) -> int:
    """Sign of permutation relative to (0,1,2,...,n-1)."""
    n = len(perm)
    seen = [False] * n
    sign = 1
    for i in range(n):
        if seen[i]:
            continue
        j = i
        cycle_len = 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len > 1:
            sign *= (-1) ** (cycle_len - 1)
    return sign


class WedgeProduct:
    """Efficient wedge product computation for GIFT Yukawa tensors.

    Precomputes the combinatorial structure for:
    - 2-form wedge 2-form -> 4-form
    - 4-form wedge 3-form -> 7-form (scalar via Hodge star)
    """

    def __init__(self):
        self.idx2 = get_2form_indices()
        self.idx3 = get_3form_indices()
        self._build_wedge_tables()

    def _build_wedge_tables(self):
        """Precompute wedge product structure constants."""
        # 2-form wedge 2-form -> 4-form
        # omega^{ij} wedge omega^{kl} contributes to the ijkl component
        # if i,j,k,l are all distinct

        idx4 = list(combinations(range(7), 4))
        n4 = len(idx4)  # C(7,4) = 35

        # Table: for each (2form_a, 2form_b), which 4form and what sign
        self.wedge_22_table = []  # List of (idx_a, idx_b, idx_4, sign)

        for a, (i, j) in enumerate(self.idx2):
            for b, (k, l) in enumerate(self.idx2):
                indices = (i, j, k, l)
                if len(set(indices)) != 4:
                    continue  # Not all distinct

                sorted_indices = tuple(sorted(indices))
                idx_4 = idx4.index(sorted_indices)

                # Sign from reordering (i,j,k,l) -> sorted
                sign = permutation_sign(tuple(sorted(range(4), key=lambda x: indices[x])))

                self.wedge_22_table.append((a, b, idx_4, sign))

        # 4-form wedge 3-form -> 7-form (volume)
        # This is non-zero only when all 7 indices are distinct
        # The result is a scalar multiple of the volume form

        self.wedge_43_table = []  # List of (idx_4, idx_3, sign)

        for c, (i, j, k) in enumerate(self.idx3):
            for d, four_idx in enumerate(idx4):
                all_7 = set(four_idx) | {i, j, k}
                if len(all_7) != 7:
                    continue  # Not all distinct

                # Sign from full permutation
                full = four_idx + (i, j, k)
                sign = permutation_sign(tuple(sorted(range(7), key=lambda x: full[x])))

                self.wedge_43_table.append((d, c, sign))

        # Convert to tensors for GPU efficiency
        if self.wedge_22_table:
            self._w22_a = torch.tensor([x[0] for x in self.wedge_22_table])
            self._w22_b = torch.tensor([x[1] for x in self.wedge_22_table])
            self._w22_4 = torch.tensor([x[2] for x in self.wedge_22_table])
            self._w22_s = torch.tensor([x[3] for x in self.wedge_22_table], dtype=torch.float)

        if self.wedge_43_table:
            self._w43_4 = torch.tensor([x[0] for x in self.wedge_43_table])
            self._w43_3 = torch.tensor([x[1] for x in self.wedge_43_table])
            self._w43_s = torch.tensor([x[2] for x in self.wedge_43_table], dtype=torch.float)

    def wedge_2_2(self, omega_a: torch.Tensor, omega_b: torch.Tensor) -> torch.Tensor:
        """Compute omega_a wedge omega_b for 2-forms.

        Args:
            omega_a: 2-form (batch, 21)
            omega_b: 2-form (batch, 21)

        Returns:
            result: 4-form (batch, 35)
        """
        device = omega_a.device
        batch = omega_a.shape[0]

        result = torch.zeros(batch, 35, device=device)

        a_idx = self._w22_a.to(device)
        b_idx = self._w22_b.to(device)
        out_idx = self._w22_4.to(device)
        signs = self._w22_s.to(device)

        # Vectorized computation
        contrib = signs * omega_a[:, a_idx] * omega_b[:, b_idx]
        result.scatter_add_(1, out_idx.unsqueeze(0).expand(batch, -1), contrib)

        return result

    def wedge_4_3(self, eta: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """Compute eta wedge Phi where eta is 4-form, Phi is 3-form.

        Result is a 7-form, represented as a scalar (coefficient of volume form).

        Args:
            eta: 4-form (batch, 35)
            Phi: 3-form (batch, 35)

        Returns:
            result: scalar (batch,)
        """
        device = eta.device

        idx_4 = self._w43_4.to(device)
        idx_3 = self._w43_3.to(device)
        signs = self._w43_s.to(device)

        # Sum over all contributions
        contrib = signs * eta[:, idx_4] * Phi[:, idx_3]
        return contrib.sum(dim=-1)


def wedge_2_2_3(
    omega_i: torch.Tensor,
    omega_j: torch.Tensor,
    Phi_k: torch.Tensor,
    wedge: WedgeProduct = None
) -> torch.Tensor:
    """Compute omega_i wedge omega_j wedge Phi_k.

    This is the integrand for Yukawa couplings:
        Y_ijk = integral omega_i wedge omega_j wedge Phi_k

    Args:
        omega_i: 2-form (batch, 21)
        omega_j: 2-form (batch, 21)
        Phi_k: 3-form (batch, 35)
        wedge: Precomputed WedgeProduct (optional)

    Returns:
        result: 7-form coefficient (batch,)
    """
    if wedge is None:
        wedge = WedgeProduct()

    # First compute 2 wedge 2 -> 4
    eta = wedge.wedge_2_2(omega_i, omega_j)

    # Then 4 wedge 3 -> 7
    return wedge.wedge_4_3(eta, Phi_k)


def wedge_3_3_to_6(Phi_a: torch.Tensor, Phi_b: torch.Tensor) -> torch.Tensor:
    """Compute Phi_a wedge Phi_b for 3-forms.

    Result is a 6-form, C(7,6) = 7 components.

    Args:
        Phi_a: 3-form (batch, 35)
        Phi_b: 3-form (batch, 35)

    Returns:
        result: 6-form (batch, 7)
    """
    device = Phi_a.device
    batch = Phi_a.shape[0]

    idx3 = get_3form_indices()
    idx6 = list(combinations(range(7), 6))

    result = torch.zeros(batch, 7, device=device)

    for a, (i, j, k) in enumerate(idx3):
        for b, (l, m, n) in enumerate(idx3):
            all_6 = {i, j, k, l, m, n}
            if len(all_6) != 6:
                continue

            sorted_6 = tuple(sorted(all_6))
            out_idx = idx6.index(sorted_6)

            full_perm = (i, j, k, l, m, n)
            sign = permutation_sign(tuple(sorted(range(6), key=lambda x: full_perm[x])))

            result[:, out_idx] += sign * Phi_a[:, a] * Phi_b[:, b]

    return result
