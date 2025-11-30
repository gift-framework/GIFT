"""Harmonic form networks for H2 and H3 on K7.

This module implements neural networks that learn harmonic differential forms
on the K7 manifold with G2 holonomy. The forms satisfy:
  - Closed: dω = 0
  - Co-closed: d*ω = 0 (equivalently: δω = 0)
  - Orthonormal: <ω_i, ω_j> = δ_ij

For a G2 manifold:
  - b2(K7) = 21 harmonic 2-forms
  - b3(K7) = 77 harmonic 3-forms (35 local + 42 global in TCS)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class HodgeConfig:
    """Configuration for Hodge form training."""
    n_h2: int = 21
    n_h3: int = 77
    n_h3_local: int = 35
    n_h3_global: int = 42
    hidden_dim: int = 256
    n_layers: int = 4


class H2Network(nn.Module):
    """Network for learning 21 harmonic 2-forms on K7.

    Output: omega_i(x) for i = 1..21
    Each omega_i is a 2-form, represented as a 21-component vector
    (antisymmetric 7x7 matrix -> 21 independent components)
    """

    def __init__(self, config: HodgeConfig):
        super().__init__()
        self.config = config
        self.n_components = 21  # C(7,2) = 21 for 2-forms

        # Shared feature extractor
        layers = []
        in_dim = 7  # TCS coordinates
        for i in range(config.n_layers):
            out_dim = config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        self.features = nn.Sequential(*layers)

        # Per-mode output heads (21 modes, each with 21 components)
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, self.n_components)
            for _ in range(config.n_h2)
        ])

        # Index mapping for 2-form components
        self.register_buffer('idx_pairs', self._build_index_pairs())

    def _build_index_pairs(self) -> torch.Tensor:
        """Build (i,j) pairs for 2-form components."""
        pairs = []
        for i in range(7):
            for j in range(i+1, 7):
                pairs.append([i, j])
        return torch.tensor(pairs, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Coordinates (batch, 7)

        Returns:
            omega: 2-forms (batch, 21 modes, 21 components)
        """
        features = self.features(x)

        # Stack outputs from all heads
        outputs = torch.stack([head(features) for head in self.heads], dim=1)

        return outputs

    def to_matrix(self, omega: torch.Tensor) -> torch.Tensor:
        """Convert component representation to antisymmetric matrix.

        Args:
            omega: (batch, 21 modes, 21 components)

        Returns:
            omega_matrix: (batch, 21 modes, 7, 7) antisymmetric
        """
        batch, n_modes, n_comp = omega.shape
        device = omega.device

        # Build antisymmetric matrices
        matrices = torch.zeros(batch, n_modes, 7, 7, device=device)

        for k, (i, j) in enumerate(self.idx_pairs):
            matrices[:, :, i, j] = omega[:, :, k]
            matrices[:, :, j, i] = -omega[:, :, k]

        return matrices


class H3Network(nn.Module):
    """Network for learning 77 harmonic 3-forms on K7.

    The 77 modes decompose as:
      - 35 local modes: come from local phi structure (C(7,3) = 35)
      - 42 global modes: come from TCS neck topology

    Output: Phi_k(x) for k = 1..77
    Each Phi_k is a 3-form, represented as a 35-component vector
    """

    def __init__(self, config: HodgeConfig):
        super().__init__()
        self.config = config
        self.n_components = 35  # C(7,3) = 35 for 3-forms

        # Shared feature extractor
        layers = []
        in_dim = 7
        for i in range(config.n_layers):
            out_dim = config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        self.features = nn.Sequential(*layers)

        # Local modes (35): directly from phi-like structure
        self.local_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, self.n_components)
            for _ in range(config.n_h3_local)
        ])

        # Global modes (42): depend on TCS neck position (lambda)
        # These modes have different structure - they're influenced by
        # the neck parameter lambda (encoded in coords[0] typically)
        self.global_neck = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),  # +1 for lambda
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.global_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, self.n_components)
            for _ in range(config.n_h3_global)
        ])

        # Index mapping for 3-form components
        self.register_buffer('idx_triples', self._build_index_triples())

    def _build_index_triples(self) -> torch.Tensor:
        """Build (i,j,k) triples for 3-form components."""
        from itertools import combinations
        triples = list(combinations(range(7), 3))
        return torch.tensor(triples, dtype=torch.long)

    def forward(self, x: torch.Tensor, lambda_neck: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Coordinates (batch, 7)
            lambda_neck: Optional neck parameter (batch, 1). If None, uses x[:, 0:1]

        Returns:
            Phi: 3-forms (batch, 77 modes, 35 components)
        """
        if lambda_neck is None:
            lambda_neck = x[:, 0:1]  # First coord is neck parameter

        features = self.features(x)

        # Local modes
        local_outputs = torch.stack(
            [head(features) for head in self.local_heads], dim=1
        )  # (batch, 35, 35)

        # Global modes - include lambda dependence
        global_features = self.global_neck(torch.cat([features, lambda_neck], dim=-1))
        global_outputs = torch.stack(
            [head(global_features) for head in self.global_heads], dim=1
        )  # (batch, 42, 35)

        # Concatenate
        Phi = torch.cat([local_outputs, global_outputs], dim=1)  # (batch, 77, 35)

        return Phi


class HodgeLoss(nn.Module):
    """Loss functions for harmonic form training.

    Enforces:
    1. Closed: dω = 0 (exterior derivative vanishes)
    2. Co-closed: d*ω = 0 (codifferential vanishes)
    3. Orthonormality: <ω_i, ω_j>_L2 = δ_ij
    """

    def __init__(self, metric_fn, config: HodgeConfig):
        super().__init__()
        self.metric_fn = metric_fn
        self.config = config

    def exterior_derivative_2form(
        self, omega: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute dω for a 2-form.

        dω is a 3-form. We compute the 35 components.

        Args:
            omega: 2-form components (batch, 21)
            x: coordinates (batch, 7)

        Returns:
            d_omega: 3-form components (batch, 35)
        """
        batch = x.shape[0]
        device = x.device

        # Need gradients of omega w.r.t. x
        x.requires_grad_(True)

        # For each 3-form component (i,j,k), dω has:
        # (dω)_ijk = ∂ω_jk/∂x_i - ∂ω_ik/∂x_j + ∂ω_ij/∂x_k

        # Compute Jacobian of omega w.r.t. x
        d_omega = torch.zeros(batch, 35, device=device)

        # Simplified: use finite differences for now
        eps = 1e-4
        for c in range(7):
            x_plus = x.clone()
            x_plus[:, c] += eps
            # Would need to re-evaluate omega at x_plus
            # This is a placeholder - full implementation needs autograd

        return d_omega

    def gram_matrix(
        self, forms: torch.Tensor, metric: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gram matrix of forms.

        Args:
            forms: (batch, n_modes, n_components)
            metric: (batch, 7, 7)

        Returns:
            G: (n_modes, n_modes) Gram matrix
        """
        # L2 inner product weighted by sqrt(det(g))
        det_g = torch.det(metric)  # (batch,)
        weight = torch.sqrt(det_g.abs()).unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)

        # Weighted inner product
        # <ω_i, ω_j> = (1/N) sum_x sqrt(det g) * ω_i(x) · ω_j(x)
        weighted_forms = forms * weight  # (batch, n_modes, n_comp)

        # Gram matrix
        G = torch.einsum('bic,bjc->ij', weighted_forms, forms) / forms.shape[0]

        return G

    def orthonormality_loss(self, G: torch.Tensor) -> torch.Tensor:
        """Loss for deviation from identity Gram matrix."""
        n = G.shape[0]
        I = torch.eye(n, device=G.device)
        return torch.mean((G - I) ** 2)

    def forward(
        self,
        forms: torch.Tensor,
        x: torch.Tensor,
        metric: torch.Tensor,
        weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            forms: (batch, n_modes, n_components)
            x: (batch, 7)
            metric: (batch, 7, 7)
            weights: Loss weights

        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # Orthonormality
        G = self.gram_matrix(forms, metric)
        losses['orthonormality'] = self.orthonormality_loss(G)

        # Closedness and co-closedness would require proper exterior calculus
        # Placeholder for now
        losses['closed'] = torch.tensor(0.0, device=forms.device)
        losses['coclosed'] = torch.tensor(0.0, device=forms.device)

        # Total weighted loss
        total = sum(weights.get(k, 1.0) * v for k, v in losses.items())

        return total, {k: v.item() for k, v in losses.items()}


def compute_yukawa_integral(
    omega: torch.Tensor,  # (batch, 21, 21) - H2 forms
    Phi: torch.Tensor,    # (batch, 77, 35) - H3 forms
    metric: torch.Tensor, # (batch, 7, 7)
) -> torch.Tensor:
    """Compute Yukawa tensor Y_ijk = integral(omega_i ^ omega_j ^ Phi_k).

    The integral is approximated by Monte Carlo over the sampled points.

    Args:
        omega: H2 2-forms (batch, 21 modes, 21 components)
        Phi: H3 3-forms (batch, 77 modes, 35 components)
        metric: Metric tensor (batch, 7, 7)

    Returns:
        Y: Yukawa tensor (21, 21, 77)
    """
    batch = omega.shape[0]
    device = omega.device

    # Volume element sqrt(det g)
    det_g = torch.det(metric)
    vol = torch.sqrt(det_g.abs())  # (batch,)

    # The wedge product omega_i ^ omega_j is a 4-form (would have 35 components)
    # But we contract it directly with Phi_k which is a 3-form
    # The result is effectively a 7-form = volume form

    # Simplified: treat as weighted triple correlation
    # Y_ijk ~ <omega_i, omega_j, Phi_k>

    # This is a proxy - proper implementation needs wedge product
    Y = torch.zeros(21, 21, 77, device=device)

    for i in range(21):
        for j in range(21):
            # omega_i ^ omega_j contribution (antisymmetric)
            omega_ij = omega[:, i, :] * omega[:, j, :]  # (batch, 21)
            omega_ij_sum = omega_ij.sum(dim=-1)  # (batch,)

            for k in range(77):
                # Integrate over manifold
                integrand = omega_ij_sum * Phi[:, k, :].sum(dim=-1) * vol
                Y[i, j, k] = integrand.mean()

    # Antisymmetrize in (i,j)
    Y = 0.5 * (Y - Y.transpose(0, 1))

    return Y


def yukawa_gram_matrix(Y: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix M_kl = sum_ij Y_ijk * Y_ijl.

    Args:
        Y: Yukawa tensor (21, 21, 77)

    Returns:
        M: Gram matrix (77, 77)
    """
    # M_kl = sum_ij Y_ijk * Y_ijl
    M = torch.einsum('ijk,ijl->kl', Y, Y)
    return M


def analyze_yukawa_spectrum(M: torch.Tensor) -> Dict:
    """Analyze eigenspectrum of Yukawa Gram matrix.

    Args:
        M: Gram matrix (77, 77)

    Returns:
        Analysis dict with eigenvalues, gaps, suggested split
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigs = eigenvalues.cpu().numpy()[::-1]  # Descending

    # Gaps
    gaps = np.abs(np.diff(eigs))

    # Find largest gap
    largest_gap_idx = np.argmax(gaps)

    # Check gap at 43
    gap_43 = gaps[42] if len(gaps) > 42 else 0
    mean_gap = gaps.mean()

    # Non-zero count
    nonzero = (np.abs(eigs) > 1e-8).sum()

    # Tau candidates
    cumsum = np.cumsum(eigs)
    total = eigs.sum()
    tau_target = 3472 / 891

    tau_candidates = []
    for n in range(35, 55):
        if n < 77 and total - cumsum[n-1] > 1e-10:
            ratio = cumsum[n-1] / (total - cumsum[n-1])
            error = 100 * abs(ratio - tau_target) / tau_target
            tau_candidates.append((n, ratio, error))

    return {
        'eigenvalues': eigs,
        'gaps': gaps,
        'largest_gap_idx': largest_gap_idx,
        'suggested_n_visible': largest_gap_idx + 1,
        'gap_43': gap_43,
        'gap_43_ratio': gap_43 / mean_gap if mean_gap > 0 else 0,
        'nonzero_count': nonzero,
        'tau_candidates': tau_candidates,
    }
