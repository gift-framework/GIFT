"""Harmonic form extraction from trained PINN.

This module extracts the b2=21 harmonic 2-forms and b3=77 harmonic 3-forms
from a trained PINN that produces the G2 metric.

Two approaches:
1. Eigenvalue method: Solve Delta omega = lambda omega, take lambda ~ 0
2. Variational method: Train networks to minimize |d omega|^2 + |delta omega|^2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.optim as optim

from .config import HarmonicConfig, default_harmonic_config
from .hodge_laplacian import HodgeLaplacian, LaplacianResult


@dataclass
class HarmonicBasis:
    """Container for extracted harmonic forms."""
    h2_forms: torch.Tensor          # (n_points, 21, 21) - 21 forms, 21 components each
    h3_forms: torch.Tensor          # (n_points, 77, 35) - 77 forms, 35 components each
    h2_gram: torch.Tensor           # (21, 21) Gram matrix of H^2
    h3_gram: torch.Tensor           # (77, 77) Gram matrix of H^3
    sample_points: torch.Tensor     # (n_points, 7) sample coordinates
    metric_at_points: torch.Tensor  # (n_points, 7, 7) metric values

    @property
    def b2_actual(self) -> int:
        """Actual number of H^2 forms extracted."""
        return self.h2_forms.shape[1]

    @property
    def b3_actual(self) -> int:
        """Actual number of H^3 forms extracted."""
        return self.h3_forms.shape[1]

    def validate(self) -> dict:
        """Check orthonormality and dimensions."""
        h2_det = torch.det(self.h2_gram)
        h3_det = torch.det(self.h3_gram)

        h2_diag = torch.diag(self.h2_gram)
        h3_diag = torch.diag(self.h3_gram)

        return {
            "b2_actual": self.b2_actual,
            "b3_actual": self.b3_actual,
            "h2_gram_det": h2_det.item(),
            "h3_gram_det": h3_det.item(),
            "h2_norm_mean": h2_diag.mean().item(),
            "h3_norm_mean": h3_diag.mean().item(),
            "h2_orthogonal": (self.h2_gram - torch.eye(21)).abs().max().item(),
            "h3_orthogonal": (self.h3_gram - torch.eye(77)).abs().max().item(),
        }


class HarmonicFormNetwork(nn.Module):
    """Neural network for learning a single harmonic form.

    Outputs a p-form at each point x that minimizes:
        L = |d omega|^2 + |delta omega|^2 + orthogonality terms
    """

    def __init__(self, p: int, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.p = p
        self.n_components = 21 if p == 2 else 35  # C(7,p)

        # Fourier features for smooth approximation
        self.n_fourier = 32
        self.register_buffer('B', torch.randn(7, self.n_fourier) * 2.0)

        # MLP
        layers = []
        in_dim = 2 * self.n_fourier + 7
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, self.n_components))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate form at points x.

        Args:
            x: coordinates (batch, 7)

        Returns:
            omega: form coefficients (batch, n_components)
        """
        proj = 2 * math.pi * x @ self.B
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        features = torch.cat([x, fourier], dim=-1)
        return self.net(features)


class HarmonicExtractor:
    """Extract harmonic forms from a trained PINN metric.

    Uses variational approach: train networks to produce forms that
    satisfy d omega = 0 and delta omega = 0.
    """

    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        config: HarmonicConfig = None,
        device: str = 'cpu'
    ):
        """Initialize extractor.

        Args:
            metric_fn: Function x -> g(x) from trained PINN
            config: Configuration parameters
            device: Torch device
        """
        self.metric_fn = metric_fn
        self.config = config or default_harmonic_config
        self.device = device
        self.laplacian = HodgeLaplacian(config)

    def sample_points(self, n_points: int) -> torch.Tensor:
        """Generate sample points on K7.

        For now, use unit cube [0,1]^7 with periodic boundary.
        """
        return torch.rand(n_points, 7, device=self.device)

    def extract_h2_eigenvalue(
        self,
        n_points: int = 10000,
        n_basis: int = 50
    ) -> Tuple[torch.Tensor, LaplacianResult]:
        """Extract H^2 forms via eigenvalue method.

        Solves generalized eigenvalue problem for Hodge Laplacian.

        Returns:
            forms: (21, 21) orthonormal basis for H^2
            result: Full eigenvalue result
        """
        points = self.sample_points(n_points)
        result = self.laplacian.compute_laplacian_2forms(
            points, self.metric_fn, n_basis
        )

        # Take the 21 lowest eigenvalues (should be near zero)
        harmonic_forms = result.eigenvectors[:, :21]

        # Orthonormalize
        harmonic_forms, _ = torch.linalg.qr(harmonic_forms)

        return harmonic_forms, result

    def extract_h3_eigenvalue(
        self,
        n_points: int = 10000,
        n_basis: int = 100
    ) -> Tuple[torch.Tensor, LaplacianResult]:
        """Extract H^3 forms via eigenvalue method.

        Returns:
            forms: (77, 35) orthonormal basis for H^3
            result: Full eigenvalue result
        """
        points = self.sample_points(n_points)
        result = self.laplacian.compute_laplacian_3forms(
            points, self.metric_fn, n_basis
        )

        # Take the 77 lowest eigenvalues
        harmonic_forms = result.eigenvectors[:, :77]
        harmonic_forms, _ = torch.linalg.qr(harmonic_forms)

        return harmonic_forms, result

    def extract_h2_variational(
        self,
        n_epochs: int = None,
        n_sample: int = None
    ) -> List[HarmonicFormNetwork]:
        """Extract H^2 forms via variational training.

        Trains 21 networks, each outputting a 2-form that:
        1. Is closed (d omega = 0)
        2. Is coclosed (delta omega = 0)
        3. Is orthogonal to previously trained forms

        Returns:
            List of 21 trained HarmonicFormNetwork modules
        """
        n_epochs = n_epochs or self.config.harmonic_epochs
        n_sample = n_sample or self.config.n_sample_points

        networks = []
        frozen_outputs = []  # Store outputs of already-trained networks

        for form_idx in range(self.config.b2):
            print(f"Training H^2 form {form_idx + 1}/{self.config.b2}")

            net = HarmonicFormNetwork(p=2).to(self.device)
            optimizer = optim.Adam(net.parameters(), lr=self.config.harmonic_lr)

            for epoch in range(n_epochs):
                optimizer.zero_grad()

                # Sample points
                x = self.sample_points(n_sample)
                g = self.metric_fn(x)
                det_g = torch.det(g)
                vol = torch.sqrt(det_g.abs().clamp(min=self.config.eps))

                # Get current form
                omega = net(x)

                # Closedness loss: |d omega|^2 (approximate via finite diff)
                loss_closed = self._closedness_loss_2form(omega, x, net)

                # Coclosedness loss: |delta omega|^2
                loss_coclosed = self._coclosedness_loss_2form(omega, x, g, net)

                # Normalization loss: ||omega||^2 - 1
                norm_sq = (omega ** 2 * vol.unsqueeze(-1)).mean()
                loss_norm = (norm_sq - 1.0) ** 2

                # Orthogonality loss with previous forms
                loss_ortho = torch.tensor(0.0, device=self.device)
                for prev_output in frozen_outputs:
                    inner = (omega * prev_output * vol.unsqueeze(-1)).mean()
                    loss_ortho = loss_ortho + inner ** 2

                # Total loss
                loss = (self.config.closedness_weight * loss_closed +
                        self.config.coclosedness_weight * loss_coclosed +
                        self.config.orthonormality_weight * (loss_norm + loss_ortho))

                loss.backward()
                optimizer.step()

                if epoch % 500 == 0:
                    print(f"  Epoch {epoch}: loss={loss.item():.6f}")

            # Freeze and store output
            net.eval()
            with torch.no_grad():
                x_eval = self.sample_points(n_sample)
                frozen_outputs.append(net(x_eval).detach())
            networks.append(net)

        return networks

    def _closedness_loss_2form(
        self,
        omega: torch.Tensor,
        x: torch.Tensor,
        net: HarmonicFormNetwork
    ) -> torch.Tensor:
        """Compute |d omega|^2 via finite differences.

        d omega for a 2-form gives a 3-form.
        We approximate partial derivatives numerically.
        """
        eps = 1e-4
        d_omega_sq = torch.zeros(1, device=x.device)

        for i in range(7):
            x_plus = x.clone()
            x_plus[:, i] += eps
            omega_plus = net(x_plus)

            x_minus = x.clone()
            x_minus[:, i] -= eps
            omega_minus = net(x_minus)

            d_omega_i = (omega_plus - omega_minus) / (2 * eps)
            d_omega_sq = d_omega_sq + (d_omega_i ** 2).mean()

        return d_omega_sq

    def _coclosedness_loss_2form(
        self,
        omega: torch.Tensor,
        x: torch.Tensor,
        g: torch.Tensor,
        net: HarmonicFormNetwork
    ) -> torch.Tensor:
        """Compute |delta omega|^2 where delta = *d*.

        For 2-forms on K7: delta: Omega^2 -> Omega^1.
        Approximate using Hodge star + d + Hodge star.
        """
        # Simplified: use divergence-like approximation
        g_inv = torch.linalg.inv(g)
        eps = 1e-4

        delta_omega = torch.zeros_like(x[:, :1])

        for i in range(7):
            x_plus = x.clone()
            x_plus[:, i] += eps
            omega_plus = net(x_plus)

            x_minus = x.clone()
            x_minus[:, i] -= eps
            omega_minus = net(x_minus)

            d_omega_i = (omega_plus - omega_minus) / (2 * eps)

            # Contract with g^{-1}
            delta_omega = delta_omega + (g_inv[:, i, :].unsqueeze(1) @ d_omega_i.unsqueeze(-1)).squeeze(-1).mean(dim=-1, keepdim=True)

        return (delta_omega ** 2).mean()

    def extract_full_basis(
        self,
        method: str = "eigenvalue"
    ) -> HarmonicBasis:
        """Extract complete harmonic basis H^2(21) and H^3(77).

        Args:
            method: "eigenvalue" or "variational"

        Returns:
            HarmonicBasis with all extracted forms
        """
        print("Extracting harmonic forms...")

        n_points = self.config.n_sample_points
        points = self.sample_points(n_points)
        g = self.metric_fn(points)

        if method == "eigenvalue":
            h2_coeff, _ = self.extract_h2_eigenvalue(n_points)
            h3_coeff, _ = self.extract_h3_eigenvalue(n_points)

            # Expand to all points: (n_points, b, n_components)
            # For eigenvalue method, coefficients are in a fixed basis
            h2_forms = h2_coeff.T.unsqueeze(0).expand(n_points, -1, -1)
            h3_forms = h3_coeff.T.unsqueeze(0).expand(n_points, -1, -1)

        else:
            raise NotImplementedError("Variational method requires training")

        # Compute Gram matrices
        det_g = torch.det(g)
        vol = torch.sqrt(det_g.abs().clamp(min=self.config.eps))

        # H^2 Gram: G_{ab} = <omega_a, omega_b>
        h2_weighted = h2_forms * vol.unsqueeze(-1).unsqueeze(-1)
        h2_gram = torch.einsum('nia,njb->ab', h2_weighted, h2_forms) / n_points

        # H^3 Gram
        h3_weighted = h3_forms * vol.unsqueeze(-1).unsqueeze(-1)
        h3_gram = torch.einsum('nia,njb->ab', h3_weighted, h3_forms) / n_points

        return HarmonicBasis(
            h2_forms=h2_forms,
            h3_forms=h3_forms,
            h2_gram=h2_gram,
            h3_gram=h3_gram,
            sample_points=points,
            metric_at_points=g
        )
