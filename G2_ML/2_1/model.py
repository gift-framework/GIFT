"""G2 Variational Network for GIFT v2.2.

This module implements a Physics-Informed Neural Network (PINN) that solves
the constrained G2 variational problem:

    Minimize: F[phi] = ||d phi||^2 + ||d* phi||^2

    Subject to:
    - det(g(phi)) = 65/32
    - kappa_T = 1/61
    - phi in Lambda^3_+ (G2 positivity)
    - (b2, b3) = (21, 77) cohomology

The network outputs the 35 independent components of a 3-form phi on R^7.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math

from .config import GIFTConfig, default_config
from .g2_geometry import (
    MetricFromPhi, G2Positivity, standard_phi_coefficients,
    normalize_phi, get_3form_indices
)


# =============================================================================
# Fourier feature encoding
# =============================================================================

class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding.

    Maps x in R^7 to sin/cos features for smooth function approximation.
    This helps the network learn smooth, periodic-like functions.
    """

    def __init__(self, in_dim: int = 7, n_features: int = 64, scale: float = 2.0):
        super().__init__()
        self.n_features = n_features

        # Random frequencies (fixed, not learned)
        B = torch.randn(in_dim, n_features) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping.

        Args:
            x: coordinates (batch, 7)

        Returns:
            features: (batch, 2*n_features)
        """
        # x @ B: (batch, n_features)
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# =============================================================================
# Main network
# =============================================================================

class G2VariationalNet(nn.Module):
    """Network solving the constrained G2 variational problem.

    Architecture:
    1. Fourier features for smooth encoding
    2. Deep MLP for function approximation
    3. Output: 35 independent 3-form components
    4. Optional: projection to G2 positive cone

    The network learns the minimizer of F[phi] = ||d phi||^2 + ||d* phi||^2
    subject to GIFT v2.2 constraints.
    """

    def __init__(self, config: GIFTConfig = None):
        super().__init__()
        self.config = config or default_config

        # Fourier feature encoding
        self.fourier = FourierFeatures(
            in_dim=self.config.dim,
            n_features=self.config.fourier_features,
            scale=self.config.fourier_scale
        )

        fourier_dim = 2 * self.config.fourier_features

        # Deep MLP
        layers = []
        in_dim = fourier_dim + self.config.dim  # Fourier + raw coords

        for i in range(self.config.n_layers):
            out_dim = self.config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())  # Smooth activation
            if i < self.config.n_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)

        # Output head: 35 independent components
        self.phi_head = nn.Linear(self.config.hidden_dim, self.config.n_phi_components)

        # Initialize near standard G2
        self._initialize_weights()

        # Geometry modules
        self.metric_extractor = MetricFromPhi()
        self.positivity = G2Positivity()

        # Store standard phi for reference
        self.register_buffer('phi_0', standard_phi_coefficients())

    def _initialize_weights(self):
        """Initialize to produce output near standard G2 structure."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Bias output head toward standard phi
        with torch.no_grad():
            phi_0 = standard_phi_coefficients()
            self.phi_head.bias.copy_(phi_0)
            self.phi_head.weight.mul_(0.01)

    def forward(self, x: torch.Tensor,
                project_positive: bool = True) -> torch.Tensor:
        """Compute phi(x) for given coordinates.

        Args:
            x: coordinates (batch, 7)
            project_positive: whether to project toward positive cone

        Returns:
            phi: 3-form components (batch, 35)
        """
        # Fourier features
        fourier_feats = self.fourier(x)

        # Concatenate with raw coordinates
        combined = torch.cat([x, fourier_feats], dim=-1)

        # MLP
        features = self.backbone(combined)

        # Output 3-form
        phi = self.phi_head(features)

        # Optional: soft projection toward positive cone
        if project_positive:
            phi = self.positivity.project_to_positive(phi, alpha=0.5)

        return phi

    def get_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Extract induced metric at given points.

        Args:
            x: coordinates (batch, 7)

        Returns:
            g: metric tensor (batch, 7, 7)
        """
        phi = self.forward(x, project_positive=True)
        return self.metric_extractor(phi)

    def get_phi_and_metric(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both phi and induced metric.

        Args:
            x: coordinates (batch, 7)

        Returns:
            phi: 3-form (batch, 35)
            g: metric (batch, 7, 7)
        """
        phi = self.forward(x, project_positive=True)
        g = self.metric_extractor(phi)
        return phi, g


# =============================================================================
# Harmonic form networks (for cohomology extraction)
# =============================================================================

class HarmonicFormsNet(nn.Module):
    """Network for learning harmonic forms from the G2 structure.

    Given a trained G2VariationalNet producing phi, this network learns:
    - 21 harmonic 2-forms (b2 = 21)
    - 77 harmonic 3-forms (b3 = 77)

    These are used to verify the cohomology constraints.
    """

    def __init__(self, config: GIFTConfig = None,
                 g2_net: Optional[G2VariationalNet] = None):
        super().__init__()
        self.config = config or default_config
        self.g2_net = g2_net

        # Fourier features (shared architecture)
        self.fourier = FourierFeatures(
            in_dim=self.config.dim,
            n_features=self.config.fourier_features,
            scale=self.config.fourier_scale
        )

        fourier_dim = 2 * self.config.fourier_features

        # Shared backbone
        layers = []
        in_dim = fourier_dim + self.config.dim

        for i in range(self.config.n_layers - 1):
            out_dim = self.config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)

        # H2 head: 21 forms, each with 21 components
        self.h2_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.n_2form_components)
            for _ in range(self.config.b2_K7)
        ])

        # H3 head: 77 forms, each with 35 components
        self.h3_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.n_phi_components)
            for _ in range(self.config.b3_K7)
        ])

    def forward_h2(self, x: torch.Tensor) -> torch.Tensor:
        """Compute harmonic 2-forms.

        Args:
            x: coordinates (batch, 7)

        Returns:
            omega: 2-forms (batch, 21 modes, 21 components)
        """
        fourier_feats = self.fourier(x)
        combined = torch.cat([x, fourier_feats], dim=-1)
        features = self.backbone(combined)

        return torch.stack([head(features) for head in self.h2_heads], dim=1)

    def forward_h3(self, x: torch.Tensor) -> torch.Tensor:
        """Compute harmonic 3-forms.

        Args:
            x: coordinates (batch, 7)

        Returns:
            Phi: 3-forms (batch, 77 modes, 35 components)
        """
        fourier_feats = self.fourier(x)
        combined = torch.cat([x, fourier_feats], dim=-1)
        features = self.backbone(combined)

        return torch.stack([head(features) for head in self.h3_heads], dim=1)

    def gram_matrix_h2(self, x: torch.Tensor,
                       metric: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for H2 forms.

        Args:
            x: coordinates (batch, 7)
            metric: metric tensor (batch, 7, 7)

        Returns:
            G: Gram matrix (21, 21)
        """
        omega = self.forward_h2(x)  # (batch, 21, 21)

        # Volume element
        det_g = torch.det(metric)
        vol = torch.sqrt(det_g.abs())

        # Weighted inner product
        weighted = omega * vol.unsqueeze(-1).unsqueeze(-1)
        G = torch.einsum('bic,bjc->ij', weighted, omega) / omega.shape[0]

        return G

    def gram_matrix_h3(self, x: torch.Tensor,
                       metric: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for H3 forms.

        Args:
            x: coordinates (batch, 7)
            metric: metric tensor (batch, 7, 7)

        Returns:
            G: Gram matrix (77, 77)
        """
        Phi = self.forward_h3(x)  # (batch, 77, 35)

        det_g = torch.det(metric)
        vol = torch.sqrt(det_g.abs())

        weighted = Phi * vol.unsqueeze(-1).unsqueeze(-1)
        G = torch.einsum('bic,bjc->ij', weighted, Phi) / Phi.shape[0]

        return G


# =============================================================================
# Combined model for full pipeline
# =============================================================================

class GIFTVariationalModel(nn.Module):
    """Complete GIFT v2.2 variational model.

    Combines:
    - G2VariationalNet: learns phi minimizing torsion
    - HarmonicFormsNet: extracts harmonic forms for cohomology

    Jointly optimized with GIFT constraints.
    """

    def __init__(self, config: GIFTConfig = None):
        super().__init__()
        self.config = config or default_config

        # Main G2 network
        self.g2_net = G2VariationalNet(config)

        # Harmonic forms network (initialized after G2 is trained)
        self.harmonic_net = HarmonicFormsNet(config, self.g2_net)

        # Track training phase
        self.current_phase = 0

    def forward_phi(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get phi and metric.

        Args:
            x: coordinates (batch, 7)

        Returns:
            phi: 3-form (batch, 35)
            g: metric (batch, 7, 7)
        """
        return self.g2_net.get_phi_and_metric(x)

    def forward_harmonic(self, x: torch.Tensor,
                         metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get harmonic forms.

        Args:
            x: coordinates (batch, 7)
            metric: metric tensor (batch, 7, 7)

        Returns:
            omega: H2 forms (batch, 21, 21)
            Phi: H3 forms (batch, 77, 35)
        """
        omega = self.harmonic_net.forward_h2(x)
        Phi = self.harmonic_net.forward_h3(x)
        return omega, Phi

    def set_phase(self, phase: int):
        """Set training phase (affects loss weights)."""
        self.current_phase = phase

    def get_phase_weights(self) -> dict:
        """Get loss weights for current phase."""
        if self.current_phase < len(self.config.phases):
            return self.config.phases[self.current_phase]["weights"]
        return self.config.phases[-1]["weights"]
