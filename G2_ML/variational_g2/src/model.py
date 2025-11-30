"""
G2VariationalNet: Neural Network for G2 Variational Problem

This module implements the Physics-Informed Neural Network architecture
for solving the constrained G2 variational problem.

The network learns to output a 3-form phi(x) on R7 that:
1. Minimizes the torsion functional F[phi] = ||d*phi||^2 + ||d*phi||^2
2. Satisfies GIFT v2.2 constraints (det(g), kappa_T, positivity)

Architecture:
    Input: x in R7 (coordinates)
    Fourier Features: Encode periodic structure
    MLP: Learn smooth 3-form field
    Output: 35 independent components -> full antisymmetric tensor
    Projection: Ensure positivity (phi in G2 cone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math

from .constraints import (
    expand_to_antisymmetric,
    metric_from_phi,
    g2_positivity_check,
    standard_g2_phi,
)


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for smooth periodic structure.

    Maps x in R7 to [sin(2*pi*B*x), cos(2*pi*B*x)] where B is a
    learnable or fixed frequency matrix.

    This encoding helps the network learn smooth, periodic functions
    and improves convergence for geometric problems.
    """

    def __init__(
        self,
        input_dim: int = 7,
        num_frequencies: int = 64,
        scale: float = 1.0,
        learnable: bool = False
    ):
        """
        Args:
            input_dim: Dimension of input (7 for R7)
            num_frequencies: Number of Fourier frequencies
            scale: Scale factor for frequencies
            learnable: Whether frequencies are learnable
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = 2 * num_frequencies

        # Initialize frequency matrix
        B = torch.randn(num_frequencies, input_dim) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Encoded tensor of shape (..., 2 * num_frequencies)
        """
        # x @ B.T: (..., num_frequencies)
        x_proj = 2 * math.pi * torch.matmul(x, self.B.T)

        # Concatenate sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim: int, activation: nn.Module = nn.SiLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class G2VariationalNet(nn.Module):
    """
    Physics-Informed Neural Network for G2 variational problem.

    This network learns a 3-form phi(x) on R7 that minimizes the
    torsion functional subject to GIFT v2.2 constraints.

    Architecture:
        1. Fourier feature encoding for periodic structure
        2. MLP with residual connections for smooth output
        3. Linear projection to 35 independent 3-form components
        4. Optional positivity projection onto G2 cone

    The output phi satisfies:
        - phi is antisymmetric (by construction)
        - phi(x) is smooth in x (Fourier + smooth activations)
        - phi > 0 optional (projection onto G2 cone)
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: List[int] = [256, 512, 512, 256],
        num_frequencies: int = 64,
        fourier_scale: float = 1.0,
        activation: str = 'silu',
        enforce_positivity: bool = True,
        use_residual: bool = True,
        dropout: float = 0.0,
        init_near_standard: bool = True,
    ):
        """
        Args:
            input_dim: Coordinate dimension (7)
            hidden_dims: List of hidden layer dimensions
            num_frequencies: Number of Fourier frequencies
            fourier_scale: Scale for Fourier features
            activation: Activation function ('silu', 'gelu', 'tanh')
            enforce_positivity: Whether to project onto G2 cone
            use_residual: Whether to use residual connections
            dropout: Dropout probability
            init_near_standard: Initialize near standard G2 form
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 35  # Independent 3-form components
        self.enforce_positivity = enforce_positivity

        # Fourier feature encoding
        self.fourier = FourierFeatures(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
            scale=fourier_scale,
            learnable=False
        )

        # Activation function
        if activation == 'silu':
            act = nn.SiLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            act = nn.SiLU()

        # Build MLP
        layers = []
        prev_dim = self.fourier.output_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            if use_residual and i > 0 and hidden_dim == hidden_dims[i - 1]:
                # Add residual connection for same-size layers
                layers.append(ResidualBlock(hidden_dim, act))

            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer: map to 35 components
        self.output_layer = nn.Linear(prev_dim, self.output_dim)

        # Learnable scale and bias for output
        self.output_scale = nn.Parameter(torch.ones(self.output_dim))
        self.output_bias = nn.Parameter(torch.zeros(self.output_dim))

        # Initialize
        self._initialize_weights(init_near_standard)

    def _initialize_weights(self, init_near_standard: bool):
        """Initialize network weights."""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if init_near_standard:
            # Initialize output bias near standard G2 form
            with torch.no_grad():
                standard = standard_g2_phi()
                self.output_bias.copy_(standard)
                # Small scale to start near standard
                self.output_scale.fill_(0.1)

    def forward(
        self,
        x: torch.Tensor,
        return_full: bool = False,
        return_metric: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: compute phi(x).

        Args:
            x: Coordinates of shape (batch, 7)
            return_full: If True, return full 7x7x7 tensor
            return_metric: If True, also return induced metric

        Returns:
            Dictionary with:
                - 'phi_components': (batch, 35) independent components
                - 'phi_full': (batch, 7, 7, 7) if return_full
                - 'metric': (batch, 7, 7) if return_metric
        """
        # Fourier encoding
        x_encoded = self.fourier(x)

        # MLP
        h = self.mlp(x_encoded)

        # Output: 35 components
        phi_raw = self.output_layer(h)

        # Apply learnable scale and bias
        phi_components = phi_raw * self.output_scale + self.output_bias

        # Optional positivity projection
        if self.enforce_positivity:
            phi_components = self._project_to_cone(phi_components)

        # Build output dictionary
        output = {'phi_components': phi_components}

        if return_full or return_metric:
            phi_full = expand_to_antisymmetric(phi_components)
            if return_full:
                output['phi_full'] = phi_full
            if return_metric:
                output['metric'] = metric_from_phi(phi_full)

        return output

    def _project_to_cone(
        self,
        phi_components: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Project phi onto G2 cone by ensuring metric positivity.

        This is a soft projection that scales phi to ensure
        positive definite metric.
        """
        phi_full = expand_to_antisymmetric(phi_components)
        g = metric_from_phi(phi_full)

        # Check eigenvalues
        eigenvalues = torch.linalg.eigvalsh(g)
        min_eigenvalue = eigenvalues.min(dim=-1)[0]

        # If any negative, scale up phi
        # (scaling phi scales g quadratically)
        needs_fix = min_eigenvalue < eps
        if needs_fix.any():
            # Simple fix: add small multiple of identity-inducing form
            standard = standard_g2_phi(device=phi_components.device, dtype=phi_components.dtype)
            correction = standard.unsqueeze(0).expand_as(phi_components)

            # Blend with standard form where needed
            alpha = torch.where(
                needs_fix.unsqueeze(-1),
                torch.full_like(phi_components, 0.1),
                torch.zeros_like(phi_components)
            )
            phi_components = phi_components * (1 - alpha) + correction * alpha

        return phi_components

    def get_phi_at_points(
        self,
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to get phi and metric at given points.

        Args:
            points: Coordinates of shape (N, 7)

        Returns:
            phi_full: (N, 7, 7, 7)
            metric: (N, 7, 7)
        """
        output = self.forward(points, return_full=True, return_metric=True)
        return output['phi_full'], output['metric']

    def compute_derivatives(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phi and its derivatives for torsion calculation.

        Args:
            x: Coordinates requiring grad, shape (batch, 7)

        Returns:
            Dictionary with phi, d_phi, metric, etc.
        """
        x = x.requires_grad_(True)

        output = self.forward(x, return_full=True, return_metric=True)
        phi = output['phi_full']
        metric = output['metric']

        # Compute gradient of each phi component w.r.t. x
        # d_phi[l,i,j,k] = partial_l phi_{ijk}
        batch_size = x.shape[0]
        d_phi = torch.zeros(batch_size, 7, 7, 7, 7, device=x.device, dtype=x.dtype)

        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if i < j < k:  # Only independent components
                        grad = torch.autograd.grad(
                            phi[:, i, j, k].sum(),
                            x,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        # Fill in antisymmetric positions
                        d_phi[:, :, i, j, k] = grad
                        d_phi[:, :, i, k, j] = -grad
                        d_phi[:, :, j, i, k] = -grad
                        d_phi[:, :, j, k, i] = grad
                        d_phi[:, :, k, i, j] = grad
                        d_phi[:, :, k, j, i] = -grad

        output['d_phi'] = d_phi
        return output


class G2VariationalNetLite(nn.Module):
    """
    Lightweight version for faster experimentation.

    Smaller network with same interface but fewer parameters.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_frequencies: int = 32,
    ):
        super().__init__()

        self.fourier = FourierFeatures(
            input_dim=input_dim,
            num_frequencies=num_frequencies,
        )

        layers = [nn.Linear(self.fourier.output_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 35))

        self.net = nn.Sequential(*layers)

        # Initialize near standard
        self.bias = nn.Parameter(standard_g2_phi())

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        x_enc = self.fourier(x)
        phi_components = self.net(x_enc) * 0.1 + self.bias
        return {'phi_components': phi_components}


def create_model(config: Dict[str, Any]) -> G2VariationalNet:
    """
    Create model from configuration dictionary.

    Args:
        config: Configuration with model parameters

    Returns:
        Configured G2VariationalNet instance
    """
    model_config = config.get('model', {})

    return G2VariationalNet(
        input_dim=model_config.get('input_dim', 7),
        hidden_dims=model_config.get('hidden_layers', [256, 512, 512, 256]),
        num_frequencies=model_config.get('fourier_features', {}).get('num_frequencies', 64),
        fourier_scale=model_config.get('fourier_features', {}).get('scale', 1.0),
        activation=model_config.get('activation', 'silu'),
        enforce_positivity=model_config.get('positivity', {}).get('method', None) is not None,
    )
