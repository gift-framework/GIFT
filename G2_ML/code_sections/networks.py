"""
Neural Network Architectures for GIFT v0.9
============================================

This module contains all neural network architectures for learning G₂ structures
on the TCS neck manifold with ACyl boundaries.

Networks:
- G2PhiNetwork_TCS: 35-component 3-form φ with boundary awareness
- MetricNetwork: 28 coefficients → 7×7 SPD metric tensor
- BoundaryNetwork: ACyl exponential decay modeling (FIXED v0.8)
- Harmonic2FormsNetwork_TCS: 21 independent b₂ harmonic forms

Critical Fixes from v0.8:
- No clone() bug in gradient computations
- Min eigenvalue ≥ 0.3 for SPD projection
- Correct ACyl decay: exp(-γ|t|/T) from center
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class G2PhiNetwork_TCS(nn.Module):
    """
    Neural network for the G₂ 3-form φ on TCS neck manifold.

    Architecture:
    - Input: Fourier-encoded coordinates (7D → ~70D)
    - Hidden layers: [256, 256, 128] with SiLU activation
    - Output: 35 components (3-form in 7D)
    - Normalization: LayerNorm + manual ||φ|| = √7

    Boundary Features:
    - Gluing rotation for boundary matching
    - Soft decay at boundaries for torsion-free matching

    Args:
        manifold: TCSNeckManifold instance
        hidden_dims: List of hidden layer dimensions (default: [256, 256, 128])
    """

    def __init__(self, manifold, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.manifold = manifold

        # Determine Fourier encoding dimension
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]

        # Build MLP with SiLU activation and LayerNorm
        layers = []
        prev_dim = encoding_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),  # Smoother gradients than ReLU
                nn.LayerNorm(h_dim)
            ])
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 35)  # 35 components for 3-form φ

        # Initialize output layer to small values
        with torch.no_grad():
            self.output.weight.mul_(0.01)
            self.output.bias.zero_()

    def forward(self, coords):
        """
        Forward pass through φ network.

        Args:
            coords: (batch, 7) tensor of manifold coordinates

        Returns:
            phi: (batch, 35) tensor of 3-form components, normalized to ||φ|| = √7
        """
        # Apply gluing rotation for boundary matching
        coords_rotated = self.manifold.apply_gluing_rotation(coords)

        # Fourier encoding for positional information
        x = self.manifold.fourier_encoding(coords_rotated)

        # Process through MLP
        x = self.mlp(x)
        phi = self.output(x)

        # Normalize to √7 (standard G₂ normalization)
        phi_norm = torch.norm(phi, dim=-1, keepdim=True)
        phi = phi * (np.sqrt(7.0) / (phi_norm + 1e-8))

        # Apply boundary decay for torsion-free matching at boundaries
        decay = self.manifold.boundary_decay_factor(coords)
        phi = phi * (1 - decay * 0.5)  # Soft decay

        return phi


class MetricNetwork(nn.Module):
    """
    Neural network for the Riemannian metric tensor g on 7D manifold.

    Architecture:
    - Input: Fourier-encoded coordinates (7D → ~70D)
    - Hidden layers: [512, 512, 256, 256, 128] (deeper for metric complexity)
    - Output: 28 coefficients → 7×7 symmetric positive-definite (SPD) matrix

    SPD Projection (CRITICAL):
    - Diagonal: exp-transformed to ensure positivity
    - Eigenvalue clamping: λ ≥ 0.3 (v0.8 fix, was 0.1 in v0.7)
    - Volume normalization: det(g) = 1

    Args:
        manifold: TCSNeckManifold instance
        hidden_dims: List of hidden layer dimensions (default: [512, 512, 256, 256, 128])
    """

    def __init__(self, manifold, hidden_dims=[512, 512, 256, 256, 128]):
        super().__init__()
        self.manifold = manifold

        # Determine Fourier encoding dimension
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]

        # Deep MLP (metrics need more capacity than forms)
        layers = []
        prev_dim = encoding_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU(),
                nn.LayerNorm(h_dim)
            ])
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 28)  # 7 diagonal + 21 off-diagonal

        # Initialize to near-identity metric
        with torch.no_grad():
            self.output.weight.mul_(0.01)
            self.output.bias.zero_()

    def forward(self, coords):
        """
        Forward pass to compute metric coefficients.

        Args:
            coords: (batch, 7) tensor of manifold coordinates

        Returns:
            coeffs: (batch, 28) tensor of metric coefficients
        """
        coords_rotated = self.manifold.apply_gluing_rotation(coords)
        x = self.manifold.fourier_encoding(coords_rotated)
        x = self.mlp(x)
        coeffs = self.output(x)

        # Soft boundary modulation (smoother near boundaries)
        decay = self.manifold.boundary_decay_factor(coords)
        boundary_mod = torch.sigmoid(10 * (1 - decay))
        coeffs = coeffs * boundary_mod.unsqueeze(-1)

        return coeffs

    def coeffs_to_metric(self, coeffs):
        """
        Convert 28 coefficients to 7×7 SPD metric tensor.

        Process:
        1. Extract diagonal (7) and off-diagonal (21) components
        2. Build symmetric matrix
        3. SPD projection via eigenvalue clamping (CRITICAL: min ≥ 0.3)
        4. Volume normalization: det(g) = 1

        Args:
            coeffs: (batch, 28) tensor of metric coefficients

        Returns:
            metric: (batch, 7, 7) SPD metric tensor
        """
        batch_size = coeffs.shape[0]
        device = coeffs.device

        # Extract diagonal (exp-transformed) and off-diagonal
        diag_raw = coeffs[:, :7]
        off_diag = coeffs[:, 7:]

        # Diagonal: exp to ensure positivity
        diag = torch.exp(diag_raw) + 0.1

        # Build symmetric matrix
        metric = torch.zeros(batch_size, 7, 7, device=device, dtype=coeffs.dtype)

        # Set diagonal
        for i in range(7):
            metric[:, i, i] = diag[:, i]

        # Set upper/lower triangular (symmetric)
        idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                metric[:, i, j] = off_diag[:, idx]
                metric[:, j, i] = off_diag[:, idx]
                idx += 1

        # SPD projection via eigenvalue clamping
        eye = torch.eye(7, device=device, dtype=coeffs.dtype).unsqueeze(0)
        metric = metric + 0.01 * eye  # Small regularization

        eigvals, eigvecs = torch.linalg.eigh(metric)
        eigvals = torch.clamp(eigvals, min=0.3)  # CRITICAL: min_eig ≥ 0.3 (v0.8 fix)
        metric = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)

        # Volume normalization: det(g) = 1
        vol = torch.sqrt(torch.abs(torch.det(metric)) + 1e-8)
        metric = metric / (vol.unsqueeze(-1).unsqueeze(-1) ** (2/7))

        return metric


class BoundaryNetwork(nn.Module):
    """
    Models ACyl boundary transitions with exponential decay from center.

    CRITICAL FIX v0.8:
    - Old v0.7: exp(-γ * dist_from_boundary) → U-shape artifact
    - New v0.8: exp(-γ * |t|/T) from center → Proper monotonic decay

    Physical Interpretation:
    - γ = 0.578: Phenomenological ACyl decay rate
    - T_neck ≈ 24.48: Neck length parameter
    - Decay: 0 at center (t=0), 1 at boundaries (|t|=T)

    Args:
        manifold: TCSNeckManifold instance
        gamma: ACyl decay rate (default: 0.578)
        acyl_width: Characteristic width scale (default: 3.0)
    """

    def __init__(self, manifold, gamma=0.578, acyl_width=3.0):
        super().__init__()
        self.manifold = manifold
        self.gamma = gamma
        self.acyl_width = acyl_width

        # Learnable fine-tuning parameters
        self.gamma_offset = nn.Parameter(torch.zeros(1))
        self.amplitude = nn.Parameter(torch.ones(1))

    def forward(self, coords):
        """
        Compute boundary transition factors.

        Returns values in [0, 1]:
        - 0: near center of neck (t ≈ 0)
        - 1: at/beyond boundaries (|t| ≈ T)

        Args:
            coords: (batch, 7) tensor of manifold coordinates

        Returns:
            boundary_factor: (batch,) tensor in [0, 1]
        """
        t = coords[:, 0]
        T = self.manifold.T_neck  # Use T_neck not T_boundary

        # Effective gamma with learnable offset
        gamma_eff = self.gamma + 0.01 * torch.tanh(self.gamma_offset)

        # FIXED v0.8: Distance from CENTER (not boundaries!)
        t_norm = torch.abs(t) / T  # |t|/T ∈ [0, 1]

        # Pure exponential decay from center
        decay = torch.exp(-gamma_eff * t_norm)

        # Smooth transition at center to ensure C² continuity
        smooth = torch.sigmoid(5.0 * (0.5 - t_norm))

        # Combine: smooth at center, exponential at boundaries
        boundary_factor = smooth + (1 - smooth) * decay

        # Convert to [0,1]: 0 at center, 1 at boundaries
        boundary_factor = 1 - boundary_factor

        return torch.clamp(boundary_factor * self.amplitude, 0, 1)

    def compute_acyl_decay(self, t):
        """
        Explicit ACyl decay formula: exp(-γ|t|/T)

        Args:
            t: (batch,) tensor of t-coordinates

        Returns:
            decay: (batch,) tensor of decay factors
        """
        T = self.manifold.T_neck
        gamma_eff = self.gamma + 0.01 * torch.tanh(self.gamma_offset)

        t_norm = torch.abs(t) / T
        decay = torch.exp(-gamma_eff * t_norm)

        smooth = torch.sigmoid(5.0 * (0.5 - t_norm))
        return smooth + (1 - smooth) * decay


class Harmonic2FormsNetwork_TCS(nn.Module):
    """
    Ensemble of 21 neural networks for harmonic 2-forms (b₂ cohomology).

    Critical Design:
    - DISTINCT initializations per form to break symmetry
    - Dropout for regularization (prevents mode collapse)
    - Form-specific perturbations during forward pass

    Architecture (per form):
    - Input: Fourier-encoded coordinates (7D → ~70D)
    - Hidden layers: [128, 128] with SiLU and Dropout
    - Output: 21 components (2-form in 7D)

    Args:
        manifold: TCSNeckManifold instance
        hidden_dims: List of hidden layer dimensions (default: [128, 128])
        n_forms: Number of harmonic forms (default: 21, b₂ for TCS)
        output_dim: Dimension of each form (default: 21)
    """

    def __init__(self, manifold, hidden_dims=[128, 128], n_forms=21, output_dim=21):
        super().__init__()
        self.n_forms = n_forms
        self.output_dim = output_dim
        self.manifold = manifold

        # Determine Fourier encoding dimension
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]

        # CRITICAL: Create networks with DIFFERENT initializations per form
        self.networks = nn.ModuleList()
        for form_idx in range(n_forms):
            # Unique seed per form to ensure diversity
            torch.manual_seed(47 + form_idx * 100)

            net = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dims[0]),
                nn.SiLU(),
                nn.Dropout(0.1),  # Regularization
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[1], output_dim)
            )

            # Unique initialization per form (prevents mode collapse)
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5 + form_idx * 0.05)
                    nn.init.constant_(layer.bias, 0.01 * form_idx)

            self.networks.append(net)

        # Reset seed for reproducibility
        torch.manual_seed(47)

    def forward(self, coords):
        """
        Forward pass through all harmonic form networks.

        Args:
            coords: (batch, 7) tensor of manifold coordinates

        Returns:
            forms: (batch, n_forms, output_dim) tensor of harmonic forms
        """
        coords_rotated = self.manifold.apply_gluing_rotation(coords)
        features = self.manifold.fourier_encoding(coords_rotated)

        forms = []
        for form_idx, net in enumerate(self.networks):
            # Add form-specific perturbation to break symmetry
            noise = torch.randn_like(features) * 0.01 * (form_idx + 1) / self.n_forms
            features_perturbed = features + noise
            form = net(features_perturbed)
            forms.append(form)

        return torch.stack(forms, dim=1)  # (batch, n_forms, output_dim)

    def compute_gram_matrix(self, coords, forms, metric):
        """
        Compute Gram matrix for harmonic forms with proper L² inner product.

        Gram[α, β] = ∫_M <ω_α, ω_β>_g √det(g) dV

        Approximated via Monte Carlo integration over batch.

        Args:
            coords: (batch, 7) tensor of coordinates
            forms: (batch, n_forms, output_dim) tensor of harmonic forms
            metric: (batch, 7, 7) tensor of metric

        Returns:
            gram_normalized: (n_forms, n_forms) normalized Gram matrix
        """
        batch_size = coords.shape[0]
        n_forms = forms.shape[1]
        gram = torch.zeros(n_forms, n_forms, device=coords.device)

        # Volume element: √det(g)
        vol = torch.sqrt(torch.abs(torch.det(metric)) + 1e-10)

        # Compute pairwise inner products
        for alpha in range(n_forms):
            for beta in range(alpha, n_forms):
                # L² inner product: <ω_α, ω_β> = sum_i ω_α^i ω_β^i
                inner = torch.sum(forms[:, alpha, :] * forms[:, beta, :], dim=-1) * vol
                gram[alpha, beta] = inner.mean()
                gram[beta, alpha] = gram[alpha, beta]  # Symmetry

        # Normalize to unit diagonal
        diag = torch.diagonal(gram)
        scale = torch.sqrt(diag + 1e-8)
        gram_normalized = gram / (scale.unsqueeze(0) * scale.unsqueeze(1))

        return gram_normalized


# ============================================================================
# Helper Functions for Metric Construction
# ============================================================================

def metric_from_phi_robust(phi, reg_strength=0.15):
    """
    Construct robust G₂ metric from φ with strong regularization.

    CRITICAL IMPROVEMENTS v0.8:
    - Regularization: 0.15 (was 0.1 in v0.7)
    - Min eigenvalue: 0.3 (was 0.1 → caused crashes)
    - Condition number monitoring

    Process:
    1. Build 7×7 symmetric matrix from 35 φ components
    2. Add regularization: g += 0.15 * I
    3. SPD projection via eigenvalue clamping (λ ≥ 0.3)
    4. Volume normalization: det(g) = 1

    Args:
        phi: (batch, 35) tensor of 3-form components
        reg_strength: Regularization strength (default: 0.15)

    Returns:
        g: (batch, 7, 7) SPD metric tensor with det(g) = 1
    """
    batch_size = phi.shape[0]
    device = phi.device

    # Base metric from φ (35 components → 7×7 symmetric)
    g = torch.zeros(batch_size, 7, 7, device=device)

    idx = 0
    for i in range(7):
        for j in range(i, 7):
            if idx < 35:
                g[:, i, j] = phi[:, idx] * 0.1 + (1.0 if i == j else 0.0)
                g[:, j, i] = g[:, i, j]
                idx += 1

    # STRONG regularization to prevent ill-conditioning
    g = g + reg_strength * torch.eye(7, device=device).unsqueeze(0)

    # Enforce symmetry (numerical stability)
    g = 0.5 * (g + g.transpose(-2, -1))

    # Add stability perturbation
    g_stable = g + 1e-4 * torch.eye(7, device=device).unsqueeze(0)

    # SPD projection via eigenvalue clamping
    try:
        eigvals, eigvecs = torch.linalg.eigh(g_stable)

        # CRITICAL: Higher floor prevents singularity
        eigvals = torch.clamp(eigvals, min=0.3)  # MUST be ≥ 0.3!

        # Check condition numbers
        condition_numbers = eigvals.max(dim=1)[0] / eigvals.min(dim=1)[0]

        if condition_numbers.max() > 100:
            # Apply stronger regularization if ill-conditioned
            eigvals = torch.clamp(eigvals, min=0.5)

        g = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)

    except RuntimeError as e:
        # Emergency fallback
        print(f"⚠ Metric computation failed: {e}")
        g = g + 0.5 * torch.eye(7, device=device).unsqueeze(0)

    # Volume normalization: det(g) = 1
    vol = torch.sqrt(torch.abs(torch.det(g)) + 1e-8)
    g = g / (vol.unsqueeze(-1).unsqueeze(-1) ** (2/7))

    return g
