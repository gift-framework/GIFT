# GIFT v0.8 → v0.9: Key Components to Reuse

## 1. IMPORTS (Working Libraries)

```python
# Standard library
import os
import sys
import json
import time
import warnings
import gc
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# IPython
from IPython.display import clear_output, display

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

# PyTorch (CRITICAL)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 2. NETWORK ARCHITECTURES

### 2A. G2PhiNetwork_TCS (35-component 3-form φ)

```python
class G2PhiNetwork_TCS(nn.Module):
    """φ network for TCS neck with boundary awareness."""
    
    def __init__(self, manifold, hidden_dims=[256, 256, 128]):
        super().__init__()
        self.manifold = manifold
        
        # Get Fourier encoding dimension
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]
        
        # MLP with SiLU activation + LayerNorm
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
        self.output = nn.Linear(prev_dim, 35)  # 35 components for 3-form φ
        
        # Initialize output small
        with torch.no_grad():
            self.output.weight.mul_(0.01)
            self.output.bias.zero_()
    
    def forward(self, coords):
        # Apply gluing rotation (boundary matching)
        coords_rotated = self.manifold.apply_gluing_rotation(coords)
        
        # Fourier encoding
        x = self.manifold.fourier_encoding(coords_rotated)
        
        # Process through MLP
        x = self.mlp(x)
        phi = self.output(x)
        
        # Normalize to √7
        phi_norm = torch.norm(phi, dim=-1, keepdim=True)
        phi = phi * (np.sqrt(7.0) / (phi_norm + 1e-8))
        
        # Apply boundary decay for torsion-free matching at boundaries
        decay = self.manifold.boundary_decay_factor(coords)
        phi = phi * (1 - decay * 0.5)  # Soft decay
        
        return phi
```

**Key features:**
- Hidden dims: [256, 256, 128] (effective for 7D manifold)
- Activation: SiLU (smoother gradients than ReLU)
- Normalization: LayerNorm after each layer + manual φ normalization
- Output: 35 components (13 diagonal + 21 off-diagonal for metric-like structure)

### 2B. MetricNetwork (28 coefficients → 7×7 SPD metric)

```python
class MetricNetwork(nn.Module):
    """Direct metric coefficient prediction with Fourier encoding."""
    
    def __init__(self, manifold, hidden_dims=[512, 512, 256, 256, 128]):
        super().__init__()
        self.manifold = manifold
        
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]
        
        # Deep MLP
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
        self.output = nn.Linear(prev_dim, 28)
        
        # Initialize to near-identity metric
        with torch.no_grad():
            self.output.weight.mul_(0.01)
            self.output.bias.zero_()
    
    def forward(self, coords):
        coords_rotated = self.manifold.apply_gluing_rotation(coords)
        x = self.manifold.fourier_encoding(coords_rotated)
        x = self.mlp(x)
        coeffs = self.output(x)
        
        # Soft boundary modulation
        decay = self.manifold.boundary_decay_factor(coords)
        boundary_mod = torch.sigmoid(10 * (1 - decay))
        coeffs = coeffs * boundary_mod.unsqueeze(-1)
        
        return coeffs
    
    def coeffs_to_metric(self, coeffs):
        """Convert 28 coeffs to 7×7 SPD metric tensor."""
        batch_size = coeffs.shape[0]
        device = coeffs.device
        
        # Extract diagonal (exp-transformed) and off-diagonal
        diag_raw = coeffs[:, :7]
        off_diag = coeffs[:, 7:]
        
        # Diagonal: exp to ensure positive
        diag = torch.exp(diag_raw) + 0.1
        
        # Build symmetric matrix
        metric = torch.zeros(batch_size, 7, 7, device=device, dtype=coeffs.dtype)
        
        # Diagonal
        for i in range(7):
            metric[:, i, i] = diag[:, i]
        
        # Upper triangular (with symmetry)
        idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                metric[:, i, j] = off_diag[:, idx]
                metric[:, j, i] = off_diag[:, idx]
                idx += 1
        
        # SPD projection via eigenvalue clamping
        eye = torch.eye(7, device=device, dtype=coeffs.dtype).unsqueeze(0)
        metric = metric + 0.01 * eye
        
        eigvals, eigvecs = torch.linalg.eigh(metric)
        eigvals = torch.clamp(eigvals, min=0.3)  # CRITICAL: min_eig ≥ 0.3
        metric = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
        
        # Volume normalization
        vol = torch.sqrt(torch.abs(torch.det(metric)) + 1e-8)
        metric = metric / (vol.unsqueeze(-1).unsqueeze(-1) ** (2/7))
        
        return metric
```

**Key features:**
- Deeper MLP: [512, 512, 256, 256, 128] (metrics need more capacity)
- Output: 28 coefficients (7 diagonal + 21 symmetric off-diagonal)
- SPD guarantee: exp() on diagonal + eigenvalue clamping (min ≥ 0.3)
- Critical fix: min eigenvalue = 0.3 (v0.7 bug used 0.1, caused numerical crashes)

### 2C. BoundaryNetwork (ACyl transition - FIXED v0.8)

```python
class BoundaryNetwork(nn.Module):
    """
    Models ACyl boundary transitions with exponential decay from CENTER.
    
    CRITICAL FIX v0.8:
    - Old v0.7: exp(-γ * dist_from_boundary) → U-shape artifact
    - New v0.8: exp(-γ * |t|/T) from center → Proper monotonic decay
    """
    
    def __init__(self, manifold, gamma=0.578, acyl_width=3.0):
        super().__init__()
        self.manifold = manifold
        self.gamma = gamma
        self.acyl_width = acyl_width
        
        # Learnable fine-tuning
        self.gamma_offset = nn.Parameter(torch.zeros(1))
        self.amplitude = nn.Parameter(torch.ones(1))
    
    def forward(self, coords):
        """
        Returns boundary transition factors (batch,) in [0, 1].
        
        0 = near center of neck (t ≈ 0)
        1 = at/beyond boundaries (|t| ≈ T)
        """
        t = coords[:, 0]
        T = self.manifold.T_neck  # Use T_neck not T_boundary
        
        # Effective gamma
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
        """Explicit ACyl decay: exp(-γ|t|/T)"""
        T = self.manifold.T_neck
        gamma_eff = self.gamma + 0.01 * torch.tanh(self.gamma_offset)
        
        t_norm = torch.abs(t) / T
        decay = torch.exp(-gamma_eff * t_norm)
        
        smooth = torch.sigmoid(5.0 * (0.5 - t_norm))
        return smooth + (1 - smooth) * decay
```

**Key formula:**
```
CORRECT v0.8 decay: exp(-γ|t|/T)
- γ = 0.578 (phenomenological ACyl decay rate)
- t = neck coordinate, T = T_neck = τ × 2π ≈ 24.48
- Decay MONOTONIC from center (t=0) to boundaries (|t|=T)
```

### 2D. Harmonic2FormsNetwork_TCS (21 b₂ forms)

```python
class Harmonic2FormsNetwork_TCS(nn.Module):
    """Harmonic forms with DISTINCT per-form initializations."""
    
    def __init__(self, manifold, hidden_dims=[128, 128], n_forms=21, output_dim=21):
        super().__init__()
        self.n_forms = n_forms
        self.output_dim = output_dim
        self.manifold = manifold
        
        test_point = torch.zeros(1, 7, device=manifold.device, dtype=manifold.dtype)
        encoding_dim = manifold.fourier_encoding(test_point).shape[-1]
        
        # CRITICAL: Create networks with DIFFERENT initializations per form
        self.networks = nn.ModuleList()
        for form_idx in range(n_forms):
            # Unique seed per form
            torch.manual_seed(47 + form_idx * 100)
            
            net = nn.Sequential(
                nn.Linear(encoding_dim, hidden_dims[0]),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[1], output_dim)
            )
            
            # Unique initialization per form
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5 + form_idx * 0.05)
                    nn.init.constant_(layer.bias, 0.01 * form_idx)
            
            self.networks.append(net)
        
        torch.manual_seed(47)  # Reset
    
    def forward(self, coords):
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
        """Gram matrix with proper normalization."""
        batch_size = coords.shape[0]
        n_forms = forms.shape[1]
        gram = torch.zeros(n_forms, n_forms, device=coords.device)
        vol = torch.sqrt(torch.abs(torch.det(metric)) + 1e-10)
        
        for alpha in range(n_forms):
            for beta in range(alpha, n_forms):
                inner = torch.sum(forms[:, alpha, :] * forms[:, beta, :], dim=-1) * vol
                gram[alpha, beta] = inner.mean()
                gram[beta, alpha] = gram[alpha, beta]
        
        # Normalize to unit diagonal
        diag = torch.diagonal(gram)
        scale = torch.sqrt(diag + 1e-8)
        gram_normalized = gram / (scale.unsqueeze(0) * scale.unsqueeze(1))
        
        return gram_normalized
```

---

## 3. MESH GENERATION (7D Grid Structure)

### TCS Neck Manifold Definition

```python
class TCSNeckManifold:
    """
    TCS-inspired neck geometry: [−T,T] × (S¹)² × T⁴
    
    Topology:
    - t ∈ [−T, T]: Non-periodic neck parameter (T ≈ 24.48)
    - θ, φ ∈ [0, 2π]: Periodic fiber circles (S¹ × S¹)
    - x₃, x₄, x₅, x₆ ∈ [0, 2π/φ]: K3-like T⁴ base (φ = golden ratio)
    """
    
    def __init__(self, gift_params, device='cpu'):
        self.device = device
        self.dim = 7
        self.dtype = torch.float32  # CRITICAL: consistent dtype
        
        tau = gift_params['tau']
        self.xi_gluing = gift_params['xi']
        self.gamma_decay = gift_params['gamma_GIFT']
        phi_golden = gift_params['phi']
        
        # Neck length (non-periodic t-direction)
        self.R_fiber = 2 * np.pi
        self.T_neck = tau * self.R_fiber  # ≈ 24.48 for τ ≈ 3.898
        
        # Fiber circles
        self.fiber_radii = torch.tensor(
            [self.R_fiber, self.R_fiber],
            device=device, dtype=self.dtype
        )
        
        # K3-like T⁴ with φ-hierarchy
        self.K3_radii = torch.tensor([
            2*np.pi,
            2*np.pi,
            2*np.pi / phi_golden,  # Smaller radii
            2*np.pi / phi_golden
        ], device=device, dtype=self.dtype)
        
        self.boundary_width = 0.15 * self.T_neck
        self._setup_fourier_modes()
    
    def sample_points(self, n_batch):
        """Uniform sampling on manifold."""
        # t ∈ [−T, T] (non-periodic)
        t = (torch.rand(n_batch, 1, device=self.device, dtype=self.dtype) * 2 - 1) * self.T_neck
        
        # θ, φ ∈ [0, 2π] (periodic)
        theta = torch.rand(n_batch, 2, device=self.device, dtype=self.dtype) * 2*np.pi
        
        # x ∈ T⁴ (periodic)
        x_K3 = torch.rand(n_batch, 4, device=self.device, dtype=self.dtype) * self.K3_radii.unsqueeze(0)
        
        return torch.cat([t, theta, x_K3], dim=1)
```

### 7D Grid for Spectral Analysis (b₃ extraction)

```python
# Create 1D grids
n_grid = 12  # CRITICAL: Must be 12 for b₃=77 extraction
coords_1d = []

# t-coordinate: [-T_neck, +T_neck]
coords_1d.append(
    torch.linspace(-manifold.T_neck, manifold.T_neck, n_grid, device='cpu')
)

# Fiber circles: [0, 2π]
for i in range(1, 3):
    coords_1d.append(
        torch.linspace(0, 2*np.pi, n_grid, device='cpu')
    )

# K3-like T⁴: [0, radius_i]
for i in range(3, 7):
    coords_1d.append(
        torch.linspace(0, manifold.K3_radii[i-3].item(), n_grid, device='cpu')
    )

# Create 7D grid via meshgrid (process t-slices sequentially to save memory)
phi_grid_7d = torch.zeros([n_grid]*7 + [35])

for t_idx in range(n_grid):
    t_val = coords_1d[0][t_idx].item()
    
    # Create 6D meshgrid for this t-slice
    grids_6d = torch.meshgrid(*coords_1d[1:], indexing='ij')
    coords_slice = torch.stack([g.flatten() for g in grids_6d], dim=1)
    
    # Add t coordinate
    t_coords = torch.full((coords_slice.shape[0], 1), t_val)
    coords_full = torch.cat([t_coords, coords_slice], dim=1)
    
    # Compute φ in batches (batch_size = 8192)
    phi_slice = []
    for i in range(0, coords_full.shape[0], 8192):
        batch = coords_full[i:i+8192].to(device)
        with torch.no_grad():
            phi_batch = phi_network(batch)
        phi_slice.append(phi_batch.cpu())
    
    phi_grid_7d[t_idx] = torch.cat(phi_slice, dim=0).reshape([n_grid]*6 + [35])

print(f"φ grid shape: {phi_grid_7d.shape}")  # [12, 12, 12, 12, 12, 12, 12, 35]
```

**Key parameters:**
- Grid size: n = 12, total points = 12⁷ = 35,831,808
- Computation: Process t-slices (12) sequentially to stay under 80GB VRAM
- FFT: Applied to each of 35 components separately (streaming)
- Mode selection: Top 250 candidates by energy, then orthogonalized to 77 forms

---

## 4. LOSS FUNCTION COMPONENTS (Working Formulas)

### 4A. Torsion Computation (FIXED v0.8 - Critical)

```python
class SafeMetrics:
    """Universal helper for robust metric operations."""
    
    @staticmethod
    def compute_torsion_safe(phi, coords, metric, use_grad=True):
        """
        Gradient-aware torsion computation.
        
        CRITICAL FIX v0.8.1: coords MUST NOT be cloned!
        Cloning breaks the computational graph.
        
        Theory:
        Torsion T = dφ + φ∧φ
        Approximation: ||∇φ|| (gradient norm of φ components)
        """
        if use_grad:
            # Training mode: needs gradients
            # FIXED: Use coords directly (don't clone!)
            
            grad_norms = []
            for i in range(min(10, phi.shape[1])):
                grad_i = torch.autograd.grad(
                    phi[:, i].sum(),
                    coords,  # Use directly, NOT clone!
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad_norms.append(grad_i.norm(dim=1))
            
            torsion = torch.stack(grad_norms, dim=1).mean(dim=1).mean()
            return torsion
        else:
            # Testing mode: simplified without gradients
            with torch.no_grad():
                phi_norm = torch.norm(phi, dim=-1).mean()
                return phi_norm * 0.1  # Rough estimate
    
    @staticmethod
    def to_json(obj):
        """Universal PyTorch/Numpy → JSON converter."""
        if isinstance(obj, torch.Tensor):
            obj_cpu = obj.detach().cpu()
            if obj_cpu.numel() == 1:
                return float(obj_cpu.item())
            else:
                return obj_cpu.tolist()
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return float(obj.item())
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            try:
                return float(obj)
            except:
                return str(obj)
    
    @staticmethod
    def safe_get(history, key, default=None):
        """Get from history dict with fallback."""
        val = history.get(key, [])
        if isinstance(val, list) and len(val) > 0:
            return val[-1]
        else:
            return default
    
    @staticmethod
    def to_scalar(obj):
        """Convert to Python float."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item()
        elif isinstance(obj, np.ndarray):
            return obj.item()
        else:
            return float(obj)
```

### 4B. Metric Construction (Robust SPD Projection)

```python
def metric_from_phi_robust(phi, reg_strength=0.15):
    """
    Robust G₂ metric from φ with strong regularization.
    
    CRITICAL IMPROVEMENTS v0.8:
    - Regularization: 0.15 (was 0.1 in v0.7)
    - Min eigenvalue: 0.3 (was 0.1 → caused crashes)
    - Condition number monitoring
    """
    batch_size = phi.shape[0]
    
    # Base metric from φ (13+21=35 components → 7×7 symmetric)
    g = torch.zeros(batch_size, 7, 7, device=phi.device)
    
    idx = 0
    for i in range(7):
        for j in range(i, 7):
            if idx < 35:
                g[:, i, j] = phi[:, idx] * 0.1 + (1.0 if i == j else 0.0)
                g[:, j, i] = g[:, i, j]
                idx += 1
    
    # STRONG regularization to prevent ill-conditioning
    g = g + reg_strength * torch.eye(7, device=phi.device).unsqueeze(0)
    
    # Enforce symmetry
    g = 0.5 * (g + g.transpose(-2, -1))
    
    # Add stability perturbation
    g_stable = g + 1e-4 * torch.eye(7, device=phi.device).unsqueeze(0)
    
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
        g = g + 0.5 * torch.eye(7, device=phi.device).unsqueeze(0)
    
    # Volume normalization: det(g) = 1
    vol = torch.sqrt(torch.abs(torch.det(g)) + 1e-8)
    g = g / (vol.unsqueeze(-1).unsqueeze(-1) ** (2/7))
    
    return g
```

**Key formula:**
```
g_ij = φ_k * 0.1 + δ_ij + 0.15 * δ_ij
     = (metric from φ) + regularization

Eigenvalue clamping: λ ← max(λ, 0.3)
Volume constraint: det(g) = 1 (normalize via g ← g / (det)^(2/7))
```

### 4C. Harmonic Loss (Gram Matrix Orthogonality - FIXED)

```python
def compute_harmonic_losses_FIXED(harmonic_network, coords, h_forms, metric):
    """
    FIXED harmonic losses from v0.6b.
    
    Critical fixes:
    - Better det loss: (det - target)² instead of just det
    - Per-element normalized orthogonality
    - Separation loss (diagonal >> off-diagonal)
    """
    # Compute Gram matrix
    gram = harmonic_network.compute_gram_matrix(coords, h_forms, metric)
    det_gram = torch.det(gram)
    
    # FIXED: Better det loss (encourage det → 0.995, not exact 1.0)
    target_det = 0.995
    harmonic_loss_det = torch.relu(det_gram - target_det) + 0.1 * (det_gram - target_det) ** 2
    
    # Orthogonality loss: ||Gram - I||² / size
    identity = torch.eye(21, device=device)
    harmonic_loss_ortho = torch.norm(gram - identity) / 21.0
    
    # Separation loss: diagonal >> off-diagonal
    diag_elements = torch.diagonal(gram)
    off_diag_mask = ~torch.eye(21, dtype=torch.bool, device=device)
    off_diag_elements = gram[off_diag_mask]
    
    separation_loss = torch.relu(
        0.5 - (diag_elements.mean() - off_diag_elements.abs().mean())
    )
    
    return harmonic_loss_det, harmonic_loss_ortho, separation_loss, det_gram
```

### 4D. Boundary & Decay Losses

```python
def compute_boundary_loss(phi, coords, manifold):
    """Penalize non-zero torsion near boundaries t=±T."""
    near_boundary = manifold.is_near_boundary(coords, threshold=0.15)
    
    if near_boundary.sum() == 0:
        return torch.tensor(0.0, device=coords.device)
    
    phi_boundary = phi[near_boundary]
    coords_boundary = coords[near_boundary].requires_grad_(True)
    phi_boundary_grad = phi_network(coords_boundary)
    
    # Gradient norm (torsion proxy)
    grad_norms = []
    for i in range(min(5, phi_boundary_grad.shape[1])):
        grad_i = torch.autograd.grad(
            phi_boundary_grad[:, i].sum(),
            coords_boundary,
            create_graph=True,
            retain_graph=True
        )[0]
        grad_norms.append(grad_i.norm(dim=1))
    
    grad_norm = torch.stack(grad_norms, dim=1).mean()
    phi_amplitude_boundary = torch.norm(phi_boundary, dim=1).mean()
    
    return grad_norm + phi_amplitude_boundary * 0.5


def compute_asymptotic_decay_loss(phi, coords, manifold):
    """Enforce exp(-γ|t|/T) decay behavior."""
    t = coords[:, 0]
    
    # Expected decay: exp(-γ × |t|/T)
    expected_decay = torch.exp(
        -manifold.gamma_decay * torch.abs(t) / manifold.T_neck
    )
    
    # Actual φ amplitude
    phi_amplitude = torch.norm(phi, dim=1)
    
    # Loss: deviation from expected decay
    decay_loss = torch.abs(phi_amplitude - expected_decay).mean()
    
    return decay_loss
```

---

## 5. TRAINING SETUP (Optimizer, Learning Rate, Curriculum)

### 5A. CONFIG Dictionary (Critical Parameters)

```python
CONFIG = {
    # Version & geometry
    'version': 'v0.8',
    'geometry': 'TCS_neck_ACyl',
    
    # Training core
    'epochs': 10000,
    'batch_size': 1536,
    'grad_accumulation_steps': 2,
    'effective_batch': 3072,
    
    # Optimization
    'lr': 1e-4,  # Starting learning rate
    'weight_decay': 1e-4,
    'grad_clip': 1.0,  # Gradient clipping threshold
    'scheduler': 'cosine',
    'warmup_epochs': 500,
    
    # Mixed precision (enabled in phase 2)
    'mixed_precision': True,
    'mixed_precision_start_epoch': 2000,
    
    # Post-training (spectral)
    'b3_grid_resolution': 12,  # CRITICAL: Must be 12 for b₃=77
    'yukawa_n_integration': 4096,
    
    # Checkpoints
    'checkpoint_interval': 500,
    'validation_interval': 1000,
    
    # Reproducibility
    'seed': 47,
    'deterministic': True,
    'use_smooth_transitions': True,
    'transition_width': 200,  # epochs to blend between phases
}
```

### 5B. Optimizer & Scheduler

```python
# Set reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])

# Optimizer: AdamW (better than Adam for regularization)
optimizer = optim.AdamW(
    list(phi_network.parameters()) + list(harmonic_network.parameters()),
    lr=CONFIG['lr'],  # 1e-4
    weight_decay=CONFIG['weight_decay']  # 1e-4
)

# Learning rate scheduler: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG['epochs'],  # 10000
    eta_min=CONFIG['lr'] / 100  # 1e-6 (final lr)
)

# Gradient clipping threshold
grad_clip = CONFIG['grad_clip']  # 1.0
```

**Learning rate schedule:**
```
lr(t) = (lr_min + (lr_init - lr_min) * (1 + cos(πt/T)) / 2)
      = (1e-6 + (1e-4 - 1e-6) * (1 + cos(π*epoch/10000)) / 2)

Starting lr: 1e-4
Final lr: 1e-6 (100× decay)
Schedule: Cosine annealing over 10000 epochs
```

### 5C. 4-Phase Curriculum (Loss Weighting)

```python
# NEW v0.7: 4-PHASE CURRICULUM
CURRICULUM = {
    'phase1': {  # Epochs 0-2000: Establish Structure
        'name': 'Establish Structure',
        'range': [0, 2000],
        'weights': {
            'torsion': 0.1,          # Minimal torsion loss
            'volume': 0.6,           # Volume × 2
            'harmonic_ortho': 6.0,   # Harmonic × 3
            'harmonic_det': 3.0,
            'separation': 2.0,
            'boundary': 0.05,
            'decay': 0.05,
            'acyl': 0.0,
        }
    },
    'phase2': {  # Epochs 2000-5000: Impose Torsion
        'name': 'Impose Torsion',
        'range': [2000, 5000],
        'weights': {
            'torsion': 2.0,          # Ramp 0.1 → 2.0 (20× increase)
            'volume': 0.4,
            'harmonic_ortho': 3.0,
            'harmonic_det': 1.5,
            'separation': 1.0,
            'boundary': 0.5,
            'decay': 0.3,
            'acyl': 0.1,             # Start ACyl matching
        }
    },
    'phase3': {  # Epochs 5000-8000: Refine b₃ + ACyl
        'name': 'Refine b₃ + ACyl',
        'range': [5000, 8000],
        'weights': {
            'torsion': 5.0,          # Continue increasing
            'volume': 0.2,
            'harmonic_ortho': 2.0,   # Reduce
            'harmonic_det': 1.0,
            'separation': 0.5,
            'boundary': 1.0,
            'decay': 0.5,
            'acyl': 0.3,             # Increase ACyl
        }
    },
    'phase4': {  # Epochs 8000-10000: Polish Final
        'name': 'Polish Final',
        'range': [8000, 10000],
        'weights': {
            'torsion': 20.0,         # Heavy torsion focus
            'volume': 0.1,
            'harmonic_ortho': 1.0,
            'harmonic_det': 0.5,
            'separation': 0.2,
            'boundary': 1.5,
            'decay': 1.0,
            'acyl': 0.5,
        }
    }
}

# Smooth transitions between phases (200-epoch blend)
def get_phase_weights_smooth(epoch, transition_width=200):
    """Get curriculum weights with smooth blending."""
    current_weights = None
    next_weights = None
    blend_factor = 0.0
    
    for phase_name, phase_cfg in CURRICULUM.items():
        phase_start, phase_end = phase_cfg['range']
        
        if epoch < phase_start:
            continue
        elif epoch < phase_end:
            current_weights = phase_cfg['weights']
            # Check if we're in transition zone
            if epoch < phase_start + transition_width:
                blend_factor = (epoch - phase_start) / transition_width
            break
    
    # Blend with next phase if in transition
    if blend_factor < 1.0 and next_weights:
        for key in current_weights:
            w_curr = current_weights[key]
            w_next = next_weights[key]
            current_weights[key] = (1 - blend_factor) * w_curr + blend_factor * w_next
    
    return current_weights, phase_name
```

### 5D. Training Loop (Core Architecture)

```python
# Initialize history (ALL keys upfront - CRITICAL FIX)
history = {
    'epoch': [],
    'loss': [],
    'torsion': [],
    'volume': [],
    'det_gram': [],
    'harmonic_ortho': [],
    'harmonic_det': [],
    'separation': [],
    'boundary': [],
    'decay': [],
    'lr': [],
    'phase': [],
    'metric_condition_avg': [],
    'metric_condition_max': [],
    'metric_det_std': []
}

test_history = {
    'epoch': [],
    'test_torsion': [],
    'test_det_gram': [],
    'test_dphi_L2': [],
    'test_dstar_phi_L2': []
}

# Training loop
for epoch in range(CONFIG['epochs']):
    try:
        phi_network.train()
        harmonic_network.train()
        
        # Get curriculum weights
        weights, phase_name = get_phase_weights_smooth(epoch)
        
        # Sample batch
        coords = manifold.sample_points(CONFIG['batch_size'])
        
        # Forward pass
        phi = phi_network(coords)
        h_forms = harmonic_network(coords)
        metric = metric_from_phi_robust(phi)
        
        # Compute individual losses
        torsion_loss = SafeMetrics.compute_torsion_safe(phi, coords, metric, use_grad=True)
        volume_loss = torch.abs(torch.det(metric).mean() - 1.0)
        harmonic_loss_det, harmonic_loss_ortho, separation_loss, det_gram = \
            compute_harmonic_losses_FIXED(harmonic_network, coords, h_forms, metric)
        boundary_loss = compute_boundary_loss(phi, coords, manifold)
        decay_loss = compute_asymptotic_decay_loss(phi, coords, manifold)
        
        # Total loss (with curriculum weighting)
        loss = (weights['torsion'] * torsion_loss +
                weights['volume'] * volume_loss +
                weights['harmonic_ortho'] * harmonic_loss_ortho +
                weights['harmonic_det'] * harmonic_loss_det +
                weights['separation'] * separation_loss +
                weights['boundary'] * boundary_loss +
                weights['decay'] * decay_loss)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(phi_network.parameters()) + list(harmonic_network.parameters()),
            CONFIG['grad_clip']
        )
        optimizer.step()
        scheduler.step()
        
        # Log metrics (EVERY epoch)
        history['epoch'].append(epoch)
        history['loss'].append(SafeMetrics.to_scalar(loss))
        history['torsion'].append(SafeMetrics.to_scalar(torsion_loss))
        history['volume'].append(SafeMetrics.to_scalar(volume_loss))
        history['det_gram'].append(SafeMetrics.to_scalar(det_gram))
        history['harmonic_ortho'].append(SafeMetrics.to_scalar(harmonic_loss_ortho))
        history['harmonic_det'].append(SafeMetrics.to_scalar(harmonic_loss_det))
        history['separation'].append(SafeMetrics.to_scalar(separation_loss))
        history['boundary'].append(SafeMetrics.to_scalar(boundary_loss))
        history['decay'].append(SafeMetrics.to_scalar(decay_loss))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['phase'].append(phase_name)
        
        # Test evaluation every 1000 epochs
        if epoch % 1000 == 0 or epoch == CONFIG['epochs'] - 1:
            phi_network.eval()
            harmonic_network.eval()
            
            test_coords.requires_grad_(True)
            phi_test = phi_network(test_coords)
            metric_test = metric_from_phi_robust(phi_test)
            
            # Torsion with gradients
            test_torsion = SafeMetrics.compute_torsion_safe(phi_test, test_coords, metric_test, use_grad=True)
            
            # PDE residuals
            dphi_components = []
            for comp_idx in range(min(10, phi_test.shape[1])):
                grad_comp = torch.autograd.grad(
                    phi_test[:, comp_idx].sum(),
                    test_coords,
                    create_graph=False,
                    retain_graph=True
                )[0]
                dphi_components.append(grad_comp)
            dphi = torch.stack(dphi_components, dim=1)
            dphi_L2 = torch.norm(dphi).item()
            
            # Harmonic properties
            with torch.no_grad():
                h_forms_test = harmonic_network(test_coords)
                _, _, _, test_det_gram = compute_harmonic_losses_FIXED(
                    harmonic_network, test_coords, h_forms_test, metric_test
                )
            
            test_history['epoch'].append(epoch)
            test_history['test_torsion'].append(SafeMetrics.to_scalar(test_torsion))
            test_history['test_det_gram'].append(SafeMetrics.to_scalar(test_det_gram))
            test_history['test_dphi_L2'].append(dphi_L2)
            
            print(f"Epoch {epoch}: torsion={test_torsion:.2e}, det(Gram)={test_det_gram:.4f}")
        
        # Checkpoints
        if epoch % CONFIG['checkpoint_interval'] == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'phi_network': phi_network.state_dict(),
                'harmonic_network': harmonic_network.state_dict(),
                'history': history,
                'test_history': test_history
            }, CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pt')
        
        # Memory cleanup
        if epoch % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    except RuntimeError as e:
        print(f"⚠ Epoch {epoch} failed: {e}")
        torch.cuda.empty_cache()
        continue

print("Training complete!")
```

---

## CRITICAL FIXES FOR v0.8 → v0.9

### 1. Torsion Computation (compute_torsion_safe)
- **MUST NOT CLONE coords**: Cloning breaks gradient graph
- Use: `grad_i = torch.autograd.grad(phi[:, i].sum(), coords, ...)`
- NOT: `grad_i = torch.autograd.grad(phi[:, i].sum(), coords.clone(), ...)`

### 2. Metric Regularization
- **Min eigenvalue ≥ 0.3**: v0.7 bug (0.1) caused singular matrices
- Formula: `eigvals = torch.clamp(eigvals, min=0.3)`
- Regularization strength: 0.15 (strong)

### 3. Boundary Decay (BoundaryNetwork)
- **CORRECT**: `exp(-γ|t|/T)` from center
- **WRONG**: `exp(-γ * dist_from_boundary)` (creates U-shape)
- γ = 0.578 (phenomenological ACyl decay)

### 4. det(Gram) Target
- **Target**: 0.995 (not 1.0!)
- Over-optimization to 1.0 causes metric singularity
- Emergency brake: Stop if det(Gram) stuck at 1.0 for 5+ consecutive test epochs

### 5. History Initialization
- **Initialize ALL keys upfront**: Prevents KeyError crashes
- Use `history.get(key, [])` with fallback

### 6. Test Coords Setup
- **Enable gradients for test torsion**:  `test_coords.requires_grad_(True)`
- Compute torsion OUTSIDE `no_grad()` context

---

## SUMMARY TABLE

| Component | Key Value | Notes |
|-----------|-----------|-------|
| **Imports** | torch, numpy, scipy, matplotlib | Standard ML stack |
| **PhiNet hidden** | [256, 256, 128] | 3-form φ |
| **MetricNet hidden** | [512, 512, 256, 256, 128] | 7×7 SPD metric |
| **BoundaryNet γ** | 0.578 | ACyl exponential decay |
| **HarmonicNet hidden** | [128, 128] | 21 b₂ forms |
| **Grid resolution** | 12 | b₃ extraction (12⁷ points) |
| **Batch size** | 1536 | Training |
| **Learning rate** | 1e-4 → 1e-6 | Cosine annealing |
| **Epochs** | 10000 | 4-phase curriculum |
| **Max eigenvalue** | ≥ 0.3 | SPD guarantee (CRITICAL) |
| **Regularization** | 0.15 | Metric robustness |
| **det(Gram) target** | 0.995 | Avoids singularity |
| **Torsion formula** | ||∇φ|| | Gradient norm proxy |
| **Decay formula** | exp(-γ\|t\|/T) | ACyl boundary behavior |

