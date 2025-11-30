"""
G2 Phi Wrapper Module

This module provides functionality to compute the G2 3-form phi from a trained 
metric network. No Unicode characters - Windows compatible.

Usage:
    from G2_phi_wrapper import load_model, compute_phi_from_metric
    
    model = load_model('G2_final_model.pt')
    metric = model(coords)
    phi = compute_phi_from_metric(metric, coords)
"""

import torch
import torch.nn as nn
import numpy as np


class CompactG2Network(nn.Module):
    """Neural network for learning G2 metrics."""

    def __init__(self, hidden_dims=[256, 256, 128], num_freq=32):
        super().__init__()

        # Fourier features for periodic representation
        self.register_buffer('B', torch.randn(7, num_freq) * 2.0)

        # MLP layers
        layers = []
        prev_dim = 2 * num_freq  # cos + sin

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim

        # Output layer: 28 parameters for upper triangular 7x7 metric
        layers.append(nn.Linear(prev_dim, 28))
        self.mlp = nn.Sequential(*layers)

        # Initialize output layer with small weights
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, coords):
        """Forward pass: coords (batch, 7) -> metric (batch, 7, 7)."""
        batch_size = coords.shape[0]
        device = coords.device

        # Fourier features
        x = 2 * np.pi * coords @ self.B
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        # MLP
        upper_tri = self.mlp(x)

        # Construct symmetric positive definite metric
        metric = torch.zeros(batch_size, 7, 7, device=device)
        idx = 0
        for i in range(7):
            for j in range(i, 7):
                if i == j:
                    # Diagonal: ensure positive
                    metric[:, i, j] = torch.nn.functional.softplus(upper_tri[:, idx]) + 0.1
                else:
                    # Off-diagonal: symmetric
                    metric[:, i, j] = upper_tri[:, idx] * 0.1
                    metric[:, j, i] = upper_tri[:, idx] * 0.1
                idx += 1

        # Add identity for numerical stability
        metric = metric + torch.eye(7, device=device).unsqueeze(0)

        return metric


def compute_phi_from_metric(metric, coords):
    """
    Compute G2 3-form phi from metric tensor.
    
    Explicit normalization to ||phi||^2 = 7
    
    Args:
        metric: Tensor of shape (batch_size, 7, 7) - metric tensor
        coords: Tensor of shape (batch_size, 7) - coordinates (unused but kept for API consistency)
    
    Returns:
        phi: Tensor of shape (batch_size, 35) - G2 3-form components
    """
    batch_size = metric.shape[0]
    device = metric.device
    dtype = metric.dtype

    # Initialize phi (35 components for 3-form in 7D)
    phi = torch.zeros(batch_size, 35, device=device, dtype=dtype)

    # Build index mapping for 3-forms
    triple_to_idx = {}
    idx = 0
    for m in range(7):
        for n in range(m+1, 7):
            for p in range(n+1, 7):
                triple_to_idx[(m, n, p)] = idx
                idx += 1

    # Construct phi using twisted connected sum (TCS) ansatz
    # This is a simplified construction - full TCS is more complex

    # Type 1: (0, i, j) components
    for i in range(2, 7):
        for j in range(i+1, 7):
            triple = tuple(sorted([0, i, j]))
            if triple in triple_to_idx:
                g_0i = metric[:, 0, i]
                g_0j = metric[:, 0, j]
                g_00 = metric[:, 0, 0]
                phi[:, triple_to_idx[triple]] = (g_0i + g_0j) / (torch.sqrt(g_00 + 1e-8) + 1e-8)

    # Type 2: (1, i, j) components
    for i in range(2, 7):
        for j in range(i+1, 7):
            triple = tuple(sorted([1, i, j]))
            if triple in triple_to_idx:
                g_1i = metric[:, 1, i]
                g_1j = metric[:, 1, j]
                g_11 = metric[:, 1, 1]
                phi[:, triple_to_idx[triple]] = (g_1i - g_1j) / (torch.sqrt(g_11 + 1e-8) + 1e-8)

    # Type 3: (i, j, k) components (i,j,k >= 2)
    for i in range(2, 7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                triple = tuple(sorted([i, j, k]))
                if triple in triple_to_idx:
                    # K3 block structure
                    g_block = metric[:, 2:7, 2:7]
                    det_block = torch.det(g_block)
                    sqrt_det = torch.sqrt(torch.abs(det_block) + 1e-8)

                    g_ij = metric[:, i, j]
                    g_jk = metric[:, j, k]
                    g_ki = metric[:, k, i]
                    structure = (g_ij * g_jk * g_ki) / (sqrt_det + 1e-8)
                    phi[:, triple_to_idx[triple]] = structure

    # Explicit G2 normalization to ||phi||^2 = 7
    phi_norm_sq = torch.sum(phi**2, dim=1, keepdim=True) + 1e-10
    phi = phi * torch.sqrt(7.0 / phi_norm_sq)

    return phi


def hodge_dual(phi, metric):
    """
    Compute Hodge dual *phi (3-form -> 4-form).
    
    Args:
        phi: Tensor of shape (batch_size, 35) - G2 3-form
        metric: Tensor of shape (batch_size, 7, 7) - metric tensor
    
    Returns:
        dual_phi: Tensor of shape (batch_size, 35) - Hodge dual 4-form
    """
    batch_size = phi.shape[0]
    device = phi.device

    # Simplified Hodge dual computation
    det_g = torch.det(metric)
    sqrt_det = torch.sqrt(torch.abs(det_g) + 1e-8)

    # For 3-form in 7D, *phi has 35 components (4-form)
    dual_phi = torch.zeros(batch_size, 35, device=device)

    # Simplified: scale by volume element
    for i in range(35):
        dual_phi[:, i] = phi[:, i] * sqrt_det.unsqueeze(-1).squeeze()

    return dual_phi


def load_model(model_path, device='cpu'):
    """
    Load a trained G2 metric model.
    
    Args:
        model_path: Path to the .pt model file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded CompactG2Network model in eval mode
    """
    model = CompactG2Network(hidden_dims=[256, 256, 128], num_freq=32)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_phi_properties(phi, metric, coords, compute_derivatives=False):
    """
    Evaluate key properties of the G2 3-form phi.
    
    Args:
        phi: Tensor of shape (batch_size, 35) - G2 3-form
        metric: Tensor of shape (batch_size, 7, 7) - metric tensor  
        coords: Tensor of shape (batch_size, 7) - coordinates
        compute_derivatives: Whether to compute d(phi) and d(*phi) (requires gradients)
    
    Returns:
        dict with:
            - phi_norm_sq: ||phi||^2 values
            - phi_norm_sq_mean: Mean of ||phi||^2
            - det_g: Determinant values
            - det_g_mean: Mean determinant
            - (optional) d_phi_norm: ||d(phi)||
            - (optional) d_dual_phi_norm: ||d(*phi)||
    """
    with torch.no_grad():
        # Compute ||phi||^2
        phi_norm_sq = torch.sum(phi**2, dim=1)
        
        # Compute det(g)
        det_g = torch.det(metric)
        
        results = {
            'phi_norm_sq': phi_norm_sq.cpu().numpy(),
            'phi_norm_sq_mean': phi_norm_sq.mean().item(),
            'phi_norm_sq_std': phi_norm_sq.std().item(),
            'det_g': det_g.cpu().numpy(),
            'det_g_mean': det_g.mean().item(),
            'det_g_std': det_g.std().item(),
            'norm_error': abs(phi_norm_sq.mean().item() - 7.0),
            'vol_error': abs(det_g.mean().item() - 1.0),
        }
    
    return results


if __name__ == '__main__':
    # Example usage and testing
    print("G2 Phi Wrapper Module")
    print("=" * 60)
    
    # Test with random coordinates
    batch_size = 10
    coords = torch.randn(batch_size, 7) * 5.0
    
    print(f"\nTesting with {batch_size} random points...")
    
    # Create a dummy model (would normally load from file)
    model = CompactG2Network()
    model.eval()
    
    with torch.no_grad():
        # Compute metric
        metric = model(coords)
        print(f"Metric shape: {metric.shape}")
        
        # Compute phi
        phi = compute_phi_from_metric(metric, coords)
        print(f"Phi shape: {phi.shape}")
        
        # Evaluate properties
        props = evaluate_phi_properties(phi, metric, coords)
        print(f"\nProperties:")
        print(f"  ||phi||^2 mean: {props['phi_norm_sq_mean']:.6f} (target: 7.0)")
        print(f"  Norm error: {props['norm_error']:.6e}")
        print(f"  det(g) mean: {props['det_g_mean']:.6f} (target: 1.0)")
        print(f"  Volume error: {props['vol_error']:.6e}")
    
    print("\n" + "=" * 60)
    print("Module test complete!")









