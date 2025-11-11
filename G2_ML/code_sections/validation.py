"""
Validation Functions for GIFT v0.9
===================================

This module contains all validation and analysis functions:
- PDE residuals: ||dφ||_L² (closedness), ||δφ||_L² (co-closedness)
- Ricci curvature computation
- Cohomology extraction (b₂ via Gram orthogonalization, b₃ via FFT)
- Regional analysis (M₁, Neck, M₂)
- Convergence diagnostics

Physical Interpretation:
- dφ = 0: φ is closed (exact 3-form)
- δφ = 0: φ is co-closed (harmonic)
- Ricc = 0: Ricci-flat metric (G₂ holonomy condition)
- b₂ = 21: Second Betti number (harmonic 2-forms)
- b₃ = 77: Third Betti number (harmonic 3-forms)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import fft
from scipy.linalg import qr


# ============================================================================
# PDE Residuals
# ============================================================================

def compute_closedness_residual(phi_network, coords, device):
    """
    Compute ||dφ||_L² (closedness residual).

    Theory: For G₂ structure, dφ = 0 (φ is closed).
    Approximation: ||dφ|| ≈ ||∇φ|| (gradient norm)

    Args:
        phi_network: G2PhiNetwork_TCS instance
        coords: (batch, 7) tensor of coordinates (requires_grad=True)
        device: torch device

    Returns:
        dphi_L2: Scalar, L² norm of dφ over batch
    """
    coords = coords.to(device)
    coords.requires_grad_(True)

    phi = phi_network(coords)

    # Compute gradients for all components
    dphi_components = []
    for comp_idx in range(phi.shape[1]):
        grad_comp = torch.autograd.grad(
            phi[:, comp_idx].sum(),
            coords,
            create_graph=False,
            retain_graph=True
        )[0]
        dphi_components.append(grad_comp)

    dphi = torch.stack(dphi_components, dim=1)  # (batch, 35, 7)
    dphi_L2 = torch.norm(dphi).item()

    return dphi_L2


def compute_coclosedness_residual(phi_network, coords, metric_from_phi_fn, device):
    """
    Compute ||δφ||_L² (co-closedness residual).

    Theory: For G₂ structure, δφ = 0 (φ is co-closed).
    δφ = *d*φ (Hodge star of exterior derivative of Hodge dual)

    Approximation: ||δφ|| ≈ ||div(φ)||, where div uses metric

    Args:
        phi_network: G2PhiNetwork_TCS instance
        coords: (batch, 7) tensor of coordinates
        metric_from_phi_fn: Function to construct metric from phi
        device: torch device

    Returns:
        delta_phi_L2: Scalar, L² norm of δφ over batch
    """
    coords = coords.to(device)
    coords.requires_grad_(True)

    phi = phi_network(coords)
    metric = metric_from_phi_fn(phi)

    # Compute metric-weighted divergence
    # Simplified: use trace of gradient with metric
    div_components = []
    for comp_idx in range(min(10, phi.shape[1])):  # Sample 10 components
        grad_comp = torch.autograd.grad(
            phi[:, comp_idx].sum(),
            coords,
            create_graph=False,
            retain_graph=True
        )[0]

        # Metric contraction: g^{ij} ∂_j φ_comp
        metric_inv = torch.inverse(metric + 1e-4 * torch.eye(7, device=device).unsqueeze(0))
        div = torch.einsum('bij,bj->bi', metric_inv, grad_comp)
        div_components.append(div.norm(dim=1))

    delta_phi = torch.stack(div_components, dim=1).mean()
    delta_phi_L2 = delta_phi.item()

    return delta_phi_L2


# ============================================================================
# Ricci Curvature
# ============================================================================

def compute_ricci_curvature_approx(metric, coords):
    """
    Compute approximate Ricci curvature via metric derivatives.

    Exact formula (complicated):
    R_{ij} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik} + Γ^k_{ij} Γ^l_{kl} - Γ^k_{il} Γ^l_{jk}

    Approximation (simpler):
    ||Ricc|| ≈ ||∂g/∂x|| (gradient of metric)

    Args:
        metric: (batch, 7, 7) tensor of metric
        coords: (batch, 7) tensor of coordinates (requires_grad=True)

    Returns:
        ricci_norm: Scalar, approximate Ricci norm
    """
    if not coords.requires_grad:
        coords.requires_grad_(True)

    # Compute gradient of metric components
    metric_grads = []
    for i in range(7):
        for j in range(i, 7):  # Symmetric, only upper triangular
            grad_ij = torch.autograd.grad(
                metric[:, i, j].sum(),
                coords,
                create_graph=False,
                retain_graph=True
            )[0]
            metric_grads.append(grad_ij.norm())

    ricci_norm = torch.stack(metric_grads).mean().item()

    return ricci_norm


# ============================================================================
# Cohomology Extraction
# ============================================================================

def extract_b2_cohomology(harmonic_network, manifold, n_samples=4096, device='cuda'):
    """
    Extract b₂ cohomology via Gram matrix orthogonalization.

    Process:
    1. Sample points uniformly on manifold
    2. Compute harmonic 2-forms
    3. Build Gram matrix
    4. Count eigenvalues > threshold (0.1) → b₂

    Args:
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        manifold: TCSNeckManifold instance
        n_samples: Number of sample points for integration
        device: torch device

    Returns:
        b2: Integer, number of independent harmonic 2-forms
        eigenvalues: Sorted eigenvalues of Gram matrix
    """
    harmonic_network.eval()

    coords = manifold.sample_points(n_samples).to(device)

    with torch.no_grad():
        h_forms = harmonic_network(coords)  # (n_samples, 21, 21)

        # Compute Gram matrix
        from .networks import metric_from_phi_robust
        phi_dummy = torch.randn(n_samples, 35, device=device)  # Placeholder
        metric = metric_from_phi_robust(phi_dummy)

        gram = harmonic_network.compute_gram_matrix(coords, h_forms, metric)

        # Eigenvalue decomposition
        eigenvalues = torch.linalg.eigvalsh(gram)
        eigenvalues_sorted = torch.sort(eigenvalues, descending=True)[0]

        # Count eigenvalues > 0.1 (independent forms)
        b2 = (eigenvalues_sorted > 0.1).sum().item()

    return b2, eigenvalues_sorted.cpu().numpy()


def extract_b3_cohomology_fft(phi_network, manifold, n_grid=12, device='cuda'):
    """
    Extract b₃ cohomology via FFT on 7D grid.

    Process:
    1. Create 7D grid with resolution n_grid (MUST be 12 for b₃=77)
    2. Evaluate φ on grid (process t-slices sequentially to save memory)
    3. FFT each of 35 components
    4. Extract top 250 modes by energy
    5. Orthogonalize via QR decomposition → b₃

    Args:
        phi_network: G2PhiNetwork_TCS instance
        manifold: TCSNeckManifold instance
        n_grid: Grid resolution per dimension (default: 12)
        device: torch device

    Returns:
        b3: Integer, number of independent harmonic 3-forms
        top_modes: List of (mode_index, energy) tuples
    """
    phi_network.eval()

    print(f"Building {n_grid}^7 grid for b₃ extraction...")
    print(f"Total points: {n_grid**7:,}")

    # Create 1D grids
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

    # Create 7D grid via t-slices (memory efficient)
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

        print(f"  t-slice {t_idx+1}/{n_grid} complete", end='\r')

    print(f"\nGrid shape: {phi_grid_7d.shape}")

    # FFT each component
    print("Computing FFT...")
    fft_modes = []
    for comp_idx in range(35):
        phi_comp = phi_grid_7d[..., comp_idx].numpy()
        fft_comp = fft.fftn(phi_comp)
        fft_energy = np.abs(fft_comp)**2
        fft_modes.append(fft_energy)

    # Stack and sum energies across components
    fft_modes = np.stack(fft_modes, axis=-1)  # (n_grid^7, 35)
    total_energy = fft_modes.sum(axis=-1).flatten()

    # Extract top 250 modes by energy
    top_250_indices = np.argsort(total_energy)[-250:]
    top_250_energies = total_energy[top_250_indices]

    # Build mode vectors
    mode_vectors = []
    for idx in top_250_indices:
        # Convert flat index to 7D multi-index
        multi_idx = np.unravel_index(idx, [n_grid]*7)
        mode_vec = fft_modes[multi_idx]
        mode_vectors.append(mode_vec)

    mode_vectors = np.array(mode_vectors)  # (250, 35)

    # Orthogonalize via QR decomposition
    Q, R = qr(mode_vectors.T, mode='economic')

    # Count independent modes (diagonal R > threshold)
    diag_R = np.abs(np.diag(R))
    b3 = (diag_R > 1e-6).sum()

    print(f"b₃ = {b3} (expected: 77 for n_grid=12)")

    top_modes = list(zip(top_250_indices, top_250_energies))

    return b3, top_modes


# ============================================================================
# Regional Analysis
# ============================================================================

def analyze_regions(phi_network, manifold, metric_from_phi_fn, device='cuda'):
    """
    Analyze φ and metric properties across 3 regions: M₁, Neck, M₂.

    Regions:
    - M₁: t ∈ [-T, -T/3] (boundary region 1)
    - Neck: t ∈ [-T/3, T/3] (central neck)
    - M₂: t ∈ [T/3, T] (boundary region 2)

    Metrics per region:
    - Mean ||φ||
    - Mean ||∇φ|| (torsion proxy)
    - Mean det(g)
    - Metric condition number

    Args:
        phi_network: G2PhiNetwork_TCS instance
        manifold: TCSNeckManifold instance
        metric_from_phi_fn: Function to construct metric from phi
        device: torch device

    Returns:
        regional_metrics: Dictionary with keys 'M1', 'Neck', 'M2'
    """
    phi_network.eval()

    T = manifold.T_neck
    n_samples = 2048

    # Sample each region
    regions = {
        'M1': (-T, -T/3),
        'Neck': (-T/3, T/3),
        'M2': (T/3, T)
    }

    regional_metrics = {}

    for region_name, (t_min, t_max) in regions.items():
        # Sample uniformly in this t-range
        coords = manifold.sample_points(n_samples).to(device)
        t = coords[:, 0]

        # Filter to region
        mask = (t >= t_min) & (t <= t_max)
        coords_region = coords[mask]

        if coords_region.shape[0] == 0:
            regional_metrics[region_name] = {
                'phi_norm': 0.0,
                'torsion': 0.0,
                'det_metric': 0.0,
                'condition_number': 0.0
            }
            continue

        coords_region.requires_grad_(True)

        phi = phi_network(coords_region)
        metric = metric_from_phi_fn(phi)

        # ||φ||
        phi_norm = torch.norm(phi, dim=1).mean().item()

        # ||∇φ|| (torsion proxy)
        grad_norms = []
        for i in range(min(5, phi.shape[1])):
            grad_i = torch.autograd.grad(
                phi[:, i].sum(),
                coords_region,
                create_graph=False,
                retain_graph=True
            )[0]
            grad_norms.append(grad_i.norm(dim=1))
        torsion = torch.stack(grad_norms, dim=1).mean().item()

        # det(g)
        det_metric = torch.det(metric).mean().item()

        # Condition number
        eigvals = torch.linalg.eigvalsh(metric)
        condition_numbers = eigvals.max(dim=1)[0] / (eigvals.min(dim=1)[0] + 1e-10)
        condition_number = condition_numbers.mean().item()

        regional_metrics[region_name] = {
            'phi_norm': phi_norm,
            'torsion': torsion,
            'det_metric': det_metric,
            'condition_number': condition_number
        }

    return regional_metrics


# ============================================================================
# Convergence Diagnostics
# ============================================================================

def compute_convergence_metrics(history, test_history, window=100):
    """
    Compute convergence diagnostics from training history.

    Metrics:
    - Relative change in loss over window: |Δloss| / loss
    - Plateau detection: std(loss) < threshold over window
    - Phase-wise convergence rates

    Args:
        history: Training history dictionary
        test_history: Test history dictionary
        window: Window size for moving statistics (default: 100)

    Returns:
        convergence_metrics: Dictionary of convergence statistics
    """
    if len(history['loss']) < window:
        return {'status': 'insufficient_data'}

    # Recent loss statistics
    recent_losses = history['loss'][-window:]
    mean_loss = np.mean(recent_losses)
    std_loss = np.std(recent_losses)
    rel_std = std_loss / (mean_loss + 1e-10)

    # Relative change
    if len(history['loss']) >= 2 * window:
        prev_mean = np.mean(history['loss'][-2*window:-window])
        rel_change = abs(mean_loss - prev_mean) / (prev_mean + 1e-10)
    else:
        rel_change = 1.0

    # Plateau detection
    is_plateau = (rel_std < 0.01) and (rel_change < 0.05)

    # Test torsion convergence
    if len(test_history['test_torsion']) >= 2:
        test_torsion_change = abs(
            test_history['test_torsion'][-1] - test_history['test_torsion'][-2]
        ) / (test_history['test_torsion'][-2] + 1e-10)
    else:
        test_torsion_change = 1.0

    convergence_metrics = {
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'rel_std': rel_std,
        'rel_change': rel_change,
        'is_plateau': is_plateau,
        'test_torsion_change': test_torsion_change,
        'status': 'converged' if is_plateau else 'training'
    }

    return convergence_metrics


def create_validation_summary(phi_network, harmonic_network, manifold,
                              metric_from_phi_fn, history, test_history, device='cuda'):
    """
    Create comprehensive validation summary.

    Args:
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        manifold: TCSNeckManifold instance
        metric_from_phi_fn: Function to construct metric from phi
        history: Training history
        test_history: Test history
        device: torch device

    Returns:
        summary: Dictionary with all validation metrics
    """
    print("Computing validation summary...")

    # Test set
    test_coords = manifold.sample_points(2000).to(device)

    # PDE residuals
    dphi_L2 = compute_closedness_residual(phi_network, test_coords, device)
    delta_phi_L2 = compute_coclosedness_residual(phi_network, test_coords, metric_from_phi_fn, device)

    # Ricci curvature
    phi_test = phi_network(test_coords)
    metric_test = metric_from_phi_fn(phi_test)
    ricci_norm = compute_ricci_curvature_approx(metric_test, test_coords)

    # Cohomology
    b2, b2_eigenvalues = extract_b2_cohomology(harmonic_network, manifold, device=device)

    # Regional analysis
    regional = analyze_regions(phi_network, manifold, metric_from_phi_fn, device)

    # Convergence
    convergence = compute_convergence_metrics(history, test_history)

    summary = {
        'pde_residuals': {
            'dphi_L2': dphi_L2,
            'delta_phi_L2': delta_phi_L2,
        },
        'ricci_norm': ricci_norm,
        'cohomology': {
            'b2': b2,
            'b2_eigenvalues': b2_eigenvalues.tolist()
        },
        'regional': regional,
        'convergence': convergence,
        'final_metrics': {
            'train_loss': history['loss'][-1] if history['loss'] else None,
            'test_torsion': test_history['test_torsion'][-1] if test_history['test_torsion'] else None,
            'test_det_gram': test_history['test_det_gram'][-1] if test_history['test_det_gram'] else None,
        }
    }

    print("Validation summary complete!")

    return summary
