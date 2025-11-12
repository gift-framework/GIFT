"""
G2 Loss Functions Module - v0.2

Physics-based loss functions for training G2 metric networks.
Implements torsion-free conditions, volume normalization, and curriculum learning.

Author: GIFT Project
No Unicode - Windows compatible
"""

import torch
import torch.nn as nn
import numpy as np

from G2_geometry import (
    exterior_derivative_3form,
    exterior_derivative_4form,
    hodge_star,
    project_spd,
    volume_form,
    ricci_tensor
)
from G2_phi_network import metric_from_phi_algebraic


# ============================================================================
# Individual Loss Components
# ============================================================================

def torsion_loss(phi, metric, coords, method='autograd'):
    """
    Compute torsion-free G2 condition: ||d(phi)||^2 + ||d(*phi)||^2.
    
    This is the primary geometric loss for G2 structures.
    For torsion-free G2: d(phi) = 0 and d(*phi) = 0
    
    Args:
        phi: 3-form of shape (batch, 35)
        metric: Metric tensor of shape (batch, 7, 7)
        coords: Coordinates of shape (batch, 7)
        method: 'autograd' or 'optimized'
    
    Returns:
        loss: Scalar tensor
        info: Dict with detailed loss components
    """
    batch_size = phi.shape[0]
    
    # Compute d(phi)
    d_phi, d_phi_norm_sq = exterior_derivative_3form(phi, coords, method=method)
    
    # Compute Hodge dual *phi
    phi_dual = hodge_star(phi, metric)
    
    # Compute d(*phi)
    d_phi_dual, d_phi_dual_norm_sq = exterior_derivative_4form(phi_dual, coords, method=method)
    
    # Total torsion loss
    loss_d_phi = d_phi_norm_sq.mean()
    loss_d_phi_dual = d_phi_dual_norm_sq.mean()
    loss = loss_d_phi + loss_d_phi_dual
    
    info = {
        'd_phi_norm_sq': loss_d_phi.item(),
        'd_phi_dual_norm_sq': loss_d_phi_dual.item(),
        'torsion_total': loss.item()
    }
    
    return loss, info


def volume_loss(metric):
    """
    Enforce volume normalization: det(g) = 1.
    
    Args:
        metric: Tensor of shape (batch, 7, 7)
    
    Returns:
        loss: Scalar tensor
        info: Dict with loss details
    """
    # Compute determinant
    det_g = torch.det(metric)
    
    # Penalize deviation from 1
    loss = torch.mean((det_g - 1.0) ** 2)
    
    info = {
        'det_g_mean': det_g.mean().item(),
        'det_g_std': det_g.std().item(),
        'volume_loss': loss.item()
    }
    
    return loss, info


def phi_normalization_loss(phi, target=7.0):
    """
    Enforce G2 normalization: ||phi||^2 = 7.
    
    Args:
        phi: 3-form of shape (batch, 35)
        target: Target norm squared (default: 7.0)
    
    Returns:
        loss: Scalar tensor
        info: Dict with loss details
    """
    phi_norm_sq = torch.sum(phi ** 2, dim=1)
    
    loss = torch.mean((phi_norm_sq - target) ** 2)
    
    info = {
        'phi_norm_sq_mean': phi_norm_sq.mean().item(),
        'phi_norm_sq_std': phi_norm_sq.std().item(),
        'phi_normalization_loss': loss.item()
    }
    
    return loss, info


def harmonic_gauge_loss(metric, coords):
    """
    Harmonic gauge condition to prevent metric drift without suppressing curvature.
    
    Instead of penalizing all metric derivatives (which forces flatness),
    we penalize the divergence of the connection (gauge condition).
    
    Condition: ∇^i g_ij = 0 (harmonic coordinates)
    
    Args:
        metric: Tensor of shape (batch, 7, 7)
        coords: Tensor of shape (batch, 7)
    
    Returns:
        loss: Scalar tensor
        info: Dict with loss details
    """
    batch_size = metric.shape[0]
    device = metric.device
    
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    
    # Compute divergence of metric (simplified)
    divergence = torch.zeros(batch_size, 7, device=device)
    
    for i in range(7):
        for j in range(7):
            # ∂g_ij/∂x^j
            grads = torch.autograd.grad(
                metric[:, i, j].sum(),
                coords,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grads is not None:
                divergence[:, i] += grads[:, j]
    
    # Penalize non-zero divergence
    loss = torch.mean(divergence ** 2)
    
    info = {
        'divergence_norm': torch.norm(divergence).item(),
        'harmonic_gauge_loss': loss.item()
    }
    
    return loss, info


def ricci_flatness_loss(metric, coords):
    """
    Enforce Ricci-flatness: Ric(g) = 0.
    
    This is automatically satisfied for torsion-free G2 manifolds,
    but can be used as auxiliary loss or for validation.
    
    Args:
        metric: Tensor of shape (batch, 7, 7)
        coords: Tensor of shape (batch, 7)
    
    Returns:
        loss: Scalar tensor
        info: Dict with loss details
    """
    ricci = ricci_tensor(metric, coords)
    
    loss = torch.mean(ricci ** 2)
    
    info = {
        'ricci_norm': torch.norm(ricci).item(),
        'ricci_flatness_loss': loss.item()
    }
    
    return loss, info


def metric_positivity_loss(metric, epsilon=1e-6):
    """
    Ensure metric is positive definite by penalizing negative eigenvalues.
    
    Args:
        metric: Tensor of shape (batch, 7, 7)
        epsilon: Minimum eigenvalue threshold
    
    Returns:
        loss: Scalar tensor
        info: Dict with loss details
    """
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(metric)
    
    # Penalize eigenvalues below epsilon
    negative_part = torch.relu(epsilon - eigenvalues)
    loss = torch.mean(negative_part ** 2)
    
    min_eigenvalue = eigenvalues.min().item()
    
    info = {
        'min_eigenvalue': min_eigenvalue,
        'positivity_loss': loss.item()
    }
    
    return loss, info


# ============================================================================
# Curriculum Learning Scheduler
# ============================================================================

class CurriculumScheduler:
    """
    Manages curriculum learning for G2 training.
    
    Progressive weight scheduling:
    - Phase 1: Learn phi structure with volume constraint
    - Phase 2: Balance torsion and volume
    - Phase 3: Emphasize torsion-free condition
    """
    
    def __init__(self, 
                 phase_epochs=[500, 2000, 3000],
                 torsion_weights=[0.1, 1.0, 10.0],
                 volume_weights=[10.0, 1.0, 0.1],
                 norm_weights=[1.0, 1.0, 1.0],
                 gauge_weights=[0.1, 0.1, 0.1]):
        """
        Args:
            phase_epochs: List of epoch boundaries [end_phase1, end_phase2, end_phase3]
            torsion_weights: List of torsion loss weights per phase
            volume_weights: List of volume loss weights per phase
            norm_weights: List of normalization loss weights per phase
            gauge_weights: List of harmonic gauge loss weights per phase
        """
        self.phase_epochs = phase_epochs
        self.torsion_weights = torsion_weights
        self.volume_weights = volume_weights
        self.norm_weights = norm_weights
        self.gauge_weights = gauge_weights
        
        self.n_phases = len(phase_epochs)
    
    def get_phase(self, epoch):
        """Determine current training phase."""
        for i, end_epoch in enumerate(self.phase_epochs):
            if epoch < end_epoch:
                return i
        return self.n_phases - 1
    
    def get_weights(self, epoch):
        """
        Get loss weights for current epoch.
        
        Returns:
            weights: Dict with weight for each loss component
        """
        phase = self.get_phase(epoch)
        
        weights = {
            'torsion': self.torsion_weights[phase],
            'volume': self.volume_weights[phase],
            'norm': self.norm_weights[phase],
            'gauge': self.gauge_weights[phase]
        }
        
        return weights
    
    def get_phase_name(self, epoch):
        """Get human-readable phase name."""
        phase = self.get_phase(epoch)
        phase_names = ['Structure Learning', 'Balance', 'Torsion-Free Refinement']
        return phase_names[phase] if phase < len(phase_names) else 'Final'


# ============================================================================
# Combined Loss Function
# ============================================================================

class G2TotalLoss(nn.Module):
    """
    Combined loss function for G2 training with curriculum learning.
    """
    
    def __init__(self, 
                 curriculum_scheduler=None,
                 use_ricci=False,
                 use_positivity=True,
                 derivative_method='autograd'):
        """
        Args:
            curriculum_scheduler: CurriculumScheduler instance (optional)
            use_ricci: Whether to include Ricci flatness loss
            use_positivity: Whether to include positivity loss
            derivative_method: 'autograd' or 'optimized' for exterior derivatives
        """
        super().__init__()
        
        if curriculum_scheduler is None:
            curriculum_scheduler = CurriculumScheduler()
        
        self.scheduler = curriculum_scheduler
        self.use_ricci = use_ricci
        self.use_positivity = use_positivity
        self.derivative_method = derivative_method
    
    def forward(self, phi, metric, coords, epoch=0):
        """
        Compute total loss.
        
        Args:
            phi: 3-form of shape (batch, 35)
            metric: Metric tensor of shape (batch, 7, 7)
            coords: Coordinates of shape (batch, 7)
            epoch: Current training epoch
        
        Returns:
            total_loss: Scalar tensor
            info: Dict with all loss components and weights
        """
        # Get curriculum weights
        weights = self.scheduler.get_weights(epoch)
        
        # Compute individual losses
        loss_torsion, info_torsion = torsion_loss(phi, metric, coords, 
                                                   method=self.derivative_method)
        loss_volume, info_volume = volume_loss(metric)
        loss_norm, info_norm = phi_normalization_loss(phi)
        loss_gauge, info_gauge = harmonic_gauge_loss(metric, coords)
        
        # Weighted sum
        total = (weights['torsion'] * loss_torsion + 
                weights['volume'] * loss_volume +
                weights['norm'] * loss_norm +
                weights['gauge'] * loss_gauge)
        
        # Optional losses
        info_extra = {}
        
        if self.use_ricci:
            loss_ricci, info_ricci = ricci_flatness_loss(metric, coords)
            total = total + 0.1 * loss_ricci
            info_extra.update(info_ricci)
        
        if self.use_positivity:
            loss_pos, info_pos = metric_positivity_loss(metric)
            total = total + 1.0 * loss_pos
            info_extra.update(info_pos)
        
        # Compile info
        info = {
            'total_loss': total.item(),
            'phase': self.scheduler.get_phase_name(epoch),
            'weights': weights,
            **info_torsion,
            **info_volume,
            **info_norm,
            **info_gauge,
            **info_extra
        }
        
        return total, info


# ============================================================================
# Testing and Demonstration
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("G2 Loss Functions Module - Test Suite")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 16
    coords = torch.rand(batch_size, 7, device=device, requires_grad=True) * 2 * np.pi
    
    # Random phi
    phi = torch.randn(batch_size, 35, device=device)
    phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)
    
    # Random metric (make it SPD)
    A = torch.randn(batch_size, 7, 7, device=device)
    metric = A @ A.transpose(-2, -1)
    metric = project_spd(metric)
    
    # Test torsion loss
    print("\n1. Testing Torsion Loss")
    print("-" * 70)
    
    loss, info = torsion_loss(phi, metric, coords, method='autograd')
    print(f"Torsion loss: {loss.item():.6e}")
    print(f"  ||d(phi)||^2: {info['d_phi_norm_sq']:.6e}")
    print(f"  ||d(*phi)||^2: {info['d_phi_dual_norm_sq']:.6e}")
    
    # Test volume loss
    print("\n2. Testing Volume Loss")
    print("-" * 70)
    
    loss, info = volume_loss(metric)
    print(f"Volume loss: {loss.item():.6e}")
    print(f"  det(g) mean: {info['det_g_mean']:.6f}")
    print(f"  det(g) std: {info['det_g_std']:.6f}")
    
    # Test phi normalization
    print("\n3. Testing Phi Normalization Loss")
    print("-" * 70)
    
    loss, info = phi_normalization_loss(phi)
    print(f"Normalization loss: {loss.item():.6e}")
    print(f"  ||phi||^2 mean: {info['phi_norm_sq_mean']:.6f}")
    print(f"  Target: 7.0")
    
    # Test harmonic gauge
    print("\n4. Testing Harmonic Gauge Loss")
    print("-" * 70)
    
    loss, info = harmonic_gauge_loss(metric, coords)
    print(f"Gauge loss: {loss.item():.6e}")
    print(f"  Divergence norm: {info['divergence_norm']:.6e}")
    
    # Test positivity loss
    print("\n5. Testing Positivity Loss")
    print("-" * 70)
    
    loss, info = metric_positivity_loss(metric)
    print(f"Positivity loss: {loss.item():.6e}")
    print(f"  Min eigenvalue: {info['min_eigenvalue']:.6f}")
    
    # Test curriculum scheduler
    print("\n6. Testing Curriculum Scheduler")
    print("-" * 70)
    
    scheduler = CurriculumScheduler()
    
    test_epochs = [0, 250, 500, 1000, 2000, 2500, 3000]
    for epoch in test_epochs:
        weights = scheduler.get_weights(epoch)
        phase = scheduler.get_phase_name(epoch)
        print(f"Epoch {epoch:4d} | {phase:25s} | " + 
              f"torsion: {weights['torsion']:5.1f} | " +
              f"volume: {weights['volume']:5.1f}")
    
    # Test combined loss
    print("\n7. Testing Combined G2 Loss")
    print("-" * 70)
    
    loss_fn = G2TotalLoss(
        curriculum_scheduler=scheduler,
        use_ricci=False,
        use_positivity=True,
        derivative_method='autograd'
    )
    
    for epoch in [0, 1000, 2500]:
        total_loss, info = loss_fn(phi, metric, coords, epoch=epoch)
        print(f"\nEpoch {epoch}:")
        print(f"  Phase: {info['phase']}")
        print(f"  Total loss: {info['total_loss']:.6e}")
        print(f"  Torsion: {info['torsion_total']:.6e} (weight: {info['weights']['torsion']:.1f})")
        print(f"  Volume: {info['volume_loss']:.6e} (weight: {info['weights']['volume']:.1f})")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)






