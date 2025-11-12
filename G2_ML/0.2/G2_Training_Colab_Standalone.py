"""
G₂ Metric Training - Complete Standalone Script for Google Colab

Version 0.2 - Torsion-Free φ-Based Architecture

This script is completely self-contained and can run on Google Colab.
Simply copy-paste into a Colab notebook or run as a script.

GIFT Project - Geometric Inference Framework Theory
"""

# ==============================================================================
# INSTALLATION AND SETUP
# ==============================================================================

import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running on Google Colab")
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'torch', 'matplotlib', 'scipy', 'scikit-learn'])

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print(f"\n✓ Using device: {device}")


# ==============================================================================
# MODULE 1: MANIFOLD (T⁷ TORUS)
# ==============================================================================

class TorusT7:
    """7-dimensional torus with periodic boundaries."""
    
    def __init__(self, radii=None, device='cpu'):
        if radii is None:
            radii = [2.0 * np.pi] * 7
        self.radii = torch.tensor(radii, dtype=torch.float32, device=device)
        self.device = device
        self.dim = 7
    
    def sample_points(self, n_batch):
        """Sample random points on T^7."""
        return torch.rand(n_batch, 7, device=self.device) * self.radii.unsqueeze(0)
    
    def volume(self):
        """Compute volume."""
        return torch.prod(self.radii).item()


# ==============================================================================
# MODULE 2: NEURAL NETWORK (φ-NETWORK)
# ==============================================================================

class FourierFeatures(nn.Module):
    """Random Fourier features."""
    
    def __init__(self, input_dim=7, n_modes=16, scale=1.0):
        super().__init__()
        B = torch.randn(input_dim, n_modes) * scale
        self.register_buffer('B', B)
    
    def forward(self, x):
        proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class G2PhiNetwork(nn.Module):
    """Neural network for G2 3-form φ."""
    
    def __init__(self, hidden_dims=[256, 256, 128], fourier_modes=16):
        super().__init__()
        
        self.encoding = FourierFeatures(input_dim=7, n_modes=fourier_modes)
        encoding_dim = 2 * fourier_modes
        
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
        self.output_layer = nn.Linear(prev_dim, 35)  # 35 = C(7,3)
        
        with torch.no_grad():
            self.output_layer.weight.mul_(0.01)
            self.output_layer.bias.zero_()
    
    def forward(self, coords):
        x = self.encoding(coords)
        x = self.mlp(x)
        phi = self.output_layer(x)
        
        # Normalize to ||phi||^2 = 7
        phi_norm = torch.norm(phi, dim=-1, keepdim=True)
        phi = phi * (np.sqrt(7.0) / (phi_norm + 1e-8))
        
        return phi


# ==============================================================================
# MODULE 3: GEOMETRY OPERATIONS
# ==============================================================================

def project_spd(metric, epsilon=1e-6):
    """Project to positive definite."""
    metric = 0.5 * (metric + metric.transpose(-2, -1))
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)


def metric_from_phi_approximate(phi):
    """Reconstruct metric from phi."""
    batch_size, device = phi.shape[0], phi.device
    metric = torch.zeros(batch_size, 7, 7, device=device)
    
    # Build index mapping
    triple_to_idx = {}
    idx = 0
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                triple_to_idx[(i, j, k)] = idx
                idx += 1
    
    # Diagonal from phi norm
    phi_norm = torch.norm(phi, dim=1, keepdim=True)
    for i in range(7):
        metric[:, i, i] = phi_norm.squeeze() / np.sqrt(7.0)
    
    # Off-diagonal
    for i in range(7):
        for j in range(i+1, 7):
            contrib, count = 0.0, 0
            for k in range(7):
                if k != i and k != j:
                    triple = tuple(sorted([i, j, k]))
                    if triple in triple_to_idx:
                        contrib += phi[:, triple_to_idx[triple]]**2
                        count += 1
            if count > 0:
                metric[:, i, j] = torch.sqrt(contrib / count + 1e-8) * 0.1
                metric[:, j, i] = metric[:, i, j]
    
    return metric


def hodge_star(phi, metric):
    """Compute Hodge dual."""
    det_g = torch.det(metric)
    vol = torch.sqrt(torch.abs(det_g) + 1e-10).unsqueeze(-1)
    phi_dual = phi * vol
    phi_dual_norm = torch.norm(phi_dual, dim=1, keepdim=True)
    return phi_dual / (phi_dual_norm + 1e-8) * np.sqrt(7.0)


def exterior_derivative_3form(phi, coords):
    """Compute exterior derivative."""
    batch_size, device = phi.shape[0], phi.device
    
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    
    d_phi = torch.zeros(batch_size, 35, device=device)
    
    for comp_idx in range(35):
        grads = torch.autograd.grad(
            phi[:, comp_idx].sum(), coords,
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if grads is not None:
            d_phi[:, comp_idx] = torch.norm(grads, dim=1)
    
    return d_phi, torch.sum(d_phi ** 2, dim=1)


# ==============================================================================
# MODULE 4: LOSS FUNCTIONS
# ==============================================================================

def torsion_loss(phi, metric, coords):
    """Torsion-free condition."""
    d_phi, d_phi_norm_sq = exterior_derivative_3form(phi, coords)
    phi_dual = hodge_star(phi, metric)
    d_phi_dual, d_phi_dual_norm_sq = exterior_derivative_3form(phi_dual, coords)
    
    loss = d_phi_norm_sq.mean() + d_phi_dual_norm_sq.mean()
    return loss, {'torsion': loss.item()}


def volume_loss(metric):
    """Volume normalization."""
    det_g = torch.det(metric)
    loss = torch.mean((det_g - 1.0) ** 2)
    return loss, {'det_g': det_g.mean().item()}


def phi_normalization_loss(phi):
    """Phi norm."""
    phi_norm_sq = torch.sum(phi ** 2, dim=1)
    loss = torch.mean((phi_norm_sq - 7.0) ** 2)
    return loss, {'phi_norm_sq': phi_norm_sq.mean().item()}


def metric_positivity_loss(metric):
    """Positive definiteness."""
    eigenvalues = torch.linalg.eigvalsh(metric)
    negative_part = torch.relu(1e-6 - eigenvalues)
    loss = torch.mean(negative_part ** 2)
    return loss, {'min_eig': eigenvalues.min().item()}


class CurriculumScheduler:
    """Curriculum learning."""
    
    def __init__(self, phase_epochs=[500, 2000, 3000],
                 torsion_w=[0.1, 1.0, 10.0], volume_w=[10.0, 1.0, 0.1]):
        self.phase_epochs = phase_epochs
        self.torsion_w = torsion_w
        self.volume_w = volume_w
    
    def get_weights(self, epoch):
        phase = 0
        for i, end in enumerate(self.phase_epochs):
            if epoch < end:
                phase = i
                break
        return {'torsion': self.torsion_w[phase], 'volume': self.volume_w[phase]}


class G2TotalLoss(nn.Module):
    """Total loss."""
    
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def forward(self, phi, metric, coords, epoch=0):
        w = self.scheduler.get_weights(epoch)
        
        l_torsion, i_torsion = torsion_loss(phi, metric, coords)
        l_volume, i_volume = volume_loss(metric)
        l_norm, i_norm = phi_normalization_loss(phi)
        l_pos, i_pos = metric_positivity_loss(metric)
        
        total = w['torsion'] * l_torsion + w['volume'] * l_volume + l_norm + l_pos
        
        info = {'total': total.item(), **i_torsion, **i_volume, **i_norm, **i_pos}
        return total, info


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

config = {
    'hidden_dims': [256, 256, 128],
    'fourier_modes': 16,
    'batch_size': 512,
    'epochs': 200,  # Set to 3000 for full training
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'plot_interval': 20,
    'seed': 42
}

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

print("\n" + "="*70)
print("Configuration")
print("="*70)
for k, v in config.items():
    print(f"  {k}: {v}")
print("="*70)


# ==============================================================================
# CREATE MODEL AND OPTIMIZER
# ==============================================================================

manifold = TorusT7(device=device)
model = G2PhiNetwork(config['hidden_dims'], config['fourier_modes']).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler_lr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=1e-7)

curriculum = CurriculumScheduler([50, 120, 200], [0.1, 1.0, 10.0], [10.0, 1.0, 0.1])
loss_fn = G2TotalLoss(curriculum)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model: {n_params:,} parameters")
print(f"✓ Device: {device}")


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

history = {'epoch': [], 'loss': [], 'torsion': [], 'phi_norm': [], 'det_g': []}

print("\n" + "="*70)
print("Starting Training")
print("="*70)

start_time = time.time()

for epoch in range(config['epochs']):
    model.train()
    
    # Sample batch
    coords = manifold.sample_points(config['batch_size']).requires_grad_(True)
    
    # Forward
    phi = model(coords)
    metric = metric_from_phi_approximate(phi)
    metric = project_spd(metric)
    
    # Loss
    total_loss, info = loss_fn(phi, metric, coords, epoch)
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    optimizer.step()
    scheduler_lr.step()
    
    # Record
    history['epoch'].append(epoch)
    history['loss'].append(info['total'])
    history['torsion'].append(info['torsion'])
    history['phi_norm'].append(info['phi_norm_sq'])
    history['det_g'].append(info['det_g'])
    
    # Plot
    if epoch % config['plot_interval'] == 0 or epoch == config['epochs'] - 1:
        clear_output(wait=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training - Epoch {epoch}/{config["epochs"]}', fontsize=14, fontweight='bold')
        
        axes[0,0].plot(history['epoch'], history['loss'], 'b-', lw=2)
        axes[0,0].set_yscale('log')
        axes[0,0].set_title('Total Loss')
        axes[0,0].grid(alpha=0.3)
        
        axes[0,1].plot(history['epoch'], history['torsion'], 'r-', lw=2)
        axes[0,1].set_yscale('log')
        axes[0,1].set_title('Torsion: ||dφ||² + ||d*φ||²')
        axes[0,1].grid(alpha=0.3)
        
        axes[1,0].plot(history['epoch'], history['phi_norm'], 'g-', lw=2)
        axes[1,0].axhline(7.0, color='k', ls='--', lw=1)
        axes[1,0].set_title('||φ||² (target: 7.0)')
        axes[1,0].grid(alpha=0.3)
        
        axes[1,1].plot(history['epoch'], history['det_g'], 'purple', lw=2)
        axes[1,1].axhline(1.0, color='k', ls='--', lw=1)
        axes[1,1].set_title('det(g) (target: 1.0)')
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{config['epochs']} | Time: {elapsed:.1f}s")
        print(f"  Loss: {info['total']:.6e} | Torsion: {info['torsion']:.6e}")
        print(f"  ||φ||²: {info['phi_norm_sq']:.6f} | det(g): {info['det_g']:.6f}")

print(f"\n{'='*70}")
print(f"✓ Training complete! Time: {(time.time()-start_time)/60:.2f} min")
print(f"{'='*70}")


# ==============================================================================
# FINAL VALIDATION
# ==============================================================================

model.eval()
print("\n" + "="*70)
print("Final Validation")
print("="*70)

with torch.no_grad():
    coords = manifold.sample_points(1000)
    phi = model(coords)
    metric = metric_from_phi_approximate(phi)
    metric = project_spd(metric)
    
    phi_norm_sq = torch.sum(phi ** 2, dim=1)
    det_g = torch.det(metric)
    eigenvalues = torch.linalg.eigvalsh(metric)
    
    print(f"\n||φ||²: {phi_norm_sq.mean():.6f} ± {phi_norm_sq.std():.6f} (target: 7.0)")
    print(f"det(g): {det_g.mean():.6f} ± {det_g.std():.6f} (target: 1.0)")
    print(f"Eigenvalues: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
    print(f"Positive definite: {eigenvalues.min() > 0}")

print("\n" + "="*70)
print("✓ Complete!")
print("="*70)
print("\nGIFT Project - Geometric Inference Framework Theory")
print("Version 0.2 - Torsion-Free φ-Based Architecture")






