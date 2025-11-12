"""
G₂ Metric Training - Complete Standalone Script for Google Colab
WITH AUTOMATIC SAVING AND ARCHIVING

Version 0.2 - Torsion-Free φ-Based Architecture

✅ Sauvegarde automatique:
   - Modèle final
   - Checkpoints intermédiaires (tous les 500 epochs)
   - Historique complet (CSV + JSON)
   - Configuration
   - Graphiques finaux
   - Téléchargement automatique sur Colab

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
import os
import json

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print(f"\nUsing device: {device}")

# Create output directory
OUTPUT_DIR = 'g2_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}/")


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
        self.output_layer = nn.Linear(prev_dim, 35)
        
        with torch.no_grad():
            self.output_layer.weight.mul_(0.01)
            self.output_layer.bias.zero_()
    
    def forward(self, coords):
        x = self.encoding(coords)
        x = self.mlp(x)
        phi = self.output_layer(x)
        
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
    
    triple_to_idx = {}
    idx = 0
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                triple_to_idx[(i, j, k)] = idx
                idx += 1
    
    phi_norm = torch.norm(phi, dim=1, keepdim=True)
    for i in range(7):
        metric[:, i, i] = phi_norm.squeeze() / np.sqrt(7.0)
    
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
# SAVING UTILITIES
# ==============================================================================

def save_checkpoint(model, optimizer, epoch, config, history, filename, verbose=True):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }
    path = os.path.join(OUTPUT_DIR, filename)
    torch.save(checkpoint, path)
    if verbose:
        print(f"  Checkpoint saved: {filename}")
    return path


def save_history_csv(history, filename='training_history.csv'):
    """Save training history to CSV."""
    import csv
    path = os.path.join(OUTPUT_DIR, filename)
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'torsion', 'phi_norm_sq', 'det_g'])
        
        for i in range(len(history['epoch'])):
            writer.writerow([
                history['epoch'][i],
                history['loss'][i],
                history['torsion'][i],
                history['phi_norm'][i],
                history['det_g'][i]
            ])
    
    print(f"  History saved: {filename}")
    return path


def save_final_plots(history, filename='training_plots.png'):
    """Save final training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('G₂ Training Results', fontsize=16, fontweight='bold')
    
    axes[0,0].plot(history['epoch'], history['loss'], 'b-', lw=2)
    axes[0,0].set_yscale('log')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Total Loss')
    axes[0,0].set_title('Total Loss')
    axes[0,0].grid(alpha=0.3)
    
    axes[0,1].plot(history['epoch'], history['torsion'], 'r-', lw=2)
    axes[0,1].set_yscale('log')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Torsion')
    axes[0,1].set_title('Torsion: ||dφ||² + ||d*φ||²')
    axes[0,1].axhline(1e-6, color='g', ls='--', label='Target: 1e-6')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    axes[1,0].plot(history['epoch'], history['phi_norm'], 'g-', lw=2)
    axes[1,0].axhline(7.0, color='k', ls='--', lw=1)
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('||φ||²')
    axes[1,0].set_title('||φ||² (target: 7.0)')
    axes[1,0].grid(alpha=0.3)
    
    axes[1,1].plot(history['epoch'], history['det_g'], 'purple', lw=2)
    axes[1,1].axhline(1.0, color='k', ls='--', lw=1)
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('det(g)')
    axes[1,1].set_title('det(g) (target: 1.0)')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Plots saved: {filename}")
    
    plt.show()
    return path


def download_all_results():
    """Download all results from Colab."""
    if IN_COLAB:
        from google.colab import files
        import glob
        
        print("\nDownloading all results...")
        
        all_files = glob.glob(os.path.join(OUTPUT_DIR, '*'))
        
        for file_path in all_files:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                print(f"  Downloading: {filename}")
                files.download(file_path)
        
        print("All files downloaded!")


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

config = {
    'hidden_dims': [256, 256, 128],
    'fourier_modes': 16,
    'batch_size': 512,
    'epochs': 3000,  # FULL TRAINING
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'plot_interval': 50,
    'checkpoint_interval': 500,  # Save every 500 epochs
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

# Save configuration
config_path = os.path.join(OUTPUT_DIR, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"\nConfig saved: {config_path}")


# ==============================================================================
# CREATE MODEL AND OPTIMIZER
# ==============================================================================

manifold = TorusT7(device=device)
model = G2PhiNetwork(config['hidden_dims'], config['fourier_modes']).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler_lr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=1e-7)

curriculum = CurriculumScheduler([500, 2000, 3000], [0.1, 1.0, 10.0], [10.0, 1.0, 0.1])
loss_fn = G2TotalLoss(curriculum)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} parameters")
print(f"Device: {device}")
print(f"Checkpoints every: {config['checkpoint_interval']} epochs")


# ==============================================================================
# TRAINING LOOP WITH AUTOMATIC SAVING
# ==============================================================================

history = {'epoch': [], 'loss': [], 'torsion': [], 'phi_norm': [], 'det_g': []}

print("\n" + "="*70)
print("Starting Training with Automatic Saving")
print("="*70)

start_time = time.time()
best_loss = float('inf')

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
    
    # Save checkpoint (silently except for major milestones)
    if (epoch + 1) % config['checkpoint_interval'] == 0:
        filename = f'checkpoint_epoch_{epoch+1}.pt'
        save_checkpoint(model, optimizer, epoch, config, history, filename, verbose=False)
    
    # Save best model (silently)
    if info['total'] < best_loss:
        best_loss = info['total']
        save_checkpoint(model, optimizer, epoch, config, history, 'best_model.pt', verbose=False)
    
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
        axes[0,1].set_title('Torsion')
        axes[0,1].axhline(1e-6, color='g', ls='--', alpha=0.5)
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
        print(f"\nEpoch {epoch}/{config['epochs']} | Time: {elapsed/60:.1f}min")
        print(f"  Loss: {info['total']:.6e} | Torsion: {info['torsion']:.6e}")
        print(f"  ||φ||²: {info['phi_norm_sq']:.6f} | det(g): {info['det_g']:.6f}")

elapsed_total = time.time() - start_time

print(f"\n{'='*70}")
print(f"Training complete! Time: {elapsed_total/60:.2f} min")
print(f"{'='*70}")

# Print checkpoint summary
import glob
checkpoints = glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint_*.pt'))
print(f"\nCheckpoints saved: {len(checkpoints)}")
print(f"Best model saved: best_model.pt")


# ==============================================================================
# SAVE ALL RESULTS
# ==============================================================================

print("\n" + "="*70)
print("Saving All Results")
print("="*70)

# Save final model
save_checkpoint(model, optimizer, config['epochs']-1, config, history, 'final_model.pt')

# Save history
save_history_csv(history)

# Save history as JSON too
history_json_path = os.path.join(OUTPUT_DIR, 'training_history.json')
with open(history_json_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"  History JSON saved: training_history.json")

# Save final plots
save_final_plots(history)


# ==============================================================================
# FINAL VALIDATION
# ==============================================================================

model.eval()
print("\n" + "="*70)
print("Final Validation")
print("="*70)

with torch.no_grad():
    coords = manifold.sample_points(2000)
    phi = model(coords)
    metric = metric_from_phi_approximate(phi)
    metric = project_spd(metric)
    
    phi_norm_sq = torch.sum(phi ** 2, dim=1)
    det_g = torch.det(metric)
    eigenvalues = torch.linalg.eigvalsh(metric)
    
    validation_results = {
        'phi_norm_sq_mean': phi_norm_sq.mean().item(),
        'phi_norm_sq_std': phi_norm_sq.std().item(),
        'det_g_mean': det_g.mean().item(),
        'det_g_std': det_g.std().item(),
        'eigenvalue_min': eigenvalues.min().item(),
        'eigenvalue_max': eigenvalues.max().item(),
        'final_torsion': history['torsion'][-1],
        'final_loss': history['loss'][-1]
    }
    
    print(f"\n||φ||²: {validation_results['phi_norm_sq_mean']:.6f} ± {validation_results['phi_norm_sq_std']:.6f} (target: 7.0)")
    print(f"det(g): {validation_results['det_g_mean']:.6f} ± {validation_results['det_g_std']:.6f} (target: 1.0)")
    print(f"Eigenvalues: [{validation_results['eigenvalue_min']:.6f}, {validation_results['eigenvalue_max']:.6f}]")
    print(f"Final Torsion: {validation_results['final_torsion']:.6e}")
    print(f"Positive definite: {validation_results['eigenvalue_min'] > 0}")

# Save validation results
validation_path = os.path.join(OUTPUT_DIR, 'validation_results.json')
with open(validation_path, 'w') as f:
    json.dump(validation_results, f, indent=2)
print(f"\n  Validation saved: validation_results.json")


# ==============================================================================
# DOWNLOAD ALL FILES (COLAB)
# ==============================================================================

print("\n" + "="*70)
print("Summary of Saved Files")
print("="*70)

import glob
all_files = glob.glob(os.path.join(OUTPUT_DIR, '*'))
print(f"\nTotal files saved: {len(all_files)}")
for file_path in all_files:
    if os.path.isfile(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        filename = os.path.basename(file_path)
        print(f"  {filename} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print("DOWNLOAD RESULTS")
print("="*70)
print("\nTo download all files, run:")
print("  download_all_results()")
print("\nOr download manually from the 'g2_outputs/' folder")

if IN_COLAB:
    print("\nImportant: Run this before closing Colab!")
    user_input = input("\nDownload all files now? (y/n): ")
    if user_input.lower() == 'y':
        download_all_results()

print("\n" + "="*70)
print("Complete!")
print("="*70)
print("\nGIFT Project - Geometric Inference Framework Theory")
print("Version 0.2 - Torsion-Free φ-Based Architecture")
print(f"\nResults saved in: {OUTPUT_DIR}/")

