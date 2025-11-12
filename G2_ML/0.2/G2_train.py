"""
G2 Training Script - v0.2

Main training script for G2 metric learning with CLI interface.
Supports both Fourier and SIREN architectures with curriculum learning.

Usage:
    python G2_train.py --encoding fourier --epochs 3000 --batch-size 512
    python G2_train.py --encoding siren --epochs 3000 --device cuda

Author: GIFT Project
No Unicode - Windows compatible
"""

import argparse
import os
import json
import time
from datetime import datetime

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
from G2_geometry import project_spd
from G2_manifold import create_manifold
from G2_losses import G2TotalLoss, CurriculumScheduler
from G2_validation import comprehensive_validation


# ============================================================================
# Training Configuration
# ============================================================================

def get_default_config():
    """Return default training configuration."""
    return {
        'encoding_type': 'fourier',
        'hidden_dims': [256, 256, 128],
        'fourier_modes': 16,
        'fourier_scale': 1.0,
        'omega_0': 30.0,
        'normalize_phi': True,
        'manifold_type': 'T7',
        'batch_size': 512,
        'epochs': 3000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'checkpoint_interval': 500,
        'validation_interval': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'output_dir': 'checkpoints',
        'use_curriculum': True,
        'derivative_method': 'autograd',
        'use_scheduler': True,
        'scheduler_T0': 500,
        'scheduler_eta_min': 1e-7
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train G2 metric network')
    
    # Architecture
    parser.add_argument('--encoding', type=str, default='fourier', 
                       choices=['fourier', 'siren'],
                       help='Encoding type (fourier or siren)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--fourier-modes', type=int, default=16,
                       help='Number of Fourier modes (for Fourier encoding)')
    parser.add_argument('--omega-0', type=float, default=30.0,
                       help='SIREN frequency parameter')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3000,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Curriculum
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--derivative-method', type=str, default='autograd',
                       choices=['autograd', 'optimized'],
                       help='Method for computing exterior derivatives')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    
    # Monitoring
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validation-interval', type=int, default=500,
                       help='Run validation every N epochs')
    
    args = parser.parse_args()
    
    # Handle auto device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, loss_fn, optimizer, manifold, config, epoch):
    """
    Train for one epoch.
    
    Args:
        model: G2PhiNetwork
        loss_fn: G2TotalLoss
        optimizer: PyTorch optimizer
        manifold: Manifold object
        config: Training configuration dict
        epoch: Current epoch number
    
    Returns:
        metrics: Dict with training metrics
    """
    model.train()
    device = config['device']
    
    # Sample batch
    coords = manifold.sample_points(config['batch_size'], method='uniform')
    coords = coords.to(device)
    coords.requires_grad = True
    
    # Forward pass
    phi = model(coords)
    
    # Reconstruct metric
    metric = metric_from_phi_algebraic(phi, use_approximation=True)
    metric = project_spd(metric)
    
    # Compute loss
    total_loss, loss_info = loss_fn(phi, metric, coords, epoch=epoch)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    if config['grad_clip'] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    
    optimizer.step()
    
    # Compile metrics
    metrics = {
        'epoch': epoch,
        'loss': total_loss.item(),
        **loss_info
    }
    
    return metrics


def validate(model, manifold, device, n_samples=1000):
    """
    Run validation.
    
    Args:
        model: G2PhiNetwork
        manifold: Manifold object
        device: PyTorch device
        n_samples: Number of validation samples
    
    Returns:
        results: Validation results dict
    """
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Basic metrics
        phi_norm_sq = torch.sum(phi ** 2, dim=1)
        det_g = torch.det(metric)
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        results = {
            'phi_norm_sq_mean': phi_norm_sq.mean().item(),
            'phi_norm_sq_std': phi_norm_sq.std().item(),
            'det_g_mean': det_g.mean().item(),
            'det_g_std': det_g.std().item(),
            'eigenvalue_min': eigenvalues.min().item(),
            'eigenvalue_max': eigenvalues.max().item()
        }
    
    return results


# ============================================================================
# Main Training Function
# ============================================================================

def train(config):
    """
    Main training function.
    
    Args:
        config: Configuration dict
    """
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = config['device']
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")
    
    # Create manifold
    print("Creating manifold...")
    manifold = create_manifold(config['manifold_type'], device=device)
    print(f"Manifold: {type(manifold).__name__}")
    print(f"Volume: {manifold.volume():.4f}\n")
    
    # Create model
    print("Creating model...")
    model = G2PhiNetwork(
        encoding_type=config['encoding_type'],
        hidden_dims=config['hidden_dims'],
        fourier_modes=config['fourier_modes'],
        fourier_scale=config['fourier_scale'],
        omega_0=config['omega_0'],
        normalize_phi=config['normalize_phi']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Encoding: {config['encoding_type']}\n")
    
    # Create loss function
    print("Creating loss function...")
    curriculum = None
    if config['use_curriculum']:
        curriculum = CurriculumScheduler()
        print("Using curriculum learning")
    
    loss_fn = G2TotalLoss(
        curriculum_scheduler=curriculum,
        use_ricci=False,
        use_positivity=True,
        derivative_method=config['derivative_method']
    )
    print(f"Derivative method: {config['derivative_method']}\n")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['scheduler_T0'],
            eta_min=config['scheduler_eta_min']
        )
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'torsion': [],
        'volume': [],
        'phi_norm': [],
        'det_g': []
    }
    
    # Training loop
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Train
        metrics = train_epoch(model, loss_fn, optimizer, manifold, config, epoch)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['epoch'].append(epoch)
        history['loss'].append(metrics['loss'])
        history['torsion'].append(metrics.get('torsion_total', 0))
        history['volume'].append(metrics.get('volume_loss', 0))
        history['phi_norm'].append(metrics.get('phi_norm_sq_mean', 0))
        history['det_g'].append(metrics.get('det_g_mean', 0))
        
        # Progress bar update
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{config['epochs']} | "
                  f"Loss: {metrics['loss']:.6e} | "
                  f"Torsion: {metrics.get('torsion_total', 0):.6e} | "
                  f"Phase: {metrics.get('phase', 'N/A')} | "
                  f"LR: {current_lr:.2e}")
        
        # Validation
        if (epoch + 1) % config['validation_interval'] == 0 or epoch == config['epochs'] - 1:
            print("\n" + "-" * 70)
            print(f"Validation at epoch {epoch}")
            val_results = validate(model, manifold, device)
            print(f"  ||phi||^2: {val_results['phi_norm_sq_mean']:.6f} (target: 7.0)")
            print(f"  det(g): {val_results['det_g_mean']:.6f} (target: 1.0)")
            print(f"  Eigenvalues: [{val_results['eigenvalue_min']:.4f}, {val_results['eigenvalue_max']:.4f}]")
            print("-" * 70 + "\n")
        
        # Checkpointing
        if (epoch + 1) % config['checkpoint_interval'] == 0 or epoch == config['epochs'] - 1:
            checkpoint_path = os.path.join(
                config['output_dir'],
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'config': config
            }, checkpoint_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_path = os.path.join(config['output_dir'], 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': metrics['loss'],
                    'config': config
                }, best_path)
                print(f"Best model updated: {best_path}")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training complete! Time elapsed: {elapsed_time/3600:.2f} hours")
    print("=" * 70)
    
    # Save final model
    final_path = os.path.join(config['output_dir'], 'final_model.pt')
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    
    # Save training history
    history_path = os.path.join(config['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")
    
    # Plot training curves
    plot_training_history(history, config['output_dir'])
    
    # Final validation
    print("\n" + "=" * 70)
    print("Running comprehensive validation...")
    print("=" * 70)
    
    report = comprehensive_validation(model, manifold, n_samples=2000, device=device, verbose=True)
    
    # Save validation report
    report_path = os.path.join(config['output_dir'], 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved: {report_path}")
    
    return model, history, report


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history, output_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Torsion loss
    ax = axes[0, 1]
    ax.plot(epochs, history['torsion'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Torsion Loss')
    ax.set_title('Torsion Loss (||d(phi)||^2 + ||d(*phi)||^2)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Phi normalization
    ax = axes[1, 0]
    ax.plot(epochs, history['phi_norm'], 'g-', linewidth=2)
    ax.axhline(y=7.0, color='k', linestyle='--', label='Target: 7.0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||phi||^2')
    ax.set_title('Phi Normalization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Volume
    ax = axes[1, 1]
    ax.plot(epochs, history['det_g'], 'purple', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--', label='Target: 1.0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('det(g)')
    ax.set_title('Metric Determinant')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining plot saved: {plot_path}")
    
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Get default config and update with args
    config = get_default_config()
    config.update({
        'encoding_type': args.encoding,
        'hidden_dims': args.hidden_dims,
        'fourier_modes': args.fourier_modes,
        'omega_0': args.omega_0,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'use_curriculum': not args.no_curriculum,
        'derivative_method': args.derivative_method,
        'device': args.device,
        'seed': args.seed,
        'output_dir': args.output_dir,
        'checkpoint_interval': args.checkpoint_interval,
        'validation_interval': args.validation_interval
    })
    
    # Print configuration
    print("\n" + "=" * 70)
    print("G2 Metric Training - v0.2")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train
    model, history, report = train(config)
    
    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()






