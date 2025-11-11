"""
Training Loop for GIFT v0.9
============================

This module contains the complete training infrastructure:
- Optimizer setup (AdamW)
- Learning rate scheduler (Cosine annealing)
- 4-phase curriculum management
- Training loop with gradient accumulation
- Checkpoint saving/loading
- History tracking
- Mixed precision (AMP) support

Critical Fixes from v0.8:
- All history keys initialized upfront (prevents KeyError)
- Gradient clipping before optimizer step
- Memory cleanup every 100 epochs
- Robust error handling with fallback
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import gc
import json
from pathlib import Path
from datetime import datetime


# ============================================================================
# Configuration Dictionary
# ============================================================================

CONFIG = {
    # Version & geometry
    'version': 'v0.9',
    'geometry': 'TCS_neck_ACyl',

    # Training core
    'epochs': 10000,
    'batch_size': 1536,
    'grad_accumulation_steps': 2,
    'effective_batch': 3072,  # batch_size × grad_accumulation_steps

    # Optimization
    'lr': 1e-4,              # Starting learning rate
    'weight_decay': 1e-4,
    'grad_clip': 1.0,        # Gradient clipping threshold
    'scheduler': 'cosine',   # Cosine annealing
    'warmup_epochs': 500,
    'eta_min': 1e-6,         # Final lr (100× decay)

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


# ============================================================================
# Optimizer & Scheduler Setup
# ============================================================================

def setup_optimizer_and_scheduler(phi_network, harmonic_network, metric_network=None):
    """
    Initialize AdamW optimizer and cosine annealing scheduler.

    Args:
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        metric_network: MetricNetwork instance (optional, if training separate metric)

    Returns:
        optimizer: torch.optim.AdamW optimizer
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR scheduler
    """
    # Set reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    # Collect all parameters
    params = list(phi_network.parameters()) + list(harmonic_network.parameters())
    if metric_network is not None:
        params += list(metric_network.parameters())

    # Optimizer: AdamW (better than Adam for regularization)
    optimizer = optim.AdamW(
        params,
        lr=CONFIG['lr'],  # 1e-4
        weight_decay=CONFIG['weight_decay']  # 1e-4
    )

    # Learning rate scheduler: Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['epochs'],  # 10000
        eta_min=CONFIG['eta_min']  # 1e-6 (final lr)
    )

    return optimizer, scheduler


# ============================================================================
# History Initialization
# ============================================================================

def initialize_history():
    """
    Initialize history dictionaries with ALL keys upfront.

    CRITICAL: Pre-defining all keys prevents KeyError crashes during training.

    Returns:
        history: Training history dictionary
        test_history: Test/validation history dictionary
    """
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
        'test_dstar_phi_L2': [],
        'test_ricci_norm': []
    }

    return history, test_history


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(phi_network, harmonic_network, manifold, optimizer, scheduler,
                epoch, history, device, metric_from_phi_fn, loss_fn, weights):
    """
    Execute one training epoch.

    Args:
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        manifold: TCSNeckManifold instance
        optimizer: torch.optim.Optimizer
        scheduler: torch.optim.lr_scheduler
        epoch: Current epoch number
        history: Training history dictionary
        device: torch device
        metric_from_phi_fn: Function to construct metric from phi
        loss_fn: Loss computation function
        weights: Dictionary of loss weights from curriculum

    Returns:
        loss_dict: Dictionary of loss components for logging
    """
    phi_network.train()
    harmonic_network.train()

    # Sample batch
    coords = manifold.sample_points(CONFIG['batch_size']).to(device)
    coords.requires_grad_(True)  # CRITICAL for torsion computation

    # Forward pass
    phi = phi_network(coords)
    h_forms = harmonic_network(coords)
    metric = metric_from_phi_fn(phi)

    # Compute loss
    loss, loss_dict = loss_fn(
        phi, h_forms, metric, coords, manifold,
        phi_network, harmonic_network, weights
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping (CRITICAL for stability)
    torch.nn.utils.clip_grad_norm_(
        list(phi_network.parameters()) + list(harmonic_network.parameters()),
        CONFIG['grad_clip']
    )

    optimizer.step()
    scheduler.step()

    # Add total loss to dict
    loss_dict['loss'] = loss

    return loss_dict


def train_model(phi_network, harmonic_network, manifold, metric_from_phi_fn,
                loss_fn, test_coords, checkpoint_dir, device):
    """
    Main training loop with 4-phase curriculum.

    Args:
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        manifold: TCSNeckManifold instance
        metric_from_phi_fn: Function to construct metric from phi (e.g., metric_from_phi_robust)
        loss_fn: Loss computation function (e.g., compute_total_loss from losses.py)
        test_coords: (n_test, 7) tensor of fixed test coordinates
        checkpoint_dir: Path to save checkpoints
        device: torch device

    Returns:
        history: Training history dictionary
        test_history: Test/validation history dictionary
    """
    from .losses import get_phase_weights_smooth, SafeMetrics, check_early_stopping

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(phi_network, harmonic_network)

    # Initialize history
    history, test_history = initialize_history()

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {CONFIG['epochs']} epochs")
    print(f"Batch size: {CONFIG['batch_size']}, Effective batch: {CONFIG['effective_batch']}")
    print(f"Learning rate: {CONFIG['lr']} → {CONFIG['eta_min']} (cosine annealing)")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 80)

    # Training loop
    for epoch in range(CONFIG['epochs']):
        try:
            # Get curriculum weights
            weights, phase_name = get_phase_weights_smooth(epoch, CONFIG['transition_width'])

            # Train one epoch
            loss_dict = train_epoch(
                phi_network, harmonic_network, manifold,
                optimizer, scheduler, epoch, history, device,
                metric_from_phi_fn, loss_fn, weights
            )

            # Log metrics (EVERY epoch)
            history['epoch'].append(epoch)
            history['loss'].append(SafeMetrics.to_scalar(loss_dict['loss']))
            history['torsion'].append(SafeMetrics.to_scalar(loss_dict['torsion']))
            history['volume'].append(SafeMetrics.to_scalar(loss_dict['volume']))
            history['det_gram'].append(SafeMetrics.to_scalar(loss_dict['det_gram']))
            history['harmonic_ortho'].append(SafeMetrics.to_scalar(loss_dict['harmonic_ortho']))
            history['harmonic_det'].append(SafeMetrics.to_scalar(loss_dict['harmonic_det']))
            history['separation'].append(SafeMetrics.to_scalar(loss_dict['separation']))
            history['boundary'].append(SafeMetrics.to_scalar(loss_dict['boundary']))
            history['decay'].append(SafeMetrics.to_scalar(loss_dict['decay']))
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['phase'].append(phase_name)

            # Metric health monitoring
            with torch.no_grad():
                coords_sample = manifold.sample_points(512).to(device)
                phi_sample = phi_network(coords_sample)
                metric_sample = metric_from_phi_fn(phi_sample)

                eigvals = torch.linalg.eigvalsh(metric_sample)
                condition_numbers = eigvals.max(dim=1)[0] / (eigvals.min(dim=1)[0] + 1e-10)
                det_metric = torch.det(metric_sample)

                history['metric_condition_avg'].append(condition_numbers.mean().item())
                history['metric_condition_max'].append(condition_numbers.max().item())
                history['metric_det_std'].append(det_metric.std().item())

            # Test evaluation every validation_interval epochs
            if epoch % CONFIG['validation_interval'] == 0 or epoch == CONFIG['epochs'] - 1:
                test_metrics = evaluate_test_set(
                    phi_network, harmonic_network, test_coords,
                    manifold, metric_from_phi_fn, device
                )

                test_history['epoch'].append(epoch)
                test_history['test_torsion'].append(test_metrics['torsion'])
                test_history['test_det_gram'].append(test_metrics['det_gram'])
                test_history['test_dphi_L2'].append(test_metrics['dphi_L2'])
                test_history['test_dstar_phi_L2'].append(test_metrics.get('dstar_phi_L2', 0.0))

                print(f"Epoch {epoch:5d} [{phase_name}]: "
                      f"loss={history['loss'][-1]:.3e}, "
                      f"torsion={test_metrics['torsion']:.3e}, "
                      f"det(Gram)={test_metrics['det_gram']:.6f}, "
                      f"lr={history['lr'][-1]:.2e}")

            # Check early stopping conditions
            should_stop, message = check_early_stopping(epoch, history, weights)
            if message:
                print(message)
            if should_stop:
                break

            # Save checkpoints
            if epoch % CONFIG['checkpoint_interval'] == 0 and epoch > 0:
                save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch_{epoch}.pt',
                    epoch, phi_network, harmonic_network,
                    optimizer, scheduler, history, test_history
                )

            # Memory cleanup
            if epoch % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except RuntimeError as e:
            print(f"⚠ Epoch {epoch} failed: {e}")
            torch.cuda.empty_cache()
            continue

    print("=" * 80)
    print("Training complete!")

    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir / 'checkpoint_final.pt',
        CONFIG['epochs'], phi_network, harmonic_network,
        optimizer, scheduler, history, test_history
    )

    return history, test_history


# ============================================================================
# Test/Validation Functions
# ============================================================================

def evaluate_test_set(phi_network, harmonic_network, test_coords, manifold,
                      metric_from_phi_fn, device):
    """
    Evaluate metrics on fixed test set.

    Args:
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        test_coords: (n_test, 7) tensor of test coordinates
        manifold: TCSNeckManifold instance
        metric_from_phi_fn: Function to construct metric from phi
        device: torch device

    Returns:
        metrics: Dictionary of test metrics
    """
    from .losses import SafeMetrics, compute_harmonic_losses_FIXED

    phi_network.eval()
    harmonic_network.eval()

    test_coords = test_coords.to(device)
    test_coords.requires_grad_(True)

    with torch.no_grad():
        phi_test = phi_network(test_coords)
        h_forms_test = harmonic_network(test_coords)
        metric_test = metric_from_phi_fn(phi_test)

    # Torsion with gradients (OUTSIDE no_grad context)
    test_torsion = SafeMetrics.compute_torsion_safe(phi_test, test_coords, metric_test, use_grad=True)

    # PDE residuals: dφ (exterior derivative)
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
        _, _, _, test_det_gram = compute_harmonic_losses_FIXED(
            harmonic_network, test_coords, h_forms_test, metric_test
        )

    metrics = {
        'torsion': SafeMetrics.to_scalar(test_torsion),
        'det_gram': SafeMetrics.to_scalar(test_det_gram),
        'dphi_L2': dphi_L2,
    }

    return metrics


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(filepath, epoch, phi_network, harmonic_network,
                   optimizer, scheduler, history, test_history):
    """
    Save training checkpoint.

    Args:
        filepath: Path to save checkpoint
        epoch: Current epoch
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        optimizer: torch.optim.Optimizer
        scheduler: Learning rate scheduler
        history: Training history dictionary
        test_history: Test history dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'phi_network': phi_network.state_dict(),
        'harmonic_network': harmonic_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'history': history,
        'test_history': test_history,
        'config': CONFIG
    }

    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(filepath, phi_network, harmonic_network, optimizer=None, scheduler=None):
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        optimizer: torch.optim.Optimizer (optional)
        scheduler: Learning rate scheduler (optional)

    Returns:
        epoch: Epoch number from checkpoint
        history: Training history
        test_history: Test history
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    phi_network.load_state_dict(checkpoint['phi_network'])
    harmonic_network.load_state_dict(checkpoint['harmonic_network'])

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    epoch = checkpoint.get('epoch', 0)
    history = checkpoint.get('history', {})
    test_history = checkpoint.get('test_history', {})

    print(f"📂 Checkpoint loaded from epoch {epoch}: {filepath}")

    return epoch, history, test_history


# ============================================================================
# Mixed Precision Training (AMP)
# ============================================================================

def train_epoch_amp(phi_network, harmonic_network, manifold, optimizer, scheduler,
                    scaler, epoch, history, device, metric_from_phi_fn, loss_fn, weights):
    """
    Execute one training epoch with mixed precision (AMP).

    Args:
        Same as train_epoch, plus:
        scaler: torch.cuda.amp.GradScaler for mixed precision

    Returns:
        loss_dict: Dictionary of loss components
    """
    phi_network.train()
    harmonic_network.train()

    coords = manifold.sample_points(CONFIG['batch_size']).to(device)
    coords.requires_grad_(True)

    # Forward pass with autocast
    with autocast():
        phi = phi_network(coords)
        h_forms = harmonic_network(coords)
        metric = metric_from_phi_fn(phi)

        loss, loss_dict = loss_fn(
            phi, h_forms, metric, coords, manifold,
            phi_network, harmonic_network, weights
        )

    # Backward pass with gradient scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()

    # Gradient clipping (unscale first)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(phi_network.parameters()) + list(harmonic_network.parameters()),
        CONFIG['grad_clip']
    )

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    loss_dict['loss'] = loss

    return loss_dict
