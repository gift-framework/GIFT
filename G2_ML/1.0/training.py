"""
Training utilities for TCS G₂ metric reconstruction.

Implements curriculum learning, optimization, and training loop.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm
import time


class CurriculumScheduler:
    """
    Five-phase curriculum scheduler for progressive training.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.curriculum = config['training']['curriculum']
        self.phases = [
            'phase1_neck_stability',
            'phase2_acyl_matching',
            'phase3_cohomology_refinement',
            'phase4_harmonic_extraction',
            'phase5_calibration_finetune'
        ]

    def get_current_phase(self, epoch: int) -> Tuple[str, Dict]:
        """
        Determine current training phase based on epoch.

        Returns:
            phase_name: Name of current phase
            phase_config: Configuration for this phase
        """
        for phase_name in self.phases:
            phase_config = self.curriculum[phase_name]
            epoch_range = phase_config['range']
            if epoch_range[0] <= epoch < epoch_range[1]:
                return phase_name, phase_config

        return self.phases[-1], self.curriculum[self.phases[-1]]

    def get_grid_resolution(self, epoch: int) -> int:
        """
        Get grid resolution for current epoch.
        """
        _, phase_config = self.get_current_phase(epoch)
        return phase_config.get('grid_n', 10)

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss component weights for current epoch.
        """
        _, phase_config = self.get_current_phase(epoch)
        return phase_config.get('loss_weights', {})

    def get_region_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get region emphasis weights for current epoch.
        """
        _, phase_config = self.get_current_phase(epoch)
        return phase_config.get('region_weights', {'m1': 0.33, 'neck': 0.34, 'm2': 0.33})


def create_optimizer(models: Dict[str, nn.Module], config: Dict) -> AdamW:
    """
    Create AdamW optimizer for all model components.

    Args:
        models: Dictionary of model components
        config: Training configuration

    Returns:
        optimizer: Configured AdamW optimizer
    """
    parameters = []
    for name, model in models.items():
        parameters.extend(list(model.parameters()))

    optimizer = AdamW(
        parameters,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    return optimizer


def create_scheduler(optimizer, config: Dict, start_epoch: int = 0):
    """
    Create learning rate scheduler with warmup and cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        start_epoch: Starting epoch for resume

    Returns:
        scheduler: Learning rate scheduler
    """
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['total_epochs']

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-7
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )

    for _ in range(start_epoch):
        scheduler.step()

    return scheduler


class GradientAccumulator:
    """
    Gradient accumulation helper for large effective batch sizes.
    """
    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self) -> bool:
        """
        Check if gradients should be applied.
        """
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss by accumulation steps.
        """
        return loss / self.accumulation_steps


def train_epoch(
    models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    topology: Any,
    curriculum: CurriculumScheduler,
    config: Dict,
    epoch: int,
    metrics_tracker: Any,
    device: torch.device
) -> Dict[str, float]:
    """
    Execute one training epoch.

    Args:
        models: Dictionary containing phi_network, harmonic_h2, harmonic_h3
        optimizer: Optimizer
        loss_fn: Composite loss function
        topology: K7Topology instance
        curriculum: Curriculum scheduler
        config: Training configuration
        epoch: Current epoch number
        metrics_tracker: Metrics tracking object
        device: Torch device

    Returns:
        epoch_metrics: Dictionary of average metrics for this epoch
    """
    for model in models.values():
        model.train()

    batch_size = config['training']['batch_size']
    grad_accum = GradientAccumulator(config['training']['grad_accumulation'])

    grid_n = curriculum.get_grid_resolution(epoch)
    loss_weights = curriculum.get_loss_weights(epoch)

    coords = topology.sample_coordinates(batch_size, grid_n=grid_n)
    coords = coords.to(device)
    coords.requires_grad_(True)

    phi_network = models['phi_network']
    harmonic_h2_network = models['harmonic_h2']
    harmonic_h3_network = models['harmonic_h3']

    phi = phi_network.get_phi_tensor(coords)

    from losses import torsion_closure_loss, torsion_coclosure_loss

    dphi_simple = torch.zeros(batch_size, 7, 7, 7, 7, device=device)
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i != j and i != k and j != k:
                    grad = torch.autograd.grad(
                        phi[:, i, j, k].sum(),
                        coords,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    for l in range(7):
                        if l not in [i, j, k]:
                            dphi_simple[:, i, j, k, l] = grad[:, l]

    dstar_phi_simple = torch.zeros(batch_size, 7, 7, device=device)

    metric = reconstruct_metric_from_phi(phi)

    star_phi = torch.zeros(batch_size, 7, 7, 7, 7, device=device)

    harmonic_h2 = harmonic_h2_network(coords)
    harmonic_h3 = harmonic_h3_network(coords)

    region_weights = topology.get_region_weights(coords)

    total_loss, components = loss_fn(
        phi=phi,
        dphi=dphi_simple,
        dstar_phi=dstar_phi_simple,
        star_phi=star_phi,
        metric=metric,
        harmonic_h2=harmonic_h2,
        harmonic_h3=harmonic_h3,
        region_weights=region_weights,
        loss_weights=loss_weights,
        epoch=epoch
    )

    scaled_loss = grad_accum.scale_loss(total_loss)
    scaled_loss.backward()

    if grad_accum.should_update():
        torch.nn.utils.clip_grad_norm_(
            [p for model in models.values() for p in model.parameters()],
            config['training']['grad_clip']
        )
        optimizer.step()
        optimizer.zero_grad()

    epoch_metrics = {
        'loss': total_loss.item(),
        **components
    }

    metrics_tracker.update(epoch, **epoch_metrics)

    return epoch_metrics


def reconstruct_metric_from_phi(phi: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct metric from 3-form (simplified version for training).
    """
    batch_size = phi.shape[0]
    metric = torch.zeros(batch_size, 7, 7, device=phi.device)

    for i in range(7):
        for j in range(7):
            for p in range(7):
                for q in range(7):
                    if p != i and q != i and p != j and q != j and p != q:
                        metric[:, i, j] += phi[:, i, p, q] * phi[:, j, p, q]

    metric = metric / 6.0
    metric = 0.5 * (metric + metric.transpose(-2, -1))

    eye = torch.eye(7, device=phi.device).unsqueeze(0)
    metric = metric + 1e-4 * eye

    return metric


def training_loop(
    models: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    topology: Any,
    curriculum: CurriculumScheduler,
    checkpoint_manager: Any,
    metrics_tracker: Any,
    config: Dict,
    start_epoch: int = 0,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Main training loop with checkpointing and validation.

    Args:
        models: Dictionary of neural networks
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Composite loss function
        topology: K7Topology instance
        curriculum: Curriculum scheduler
        checkpoint_manager: Checkpoint management object
        metrics_tracker: Metrics tracking object
        config: Training configuration
        start_epoch: Starting epoch (for resume)
        device: Torch device

    Returns:
        final_results: Dictionary containing final metrics and paths
    """
    total_epochs = config['training']['total_epochs']
    checkpoint_interval = config['checkpointing']['interval']
    validation_interval = config['validation']['interval']

    print(f"Starting training from epoch {start_epoch} to {total_epochs}")
    print(f"Device: {device}")

    training_start_time = time.time()

    for epoch in tqdm(range(start_epoch, total_epochs), desc="Training"):
        epoch_start = time.time()

        phase_name, phase_config = curriculum.get_current_phase(epoch)

        epoch_metrics = train_epoch(
            models=models,
            optimizer=optimizer,
            loss_fn=loss_fn,
            topology=topology,
            curriculum=curriculum,
            config=config,
            epoch=epoch,
            metrics_tracker=metrics_tracker,
            device=device
        )

        scheduler.step()

        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{total_epochs} [{phase_name}]")
            print(f"  Loss: {epoch_metrics['loss']:.6f}")
            print(f"  Torsion closure: {epoch_metrics['torsion_closure']:.6e}")
            print(f"  Torsion coclosure: {epoch_metrics['torsion_coclosure']:.6e}")
            print(f"  Rank H²: {epoch_metrics['rank_h2']}/21")
            print(f"  Rank H³: {epoch_metrics['rank_h3']}/77")
            print(f"  LR: {current_lr:.2e}")
            print(f"  Time: {time.time() - epoch_start:.2f}s")

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_manager.save(
                epoch=epoch,
                models=models,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics
            )
            print(f"Checkpoint saved at epoch {epoch}")

    training_time = time.time() - training_start_time

    final_checkpoint = checkpoint_manager.save(
        epoch=total_epochs,
        models=models,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=epoch_metrics
    )

    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Final checkpoint: {final_checkpoint}")

    final_results = {
        'total_epochs': total_epochs,
        'training_time_hours': training_time / 3600,
        'final_metrics': epoch_metrics,
        'checkpoint_path': str(final_checkpoint)
    }

    return final_results
