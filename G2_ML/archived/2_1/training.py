"""Training protocol for GIFT v2.2 variational G2 extraction.

Implements phased training:
1. Initialization: establish valid G2 structure
2. Constraint satisfaction: achieve det(g) = 65/32
3. Torsion targeting: achieve kappa_T = 1/61
4. Cohomology refinement: verify (b2, b3) = (21, 77)

Uses Adam optimizer with cosine annealing learning rate schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from .config import GIFTConfig, default_config, TrainingState
from .model import GIFTVariationalModel, G2VariationalNet
from .loss import VariationalLoss, PhasedLossManager, format_loss_dict, log_constraints
from .g2_geometry import MetricFromPhi


# =============================================================================
# Coordinate sampling
# =============================================================================

def sample_coordinates(batch_size: int, device: str = 'cpu',
                       bounds: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Sample random coordinates in R^7.

    Args:
        batch_size: number of samples
        device: torch device
        bounds: (min, max) coordinate bounds

    Returns:
        x: (batch_size, 7)
    """
    return torch.rand(batch_size, 7, device=device) * (bounds[1] - bounds[0]) + bounds[0]


def sample_grid(resolution: int, device: str = 'cpu',
                bounds: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Sample regular grid in R^7.

    Args:
        resolution: points per dimension
        device: torch device
        bounds: (min, max) coordinate bounds

    Returns:
        x: (resolution^7, 7) - WARNING: can be very large!
    """
    # For 7D, even resolution=4 gives 4^7 = 16384 points
    coords_1d = torch.linspace(bounds[0], bounds[1], resolution, device=device)

    # Create meshgrid
    grids = torch.meshgrid([coords_1d] * 7, indexing='ij')
    x = torch.stack([g.flatten() for g in grids], dim=-1)

    return x


def sample_stratified(batch_size: int, n_strata: int = 4,
                      device: str = 'cpu') -> torch.Tensor:
    """Stratified sampling for better coverage.

    Args:
        batch_size: total samples
        n_strata: strata per dimension
        device: torch device

    Returns:
        x: (batch_size, 7)
    """
    samples_per_stratum = batch_size // (n_strata ** 7) + 1
    samples = []

    for stratum in range(n_strata ** 7):
        # Decode stratum index
        idx = stratum
        offsets = []
        for _ in range(7):
            offsets.append(idx % n_strata)
            idx //= n_strata

        # Sample within stratum
        low = torch.tensor([o / n_strata for o in offsets], device=device)
        high = torch.tensor([(o + 1) / n_strata for o in offsets], device=device)

        stratum_samples = torch.rand(samples_per_stratum, 7, device=device)
        stratum_samples = stratum_samples * (high - low) + low

        samples.append(stratum_samples)

    x = torch.cat(samples, dim=0)[:batch_size]
    return 2 * x - 1  # Map to [-1, 1]


# =============================================================================
# Training loop
# =============================================================================

@dataclass
class TrainingHistory:
    """Training history storage."""
    loss_history: List[float] = field(default_factory=list)
    det_history: List[float] = field(default_factory=list)
    torsion_history: List[float] = field(default_factory=list)
    positivity_history: List[float] = field(default_factory=list)
    phase_boundaries: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'loss': self.loss_history,
            'det': self.det_history,
            'torsion': self.torsion_history,
            'positivity': self.positivity_history,
            'phase_boundaries': self.phase_boundaries
        }


class Trainer:
    """Main trainer for GIFT variational G2 extraction."""

    def __init__(self, model: G2VariationalNet,
                 config: GIFTConfig = None,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config or default_config
        self.device = device

        # Loss function
        self.loss_fn = VariationalLoss(config)
        self.phase_manager = PhasedLossManager(config)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.phase_manager.total_epochs,
            eta_min=1e-6
        )

        # History
        self.history = TrainingHistory()

        # Best model tracking
        self.best_loss = float('inf')
        self.best_state = None

    def train_epoch(self, batch_size: int = None) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            batch_size: samples per epoch (default from config)

        Returns:
            loss_dict: losses for this epoch
        """
        batch_size = batch_size or self.config.batch_size
        self.model.train()

        # Sample coordinates
        x = sample_coordinates(batch_size, self.device)

        # Forward pass
        phi = self.model(x, project_positive=True)

        # Get loss weights for current phase
        weights = self.phase_manager.get_weights()

        # Compute loss
        def phi_fn(x_new):
            return self.model(x_new, project_positive=True)

        loss, loss_dict = self.loss_fn(phi, x, phi_fn, weights)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return loss_dict

    def train(self, n_epochs: int = None,
              log_interval: int = 100,
              save_interval: int = 1000,
              save_path: Optional[Path] = None) -> TrainingHistory:
        """Full training loop.

        Args:
            n_epochs: total epochs (default from config)
            log_interval: epochs between logging
            save_interval: epochs between checkpoints
            save_path: where to save checkpoints

        Returns:
            history: training history
        """
        n_epochs = n_epochs or self.phase_manager.total_epochs

        print("="*60)
        print("GIFT v2.2 Variational G2 Training")
        print("="*60)
        print(f"Total epochs: {n_epochs}")
        print(f"Phases: {len(self.config.phases)}")
        print(f"Device: {self.device}")
        print("="*60)

        start_time = time.time()

        for epoch in range(n_epochs):
            # Train one epoch
            loss_dict = self.train_epoch()

            # Update history
            self.history.loss_history.append(loss_dict['total'])
            if 'det_value' in loss_dict:
                self.history.det_history.append(loss_dict['det_value'])
            if 'torsion_value' in loss_dict:
                self.history.torsion_history.append(loss_dict['torsion_value'])
            if 'positivity' in loss_dict:
                self.history.positivity_history.append(loss_dict['positivity'])

            # Track best model
            if loss_dict['total'] < self.best_loss:
                self.best_loss = loss_dict['total']
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}

            # Phase transition
            if self.phase_manager.step():
                self.history.phase_boundaries.append(epoch)
                print(f"\n{'='*60}")
                print(f"Phase transition at epoch {epoch}")
                print(f"New phase: {self.phase_manager.get_phase_name()}")
                print(f"{'='*60}\n")

            # Logging
            if epoch % log_interval == 0:
                phase = self.phase_manager.get_phase_name()
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                print(f"[{epoch:5d}/{n_epochs}] Phase: {phase:20s} | "
                      f"LR: {lr:.2e} | Time: {elapsed:.0f}s")
                print(f"  {format_loss_dict(loss_dict)}")

            # Checkpointing
            if save_path and epoch % save_interval == 0 and epoch > 0:
                self.save_checkpoint(save_path / f"checkpoint_{epoch}.pt")

        # Final summary
        print("\n" + "="*60)
        print("Training Complete")
        print("="*60)
        print(f"Final loss: {self.history.loss_history[-1]:.6f}")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Total time: {time.time() - start_time:.0f}s")

        # Constraint satisfaction
        self.model.eval()
        with torch.no_grad():
            x = sample_coordinates(1000, self.device)
            phi = self.model(x, project_positive=True)
            weights = self.config.phases[-1]["weights"]

            def phi_fn(x_new):
                return self.model(x_new, project_positive=True)

            _, final_losses = self.loss_fn(phi, x, phi_fn, weights)

        print("\nFinal Constraint Status:")
        print(log_constraints(final_losses, self.config))

        return self.history

    def save_checkpoint(self, path: Path):
        """Save training checkpoint.

        Args:
            path: save path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'phase': self.phase_manager.current_phase,
            'epoch_in_phase': self.phase_manager.epoch_in_phase,
            'best_loss': self.best_loss,
            'best_state': self.best_state,
            'history': self.history.to_dict(),
            'config': {
                'det_g_target': self.config.det_g_target,
                'kappa_T': self.config.kappa_T,
                'b2_K7': self.config.b2_K7,
                'b3_K7': self.config.b3_K7,
            }
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load training checkpoint.

        Args:
            path: checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.phase_manager.current_phase = checkpoint['phase']
        self.phase_manager.epoch_in_phase = checkpoint['epoch_in_phase']
        self.best_loss = checkpoint['best_loss']
        self.best_state = checkpoint['best_state']

        history = checkpoint['history']
        self.history.loss_history = history['loss']
        self.history.det_history = history['det']
        self.history.torsion_history = history['torsion']
        self.history.positivity_history = history['positivity']
        self.history.phase_boundaries = history['phase_boundaries']

        print(f"Checkpoint loaded from {path}")

    def restore_best(self):
        """Restore best model state."""
        if self.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
            print(f"Restored best model (loss: {self.best_loss:.6f})")


# =============================================================================
# Quick training function
# =============================================================================

def train_gift_g2(config: GIFTConfig = None,
                  device: str = 'cpu',
                  save_path: Optional[Path] = None,
                  verbose: bool = True) -> Tuple[G2VariationalNet, TrainingHistory]:
    """Quick training function for GIFT G2 extraction.

    Args:
        config: GIFT configuration
        device: torch device
        save_path: where to save results
        verbose: print progress

    Returns:
        model: trained model
        history: training history
    """
    config = config or default_config

    # Create model
    model = G2VariationalNet(config)

    # Create trainer
    trainer = Trainer(model, config, device)

    # Train
    history = trainer.train(
        log_interval=100 if verbose else 10000,
        save_path=save_path
    )

    # Restore best
    trainer.restore_best()

    # Save final model
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path / "model_final.pt")
        torch.save({
            'model_state': trainer.best_state,
            'loss': trainer.best_loss,
            'history': history.to_dict(),
        }, save_path / "best_model.pt")

        with open(save_path / "config.json", 'w') as f:
            json.dump({
                'det_g_target': config.det_g_target,
                'kappa_T': config.kappa_T,
                'b2_K7': config.b2_K7,
                'b3_K7': config.b3_K7,
                'tau': config.tau,
                'sin2_theta_W': config.sin2_theta_W,
            }, f, indent=2)

    return model, history
