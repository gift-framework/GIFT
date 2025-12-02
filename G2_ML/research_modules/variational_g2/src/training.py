"""
Phased Training Protocol for G2 Variational Problem

This module implements the multi-phase training strategy for the GIFT v2.2
variational problem. Training proceeds in phases with different loss weight
configurations to progressively satisfy all constraints.

Phases:
1. Initialization: Establish valid G2 structure
2. Constraint Satisfaction: Achieve det(g) = 65/32
3. Torsion Targeting: Reach kappa_T = 1/61
4. Cohomology Refinement: Fine-tune (b2, b3) = (21, 77)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time
import yaml

from .model import G2VariationalNet, create_model
from .loss import VariationalLoss, LossWeights, LossOutput, create_loss
from .harmonic import sample_grid_points, extract_betti_numbers
from .constraints import expand_to_antisymmetric


@dataclass
class PhaseConfig:
    """Configuration for a single training phase."""
    name: str
    epochs: int
    focus: str
    weights: Dict[str, float]
    lr: float


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    phases: List[PhaseConfig]
    total_epochs: int
    batch_size: int
    grid_resolution: int
    domain: tuple = (-1.0, 1.0)
    optimizer: str = 'adam'
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    checkpoint_freq: int = 500
    eval_freq: int = 100
    log_freq: int = 50
    seed: int = 42
    device: str = 'auto'

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        training = config.get('training', {})
        phases = []

        for phase_dict in training.get('phases', []):
            phases.append(PhaseConfig(
                name=phase_dict['name'],
                epochs=int(phase_dict['epochs']),
                focus=phase_dict['focus'],
                weights={k: float(v) for k, v in phase_dict['weights'].items()},
                lr=float(phase_dict['lr']),
            ))

        return cls(
            phases=phases,
            total_epochs=int(training.get('total_epochs', 10000)),
            batch_size=int(training.get('batch_size', 2048)),
            grid_resolution=int(training.get('grid', {}).get('resolution', 16)),
            domain=tuple(float(x) for x in training.get('grid', {}).get('domain', [-1.0, 1.0])),
            optimizer=str(training.get('optimizer', 'adam')),
            weight_decay=float(training.get('weight_decay', 1e-6)),
            grad_clip=float(training.get('grad_clip', 1.0)),
            checkpoint_freq=int(training.get('checkpoint_freq', 500)),
            eval_freq=int(config.get('validation', {}).get('eval_freq', 100)),
            log_freq=int(config.get('logging', {}).get('log_freq', 50)),
            seed=int(config.get('seed', 42)),
        )


class TrainingMetrics:
    """Track and log training metrics."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir
        self.history: Dict[str, List[float]] = {
            'epoch': [],
            'phase': [],
            'loss_total': [],
            'loss_torsion': [],
            'loss_det': [],
            'loss_positivity': [],
            'loss_cohomology': [],
            'det_g': [],
            'torsion_norm': [],
            'min_eigenvalue': [],
            'lr': [],
        }
        self.logger = logging.getLogger('G2Training')

    def log(
        self,
        epoch: int,
        phase: str,
        loss_output: LossOutput,
        lr: float
    ):
        """Log metrics for current epoch."""
        self.history['epoch'].append(epoch)
        self.history['phase'].append(phase)
        self.history['loss_total'].append(loss_output.total.item())
        self.history['lr'].append(lr)

        # Loss components
        for key in ['torsion', 'det', 'positivity', 'cohomology']:
            val = loss_output.components.get(key, torch.tensor(0.0))
            self.history[f'loss_{key}'].append(val.item())

        # Metrics
        for key in ['det_g', 'torsion_norm', 'min_eigenvalue']:
            val = loss_output.metrics.get(key, torch.tensor(0.0))
            self.history[key].append(val.item() if torch.is_tensor(val) else val)

    def save(self, path: Path):
        """Save metrics history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_latest(self) -> Dict[str, float]:
        """Get most recent metrics."""
        if not self.history['epoch']:
            return {}
        return {k: v[-1] for k, v in self.history.items()}


class Trainer:
    """
    Trainer for G2 variational problem with phased training.

    Implements the four-phase training protocol:
    1. Initialization: Focus on G2 positivity
    2. Constraint Satisfaction: Focus on det(g) = 65/32
    3. Torsion Targeting: Focus on kappa_T = 1/61
    4. Cohomology Refinement: Focus on (b2, b3) = (21, 77)
    """

    def __init__(
        self,
        model: G2VariationalNet,
        loss_fn: VariationalLoss,
        config: TrainingConfig,
        output_dir: Optional[Path] = None,
    ):
        """
        Args:
            model: G2VariationalNet model
            loss_fn: VariationalLoss instance
            config: Training configuration
            output_dir: Directory for outputs
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path('outputs')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Setup logging
        self.logger = logging.getLogger('G2Training')
        self.metrics = TrainingMetrics(self.output_dir)

        # Training state
        self.current_epoch = 0
        self.current_phase = 0
        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(self, phase: PhaseConfig):
        """Setup optimizer and scheduler for a training phase."""
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=phase.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=phase.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=phase.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )

        # Cosine annealing within phase
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=phase.epochs,
            eta_min=phase.lr * 0.01
        )

    def sample_batch(self) -> torch.Tensor:
        """Sample a batch of training points."""
        points = sample_grid_points(
            self.config.batch_size,
            domain=self.config.domain,
            device=self.device,
        )
        return points.requires_grad_(True)

    def train_step(self, x: torch.Tensor) -> LossOutput:
        """Execute single training step."""
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(x, return_full=True)
        phi = output['phi_full']

        # Compute loss
        loss_output = self.loss_fn(phi, x)

        # Backward pass
        loss_output.total.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        # Update
        self.optimizer.step()

        return loss_output

    def train_phase(self, phase: PhaseConfig) -> Dict[str, float]:
        """
        Train for one phase.

        Args:
            phase: Phase configuration

        Returns:
            Final metrics for phase
        """
        self.logger.info(f"Starting phase: {phase.name}")
        self.logger.info(f"  Focus: {phase.focus}")
        self.logger.info(f"  Epochs: {phase.epochs}")
        self.logger.info(f"  Weights: {phase.weights}")

        # Setup optimizer
        self.setup_optimizer(phase)

        # Update loss weights
        self.loss_fn.update_weights(LossWeights.from_dict(phase.weights))

        # Training loop
        self.model.train()
        phase_start = self.current_epoch

        for local_epoch in range(phase.epochs):
            self.current_epoch = phase_start + local_epoch

            # Sample batch and train
            x = self.sample_batch()
            loss_output = self.train_step(x)

            # Step scheduler
            self.scheduler.step()

            # Logging
            if local_epoch % self.config.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.metrics.log(self.current_epoch, phase.name, loss_output, lr)

                self.logger.info(
                    f"Epoch {self.current_epoch:5d} | "
                    f"Loss: {loss_output.total.item():.6f} | "
                    f"det(g): {loss_output.metrics.get('det_g', 0):.4f} | "
                    f"kappa_T: {loss_output.metrics.get('torsion_norm', 0):.6f}"
                )

            # Evaluation
            if local_epoch % self.config.eval_freq == 0:
                self.evaluate()

            # Checkpointing
            if local_epoch % self.config.checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pt")

        return self.metrics.get_latest()

    def train(self) -> Dict[str, Any]:
        """
        Run complete multi-phase training.

        Returns:
            Final training results
        """
        # Set seed
        torch.manual_seed(self.config.seed)

        self.logger.info("="*60)
        self.logger.info("G2 Variational Training - GIFT v2.2")
        self.logger.info("="*60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Total epochs: {self.config.total_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Number of phases: {len(self.config.phases)}")

        start_time = time.time()
        phase_results = []

        # Train each phase
        for i, phase in enumerate(self.config.phases):
            self.current_phase = i
            results = self.train_phase(phase)
            phase_results.append({
                'phase': phase.name,
                'results': results
            })

        # Final evaluation
        final_metrics = self.evaluate(detailed=True)

        # Save final model and metrics
        self.save_checkpoint("final_model.pt")
        self.metrics.save(self.output_dir / "training_history.json")

        elapsed = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"Training complete in {elapsed:.1f}s")
        self.logger.info(f"Final det(g): {final_metrics.get('det_g', 'N/A')}")
        self.logger.info(f"Final kappa_T: {final_metrics.get('torsion_norm', 'N/A')}")
        self.logger.info("="*60)

        return {
            'phase_results': phase_results,
            'final_metrics': final_metrics,
            'elapsed_time': elapsed,
        }

    def evaluate(self, detailed: bool = False) -> Dict[str, float]:
        """
        Evaluate current model.

        Args:
            detailed: If True, also compute Betti numbers (expensive)

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Sample evaluation points
            x = sample_grid_points(
                self.config.batch_size * 2,
                domain=self.config.domain,
                device=self.device,
            )

            # Forward pass
            output = self.model(x, return_full=True, return_metric=True)
            phi = output['phi_full']
            metric = output['metric']

            # Compute metrics
            det_g = torch.det(metric).mean().item()
            eigenvalues = torch.linalg.eigvalsh(metric)
            min_eig = eigenvalues.min().item()
            g_positive = min_eig > 0

            metrics = {
                'det_g': det_g,
                'det_g_target': 65.0 / 32.0,
                'det_g_error': abs(det_g - 65.0 / 32.0),
                'min_eigenvalue': min_eig,
                'g_positive': g_positive,
            }

            # Betti numbers (expensive)
            if detailed:
                b2, b3 = extract_betti_numbers(
                    self.model,
                    resolution=self.config.grid_resolution,
                    num_samples=1000,
                    device=self.device,
                )
                metrics['b2_effective'] = b2
                metrics['b3_effective'] = b3
                metrics['h_star'] = b2 + b3 + 1

        self.model.train()
        return metrics

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config.__dict__,
        }

        path = self.output_dir / 'checkpoints' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_phase = checkpoint['phase']

        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_from_config(config_path: str, output_dir: str = 'outputs') -> Dict:
    """
    Convenience function to train from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration
        output_dir: Output directory

    Returns:
        Training results
    """
    # Load config
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    training_config = TrainingConfig.from_yaml(config_path)

    # Create model and loss
    model = create_model(full_config)
    loss_fn = create_loss(full_config)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / 'training.log'),
        ]
    )

    # Create trainer and run
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=training_config,
        output_dir=output_dir,
    )

    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train G2 Variational Network")
    parser.add_argument(
        "--config",
        type=str,
        default="config/gift_v22.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory"
    )
    args = parser.parse_args()

    results = train_from_config(args.config, args.output)
    print("Training complete!")
    print(f"Final metrics: {results['final_metrics']}")
