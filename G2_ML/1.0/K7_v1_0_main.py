"""
K₇ Metric Reconstruction v1.0 - Main Training Script

Complete TCS with calibration and Yukawa computation.
Can be run as Python script or converted to Jupyter notebook.
"""

import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from losses import CompositeLoss
from training import (
    CurriculumScheduler,
    create_optimizer,
    create_scheduler,
    training_loop
)
from validation import GeometricValidator
from yukawa import compute_and_analyze_yukawa


print("="*70)
print("K₇ METRIC RECONSTRUCTION v1.0")
print("Complete Torsion Cohomology Solver for G₂ Manifolds")
print("="*70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


WORK_DIR = Path("./v1_0_output")
WORK_DIR.mkdir(exist_ok=True)
print(f"Output directory: {WORK_DIR}")

with open("config_v1_0.json", 'r') as f:
    CONFIG = json.load(f)

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

print("\nConfiguration loaded:")
print(f"  Total epochs: {CONFIG['training']['total_epochs']}")
print(f"  Batch size: {CONFIG['training']['batch_size']}")
print(f"  Learning rate: {CONFIG['training']['lr']}")


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, n_frequencies, scale=1.0):
        super().__init__()
        B = torch.randn(input_dim, n_frequencies) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ModularPhiNetwork(nn.Module):
    def __init__(self, hidden_dims, n_fourier):
        super().__init__()
        self.fourier = FourierFeatures(7, n_fourier, scale=1.0)

        layers = []
        in_dim = self.fourier.B.shape[0] * self.fourier.B.shape[1] * 2
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.SiLU()])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 35))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        features = self.fourier(x)
        return self.network(features)

    def get_phi_tensor(self, x):
        phi_flat = self.forward(x)
        batch_size = x.shape[0]
        phi = torch.zeros(batch_size, 7, 7, 7, device=x.device)

        idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                for k in range(j+1, 7):
                    val = phi_flat[:, idx]
                    phi[:, i, j, k] = val
                    phi[:, i, k, j] = -val
                    phi[:, j, i, k] = -val
                    phi[:, j, k, i] = val
                    phi[:, k, i, j] = val
                    phi[:, k, j, i] = -val
                    idx += 1

        return phi


class HarmonicFormsNetwork(nn.Module):
    def __init__(self, p, n_forms, hidden_dim, n_fourier):
        super().__init__()
        self.p = p
        self.n_forms = n_forms
        self.n_components = 21 if p == 2 else 35

        self.networks = nn.ModuleList()
        for i in range(n_forms):
            hidden_var = hidden_dim + (i % 5) * 8
            fourier = FourierFeatures(7, n_fourier, scale=1.0)
            fourier_dim = 7 * n_fourier * 2
            net = nn.Sequential(
                nn.Linear(fourier_dim, hidden_var),
                nn.SiLU(),
                nn.Linear(hidden_var, hidden_var),
                nn.SiLU(),
                nn.Linear(hidden_var, self.n_components),
            )
            self.networks.append(nn.Sequential(fourier, net))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.n_forms, self.n_components, device=x.device)

        for i, network in enumerate(self.networks):
            outputs[:, i, :] = network(x)

        return outputs


class K7Topology:
    def __init__(self, gift_params):
        self.params = gift_params
        self.epsilon = gift_params['epsilon0']

    def sample_coordinates(self, n_samples, grid_n=10):
        coords_1d = torch.linspace(0, 2*np.pi, grid_n)
        grid_7d = torch.stack(torch.meshgrid(*[coords_1d]*7, indexing='ij'), dim=-1)
        grid_flat = grid_7d.reshape(-1, 7)

        n_grid = min(n_samples // 2, grid_flat.shape[0])
        idx_grid = torch.randperm(grid_flat.shape[0])[:n_grid]
        samples_grid = grid_flat[idx_grid]

        n_random = n_samples - n_grid
        samples_random = torch.rand(n_random, 7) * 2 * np.pi

        return torch.cat([samples_grid, samples_random], dim=0)

    def get_region_weights(self, x):
        t = x[:, 0]
        w_m1 = torch.sigmoid((np.pi - t) / 0.3)
        w_m2 = torch.sigmoid((t - np.pi) / 0.3)
        w_neck = 1.0 - w_m1 - w_m2
        return {'m1': w_m1, 'neck': w_neck, 'm2': w_m2}

    def define_associative_cycles(self, n_cycles=6):
        cycles = []
        for region, t_vals in [('M1', [np.pi/4, np.pi/3]),
                                ('neck', [np.pi, 5*np.pi/4]),
                                ('M2', [3*np.pi/2, 7*np.pi/4])]:
            for t in t_vals:
                cycles.append({
                    'region': region,
                    't_fixed': t,
                    'type': 'T3',
                    'indices': [1, 2, 3],
                })
        return cycles[:n_cycles]

    def define_coassociative_cycles(self, n_cycles=6):
        cycles = []
        for region, t_vals in [('M1', [np.pi/4]),
                                ('neck', [np.pi, 5*np.pi/4]),
                                ('M2', [3*np.pi/2, 7*np.pi/4])]:
            for t in t_vals:
                cycles.append({
                    'region': region,
                    't_fixed': t,
                    'type': 'T4',
                    'indices': [0, 4, 5, 6],
                })
        return cycles[:n_cycles]

    def sample_on_cycle(self, cycle, n_samples=512):
        samples = torch.rand(n_samples, 7) * 2 * np.pi
        samples[:, 0] = cycle['t_fixed']
        return samples


print("\nInitializing models...")

topology = K7Topology(CONFIG['gift_params'])
assoc_cycles = topology.define_associative_cycles(6)
coassoc_cycles = topology.define_coassociative_cycles(6)

models = {
    'phi_network': ModularPhiNetwork(
        hidden_dims=CONFIG['architecture']['phi_hidden_dims'],
        n_fourier=CONFIG['architecture']['phi_n_fourier']
    ).to(DEVICE),

    'harmonic_h2': HarmonicFormsNetwork(
        p=2,
        n_forms=CONFIG['gift_params']['b2'],
        hidden_dim=CONFIG['architecture']['harmonic_h2_hidden'],
        n_fourier=CONFIG['architecture']['harmonic_h2_fourier']
    ).to(DEVICE),

    'harmonic_h3': HarmonicFormsNetwork(
        p=3,
        n_forms=CONFIG['gift_params']['b3'],
        hidden_dim=CONFIG['architecture']['harmonic_h3_hidden'],
        n_fourier=CONFIG['architecture']['harmonic_h3_fourier']
    ).to(DEVICE),
}

print(f"  PhiNetwork parameters: {sum(p.numel() for p in models['phi_network'].parameters()):,}")
print(f"  Harmonic H² parameters: {sum(p.numel() for p in models['harmonic_h2'].parameters()):,}")
print(f"  Harmonic H³ parameters: {sum(p.numel() for p in models['harmonic_h3'].parameters()):,}")

optimizer = create_optimizer(models, CONFIG)
scheduler = create_scheduler(optimizer, CONFIG)

loss_fn = CompositeLoss(topology, assoc_cycles, coassoc_cycles).to(DEVICE)

curriculum = CurriculumScheduler(CONFIG)

from training import MetricsTracker, CheckpointManager

metrics_tracker = MetricsTracker()
checkpoint_manager = CheckpointManager(
    save_dir=str(WORK_DIR / "checkpoints"),
    keep_best=CONFIG['checkpointing']['keep_best']
)

start_epoch = 0
if CONFIG['checkpointing']['auto_resume']:
    checkpoint = checkpoint_manager.load_latest()
    if checkpoint is not None:
        print(f"\nResuming from epoch {checkpoint['epoch']}")
        for name in models:
            models[name].load_state_dict(checkpoint['models'][name])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1


print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

final_results = training_loop(
    models=models,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    topology=topology,
    curriculum=curriculum,
    checkpoint_manager=checkpoint_manager,
    metrics_tracker=metrics_tracker,
    config=CONFIG,
    start_epoch=start_epoch,
    device=DEVICE
)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nTotal time: {final_results['training_time_hours']:.2f} hours")
print(f"Final torsion closure: {final_results['final_metrics']['torsion_closure']:.6e}")
print(f"Final torsion coclosure: {final_results['final_metrics']['torsion_coclosure']:.6e}")

metrics_tracker.save(str(WORK_DIR / "training_history.npz"))
print(f"\nMetrics saved to {WORK_DIR / 'training_history.npz'}")

print("\n" + "="*70)
print("GEOMETRIC VALIDATION")
print("="*70)

validator = GeometricValidator(CONFIG)
validation_report = validator.final_validation(models, DEVICE)

validator.save_validation_report(
    validation_report,
    str(WORK_DIR / "geometric_validation.json")
)

print("\n" + "="*70)
print("YUKAWA COMPUTATION")
print("="*70)

yukawa_results, yukawa_tensor, yukawa_uncertainty = compute_and_analyze_yukawa(
    models=models,
    topology=topology,
    config=CONFIG,
    device=DEVICE
)

np.save(str(WORK_DIR / "yukawa_tensor.npy"), yukawa_tensor.cpu().numpy())
np.save(str(WORK_DIR / "yukawa_uncertainty.npy"), yukawa_uncertainty.cpu().numpy())

with open(str(WORK_DIR / "yukawa_analysis.json"), 'w') as f:
    json.dump(yukawa_results, f, indent=2)

print(f"\nYukawa tensor saved to {WORK_DIR / 'yukawa_tensor.npy'}")
print(f"Analysis saved to {WORK_DIR / 'yukawa_analysis.json'}")

print("\n" + "="*70)
print("COMPLETE RESULTS SUMMARY")
print("="*70)

summary = {
    'version': CONFIG['version'],
    'training_results': final_results,
    'validation': validation_report,
    'yukawa': yukawa_results
}

with open(str(WORK_DIR / "complete_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nComplete summary saved to {WORK_DIR / 'complete_summary.json'}")

print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {WORK_DIR.absolute()}")
