#!/usr/bin/env python
"""Training script for v1.9 Hodge Pure.

Phase 1: Learn H2 harmonic 2-forms (21 modes)
Phase 2: Learn H3 harmonic 3-forms (77 modes)
Phase 3: Compute Yukawa and verify 43/77 structure
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from hodge_forms import (
    HodgeConfig, H2Network, H3Network, HodgeLoss,
    compute_yukawa_integral, yukawa_gram_matrix, analyze_yukawa_spectrum
)


def load_metric_data(metric_source: Path) -> dict:
    """Load metric and coordinates from v1.8."""
    data = np.load(metric_source)
    return {
        'coords': torch.from_numpy(data['coords']).float(),
        'metric': torch.from_numpy(data['metric']).float(),
        'phi': torch.from_numpy(data['phi']).float(),
    }


def train_h2(
    data: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """Train H2 network for 21 harmonic 2-forms."""
    print("\n" + "="*70)
    print("PHASE 1: Training H2 (21 harmonic 2-forms)")
    print("="*70)

    hodge_config = HodgeConfig(n_h2=21)
    model = H2Network(hodge_config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr_h2'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor']
    )

    coords = data['coords'].to(device)
    metric = data['metric'].to(device)
    n_samples = coords.shape[0]
    batch_size = config['training']['batch_size']

    loss_fn = HodgeLoss(None, hodge_config)
    weights = config['hodge']['h2_modes']['loss_weights']

    best_loss = float('inf')
    best_state = None

    for epoch in range(config['training']['n_epochs_h2']):
        # Random batch
        idx = torch.randperm(n_samples)[:batch_size]
        x_batch = coords[idx]
        g_batch = metric[idx]

        # Forward
        omega = model(x_batch)

        # Loss
        loss, loss_dict = loss_fn(omega, x_batch, g_batch, weights)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

        if (epoch + 1) % config['logging']['log_every'] == 0:
            print(f"Epoch {epoch+1:5d}: loss={loss.item():.6f}, ortho={loss_dict['orthonormality']:.6f}")

    model.load_state_dict(best_state)
    print(f"\nBest H2 loss: {best_loss:.6f}")

    return model, best_loss


def train_h3(
    data: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """Train H3 network for 77 harmonic 3-forms."""
    print("\n" + "="*70)
    print("PHASE 2: Training H3 (77 harmonic 3-forms)")
    print("="*70)

    hodge_config = HodgeConfig(n_h3=77, n_h3_local=35, n_h3_global=42)
    model = H3Network(hodge_config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr_h3'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=config['training']['scheduler_patience'],
        factor=config['training']['scheduler_factor']
    )

    coords = data['coords'].to(device)
    metric = data['metric'].to(device)
    phi = data['phi'].to(device)  # Use v1.8 phi as initialization hint
    n_samples = coords.shape[0]
    batch_size = config['training']['batch_size']

    loss_fn = HodgeLoss(None, hodge_config)
    weights = config['hodge']['h3_modes']['loss_weights']

    best_loss = float('inf')
    best_state = None

    for epoch in range(config['training']['n_epochs_h3']):
        # Random batch
        idx = torch.randperm(n_samples)[:batch_size]
        x_batch = coords[idx]
        g_batch = metric[idx]

        # Forward
        Phi = model(x_batch)

        # Loss
        loss, loss_dict = loss_fn(Phi, x_batch, g_batch, weights)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

        if (epoch + 1) % config['logging']['log_every'] == 0:
            print(f"Epoch {epoch+1:5d}: loss={loss.item():.6f}, ortho={loss_dict['orthonormality']:.6f}")

    model.load_state_dict(best_state)
    print(f"\nBest H3 loss: {best_loss:.6f}")

    return model, best_loss


def compute_and_analyze_yukawa(
    h2_model: H2Network,
    h3_model: H3Network,
    data: dict,
    config: dict,
    device: torch.device,
) -> dict:
    """Compute Yukawa tensor and analyze spectrum."""
    print("\n" + "="*70)
    print("PHASE 3: Computing Yukawa and analyzing spectrum")
    print("="*70)

    coords = data['coords'].to(device)
    metric = data['metric'].to(device)
    n_points = config['yukawa']['n_integration_points']

    # Sample points for integration
    idx = torch.randperm(coords.shape[0])[:n_points]
    x = coords[idx]
    g = metric[idx]

    # Evaluate forms
    with torch.no_grad():
        omega = h2_model(x)  # (n_points, 21, 21)
        Phi = h3_model(x)    # (n_points, 77, 35)

    # Compute Yukawa
    print("Computing Yukawa integral...")
    Y = compute_yukawa_integral(omega, Phi, g)

    # Gram matrix
    print("Computing Gram matrix...")
    M = yukawa_gram_matrix(Y)

    # Analyze spectrum
    print("Analyzing spectrum...")
    analysis = analyze_yukawa_spectrum(M)

    # Print results
    print("\n" + "-"*50)
    print("YUKAWA SPECTRUM ANALYSIS")
    print("-"*50)
    print(f"Non-zero eigenvalues: {analysis['nonzero_count']}")
    print(f"Suggested n_visible: {analysis['suggested_n_visible']}")
    print(f"Largest gap at index: {analysis['largest_gap_idx']}")
    print(f"Gap at 43: {analysis['gap_43']:.6f} ({analysis['gap_43_ratio']:.2f}x mean)")

    if analysis['suggested_n_visible'] in [42, 43, 44]:
        print("\n*** 43/77 STRUCTURE CONFIRMED! ***")
    else:
        print(f"\nNote: spectral gap suggests {analysis['suggested_n_visible']}/77 split")

    print("\nTau candidates:")
    for n, ratio, error in analysis['tau_candidates'][:5]:
        marker = " <-- CLOSE!" if error < 20 else ""
        print(f"  Split {n}: ratio={ratio:.4f}, error={error:.1f}%{marker}")

    return {
        'Y': Y.cpu().numpy(),
        'M': M.cpu().numpy(),
        'analysis': {k: v.tolist() if hasattr(v, 'tolist') else v
                     for k, v in analysis.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train v1.9 Hodge Pure")
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--phase', type=str, choices=['h2', 'h3', 'yukawa', 'all'], default='all')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load metric from v1.8
    metric_source = Path(__file__).parent / config['hodge']['metric_source']
    print(f"Loading metric from {metric_source}")
    data = load_metric_data(metric_source)
    print(f"Loaded {data['coords'].shape[0]} samples")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / config['logging']['checkpoint_dir'] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Phase 1: H2
    if args.phase in ['h2', 'all']:
        h2_model, h2_loss = train_h2(data, config, device)
        torch.save(h2_model.state_dict(), output_dir / 'h2_model.pt')
        results['h2_loss'] = h2_loss

    # Phase 2: H3
    if args.phase in ['h3', 'all']:
        h3_model, h3_loss = train_h3(data, config, device)
        torch.save(h3_model.state_dict(), output_dir / 'h3_model.pt')
        results['h3_loss'] = h3_loss

    # Phase 3: Yukawa
    if args.phase in ['yukawa', 'all']:
        if args.phase == 'yukawa':
            # Load saved models
            h2_model = H2Network(HodgeConfig(n_h2=21)).to(device)
            h3_model = H3Network(HodgeConfig(n_h3=77)).to(device)
            h2_model.load_state_dict(torch.load(output_dir / 'h2_model.pt'))
            h3_model.load_state_dict(torch.load(output_dir / 'h3_model.pt'))

        yukawa_results = compute_and_analyze_yukawa(h2_model, h3_model, data, config, device)
        np.savez(output_dir / 'yukawa.npz', **{k: v for k, v in yukawa_results.items() if k != 'analysis'})
        with open(output_dir / 'yukawa_analysis.json', 'w') as f:
            json.dump(yukawa_results['analysis'], f, indent=2)
        results['yukawa'] = yukawa_results['analysis']

    # Save final results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
