#!/usr/bin/env python3
"""
TCS/Joyce v2.0 Analysis Script

Run this script locally where you have the real checkpoint files (not LFS pointers).

Usage:
    python run_analysis.py --checkpoint path/to/checkpoint.pt
    python run_analysis.py --samples path/to/samples.npz
    python run_analysis.py --version 1_8

This will:
1. Load real phi values from your checkpoints
2. Build 77 H3 modes using TCS/Joyce construction
3. Compute the Yukawa Gram matrix
4. Analyze the spectrum for 43/77 split and tau = 3472/891
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phi_from_checkpoint(checkpoint_path: Path) -> tuple:
    """Load phi and coords from a checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")

        # Try to find phi
        phi = None
        for key in ['phi', 'phi_values', 'phi_samples', 'outputs']:
            if key in data:
                phi = data[key]
                print(f"  Found phi in '{key}': shape {phi.shape if hasattr(phi, 'shape') else type(phi)}")
                break

        # Try to find coords
        coords = None
        for key in ['coords', 'coordinates', 'x', 'inputs']:
            if key in data:
                coords = data[key]
                print(f"  Found coords in '{key}': shape {coords.shape if hasattr(coords, 'shape') else type(coords)}")
                break

        if phi is None:
            raise ValueError("Could not find phi in checkpoint")
        if coords is None:
            print("  No coords found, generating random...")
            coords = torch.rand(phi.shape[0], 7)

        phi = torch.as_tensor(phi, dtype=torch.float32)
        coords = torch.as_tensor(coords, dtype=torch.float32)

        return phi, coords
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(data)}")


def load_phi_from_samples(samples_path: Path) -> tuple:
    """Load phi and coords from a samples.npz file."""
    print(f"Loading samples: {samples_path}")
    data = np.load(samples_path)

    print(f"  Keys: {list(data.keys())}")

    phi = torch.from_numpy(data['phi']).float() if 'phi' in data else None
    coords = torch.from_numpy(data['coords']).float() if 'coords' in data else None

    if phi is None:
        raise ValueError("No 'phi' in samples file")
    if coords is None:
        print("  No coords, generating random...")
        coords = torch.rand(phi.shape[0], 7)

    print(f"  phi shape: {phi.shape}")
    print(f"  coords shape: {coords.shape}")

    return phi, coords


def build_tcs_global_modes(coords: torch.Tensor) -> torch.Tensor:
    """Build 42 TCS/Joyce global modes."""
    N = coords.shape[0]
    lam = coords[:, 0]
    xi = coords[:, 1:]

    # Profile functions
    def smooth_profile(x, center, width):
        return torch.exp(-((x - center) / width)**2)

    global_modes = []

    # Left modes (14): peaked at lambda ~ 0.2
    for j in range(14):
        profile = smooth_profile(lam, 0.2, 0.15)
        xi_mod = 1 + 0.2 * xi[:, j % 6]
        global_modes.append(profile * xi_mod)

    # Right modes (14): peaked at lambda ~ 0.8
    for j in range(14):
        profile = smooth_profile(lam, 0.8, 0.15)
        xi_mod = 1 + 0.2 * xi[:, (5 - j) % 6]
        global_modes.append(profile * xi_mod)

    # Neck modes (14): peaked at lambda ~ 0.5
    for j in range(14):
        profile = smooth_profile(lam, 0.5, 0.1)
        xi_mod = 1 + 0.15 * (xi**2).sum(dim=1)
        global_modes.append(profile * xi_mod)

    return torch.stack(global_modes, dim=1)


def analyze_spectrum(phi: torch.Tensor, coords: torch.Tensor) -> dict:
    """Run the full TCS/Joyce spectral analysis."""
    N = phi.shape[0]

    print("\n" + "=" * 70)
    print("TCS/JOYCE SPECTRAL ANALYSIS")
    print("=" * 70)

    # Build H3 modes
    print("\n[1] Building 77 H3 modes...")
    local_modes = phi  # 35 components
    global_modes = build_tcs_global_modes(coords)  # 42 components

    h3_modes = torch.cat([local_modes, global_modes], dim=1)

    # Normalize
    norms = torch.norm(h3_modes, dim=0, keepdim=True) + 1e-10
    h3_modes = h3_modes / norms

    print(f"    Local modes: {local_modes.shape}")
    print(f"    Global modes: {global_modes.shape}")
    print(f"    H3 total: {h3_modes.shape}")

    # Compute Gram matrix
    print("\n[2] Computing Yukawa Gram matrix...")
    M = (h3_modes.T @ h3_modes) / N
    M = 0.5 * (M + M.T)
    print(f"    Gram shape: {M.shape}")

    # Eigendecomposition
    print("\n[3] Eigendecomposition...")
    eigenvalues, _ = torch.linalg.eigh(M)
    eigenvalues = eigenvalues.flip(0)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    print(f"    Top 5: {eigenvalues[:5].tolist()}")
    print(f"    Around 43: {eigenvalues[40:46].tolist()}")

    # Gap analysis
    print("\n[4] Gap analysis...")
    gaps = torch.abs(eigenvalues[:-1] - eigenvalues[1:])
    mean_gap = gaps[20:55].mean()

    print(f"    Mean gap (20-55): {mean_gap:.6f}")
    print()
    print("    Pos | Eigenvalue | Gap      | Ratio")
    print("    " + "-" * 42)

    for pos in range(40, 47):
        ev = eigenvalues[pos].item()
        g = gaps[pos].item()
        r = g / (mean_gap + 1e-10)
        mark = " *** 43/77 ***" if pos == 42 else ""
        print(f"    {pos:>3} | {ev:>10.6f} | {g:.6f} | {r:>5.2f}x{mark}")

    # Find largest gap
    largest_idx = gaps[35:50].argmax().item() + 35
    print(f"\n    Largest gap [35-50]: position {largest_idx}")

    # Tau analysis
    print("\n[5] Tau analysis...")
    tau_target = 3472.0 / 891.0

    sum_43 = eigenvalues[:43].sum().item()
    sum_34 = eigenvalues[43:].sum().item()
    tau_actual = sum_43 / (sum_34 + 1e-10)
    error = abs(tau_actual - tau_target) / tau_target * 100

    print(f"    Sum(1:43) = {sum_43:.6f}")
    print(f"    Sum(44:77) = {sum_34:.6f}")
    print(f"    Tau = {tau_actual:.6f}")
    print(f"    Target = {tau_target:.6f}")
    print(f"    Error = {error:.2f}%")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gap at 42-43: {gaps[42]:.6f} ({gaps[42]/mean_gap:.2f}x mean)")
    print(f"  Largest gap: position {largest_idx}")
    print(f"  Tau = {tau_actual:.4f} (target: {tau_target:.4f})")
    print(f"  Error = {error:.2f}%")

    if largest_idx == 42:
        print("\n  *** GAP AT 43 DETECTED! ***")
    if error < 10:
        print("  *** TAU WITHIN 10%! ***")

    print("=" * 70)

    return {
        'eigenvalues': eigenvalues,
        'gaps': gaps,
        'largest_gap_position': largest_idx,
        'tau': tau_actual,
        'tau_error': error,
        'gap_43_ratio': (gaps[42] / mean_gap).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="TCS/Joyce v2.0 Spectral Analysis")
    parser.add_argument('--checkpoint', '-c', type=Path, help='Path to checkpoint.pt')
    parser.add_argument('--samples', '-s', type=Path, help='Path to samples.npz')
    parser.add_argument('--version', '-v', type=str, help='Version directory (e.g., 1_8)')

    args = parser.parse_args()

    # Find data source
    phi, coords = None, None

    if args.checkpoint:
        phi, coords = load_phi_from_checkpoint(args.checkpoint)
    elif args.samples:
        phi, coords = load_phi_from_samples(args.samples)
    elif args.version:
        # Try common paths
        base = Path(__file__).parent.parent / args.version
        for name in ['samples.npz', 'checkpoint.pt', f'models_v{args.version}.pt']:
            p = base / name
            if p.exists():
                if name.endswith('.npz'):
                    phi, coords = load_phi_from_samples(p)
                else:
                    phi, coords = load_phi_from_checkpoint(p)
                break
    else:
        print("No data source specified. Use --checkpoint, --samples, or --version")
        print("Generating synthetic data for demo...")

        N = 500
        coords = torch.rand(N, 7)
        phi = torch.randn(N, 35) * 0.155
        phi = phi * np.sqrt(7.0) / (torch.norm(phi, dim=1, keepdim=True) + 1e-8)

    # Run analysis
    results = analyze_spectrum(phi, coords)

    return results


if __name__ == '__main__':
    main()
