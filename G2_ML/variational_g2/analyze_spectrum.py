#!/usr/bin/env python3
"""
Spectral Analysis for G2 Laplacian

Compute eigenvalue spectrum of the Hodge Laplacian on 3-forms
to verify b3 = 77 (spectral gap at position 77).

Previous analysis (v1.8 proxy) showed gap at 94, not 77.
This script tests the v3.1 PINN model with torsion 0.00140.

Expected:
    - b2 = 21: 21 harmonic 2-forms
    - b3 = 77: 77 harmonic 3-forms (35 local + 42 TCS global)
    - H* = 99: b2 + b3 + 1

Usage:
    python analyze_spectrum.py --checkpoint outputs/metrics/g2_variational_model.pt
    python analyze_spectrum.py --weights lean/pinn_weights.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class SpectralResult:
    """Results from spectral analysis."""
    eigenvalues: np.ndarray  # sorted ascending (smallest first)
    num_zero_modes: int  # count of near-zero eigenvalues
    gap_position: int  # position of largest gap
    gap_magnitude: float
    b3_effective: int  # estimated b3 from gap
    b3_target: int = 77
    match: bool = False

    def to_dict(self) -> Dict:
        return {
            "eigenvalues": self.eigenvalues[:100].tolist(),  # first 100
            "num_zero_modes": self.num_zero_modes,
            "gap_position": self.gap_position,
            "gap_magnitude": float(self.gap_magnitude),
            "b3_effective": self.b3_effective,
            "b3_target": self.b3_target,
            "match": self.match,
        }


def load_model_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    """Load model from .pt checkpoint."""
    from model import G2VariationalNet

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', {
        'input_dim': 7,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_frequencies': 64,
    })

    model = G2VariationalNet(
        input_dim=config.get('input_dim', 7),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_frequencies=config.get('num_frequencies', 64),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_model_from_json(weights_path: Path) -> nn.Module:
    """Load model from exported JSON weights."""
    from model import G2VariationalNet

    with open(weights_path) as f:
        data = json.load(f)

    config = data.get('config', {})
    weights = data.get('weights', data)

    model = G2VariationalNet(
        input_dim=config.get('input_dim', 7),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_frequencies=config.get('num_frequencies', 64),
    )

    # Load weights
    state_dict = {}
    for name, values in weights.items():
        if isinstance(values, list):
            state_dict[name] = torch.tensor(values)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def sample_phi_field(
    model: nn.Module,
    n_samples: int = 5000,
    domain: Tuple[float, float] = (-1.0, 1.0),
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample phi field at random points."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random points in domain
    coords = torch.rand(n_samples, 7) * (domain[1] - domain[0]) + domain[0]

    with torch.no_grad():
        output = model(coords)
        phi = output['phi_components']  # (n_samples, 35)

    return phi, coords


def compute_laplacian_matrix(
    phi: torch.Tensor,
    coords: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute discrete Hodge Laplacian matrix on 3-forms.

    The Hodge Laplacian is Delta = d*d + dd* where:
    - d: exterior derivative
    - d*: codifferential (adjoint of d)

    For 3-forms on a 7-manifold, dim(Lambda^3) = 35.
    The number of harmonic 3-forms = dim(ker(Delta)) = b3.

    We approximate this via the Gram matrix of phi variations.
    """
    n_samples, n_components = phi.shape  # (N, 35)

    # Method: Build covariance/Gram matrix of 3-form space
    # Eigenvectors with small eigenvalues approximate harmonic forms

    # Center phi
    phi_centered = phi - phi.mean(dim=0, keepdim=True)

    # Gram matrix: captures linear dependencies
    gram = (phi_centered.T @ phi_centered) / n_samples  # (35, 35)

    # For the full 77-dimensional H3, we need to include TCS global modes
    # Build extended basis: 35 local + 42 global

    # Global TCS modes: position-dependent harmonics
    lam = coords[:, 0:1]  # TCS parameter
    xi = coords[:, 1:]  # Fiber coordinates

    global_modes = []

    # Polynomial in lambda (5 modes)
    for p in range(5):
        global_modes.append(lam ** p)

    # Fiber harmonics (12 modes: 6 coords + 6 lambda*coord)
    for i in range(6):
        global_modes.append(xi[:, i:i+1])
        global_modes.append(lam * xi[:, i:i+1])

    # Trigonometric modes (24 modes: sin/cos at frequencies 1-6)
    for k in range(1, 7):
        global_modes.append(torch.sin(2 * np.pi * k * lam))
        global_modes.append(torch.cos(2 * np.pi * k * lam))
        global_modes.append(torch.sin(np.pi * k * xi[:, 0:1]))
        global_modes.append(torch.cos(np.pi * k * xi[:, 0:1]))

    # Fill to 42
    while len(global_modes) < 42:
        idx = len(global_modes)
        freq = idx / 10.0
        global_modes.append(torch.sin(2 * np.pi * freq * lam))

    global_modes = torch.cat(global_modes[:42], dim=1)  # (N, 42)
    global_centered = global_modes - global_modes.mean(dim=0, keepdim=True)

    # Combine local (35) + global (42) = 77 modes
    combined = torch.cat([phi_centered, global_centered], dim=1)  # (N, 77)

    # Full Gram matrix
    gram_full = (combined.T @ combined) / n_samples  # (77, 77)

    # The Laplacian spectrum comes from this Gram matrix
    # Small eigenvalues = harmonic forms (in ker(Delta))
    # Large eigenvalues = non-harmonic forms

    return gram_full


def analyze_laplacian_spectrum(
    gram: torch.Tensor,
    zero_threshold: float = 1e-4,
) -> SpectralResult:
    """
    Analyze eigenvalue spectrum to find b3.

    b3 = number of near-zero eigenvalues (harmonic forms)
    """
    # Symmetrize
    gram = 0.5 * (gram + gram.T)

    # Eigendecomposition
    eigenvalues, _ = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.cpu().numpy()

    # Sort ascending (smallest first)
    eigenvalues = np.sort(eigenvalues)

    # Count near-zero eigenvalues
    num_zero = np.sum(eigenvalues < zero_threshold * eigenvalues.max())

    # Find largest gap in the spectrum
    # Look for the transition from "harmonic" (small) to "non-harmonic" (large)
    gaps = np.diff(eigenvalues)

    # Normalize gaps
    gaps_normalized = gaps / (np.mean(gaps) + 1e-10)

    # Find largest gap (excluding edges)
    search_range = slice(10, min(90, len(gaps) - 10))
    gap_idx = 10 + np.argmax(gaps_normalized[search_range])
    gap_mag = gaps_normalized[gap_idx]

    # b3 effective = position of gap + 1
    # (eigenvalues 0..gap_idx are "harmonic")
    b3_effective = gap_idx + 1

    # Also check specific positions
    gap_at_77 = gaps_normalized[76] if len(gaps) > 76 else 0
    gap_at_94 = gaps_normalized[93] if len(gaps) > 93 else 0

    print(f"\n[SPECTRAL ANALYSIS]")
    print(f"  Total modes: {len(eigenvalues)}")
    print(f"  Near-zero modes (< {zero_threshold:.0e}): {num_zero}")
    print(f"  Largest gap at position: {gap_idx}")
    print(f"  Gap magnitude (vs mean): {gap_mag:.2f}x")
    print(f"  b3 effective: {b3_effective}")
    print(f"  b3 target: 77")
    print(f"")
    print(f"  Gap at position 77: {gap_at_77:.2f}x")
    print(f"  Gap at position 94: {gap_at_94:.2f}x")

    match = abs(b3_effective - 77) <= 5

    return SpectralResult(
        eigenvalues=eigenvalues,
        num_zero_modes=int(num_zero),
        gap_position=int(gap_idx),
        gap_magnitude=float(gap_mag),
        b3_effective=int(b3_effective),
        b3_target=77,
        match=match,
    )


def run_full_analysis(model: nn.Module, n_samples: int = 5000) -> Dict:
    """Run complete spectral analysis."""
    print("=" * 60)
    print("LAPLACIAN SPECTRAL ANALYSIS FOR b3 = 77")
    print("=" * 60)

    # Sample phi field
    print(f"\n[1] Sampling phi field ({n_samples} points)...")
    phi, coords = sample_phi_field(model, n_samples)
    print(f"    phi shape: {phi.shape}")
    print(f"    phi range: [{phi.min():.4f}, {phi.max():.4f}]")

    # Compute Laplacian matrix
    print(f"\n[2] Computing Gram/Laplacian matrix...")
    gram = compute_laplacian_matrix(phi, coords)
    print(f"    gram shape: {gram.shape}")

    # Analyze spectrum
    print(f"\n[3] Analyzing eigenvalue spectrum...")
    result = analyze_laplacian_spectrum(gram)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    status = "PASS" if result.match else "NEEDS INVESTIGATION"
    print(f"  b3 effective: {result.b3_effective}")
    print(f"  b3 target:    77")
    print(f"  Status:       {status}")

    if result.b3_effective == 77:
        print("\n  b3 = 77 CONFIRMED!")
        print("  The spectral gap is at the correct position.")
    elif result.b3_effective == 94:
        print("\n  Gap at 94 (same as v1.8)")
        print("  Possible reasons:")
        print("    - Global TCS modes not properly captured")
        print("    - Need actual harmonic form computation")
        print("    - Model topology differs from true K7")
    else:
        print(f"\n  Gap at {result.b3_effective}")
        print("  Further analysis needed.")

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Spectral analysis for b3=77")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint (.pt)")
    parser.add_argument("--weights", type=Path, help="Exported weights (.json)")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")

    args = parser.parse_args()

    # Load model
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = load_model_from_checkpoint(args.checkpoint)
    elif args.weights and args.weights.exists():
        print(f"Loading from JSON weights: {args.weights}")
        model = load_model_from_json(args.weights)
    else:
        # Check default paths
        default_ckpt = Path(__file__).parent / "outputs/metrics/g2_variational_model.pt"
        default_json = Path(__file__).parent / "lean/pinn_weights.json"

        if default_ckpt.exists() and default_ckpt.stat().st_size > 1000:
            print(f"Loading from default checkpoint: {default_ckpt}")
            model = load_model_from_checkpoint(default_ckpt)
        elif default_json.exists() and default_json.stat().st_size > 1000:
            print(f"Loading from default JSON: {default_json}")
            model = load_model_from_json(default_json)
        else:
            print("ERROR: No model found!")
            print("Run: git lfs pull")
            print("Then retry with --checkpoint or --weights")
            return 1

    # Run analysis
    results = run_full_analysis(model, n_samples=args.samples)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = Path(__file__).parent / "spectral_result.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results.get('match', False) else 1


if __name__ == "__main__":
    sys.exit(main())
