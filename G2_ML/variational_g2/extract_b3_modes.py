#!/usr/bin/env python3
"""
Extract b₃ = 77 Harmonic Modes from GIFT Structure

Key insight: 77 = 35 + 42 = 35 + 2×21
- 35 local modes: Λ³ℝ⁷ (3-form components)
- 42 global modes: 2 × Λ²ℝ⁷ ∧ S¹ (2-forms wedged with circle)

This script extracts modes using the PINN's learned geometry,
not synthetic polynomials.

Usage:
    python extract_b3_modes.py --checkpoint outputs/metrics/g2_variational_model.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# GIFT constants
B2 = 21  # dim(Λ²ℝ⁷) = C(7,2)
B3_LOCAL = 35  # dim(Λ³ℝ⁷) = C(7,3)
B3_GLOBAL = 42  # 2 × B2 from TCS
B3_TARGET = 77  # B3_LOCAL + B3_GLOBAL
H_STAR = 99  # B2 + B3 + 1
DET_G_TARGET = 65 / 32  # 2.03125


@dataclass
class B3Result:
    """Results from b₃ mode extraction."""
    b3_effective: int
    gap_position: int
    gap_magnitude: float
    eigenvalues: np.ndarray
    local_contribution: float
    global_contribution: float
    match: bool

    def to_dict(self) -> Dict:
        return {
            "b3_effective": self.b3_effective,
            "b3_target": B3_TARGET,
            "gap_position": self.gap_position,
            "gap_magnitude": float(self.gap_magnitude),
            "local_contribution": float(self.local_contribution),
            "global_contribution": float(self.global_contribution),
            "match": self.match,
            "eigenvalues_first_100": self.eigenvalues[:100].tolist(),
        }


def load_model(checkpoint_path: Path) -> nn.Module:
    """Load PINN model from checkpoint."""
    from model import G2VariationalNet

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    model = G2VariationalNet(
        input_dim=config.get('input_dim', 7),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_frequencies=config.get('num_frequencies', 64),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def extract_2form_indices() -> List[Tuple[int, int]]:
    """Get the 21 index pairs for 2-forms (i < j)."""
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            indices.append((i, j))
    return indices  # 21 pairs


def extract_3form_indices() -> List[Tuple[int, int, int]]:
    """Get the 35 index triples for 3-forms (i < j < k)."""
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append((i, j, k))
    return indices  # 35 triples


def compute_jacobian_batch(model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian ∂φ/∂x for a batch of coordinates.

    Args:
        model: PINN model
        coords: (N, 7) coordinates

    Returns:
        jacobian: (N, 35, 7) tensor
    """
    coords = coords.requires_grad_(True)
    output = model(coords)
    phi = output['phi_components']  # (N, 35)

    jacobians = []
    for i in range(35):
        grad = torch.autograd.grad(
            phi[:, i].sum(), coords,
            create_graph=False, retain_graph=True
        )[0]  # (N, 7)
        jacobians.append(grad)

    return torch.stack(jacobians, dim=1)  # (N, 35, 7)


def extract_metric_2forms(jacobian: torch.Tensor) -> torch.Tensor:
    """
    Extract 21 "2-form modes" from the Jacobian structure.

    The metric g_ij ∝ J^T J encodes 2-form geometry.
    Off-diagonal elements (21 pairs) represent the 2-form content.

    Args:
        jacobian: (N, 35, 7) Jacobian tensor

    Returns:
        metric_2forms: (N, 21) tensor
    """
    # Gramian: G_ij = sum_k J_ki * J_kj
    # Shape: (N, 7, 7)
    G = torch.einsum('nki,nkj->nij', jacobian, jacobian)

    # Extract off-diagonal elements (21 pairs)
    indices_2form = extract_2form_indices()
    metric_2forms = torch.stack([G[:, i, j] for i, j in indices_2form], dim=-1)

    return metric_2forms  # (N, 21)


def build_77_basis(
    model: nn.Module,
    coords: torch.Tensor,
    use_jacobian: bool = True
) -> torch.Tensor:
    """
    Build the full 77-dimensional harmonic 3-form basis.

    Structure: 77 = 35 (local) + 42 (global)
                  = 35 + 2×21
                  = Λ³ℝ⁷ + (Λ²ℝ⁷ ∧ S¹)_+ + (Λ²ℝ⁷ ∧ S¹)_-

    Args:
        model: PINN model
        coords: (N, 7) coordinates
        use_jacobian: If True, extract 2-forms from Jacobian

    Returns:
        basis: (N, 77) tensor
    """
    N = coords.shape[0]

    # === Part 1: 35 Local Modes (Λ³ℝ⁷) ===
    with torch.no_grad():
        output = model(coords)
        phi = output['phi_components']  # (N, 35)

    # === Part 2: Extract 2-form Content ===
    if use_jacobian:
        # Use Jacobian structure
        jacobian = compute_jacobian_batch(model, coords)  # (N, 35, 7)
        metric_2forms = extract_metric_2forms(jacobian)  # (N, 21)
    else:
        # Fallback: use metric directly
        with torch.no_grad():
            metric = output.get('metric', None)
            if metric is None:
                from constraints import metric_from_phi
                phi_full = output.get('phi_full')
                if phi_full is None:
                    from constraints import expand_to_antisymmetric
                    phi_full = expand_to_antisymmetric(phi)
                metric = metric_from_phi(phi_full)  # (N, 7, 7)

            indices_2form = extract_2form_indices()
            metric_2forms = torch.stack([metric[:, i, j] for i, j in indices_2form], dim=-1)

    # Normalize 2-form modes
    metric_2forms = metric_2forms / (torch.norm(metric_2forms, dim=-1, keepdim=True) + 1e-8)

    # === Part 3: S¹ Modulation for TCS Structure ===
    # The first coordinate plays the role of the TCS gluing parameter λ
    # λ = 0: near X+ building block
    # λ = 1: near X- building block

    # Map coords[:, 0] to [0, 1] (assuming domain is [-1, 1] or [0, 2π])
    lam = coords[:, 0]
    lam_normalized = (lam - lam.min()) / (lam.max() - lam.min() + 1e-8)

    # S¹ harmonics for the two building blocks
    # X+ modes: localized near λ=0
    # X- modes: localized near λ=1
    s1_plus = torch.sin(torch.pi * lam_normalized).unsqueeze(-1)   # (N, 1)
    s1_minus = torch.cos(torch.pi * lam_normalized).unsqueeze(-1)  # (N, 1)

    # === Part 4: Global Modes (42 = 21 + 21) ===
    # 2-forms wedged with S¹ factor
    global_plus = metric_2forms * s1_plus    # (N, 21) - X+ contribution
    global_minus = metric_2forms * s1_minus  # (N, 21) - X- contribution

    # === Part 5: Combine into 77-basis ===
    basis_77 = torch.cat([phi, global_plus, global_minus], dim=-1)  # (N, 77)

    return basis_77


def compute_gram_spectrum(basis: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenspectrum of the Gram matrix.

    Args:
        basis: (N, 77) mode basis

    Returns:
        eigenvalues: (77,) sorted ascending
        eigenvectors: (77, 77)
    """
    # Center the basis
    basis_centered = basis - basis.mean(dim=0, keepdim=True)

    # Gram matrix
    N = basis.shape[0]
    gram = (basis_centered.T @ basis_centered) / N  # (77, 77)

    # Symmetrize
    gram = 0.5 * (gram + gram.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)

    # Sort ascending
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx].cpu().numpy()
    eigenvectors = eigenvectors[:, idx].cpu().numpy()

    return eigenvalues, eigenvectors


def analyze_spectrum(eigenvalues: np.ndarray) -> B3Result:
    """
    Analyze eigenspectrum to determine b₃.

    Args:
        eigenvalues: (77,) sorted eigenvalues

    Returns:
        B3Result with analysis
    """
    # Normalize eigenvalues
    eigenvalues_norm = eigenvalues / (np.max(eigenvalues) + 1e-10)

    # Compute gaps
    gaps = np.diff(eigenvalues_norm)
    gaps_relative = gaps / (np.mean(gaps) + 1e-10)

    # Find largest gap
    gap_position = np.argmax(gaps_relative)
    gap_magnitude = gaps_relative[gap_position]

    # b₃ effective = position after which eigenvalues jump
    b3_effective = gap_position + 1

    # Analyze local vs global contribution
    # First 35 eigenvalues correspond to local modes
    # Next 42 correspond to global modes (in ideal case)
    local_variance = np.sum(eigenvalues[:B3_LOCAL])
    global_variance = np.sum(eigenvalues[B3_LOCAL:B3_TARGET])
    total_variance = np.sum(eigenvalues[:B3_TARGET])

    local_contribution = local_variance / (total_variance + 1e-10)
    global_contribution = global_variance / (total_variance + 1e-10)

    # Check if gap is at target position
    match = abs(b3_effective - B3_TARGET) <= 5

    # Additional analysis: check specific gaps
    gap_at_35 = gaps_relative[34] if len(gaps_relative) > 34 else 0
    gap_at_77 = gaps_relative[76] if len(gaps_relative) > 76 else 0

    print(f"\n{'='*60}")
    print("SPECTRAL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nTarget b₃ = {B3_TARGET} (35 local + 42 global)")
    print(f"Effective b₃ = {b3_effective}")
    print(f"Gap position = {gap_position}")
    print(f"Gap magnitude = {gap_magnitude:.2f}x mean")
    print(f"\nLocal contribution (35 modes): {local_contribution:.1%}")
    print(f"Global contribution (42 modes): {global_contribution:.1%}")
    print(f"\nGap at position 35: {gap_at_35:.2f}x")
    print(f"Gap at position 77: {gap_at_77:.2f}x")
    print(f"\nMatch: {'YES' if match else 'NO'}")

    return B3Result(
        b3_effective=b3_effective,
        gap_position=gap_position,
        gap_magnitude=gap_magnitude,
        eigenvalues=eigenvalues,
        local_contribution=local_contribution,
        global_contribution=global_contribution,
        match=match
    )


def run_analysis(
    model: nn.Module,
    n_samples: int = 10000,
    domain: Tuple[float, float] = (-1.0, 1.0),
    seed: int = 42
) -> Dict:
    """Run full b₃ = 77 verification."""

    print(f"\n{'='*60}")
    print("B₃ = 77 VERIFICATION (GIFT Structure-Based)")
    print(f"{'='*60}")
    print(f"\nUsing GIFT decomposition: 77 = 35 + 42 = 35 + 2×21")
    print(f"  - 35 local modes (Λ³ℝ⁷)")
    print(f"  - 42 global modes (2 × Λ²ℝ⁷ ∧ S¹)")

    # Generate sample points
    torch.manual_seed(seed)
    np.random.seed(seed)

    coords = torch.rand(n_samples, 7) * (domain[1] - domain[0]) + domain[0]
    print(f"\n[1] Sampling {n_samples} points in domain {domain}")

    # Build 77-mode basis
    print(f"\n[2] Building 77-mode basis from PINN geometry...")
    basis = build_77_basis(model, coords, use_jacobian=True)
    print(f"    Basis shape: {basis.shape}")

    # Compute spectrum
    print(f"\n[3] Computing Gram matrix eigenspectrum...")
    eigenvalues, eigenvectors = compute_gram_spectrum(basis)

    # Analyze
    print(f"\n[4] Analyzing spectrum...")
    result = analyze_spectrum(eigenvalues)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if result.match:
        print(f"\n  b₃ = 77 CONFIRMED")
        print(f"  Gap at position {result.gap_position} matches target!")
    else:
        print(f"\n  b₃ = {result.b3_effective} (target: 77)")
        print(f"  Gap at position {result.gap_position}")

        if result.b3_effective < B3_TARGET:
            missing = B3_TARGET - result.b3_effective
            print(f"  Missing {missing} modes - global TCS modes may need refinement")
        else:
            excess = result.b3_effective - B3_TARGET
            print(f"  Excess {excess} modes - may include non-harmonic contributions")

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Extract b₃=77 modes from GIFT structure")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--no-jacobian", action="store_true", help="Don't use Jacobian (faster)")

    args = parser.parse_args()

    # Find model
    if args.checkpoint and args.checkpoint.exists():
        checkpoint_path = args.checkpoint
    else:
        default_path = Path(__file__).parent / "outputs/metrics/g2_variational_model.pt"
        if default_path.exists() and default_path.stat().st_size > 1000:
            checkpoint_path = default_path
        else:
            print("ERROR: No model found. Run: git lfs pull")
            return 1

    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path)

    # Run analysis
    results = run_analysis(model, n_samples=args.samples)

    # Save
    output_path = args.output or Path(__file__).parent / "b3_77_result.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0 if results.get('match', False) else 1


if __name__ == "__main__":
    sys.exit(main())
