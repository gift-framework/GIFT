#!/usr/bin/env python3
"""
Complete Pipeline: Verify det(g) = 65/32 from PINN weights

This script:
1. Loads PINN weights (from .pt or .json)
2. Propagates intervals through the network
3. Computes det(g) with interval bounds
4. Verifies det(g) = 65/32 within tolerance

Usage:
    # With real weights (requires PyTorch + Git LFS)
    python verify_det_g.py --checkpoint path/to/model.pt

    # With exported JSON weights (no PyTorch needed)
    python verify_det_g.py --weights weights.json

    # Demo with synthetic weights
    python verify_det_g.py --demo

Author: GIFT Framework
Date: 2025-11-30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

# Import interval arithmetic from our module
from interval_det_g import (
    Interval,
    interval_silu,
    interval_matmul,
    metric_from_phi_interval,
    interval_det_7x7,
    verify_det_g,
)


@dataclass
class NetworkConfig:
    """Configuration for G2VariationalNet."""
    input_dim: int = 7
    num_frequencies: int = 64
    hidden_dims: List[int] = None
    output_dim: int = 35

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 512, 256]


def load_weights_from_json(path: Path) -> Tuple[Dict[str, Any], NetworkConfig]:
    """Load weights from exported JSON file."""
    with open(path) as f:
        data = json.load(f)

    config = NetworkConfig(
        input_dim=data.get("metadata", {}).get("input_dim", 7),
        num_frequencies=data.get("metadata", {}).get("num_frequencies", 64),
        output_dim=data.get("metadata", {}).get("output_dim", 35),
    )

    return data, config


def load_weights_from_checkpoint(path: Path) -> Tuple[Dict[str, Any], NetworkConfig]:
    """Load weights from PyTorch checkpoint."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch required for loading .pt files")
        print("Use --weights with exported JSON instead")
        sys.exit(1)

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    # Convert to nested lists
    weights = {
        "metadata": {"format": "pytorch"},
        "layers": [],
        "fourier": {},
        "output": {},
    }

    for key, value in state_dict.items():
        tensor = value.detach().cpu().numpy().tolist()
        if 'fourier' in key and 'B' in key:
            weights["fourier"]["B"] = tensor
        elif 'output_layer.weight' in key:
            weights["output"]["layer_weight"] = tensor
        elif 'output_layer.bias' in key:
            weights["output"]["layer_bias"] = tensor
        elif 'output_scale' in key:
            weights["output"]["scale"] = tensor
        elif 'output_bias' in key:
            weights["output"]["bias"] = tensor

    # Extract MLP layers
    layer_idx = 0
    while True:
        w_key = f"mlp.{layer_idx}.weight"
        b_key = f"mlp.{layer_idx}.bias"
        if w_key in state_dict:
            weights["layers"].append({
                "weight": state_dict[w_key].detach().cpu().numpy().tolist(),
                "bias": state_dict[b_key].detach().cpu().numpy().tolist() if b_key in state_dict else None,
            })
        elif layer_idx > 20:
            break
        layer_idx += 1

    config = NetworkConfig()
    return weights, config


def create_synthetic_weights(config: NetworkConfig) -> Dict[str, Any]:
    """
    Create synthetic weights that produce standard G2 form.

    This is for testing the verification pipeline.
    """
    import random
    random.seed(42)

    weights = {
        "metadata": {"format": "synthetic", "description": "Test weights"},
        "fourier": {},
        "layers": [],
        "output": {},
    }

    # Fourier matrix (random but fixed seed)
    weights["fourier"]["B"] = [
        [random.gauss(0, 1) for _ in range(config.input_dim)]
        for _ in range(config.num_frequencies)
    ]

    # MLP layers: just use identity-ish weights for simplicity
    fourier_out = 2 * config.num_frequencies  # sin + cos

    prev_dim = fourier_out
    for hidden_dim in config.hidden_dims:
        # Small random weights
        weights["layers"].append({
            "weight": [
                [random.gauss(0, 0.01) for _ in range(prev_dim)]
                for _ in range(hidden_dim)
            ],
            "bias": [0.0] * hidden_dim,
        })
        prev_dim = hidden_dim

    # Output layer
    weights["output"]["layer_weight"] = [
        [random.gauss(0, 0.01) for _ in range(prev_dim)]
        for _ in range(config.output_dim)
    ]
    weights["output"]["layer_bias"] = [0.0] * config.output_dim

    # Output scale and bias
    weights["output"]["scale"] = [0.1] * config.output_dim

    # Initialize bias to standard G2 form
    g2_indices = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    g2_signs = [1, 1, 1, 1, -1, -1, -1]

    # Map to linear indices
    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append((i, j, k))

    bias = [0.0] * 35
    for idx, sign in zip(g2_indices, g2_signs):
        linear_idx = indices.index(idx)
        # Scale to get det(g) = 65/32
        scale_factor = (65.0 / 32.0) ** (1.0 / 14.0)
        bias[linear_idx] = float(sign) * scale_factor

    weights["output"]["bias"] = bias

    return weights


def interval_fourier(
    x: List[Interval],
    B: List[List[float]]
) -> List[Interval]:
    """
    Apply Fourier feature encoding with intervals.

    fourier(x) = [sin(2*pi*B*x), cos(2*pi*B*x)]
    """
    num_freq = len(B)
    result = []

    for row in B:
        # Compute B[i] @ x
        dot = Interval.point(0.0)
        for j, (b_val, x_val) in enumerate(zip(row, x)):
            dot = dot + Interval.point(b_val) * x_val

        # 2 * pi * dot
        scaled = Interval.point(2 * math.pi) * dot

        # sin and cos (conservative: [-1, 1])
        # For better bounds, could track periodicity
        result.append(Interval(-1.0, 1.0))  # sin
        result.append(Interval(-1.0, 1.0))  # cos

    return result


def propagate_network(
    x: List[Interval],
    weights: Dict[str, Any],
    config: NetworkConfig
) -> List[Interval]:
    """
    Propagate intervals through the full network.

    Returns phi_components as list of 35 intervals.
    """
    # 1. Fourier encoding
    if "B" in weights.get("fourier", {}):
        h = interval_fourier(x, weights["fourier"]["B"])
    else:
        # No Fourier features, just use input
        h = x

    # 2. MLP layers
    for layer in weights.get("layers", []):
        W = [[Interval.point(w) for w in row] for row in layer["weight"]]
        b = [Interval.point(bi) for bi in layer["bias"]] if layer["bias"] else None

        h = interval_matmul(W, h, b)
        # Apply SiLU activation
        h = [interval_silu(hi) for hi in h]

    # 3. Output layer
    if "layer_weight" in weights.get("output", {}):
        W_out = [[Interval.point(w) for w in row]
                 for row in weights["output"]["layer_weight"]]
        b_out = ([Interval.point(b) for b in weights["output"]["layer_bias"]]
                 if "layer_bias" in weights["output"] else None)
        h = interval_matmul(W_out, h, b_out)

    # 4. Apply scale and bias
    if "scale" in weights.get("output", {}):
        scale = weights["output"]["scale"]
        h = [hi * Interval.point(si) for hi, si in zip(h, scale)]

    if "bias" in weights.get("output", {}):
        bias = weights["output"]["bias"]
        h = [hi + Interval.point(bi) for hi, bi in zip(h, bias)]

    return h


def verify_det_g_from_network(
    weights: Dict[str, Any],
    config: NetworkConfig,
    input_bounds: Tuple[float, float] = (-1.0, 1.0),
    n_subdivisions: int = 1,
    target: float = 65.0 / 32.0,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Verify det(g) = 65/32 from network weights.

    Args:
        weights: Network weights dictionary
        config: Network configuration
        input_bounds: Domain bounds for inputs
        n_subdivisions: Number of subdivisions per dimension (for tighter bounds)
        target: Target det(g) value
        tolerance: Allowed tolerance

    Returns:
        Verification result dictionary
    """
    lo, hi = input_bounds

    # For demonstration, use single input interval (full domain)
    # For tighter bounds, subdivide the domain
    x = [Interval(lo, hi) for _ in range(config.input_dim)]

    print(f"\nInput domain: [{lo}, {hi}]^{config.input_dim}")
    print(f"Network: {config.num_frequencies} Fourier + {config.hidden_dims} MLP")

    # Propagate through network
    print("\nPropagating intervals through network...")
    phi = propagate_network(x, weights, config)

    print(f"Output phi has {len(phi)} components")

    # Show some phi bounds
    print("\nFirst 7 phi components (intervals):")
    for i, p in enumerate(phi[:7]):
        print(f"  phi[{i}] = {p}")

    # Compute metric
    print("\nComputing metric g_ij...")
    g = metric_from_phi_interval(phi)

    # Show diagonal
    print("Metric diagonal:")
    for i in range(7):
        print(f"  g[{i},{i}] = {g[i][i]}")

    # Compute determinant
    print("\nComputing det(g)...")
    det_g = interval_det_7x7(g)

    print(f"\ndet(g) = {det_g}")
    print(f"Width: {det_g.width():.6f}")
    print(f"Target: {target}")

    # Verify
    success, det_result, msg = verify_det_g(phi, target=target, tolerance=tolerance)

    result = {
        "success": success,
        "det_g": {"lo": det_g.lo, "hi": det_g.hi},
        "target": target,
        "tolerance": tolerance,
        "width": det_g.width(),
        "message": msg,
    }

    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULT: {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}")
    print(f"det(g) in [{det_g.lo:.6f}, {det_g.hi:.6f}]")
    print(f"Target: {target} ± {tolerance}")
    print(f"Message: {msg}")

    return result


def test_direct_verification():
    """
    Test det(g) verification directly from phi values.

    This bypasses the network and tests the core formula.
    """
    print("=" * 60)
    print("DIRECT VERIFICATION TEST (no network)")
    print("=" * 60)

    # Standard G2 form scaled for det(g) = 65/32
    g2_indices = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    g2_signs = [1, 1, 1, 1, -1, -1, -1]

    indices = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                indices.append((i, j, k))

    # Scale factor for det(g) = 65/32
    scale = (65.0 / 32.0) ** (1.0 / 14.0)
    print(f"\nScale factor: {scale:.6f}")

    # Create phi as point intervals
    phi = [Interval.point(0.0) for _ in range(35)]
    for idx, sign in zip(g2_indices, g2_signs):
        linear_idx = indices.index(idx)
        phi[linear_idx] = Interval.point(float(sign) * scale)

    print("\nPhi values (point intervals):")
    for idx, sign in zip(g2_indices, g2_signs):
        linear_idx = indices.index(idx)
        print(f"  phi[{linear_idx}] ({idx}) = {phi[linear_idx]}")

    # Verify with point values
    success, det_g, msg = verify_det_g(phi, target=65.0/32.0, tolerance=1e-6)
    print(f"\nPoint verification: {msg}")

    # Now add small uncertainty
    epsilon = 0.001
    print(f"\nAdding uncertainty: epsilon = {epsilon}")
    phi_uncertain = [
        Interval(p.lo - epsilon, p.hi + epsilon) if abs(p.lo) > 0.5 else p
        for p in phi
    ]

    success2, det_g2, msg2 = verify_det_g(phi_uncertain, target=65.0/32.0, tolerance=0.01)
    print(f"Uncertain verification: {msg2}")

    return success and success2


def main():
    parser = argparse.ArgumentParser(
        description="Verify det(g) = 65/32 from PINN weights"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--weights", type=str, help="Path to exported JSON weights")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic weights")
    parser.add_argument("--direct", action="store_true", help="Test direct verification (no network)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Verification tolerance")

    args = parser.parse_args()

    # Quick direct test
    if args.direct:
        success = test_direct_verification()
        return 0 if success else 1

    print("=" * 60)
    print("GIFT det(g) = 65/32 Interval Verification")
    print("=" * 60)

    config = NetworkConfig()

    if args.demo:
        print("\nUsing SYNTHETIC weights (demo mode)")
        print("These weights are designed to produce standard G2 form")
        print("\nWARNING: Random MLP weights cause interval blowup!")
        print("Use --direct for clean formula test")
        weights = create_synthetic_weights(config)
    elif args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        weights, config = load_weights_from_checkpoint(Path(args.checkpoint))
    elif args.weights:
        print(f"\nLoading weights: {args.weights}")
        weights, config = load_weights_from_json(Path(args.weights))
    else:
        print("\nNo weights specified, running direct test")
        return 0 if test_direct_verification() else 1

    result = verify_det_g_from_network(
        weights,
        config,
        tolerance=args.tolerance,
    )

    # Save result
    output_path = Path("verification_result.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {output_path}")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
