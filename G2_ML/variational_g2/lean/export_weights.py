#!/usr/bin/env python3
"""
Export PINN weights to JSON for Lean verification.

This script extracts the trained neural network weights and exports them
in a format suitable for formal verification in Lean 4.

Usage:
    python export_weights.py --model path/to/model.pt --output weights.json

Output format:
    {
        "metadata": {...},
        "fourier": {"B": [[...]]},
        "layers": [
            {"weight": [[...]], "bias": [...]},
            ...
        ],
        "output": {"scale": [...], "bias": [...]}
    }

Author: GIFT Framework
Date: 2025-11-30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


def export_tensor(t: torch.Tensor) -> List:
    """Convert tensor to nested Python list."""
    return t.detach().cpu().numpy().tolist()


def export_model_weights(model_path: Path) -> Dict[str, Any]:
    """
    Export model weights to dictionary.

    Args:
        model_path: Path to saved .pt model file

    Returns:
        Dictionary with all weights in nested list format
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different save formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # Assume it's the model itself
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else {}

    result = {
        "metadata": {
            "format_version": "1.0",
            "model_type": "G2VariationalNet",
            "input_dim": 7,
            "output_dim": 35,
            "description": "PINN for G2 variational problem",
        },
        "fourier": {},
        "layers": [],
        "residual_blocks": [],
        "output": {},
    }

    # Extract Fourier features
    for key, value in state_dict.items():
        if 'fourier' in key and 'B' in key:
            result["fourier"]["B"] = export_tensor(value)
            result["metadata"]["num_frequencies"] = value.shape[0]

    # Extract MLP layers
    layer_idx = 0
    while True:
        weight_key = f"mlp.{layer_idx}.weight"
        bias_key = f"mlp.{layer_idx}.bias"

        if weight_key in state_dict:
            layer = {
                "index": layer_idx // 2 if layer_idx % 2 == 0 else layer_idx // 2,
                "weight": export_tensor(state_dict[weight_key]),
                "bias": export_tensor(state_dict[bias_key]) if bias_key in state_dict else None,
                "input_dim": state_dict[weight_key].shape[1],
                "output_dim": state_dict[weight_key].shape[0],
            }
            result["layers"].append(layer)
        elif layer_idx > 20:  # Safety limit
            break

        layer_idx += 1

    # Extract residual blocks
    for key, value in state_dict.items():
        if 'residual' in key.lower() or 'res' in key.lower():
            # Parse residual block structure
            pass  # TODO: handle residual blocks if present

    # Extract output layer
    if 'output_layer.weight' in state_dict:
        result["output"]["layer_weight"] = export_tensor(state_dict['output_layer.weight'])
        result["output"]["layer_bias"] = export_tensor(state_dict['output_layer.bias'])

    if 'output_scale' in state_dict:
        result["output"]["scale"] = export_tensor(state_dict['output_scale'])

    if 'output_bias' in state_dict:
        result["output"]["bias"] = export_tensor(state_dict['output_bias'])

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    result["metadata"]["total_parameters"] = total_params

    return result


def export_to_lean_format(weights: Dict[str, Any], output_path: Path) -> None:
    """
    Export weights in Lean-friendly format.

    Creates both JSON and a summary Lean file.
    """
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)

    # Create Lean-compatible summary
    lean_path = output_path.with_suffix('.lean')

    lean_content = f'''/-
  GIFT PINN Weights - Auto-generated

  Total parameters: {weights["metadata"].get("total_parameters", "unknown")}
  Input dim: {weights["metadata"]["input_dim"]}
  Output dim: {weights["metadata"]["output_dim"]}

  This file provides Lean definitions for the frozen neural network.
-/

namespace GIFT.PINN

-- Network architecture constants
def input_dim : ℕ := {weights["metadata"]["input_dim"]}
def output_dim : ℕ := {weights["metadata"]["output_dim"]}
def num_frequencies : ℕ := {weights["metadata"].get("num_frequencies", 64)}
def num_layers : ℕ := {len(weights["layers"])}

-- Layer dimensions
def layer_dims : List (ℕ × ℕ) := [
'''

    for layer in weights["layers"]:
        lean_content += f'  ({layer["input_dim"]}, {layer["output_dim"]}),\n'

    lean_content += ''']

-- Weights are stored in external JSON file
-- Load with: `json_weights := load_json "weights.json"`

-- Standard G2 form bias (output initialization)
def standard_g2_indices : List (ℕ × ℕ × ℕ) := [
  (0, 1, 2),  -- e^{123}
  (0, 3, 4),  -- e^{145}
  (0, 5, 6),  -- e^{167}
  (1, 3, 5),  -- e^{246}
  (1, 4, 6),  -- e^{257} sign -1
  (2, 3, 6),  -- e^{347} sign -1
  (2, 4, 5),  -- e^{356} sign -1
]

def standard_g2_signs : List ℤ := [1, 1, 1, 1, -1, -1, -1]

end GIFT.PINN
'''

    with open(lean_path, 'w') as f:
        f.write(lean_content)

    print(f"Exported weights to: {output_path}")
    print(f"Lean summary at: {lean_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PINN weights for Lean verification")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--output", type=str, default="weights.json", help="Output JSON path")

    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading model from: {model_path}")
    weights = export_model_weights(model_path)

    print(f"Extracted {len(weights['layers'])} layers")
    print(f"Total parameters: {weights['metadata'].get('total_parameters', 'unknown')}")

    export_to_lean_format(weights, output_path)


if __name__ == "__main__":
    main()
