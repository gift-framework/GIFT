"""
G2 Evaluation Script - v0.2

Comprehensive evaluation of trained G2 metric models.
Provides CLI interface for model analysis and testing.

Usage:
    python G2_eval.py --model checkpoints/final_model.pt --samples 2000
    python G2_eval.py --model checkpoints/best_model.pt --point 0,0,0,0,0,0,0

Author: GIFT Project
No Unicode - Windows compatible
"""

import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
from G2_geometry import project_spd, volume_form
from G2_manifold import create_manifold
from G2_validation import comprehensive_validation, plot_validation_summary


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_path, device='cpu'):
    """
    Load a trained G2 model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model on
    
    Returns:
        model: Loaded G2PhiNetwork
        config: Training configuration dict
    """
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default configuration if not saved
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            'encoding_type': 'fourier',
            'hidden_dims': [256, 256, 128],
            'fourier_modes': 16,
            'fourier_scale': 1.0,
            'omega_0': 30.0,
            'normalize_phi': True
        }
    
    # Create model
    model = G2PhiNetwork(
        encoding_type=config['encoding_type'],
        hidden_dims=config['hidden_dims'],
        fourier_modes=config.get('fourier_modes', 16),
        fourier_scale=config.get('fourier_scale', 1.0),
        omega_0=config.get('omega_0', 30.0),
        normalize_phi=config.get('normalize_phi', True)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"  Trained for: {checkpoint['epoch']} epochs")
    if 'loss' in checkpoint:
        print(f"  Final loss: {checkpoint['loss']:.6e}")
    
    return model, config


# ============================================================================
# Point Evaluation
# ============================================================================

def evaluate_at_point(model, coords, device='cpu'):
    """
    Evaluate model at specific coordinate point(s).
    
    Args:
        model: G2PhiNetwork
        coords: Coordinates tensor of shape (n, 7) or (7,)
        device: Device
    
    Returns:
        results: Dict with phi, metric, and derived quantities
    """
    model.eval()
    
    # Ensure coords is 2D
    if coords.ndim == 1:
        coords = coords.unsqueeze(0)
    
    coords = coords.to(device)
    
    with torch.no_grad():
        # Compute phi
        phi = model(coords)
        
        # Reconstruct metric
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Compute derived quantities
        phi_norm_sq = torch.sum(phi ** 2, dim=1)
        det_g = torch.det(metric)
        vol = volume_form(metric)
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        results = {
            'coords': coords.cpu().numpy(),
            'phi': phi.cpu().numpy(),
            'metric': metric.cpu().numpy(),
            'phi_norm_sq': phi_norm_sq.cpu().numpy(),
            'det_g': det_g.cpu().numpy(),
            'volume': vol.cpu().numpy(),
            'eigenvalues': eigenvalues.cpu().numpy()
        }
    
    return results


def print_point_evaluation(results, idx=0):
    """Print evaluation results for a single point."""
    print("\n" + "=" * 70)
    print("Point Evaluation")
    print("=" * 70)
    
    print(f"\nCoordinates: {results['coords'][idx]}")
    print(f"\n||phi||^2 = {results['phi_norm_sq'][idx]:.6f} (target: 7.0)")
    print(f"det(g) = {results['det_g'][idx]:.6f} (target: 1.0)")
    print(f"sqrt(det(g)) = {results['volume'][idx]:.6f}")
    
    print(f"\nMetric eigenvalues:")
    eigenvalues = results['eigenvalues'][idx]
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {ev:.6f}")
    
    print(f"\nCondition number: {eigenvalues.max() / eigenvalues.min():.2f}")
    print(f"Positive definite: {eigenvalues.min() > 0}")
    
    print("\nPhi components (first 10):")
    phi = results['phi'][idx]
    for i in range(min(10, len(phi))):
        print(f"  phi_{i} = {phi[i]:.6f}")
    
    print("=" * 70)


# ============================================================================
# Batch Evaluation
# ============================================================================

def evaluate_on_grid(model, manifold, n_points_per_dim=5, device='cpu'):
    """
    Evaluate model on regular grid.
    
    Args:
        model: G2PhiNetwork
        manifold: Manifold object
        n_points_per_dim: Points per dimension
        device: Device
    
    Returns:
        results: Dict with evaluation results
    """
    print(f"\nEvaluating on {n_points_per_dim}^7 = {n_points_per_dim**7} grid points...")
    
    model.eval()
    
    grid_coords = manifold.create_validation_grid(n_points_per_dim)
    grid_coords = grid_coords.to(device)
    
    # Evaluate in batches to avoid memory issues
    batch_size = 1000
    n_points = grid_coords.shape[0]
    
    all_phi_norm_sq = []
    all_det_g = []
    all_eigenvalues = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = grid_coords[i:i+batch_size]
            
            phi = model(batch_coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)
            
            phi_norm_sq = torch.sum(phi ** 2, dim=1)
            det_g = torch.det(metric)
            eigenvalues = torch.linalg.eigvalsh(metric)
            
            all_phi_norm_sq.append(phi_norm_sq.cpu())
            all_det_g.append(det_g.cpu())
            all_eigenvalues.append(eigenvalues.cpu())
    
    # Concatenate results
    all_phi_norm_sq = torch.cat(all_phi_norm_sq)
    all_det_g = torch.cat(all_det_g)
    all_eigenvalues = torch.cat(all_eigenvalues)
    
    results = {
        'phi_norm_sq': {
            'mean': all_phi_norm_sq.mean().item(),
            'std': all_phi_norm_sq.std().item(),
            'min': all_phi_norm_sq.min().item(),
            'max': all_phi_norm_sq.max().item(),
            'error_from_7': torch.abs(all_phi_norm_sq - 7.0).mean().item()
        },
        'det_g': {
            'mean': all_det_g.mean().item(),
            'std': all_det_g.std().item(),
            'min': all_det_g.min().item(),
            'max': all_det_g.max().item(),
            'error_from_1': torch.abs(all_det_g - 1.0).mean().item()
        },
        'eigenvalues': {
            'min': all_eigenvalues.min().item(),
            'max': all_eigenvalues.max().item(),
            'mean': all_eigenvalues.mean().item(),
            'condition_number_mean': (all_eigenvalues.max(dim=1)[0] / all_eigenvalues.min(dim=1)[0]).mean().item()
        }
    }
    
    return results


def print_grid_evaluation(results):
    """Print grid evaluation results."""
    print("\n" + "=" * 70)
    print("Grid Evaluation Results")
    print("=" * 70)
    
    print("\nPhi Normalization:")
    print(f"  Mean: {results['phi_norm_sq']['mean']:.6f}")
    print(f"  Std:  {results['phi_norm_sq']['std']:.6f}")
    print(f"  Range: [{results['phi_norm_sq']['min']:.6f}, {results['phi_norm_sq']['max']:.6f}]")
    print(f"  Error from 7.0: {results['phi_norm_sq']['error_from_7']:.6e}")
    
    print("\nMetric Determinant:")
    print(f"  Mean: {results['det_g']['mean']:.6f}")
    print(f"  Std:  {results['det_g']['std']:.6f}")
    print(f"  Range: [{results['det_g']['min']:.6f}, {results['det_g']['max']:.6f}]")
    print(f"  Error from 1.0: {results['det_g']['error_from_1']:.6e}")
    
    print("\nMetric Eigenvalues:")
    print(f"  Mean: {results['eigenvalues']['mean']:.6f}")
    print(f"  Range: [{results['eigenvalues']['min']:.6f}, {results['eigenvalues']['max']:.6f}]")
    print(f"  Condition number (mean): {results['eigenvalues']['condition_number_mean']:.2f}")
    
    print("=" * 70)


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description='Evaluate trained G2 model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of validation samples')
    parser.add_argument('--point', type=str, default=None,
                       help='Evaluate at specific point (comma-separated 7 values)')
    parser.add_argument('--grid', type=int, default=0,
                       help='Evaluate on grid (specify points per dimension, 0=disabled)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive validation (includes torsion, curvature, etc.)')
    
    args = parser.parse_args()
    
    # Handle auto device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "=" * 70)
    print("G2 Model Evaluation - v0.2")
    print("=" * 70)
    print(f"\nDevice: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, config = load_model(args.model, device=args.device)
    
    # Create manifold
    manifold = create_manifold(
        config.get('manifold_type', 'T7'),
        device=args.device
    )
    
    # Evaluate at specific point
    if args.point is not None:
        try:
            coords_list = [float(x) for x in args.point.split(',')]
            if len(coords_list) != 7:
                raise ValueError("Point must have exactly 7 coordinates")
            
            coords = torch.tensor(coords_list, dtype=torch.float32)
            results = evaluate_at_point(model, coords, device=args.device)
            print_point_evaluation(results)
            
            # Save results
            output_path = os.path.join(args.output_dir, 'point_evaluation.json')
            with open(output_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                results_json = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in results.items()}
                json.dump(results_json, f, indent=2)
            print(f"\nResults saved to: {output_path}")
            
        except Exception as e:
            print(f"Error evaluating at point: {e}")
    
    # Evaluate on grid
    if args.grid > 0:
        grid_results = evaluate_on_grid(model, manifold, args.grid, device=args.device)
        print_grid_evaluation(grid_results)
        
        # Save results
        output_path = os.path.join(args.output_dir, 'grid_evaluation.json')
        with open(output_path, 'w') as f:
            json.dump(grid_results, f, indent=2)
        print(f"\nGrid results saved to: {output_path}")
    
    # Comprehensive validation
    if args.comprehensive:
        print("\n" + "=" * 70)
        print("Running Comprehensive Validation...")
        print("=" * 70)
        
        report = comprehensive_validation(
            model, manifold,
            n_samples=args.samples,
            device=args.device,
            verbose=True
        )
        
        # Save report
        report_path = os.path.join(args.output_dir, 'comprehensive_validation.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nValidation report saved to: {report_path}")
        
        # Plot validation summary
        plot_path = os.path.join(args.output_dir, 'validation_summary.png')
        plot_validation_summary(report, save_path=plot_path)
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()






