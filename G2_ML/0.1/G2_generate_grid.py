"""
G2 Validation Grid Generation

Generate a grid of evaluated points for validation and reference.
No Unicode - Windows compatible.

Usage:
    python G2_generate_grid.py --model G2_final_model.pt --samples 1000
"""

import argparse
import torch
import numpy as np
from G2_phi_wrapper import load_model, compute_phi_from_metric
import time


def generate_validation_grid(model, n_samples=1000, domain_size=5.0, device='cpu', 
                            compute_ricci=False, seed=42):
    """
    Generate validation grid with comprehensive data at each point.
    
    Args:
        model: Trained G2 network
        n_samples: Number of sample points
        domain_size: Sampling domain size
        device: Device for computation
        compute_ricci: Whether to compute Ricci tensor (slow)
        seed: Random seed for reproducibility
    
    Returns:
        dict with all grid data
    """
    print(f"\nGenerating validation grid with {n_samples} samples...")
    print(f"  Domain size: [-{domain_size}, {domain_size}]^7")
    print(f"  Random seed: {seed}")
    print(f"  Device: {device}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Sample coordinates
    coords = torch.randn(n_samples, 7) * domain_size
    coords = coords.to(device)
    
    # Allocate arrays
    grid_data = {
        'coordinates': np.zeros((n_samples, 7)),
        'metric': np.zeros((n_samples, 7, 7)),
        'phi': np.zeros((n_samples, 35)),
        'phi_norm_sq': np.zeros(n_samples),
        'det_g': np.zeros(n_samples),
        'eigenvalues': np.zeros((n_samples, 7)),
        'min_eigenvalue': np.zeros(n_samples),
        'max_eigenvalue': np.zeros(n_samples),
        'condition_number': np.zeros(n_samples),
    }
    
    if compute_ricci:
        grid_data['ricci_norm_sq'] = np.zeros(n_samples)
    
    # Process in batches
    batch_size = 100
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    model.eval()
    
    print("\nProcessing batches...")
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_coords = coords[start_idx:end_idx]
        
        # Enable gradients if computing Ricci
        if compute_ricci:
            batch_coords.requires_grad_(True)
        
        with torch.set_grad_enabled(compute_ricci):
            # Compute metric and phi
            metric = model(batch_coords)
            phi = compute_phi_from_metric(metric, batch_coords)
            
            # Store basic quantities
            grid_data['coordinates'][start_idx:end_idx] = batch_coords.detach().cpu().numpy()
            grid_data['metric'][start_idx:end_idx] = metric.detach().cpu().numpy()
            grid_data['phi'][start_idx:end_idx] = phi.detach().cpu().numpy()
            
            # Compute phi norm
            phi_norm_sq = torch.sum(phi**2, dim=1)
            grid_data['phi_norm_sq'][start_idx:end_idx] = phi_norm_sq.detach().cpu().numpy()
            
            # Compute determinant
            det_g = torch.det(metric)
            grid_data['det_g'][start_idx:end_idx] = det_g.detach().cpu().numpy()
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(metric)
            grid_data['eigenvalues'][start_idx:end_idx] = eigenvalues.detach().cpu().numpy()
            grid_data['min_eigenvalue'][start_idx:end_idx] = eigenvalues.min(dim=1)[0].detach().cpu().numpy()
            grid_data['max_eigenvalue'][start_idx:end_idx] = eigenvalues.max(dim=1)[0].detach().cpu().numpy()
            grid_data['condition_number'][start_idx:end_idx] = (
                eigenvalues.max(dim=1)[0] / eigenvalues.min(dim=1)[0]
            ).detach().cpu().numpy()
            
            # Compute Ricci if requested
            if compute_ricci:
                metric_inv = torch.linalg.inv(metric + 1e-8 * torch.eye(7, device=device).unsqueeze(0))
                ricci = torch.zeros(end_idx - start_idx, 7, 7, device=device)
                
                for i in range(7):
                    grad_metric = torch.autograd.grad(
                        metric[:, :, :].sum(), batch_coords,
                        create_graph=True, retain_graph=True
                    )[0]
                    
                    for j in range(7):
                        ricci[:, i, j] = torch.sum(
                            metric_inv[:, i, :] * grad_metric[:, j].unsqueeze(-1),
                            dim=1
                        )
                
                ricci_norm_sq = torch.sum(ricci**2, dim=(1, 2))
                grid_data['ricci_norm_sq'][start_idx:end_idx] = ricci_norm_sq.detach().cpu().numpy()
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / n_batches
            eta = elapsed / progress - elapsed
            print(f"  Batch {batch_idx+1}/{n_batches} ({progress*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nGrid generation complete in {total_time:.1f}s")
    
    return grid_data


def save_grid(grid_data, output_path, metadata=None):
    """
    Save validation grid to .npz file.
    
    Args:
        grid_data: Dictionary of grid arrays
        output_path: Path for .npz file
        metadata: Optional metadata dictionary
    """
    # Prepare save dict
    save_dict = dict(grid_data)
    
    # Add metadata as string
    if metadata is not None:
        import json
        save_dict['metadata_json'] = json.dumps(metadata)
    
    # Save
    np.savez_compressed(output_path, **save_dict)
    print(f"\nGrid saved to: {output_path}")
    
    # Print file size
    import os
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")


def print_grid_statistics(grid_data):
    """Print summary statistics of the grid."""
    print("\n" + "=" * 70)
    print("GRID STATISTICS")
    print("=" * 70)
    
    n_samples = grid_data['coordinates'].shape[0]
    
    print(f"\nNumber of samples: {n_samples}")
    
    # Phi normalization
    phi_norm = grid_data['phi_norm_sq']
    print(f"\nPhi normalization:")
    print(f"  ||phi||^2:  {phi_norm.mean():.8f} +/- {phi_norm.std():.8f}")
    print(f"  Range:      [{phi_norm.min():.8f}, {phi_norm.max():.8f}]")
    print(f"  Target:     7.0")
    print(f"  Error:      {abs(phi_norm.mean() - 7.0):.6e}")
    
    # Determinant
    det = grid_data['det_g']
    print(f"\nDeterminant:")
    print(f"  det(g):     {det.mean():.8f} +/- {det.std():.8f}")
    print(f"  Range:      [{det.min():.8f}, {det.max():.8f}]")
    print(f"  Target:     1.0")
    print(f"  Error:      {abs(det.mean() - 1.0):.6e}")
    
    # Eigenvalues
    eig_min = grid_data['min_eigenvalue']
    eig_max = grid_data['max_eigenvalue']
    condition = grid_data['condition_number']
    print(f"\nEigenvalues:")
    print(f"  Min:        {eig_min.min():.6f} (mean: {eig_min.mean():.6f})")
    print(f"  Max:        {eig_max.max():.6f} (mean: {eig_max.mean():.6f})")
    print(f"  Condition:  {condition.mean():.2f} +/- {condition.std():.2f}")
    
    # Ricci if available
    if 'ricci_norm_sq' in grid_data:
        ricci = grid_data['ricci_norm_sq']
        print(f"\nRicci curvature:")
        print(f"  ||Ric||^2:  {ricci.mean():.6e} +/- {ricci.std():.6e}")
        print(f"  Range:      [{ricci.min():.6e}, {ricci.max():.6e}]")
        print(f"  95th perc:  {np.percentile(ricci, 95):.6e}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Generate G2 validation grid')
    parser.add_argument('--model', type=str, default='G2_final_model.pt',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='G2_validation_grid.npz',
                        help='Output path for grid')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples')
    parser.add_argument('--domain-size', type=float, default=5.0,
                        help='Domain size for sampling')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--compute-ricci', action='store_true',
                        help='Compute Ricci tensor (slow)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("\nG2 Validation Grid Generation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.samples}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, device=args.device)
    print("Model loaded successfully!")
    
    # Generate grid
    grid_data = generate_validation_grid(
        model,
        n_samples=args.samples,
        domain_size=args.domain_size,
        device=args.device,
        compute_ricci=args.compute_ricci,
        seed=args.seed
    )
    
    # Print statistics
    print_grid_statistics(grid_data)
    
    # Prepare metadata
    metadata = {
        'model_path': args.model,
        'n_samples': args.samples,
        'domain_size': args.domain_size,
        'seed': args.seed,
        'compute_ricci': args.compute_ricci,
        'description': f'{args.samples} sample points of G2 metric on K7 manifold'
    }
    
    # Save grid
    save_grid(grid_data, args.output, metadata=metadata)
    
    print("\n" + "=" * 70)
    print("Grid generation complete!")
    print("\nUsage:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{args.output}')")
    print(f"  coords = data['coordinates']")
    print(f"  metric = data['metric']")
    print(f"  phi = data['phi']")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())









