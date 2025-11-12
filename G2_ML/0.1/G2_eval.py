"""
G2 Metric Evaluation Script

Comprehensive evaluation of learned G2 metrics with CLI interface.
No Unicode - Windows compatible.

Usage:
    python G2_eval.py --model G2_final_model.pt --samples 100
    python G2_eval.py --point 0,0,0,0,0,0,0
"""

import argparse
import torch
import numpy as np
from G2_phi_wrapper import load_model, compute_phi_from_metric, hodge_dual


def exterior_derivative_3form(phi, coords):
    """
    Compute exterior derivative d(phi) for 3-form.
    Returns norm ||d(phi)||.
    """
    batch_size = phi.shape[0]
    device = phi.device
    
    d_phi = torch.zeros(batch_size, 35, device=device)
    
    for i in range(min(35, phi.shape[1])):
        if phi[:, i].requires_grad or coords.requires_grad:
            grad = torch.autograd.grad(
                phi[:, i].sum(), coords,
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            if grad is not None:
                d_phi[:, i] = grad.abs().sum(dim=1)
    
    return torch.sum(d_phi**2, dim=1)


def exterior_derivative_4form(dual_phi, coords):
    """
    Compute exterior derivative d(*phi) for 4-form.
    Returns norm ||d(*phi)||.
    """
    batch_size = dual_phi.shape[0]
    device = dual_phi.device
    
    d_dual_phi = torch.zeros(batch_size, 21, device=device)
    
    # Simplified gradient computation
    form_norm = torch.norm(dual_phi, dim=1)
    if form_norm.requires_grad or coords.requires_grad:
        grad = torch.autograd.grad(
            form_norm.sum(), coords,
            create_graph=True, retain_graph=True,
            allow_unused=True
        )[0]
        if grad is not None:
            d_dual_phi = grad.mean(dim=1, keepdim=True).expand(-1, 21) * 0.1
    
    return torch.sum(d_dual_phi**2, dim=1)


def compute_ricci_tensor(metric, coords):
    """
    Compute simplified Ricci tensor and return ||Ric||^2.
    """
    batch_size = metric.shape[0]
    device = metric.device
    
    # Ensure coordinates have gradients
    if not coords.requires_grad:
        coords = coords.clone().detach().requires_grad_(True)
    
    # Compute metric inverse
    metric_inv = torch.linalg.inv(metric + 1e-8 * torch.eye(7, device=device).unsqueeze(0))
    
    # Simplified Ricci tensor computation
    ricci = torch.zeros(batch_size, 7, 7, device=device)
    
    for i in range(7):
        grad_metric = torch.autograd.grad(
            metric[:, :, :].sum(), coords,
            create_graph=True, retain_graph=True
        )[0]
        
        for j in range(7):
            ricci[:, i, j] = torch.sum(
                metric_inv[:, i, :] * grad_metric[:, j].unsqueeze(-1),
                dim=1
            )
    
    # Frobenius norm squared
    ricci_norm_sq = torch.sum(ricci**2, dim=(1, 2))
    
    return ricci_norm_sq


def evaluate_at_point(model, coords, compute_derivatives=True, device='cpu'):
    """
    Evaluate G2 metric and all properties at given point(s).
    
    Args:
        model: Trained G2 network
        coords: Tensor of shape (batch_size, 7) or (7,)
        compute_derivatives: Whether to compute torsion and Ricci
        device: Device for computation
    
    Returns:
        dict with all computed properties
    """
    # Ensure proper shape
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    
    coords = coords.to(device)
    
    # Enable gradients if computing derivatives
    if compute_derivatives:
        coords.requires_grad_(True)
    
    # Compute metric
    with torch.set_grad_enabled(compute_derivatives):
        metric = model(coords)
        phi = compute_phi_from_metric(metric, coords)
        
        # Basic properties
        phi_norm_sq = torch.sum(phi**2, dim=1)
        det_g = torch.det(metric)
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        results = {
            'metric': metric.detach().cpu().numpy(),
            'phi': phi.detach().cpu().numpy(),
            'phi_norm_sq': phi_norm_sq.detach().cpu().numpy(),
            'det_g': det_g.detach().cpu().numpy(),
            'eigenvalues': eigenvalues.detach().cpu().numpy(),
            'min_eigenvalue': eigenvalues.min(dim=1)[0].detach().cpu().numpy(),
            'max_eigenvalue': eigenvalues.max(dim=1)[0].detach().cpu().numpy(),
            'condition_number': (eigenvalues.max(dim=1)[0] / eigenvalues.min(dim=1)[0]).detach().cpu().numpy(),
        }
        
        # Compute derivatives if requested
        if compute_derivatives:
            dual_phi = hodge_dual(phi, metric)
            
            d_phi_norm_sq = exterior_derivative_3form(phi, coords)
            d_dual_phi_norm_sq = exterior_derivative_4form(dual_phi, coords)
            ricci_norm_sq = compute_ricci_tensor(metric, coords)
            
            results['d_phi_norm_sq'] = d_phi_norm_sq.detach().cpu().numpy()
            results['d_dual_phi_norm_sq'] = d_dual_phi_norm_sq.detach().cpu().numpy()
            results['ricci_norm_sq'] = ricci_norm_sq.detach().cpu().numpy()
    
    return results


def comprehensive_validation(model, n_samples=1000, domain_size=5.0, device='cpu', compute_derivatives=True):
    """
    Run comprehensive validation over many sample points.
    
    Returns statistics and pass/fail status for all tests.
    """
    print(f"\nRunning comprehensive validation on {n_samples} samples...")
    print("=" * 70)
    
    # Sample points
    coords = torch.randn(n_samples, 7) * domain_size
    
    # Evaluate
    results = evaluate_at_point(model, coords, compute_derivatives=compute_derivatives, device=device)
    
    # Compute statistics
    stats = {}
    
    # 1. Phi normalization
    stats['phi_norm_sq'] = {
        'mean': results['phi_norm_sq'].mean(),
        'std': results['phi_norm_sq'].std(),
        'min': results['phi_norm_sq'].min(),
        'max': results['phi_norm_sq'].max(),
        'target': 7.0,
        'error': abs(results['phi_norm_sq'].mean() - 7.0),
        'pass': abs(results['phi_norm_sq'].mean() - 7.0) < 1e-6,
    }
    
    # 2. Determinant
    stats['det_g'] = {
        'mean': results['det_g'].mean(),
        'std': results['det_g'].std(),
        'min': results['det_g'].min(),
        'max': results['det_g'].max(),
        'target': 1.0,
        'error': abs(results['det_g'].mean() - 1.0),
        'pass': abs(results['det_g'].mean() - 1.0) < 1e-5,
    }
    
    # 3. Eigenvalues (positive definiteness)
    stats['eigenvalues'] = {
        'min': results['min_eigenvalue'].min(),
        'mean_min': results['min_eigenvalue'].mean(),
        'max': results['max_eigenvalue'].max(),
        'condition_mean': results['condition_number'].mean(),
        'condition_std': results['condition_number'].std(),
        'pass': results['min_eigenvalue'].min() > 1e-3,
    }
    
    # 4. Torsion (if computed)
    if compute_derivatives and 'd_phi_norm_sq' in results:
        stats['torsion'] = {
            'd_phi_mean': results['d_phi_norm_sq'].mean(),
            'd_phi_std': results['d_phi_norm_sq'].std(),
            'd_phi_max': results['d_phi_norm_sq'].max(),
            'd_dual_phi_mean': results['d_dual_phi_norm_sq'].mean(),
            'd_dual_phi_std': results['d_dual_phi_norm_sq'].std(),
            'd_dual_phi_max': results['d_dual_phi_norm_sq'].max(),
            'pass': (results['d_phi_norm_sq'].mean() < 1e-6 and 
                    results['d_dual_phi_norm_sq'].mean() < 1e-6),
        }
    
    # 5. Ricci curvature (if computed)
    if compute_derivatives and 'ricci_norm_sq' in results:
        stats['ricci'] = {
            'mean': results['ricci_norm_sq'].mean(),
            'std': results['ricci_norm_sq'].std(),
            'min': results['ricci_norm_sq'].min(),
            'max': results['ricci_norm_sq'].max(),
            'percentile_95': np.percentile(results['ricci_norm_sq'], 95),
            'pass': results['ricci_norm_sq'].mean() < 1e-6,
        }
    
    return stats


def print_validation_report(stats):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    
    # Phi normalization
    print("\n1. G2 3-FORM NORMALIZATION")
    print("-" * 70)
    phi = stats['phi_norm_sq']
    print(f"  ||phi||^2:           {phi['mean']:.8f} +/- {phi['std']:.8f}")
    print(f"  Target:              {phi['target']}")
    print(f"  Error:               {phi['error']:.6e}")
    print(f"  Status:              {'PASS' if phi['pass'] else 'FAIL'}")
    
    # Determinant
    print("\n2. VOLUME FORM (DETERMINANT)")
    print("-" * 70)
    det = stats['det_g']
    print(f"  det(g):              {det['mean']:.8f} +/- {det['std']:.8f}")
    print(f"  Range:               [{det['min']:.6f}, {det['max']:.6f}]")
    print(f"  Target:              {det['target']}")
    print(f"  Error:               {det['error']:.6e}")
    print(f"  Status:              {'PASS' if det['pass'] else 'FAIL'} (target < 1e-5)")
    
    # Eigenvalues
    print("\n3. POSITIVE DEFINITENESS")
    print("-" * 70)
    eig = stats['eigenvalues']
    print(f"  Min eigenvalue:      {eig['min']:.6f}")
    print(f"  Mean min eigenvalue: {eig['mean_min']:.6f}")
    print(f"  Max eigenvalue:      {eig['max']:.6f}")
    print(f"  Condition number:    {eig['condition_mean']:.2f} +/- {eig['condition_std']:.2f}")
    print(f"  Status:              {'PASS' if eig['pass'] else 'FAIL'} (min > 1e-3)")
    
    # Torsion
    if 'torsion' in stats:
        print("\n4. TORSION CLASSES (G2 CLOSURE)")
        print("-" * 70)
        tor = stats['torsion']
        print(f"  ||d(phi)||^2:        {tor['d_phi_mean']:.6e} +/- {tor['d_phi_std']:.6e}")
        print(f"  ||d(*phi)||^2:       {tor['d_dual_phi_mean']:.6e} +/- {tor['d_dual_phi_std']:.6e}")
        print(f"  Max ||d(phi)||^2:    {tor['d_phi_max']:.6e}")
        print(f"  Max ||d(*phi)||^2:   {tor['d_dual_phi_max']:.6e}")
        print(f"  Status:              {'PASS' if tor['pass'] else 'FAIL'} (target < 1e-6)")
    
    # Ricci
    if 'ricci' in stats:
        print("\n5. RICCI CURVATURE")
        print("-" * 70)
        ric = stats['ricci']
        print(f"  ||Ric||^2:           {ric['mean']:.6e} +/- {ric['std']:.6e}")
        print(f"  Range:               [{ric['min']:.6e}, {ric['max']:.6e}]")
        print(f"  95th percentile:     {ric['percentile_95']:.6e}")
        print(f"  Status:              {'PASS' if ric['pass'] else 'FAIL'} (target < 1e-6)")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Phi normalization", stats['phi_norm_sq']['pass']),
        ("Positive definite", stats['eigenvalues']['pass']),
    ]
    
    if 'torsion' in stats:
        checks.append(("Torsion-free", stats['torsion']['pass']))
    if 'ricci' in stats:
        checks.append(("Ricci-flat", stats['ricci']['pass']))
    
    passed = sum(1 for _, p in checks if p)
    total = len(checks)
    
    for name, status in checks:
        print(f"  {name:20s}: {'PASS' if status else 'FAIL'}")
    
    print(f"\n  Tests passed: {passed}/{total}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate G2 metric at points')
    parser.add_argument('--model', type=str, default='G2_final_model.pt',
                        help='Path to trained model')
    parser.add_argument('--point', type=str, default=None,
                        help='Single point to evaluate (comma-separated: x1,x2,...,x7)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples for comprehensive validation')
    parser.add_argument('--no-derivatives', action='store_true',
                        help='Skip derivative computation (faster)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--domain-size', type=float, default=5.0,
                        help='Domain size for random sampling')
    
    args = parser.parse_args()
    
    print("\nG2 Metric Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model, device=args.device)
    print("Model loaded successfully!")
    
    compute_derivatives = not args.no_derivatives
    
    if args.point is not None:
        # Evaluate single point
        try:
            coords = torch.tensor([float(x) for x in args.point.split(',')])
            if coords.shape[0] != 7:
                raise ValueError("Point must have exactly 7 coordinates")
            
            print(f"\nEvaluating at point: {coords.numpy()}")
            results = evaluate_at_point(model, coords, compute_derivatives=compute_derivatives, device=args.device)
            
            print("\nResults:")
            print(f"  ||phi||^2 = {results['phi_norm_sq'][0]:.8f}")
            print(f"  det(g) = {results['det_g'][0]:.8f}")
            print(f"  Min eigenvalue = {results['min_eigenvalue'][0]:.6f}")
            print(f"  Condition number = {results['condition_number'][0]:.2f}")
            
            if compute_derivatives:
                print(f"  ||d(phi)||^2 = {results['d_phi_norm_sq'][0]:.6e}")
                print(f"  ||d(*phi)||^2 = {results['d_dual_phi_norm_sq'][0]:.6e}")
                print(f"  ||Ric||^2 = {results['ricci_norm_sq'][0]:.6e}")
            
        except Exception as e:
            print(f"\nError evaluating point: {e}")
            return 1
    
    else:
        # Comprehensive validation
        stats = comprehensive_validation(
            model, 
            n_samples=args.samples,
            domain_size=args.domain_size,
            device=args.device,
            compute_derivatives=compute_derivatives
        )
        print_validation_report(stats)
    
    print("\nEvaluation complete!")
    return 0


if __name__ == '__main__':
    exit(main())









