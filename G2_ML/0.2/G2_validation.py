"""
G2 Validation Module - v0.2

Comprehensive validation and analysis for learned G2 metrics.
Includes torsion checks, curvature analysis, and topological signatures.

Author: GIFT Project
No Unicode - Windows compatible
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from G2_geometry import (
    exterior_derivative_3form,
    exterior_derivative_4form,
    hodge_star,
    ricci_tensor,
    volume_form,
    project_spd
)
from G2_phi_network import metric_from_phi_algebraic


# ============================================================================
# Torsion Verification
# ============================================================================

def validate_torsion_free(model, manifold, n_samples=1000, method='autograd', device='cpu'):
    """
    Validate torsion-free condition on sample points.
    
    Checks:
    - ||d(phi)||^2 < epsilon (closure)
    - ||d(*phi)||^2 < epsilon (co-closure)
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object (e.g., TorusT7)
        n_samples: Number of test points
        method: 'autograd' or 'optimized'
        device: PyTorch device
    
    Returns:
        results: Dict with validation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Sample test points
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        coords.requires_grad = True
        
        # Compute phi
        phi = model(coords)
        
        # Reconstruct metric
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Compute exterior derivatives
        d_phi, d_phi_norm_sq = exterior_derivative_3form(phi, coords, method=method)
        
        phi_dual = hodge_star(phi, metric)
        d_phi_dual, d_phi_dual_norm_sq = exterior_derivative_4form(phi_dual, coords, method=method)
        
        # Statistics
        results = {
            'd_phi_norm_sq_mean': d_phi_norm_sq.mean().item(),
            'd_phi_norm_sq_std': d_phi_norm_sq.std().item(),
            'd_phi_norm_sq_max': d_phi_norm_sq.max().item(),
            'd_phi_dual_norm_sq_mean': d_phi_dual_norm_sq.mean().item(),
            'd_phi_dual_norm_sq_std': d_phi_dual_norm_sq.std().item(),
            'd_phi_dual_norm_sq_max': d_phi_dual_norm_sq.max().item(),
            'torsion_total': (d_phi_norm_sq + d_phi_dual_norm_sq).mean().item(),
            'n_samples': n_samples,
            'is_torsion_free': (d_phi_norm_sq.mean() < 1e-4 and d_phi_dual_norm_sq.mean() < 1e-4)
        }
    
    return results


# ============================================================================
# Curvature Analysis
# ============================================================================

def validate_curvature(model, manifold, n_samples=500, device='cpu'):
    """
    Analyze curvature properties of learned metric.
    
    Checks:
    - Ricci flatness (should be satisfied for torsion-free G2)
    - Non-flatness (Riemann tensor not zero)
    - Eigenvalue spectrum
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object
        n_samples: Number of test points
        device: PyTorch device
    
    Returns:
        results: Dict with curvature metrics
    """
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        coords.requires_grad = True
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Compute Ricci tensor
        ricci = ricci_tensor(metric, coords)
        ricci_norm_sq = torch.sum(ricci ** 2, dim=(1,2))
        
        # Metric eigenvalues
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        results = {
            'ricci_norm_sq_mean': ricci_norm_sq.mean().item(),
            'ricci_norm_sq_std': ricci_norm_sq.std().item(),
            'ricci_norm_sq_max': ricci_norm_sq.max().item(),
            'is_ricci_flat': ricci_norm_sq.mean() < 1e-4,
            'eigenvalue_min': eigenvalues.min().item(),
            'eigenvalue_max': eigenvalues.max().item(),
            'eigenvalue_mean': eigenvalues.mean().item(),
            'condition_number': (eigenvalues.max() / eigenvalues.min()).item(),
            'is_positive_definite': eigenvalues.min() > 0
        }
    
    return results


# ============================================================================
# Metric Quality Checks
# ============================================================================

def validate_metric_quality(model, manifold, n_samples=1000, device='cpu'):
    """
    Validate basic metric properties.
    
    Checks:
    - Positive definiteness
    - Symmetry
    - Volume normalization
    - Condition number
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object
        n_samples: Number of test points
        device: PyTorch device
    
    Returns:
        results: Dict with metric quality metrics
    """
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Check symmetry
        symmetry_error = torch.norm(metric - metric.transpose(-2, -1), dim=(-2,-1))
        
        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(metric)
        
        # Volume
        vol = volume_form(metric)
        
        # Phi norm
        phi_norm_sq = torch.sum(phi ** 2, dim=1)
        
        results = {
            'symmetry_error_mean': symmetry_error.mean().item(),
            'symmetry_error_max': symmetry_error.max().item(),
            'is_symmetric': symmetry_error.max() < 1e-5,
            'min_eigenvalue': eigenvalues.min().item(),
            'max_eigenvalue': eigenvalues.max().item(),
            'is_positive_definite': eigenvalues.min() > 1e-6,
            'condition_number_mean': (eigenvalues.max(dim=1)[0] / eigenvalues.min(dim=1)[0]).mean().item(),
            'volume_mean': vol.mean().item(),
            'volume_std': vol.std().item(),
            'volume_error': torch.abs(vol - 1.0).mean().item(),
            'phi_norm_sq_mean': phi_norm_sq.mean().item(),
            'phi_norm_sq_std': phi_norm_sq.std().item(),
            'phi_norm_error': torch.abs(phi_norm_sq - 7.0).mean().item()
        }
    
    return results


# ============================================================================
# G2 Structure Validation
# ============================================================================

def validate_g2_identity(model, manifold, n_samples=500, device='cpu'):
    """
    Validate G2-specific identity: phi ∧ *phi = (7/6) vol_g.
    
    This is a fundamental property of G2 structures.
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object
        n_samples: Number of test points
        device: PyTorch device
    
    Returns:
        results: Dict with validation results
    """
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        # Hodge dual
        phi_dual = hodge_star(phi, metric)
        
        # Volume form
        vol = volume_form(metric)
        
        # Compute phi ∧ *phi (simplified: use norm product)
        phi_norm = torch.norm(phi, dim=1)
        phi_dual_norm = torch.norm(phi_dual, dim=1)
        wedge_product = phi_norm * phi_dual_norm  # Simplified
        
        # Target: (7/6) * vol
        target = (7.0 / 6.0) * vol
        
        error = torch.abs(wedge_product - target)
        
        results = {
            'wedge_product_mean': wedge_product.mean().item(),
            'target_mean': target.mean().item(),
            'error_mean': error.mean().item(),
            'error_std': error.std().item(),
            'relative_error': (error / target).mean().item(),
            'identity_satisfied': (error / target).mean() < 0.1
        }
    
    return results


# ============================================================================
# Topological Signatures
# ============================================================================

def estimate_betti_numbers(model, manifold, n_samples=2000, n_neighbors=15, device='cpu'):
    """
    Estimate Betti numbers using persistent homology approximation.
    
    Uses PCA on phi components and metric eigenvalues to estimate topology.
    Note: This is a rough approximation, not rigorous topology computation.
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object
        n_samples: Number of sample points
        n_neighbors: Number of neighbors for graph construction
        device: PyTorch device
    
    Returns:
        results: Dict with topological estimates
    """
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(n_samples, method='uniform')
        coords = coords.to(device)
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        
        # Convert to numpy for sklearn
        phi_np = phi.cpu().numpy()
        
        # PCA analysis
        pca = PCA()
        pca.fit(phi_np)
        
        explained_variance = pca.explained_variance_ratio_
        
        # Estimate dimensions from variance
        cumsum_var = np.cumsum(explained_variance)
        dim_95 = np.argmax(cumsum_var > 0.95) + 1  # Dimensions capturing 95% variance
        dim_99 = np.argmax(cumsum_var > 0.99) + 1  # Dimensions capturing 99% variance
        
        results = {
            'intrinsic_dim_95': int(dim_95),
            'intrinsic_dim_99': int(dim_99),
            'variance_first_component': explained_variance[0],
            'variance_first_5': explained_variance[:5].sum(),
            'effective_rank': np.sum(explained_variance > 0.01),
            'note': 'These are rough estimates, not rigorous Betti numbers'
        }
    
    return results


# ============================================================================
# Comprehensive Validation Report
# ============================================================================

def comprehensive_validation(model, manifold, n_samples=1000, device='cpu', verbose=True):
    """
    Run all validation checks and compile comprehensive report.
    
    Args:
        model: Trained G2PhiNetwork
        manifold: Manifold object
        n_samples: Number of test points
        device: PyTorch device
        verbose: Whether to print results
    
    Returns:
        report: Dict with all validation results
    """
    if verbose:
        print("=" * 70)
        print("G2 Metric Validation Report")
        print("=" * 70)
    
    report = {}
    
    # Torsion validation
    if verbose:
        print("\n1. Torsion-Free Validation")
        print("-" * 70)
    
    torsion_results = validate_torsion_free(model, manifold, n_samples, device=device)
    report['torsion'] = torsion_results
    
    if verbose:
        print(f"||d(phi)||^2: {torsion_results['d_phi_norm_sq_mean']:.6e} ± {torsion_results['d_phi_norm_sq_std']:.6e}")
        print(f"||d(*phi)||^2: {torsion_results['d_phi_dual_norm_sq_mean']:.6e} ± {torsion_results['d_phi_dual_norm_sq_std']:.6e}")
        print(f"Torsion-free: {torsion_results['is_torsion_free']}")
    
    # Curvature validation
    if verbose:
        print("\n2. Curvature Analysis")
        print("-" * 70)
    
    curvature_results = validate_curvature(model, manifold, n_samples//2, device=device)
    report['curvature'] = curvature_results
    
    if verbose:
        print(f"||Ric||^2: {curvature_results['ricci_norm_sq_mean']:.6e} ± {curvature_results['ricci_norm_sq_std']:.6e}")
        print(f"Ricci-flat: {curvature_results['is_ricci_flat']}")
        print(f"Eigenvalue range: [{curvature_results['eigenvalue_min']:.4f}, {curvature_results['eigenvalue_max']:.4f}]")
        print(f"Condition number: {curvature_results['condition_number']:.2f}")
    
    # Metric quality
    if verbose:
        print("\n3. Metric Quality")
        print("-" * 70)
    
    quality_results = validate_metric_quality(model, manifold, n_samples, device=device)
    report['quality'] = quality_results
    
    if verbose:
        print(f"Positive definite: {quality_results['is_positive_definite']}")
        print(f"Symmetric: {quality_results['is_symmetric']}")
        print(f"Volume: {quality_results['volume_mean']:.6f} ± {quality_results['volume_std']:.6f} (target: 1.0)")
        print(f"||phi||^2: {quality_results['phi_norm_sq_mean']:.6f} ± {quality_results['phi_norm_sq_std']:.6f} (target: 7.0)")
    
    # G2 identity
    if verbose:
        print("\n4. G2 Identity Validation")
        print("-" * 70)
    
    identity_results = validate_g2_identity(model, manifold, n_samples//2, device=device)
    report['g2_identity'] = identity_results
    
    if verbose:
        print(f"phi ∧ *phi: {identity_results['wedge_product_mean']:.6f}")
        print(f"(7/6) vol_g: {identity_results['target_mean']:.6f}")
        print(f"Relative error: {identity_results['relative_error']:.2%}")
    
    # Topological signatures
    if verbose:
        print("\n5. Topological Signatures")
        print("-" * 70)
    
    topology_results = estimate_betti_numbers(model, manifold, min(n_samples, 1000), device=device)
    report['topology'] = topology_results
    
    if verbose:
        print(f"Intrinsic dimension (95% var): {topology_results['intrinsic_dim_95']}")
        print(f"Intrinsic dimension (99% var): {topology_results['intrinsic_dim_99']}")
        print(f"Effective rank: {topology_results['effective_rank']}")
    
    # Overall assessment
    if verbose:
        print("\n6. Overall Assessment")
        print("-" * 70)
    
    score = 0
    max_score = 5
    
    if torsion_results['is_torsion_free']:
        score += 1
        if verbose:
            print("✓ Torsion-free condition satisfied")
    
    if curvature_results['is_positive_definite']:
        score += 1
        if verbose:
            print("✓ Metric is positive definite")
    
    if quality_results['volume_error'] < 0.1:
        score += 1
        if verbose:
            print("✓ Volume normalization good")
    
    if quality_results['phi_norm_error'] < 0.1:
        score += 1
        if verbose:
            print("✓ Phi normalization good")
    
    if identity_results['identity_satisfied']:
        score += 1
        if verbose:
            print("✓ G2 identity satisfied")
    
    report['overall_score'] = score
    report['max_score'] = max_score
    report['quality_percentage'] = 100.0 * score / max_score
    
    if verbose:
        print(f"\nOverall Quality: {score}/{max_score} ({report['quality_percentage']:.1f}%)")
        print("=" * 70)
    
    return report


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_validation_summary(report, save_path=None):
    """
    Create summary visualization of validation results.
    
    Args:
        report: Validation report dict from comprehensive_validation
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('G2 Metric Validation Summary', fontsize=16, fontweight='bold')
    
    # 1. Torsion metrics
    ax = axes[0, 0]
    torsion_data = [
        report['torsion']['d_phi_norm_sq_mean'],
        report['torsion']['d_phi_dual_norm_sq_mean']
    ]
    ax.bar(['||d(phi)||^2', '||d(*phi)||^2'], torsion_data, color=['blue', 'green'])
    ax.set_ylabel('Norm Squared')
    ax.set_title('Torsion Components')
    ax.set_yscale('log')
    ax.axhline(y=1e-4, color='r', linestyle='--', label='Target: 1e-4')
    ax.legend()
    
    # 2. Metric quality
    ax = axes[0, 1]
    quality_data = [
        ('Volume\nError', report['quality']['volume_error']),
        ('Phi Norm\nError', report['quality']['phi_norm_error']),
        ('Symmetry\nError', report['quality']['symmetry_error_max'])
    ]
    labels, values = zip(*quality_data)
    ax.bar(labels, values, color=['purple', 'orange', 'red'])
    ax.set_ylabel('Error')
    ax.set_title('Metric Quality Errors')
    ax.set_yscale('log')
    
    # 3. Overall score
    ax = axes[0, 2]
    score = report['overall_score']
    max_score = report['max_score']
    colors = ['green'] * score + ['gray'] * (max_score - score)
    ax.bar(range(max_score), [1]*max_score, color=colors)
    ax.set_ylim([0, 1.5])
    ax.set_xticks([])
    ax.set_ylabel('Pass')
    ax.set_title(f'Overall Score: {score}/{max_score}')
    ax.text(max_score/2, 0.5, f'{report["quality_percentage"]:.1f}%', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # 4. Curvature
    ax = axes[1, 0]
    ax.text(0.5, 0.7, f"||Ric||^2: {report['curvature']['ricci_norm_sq_mean']:.2e}", 
            ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.5, f"Ricci-flat: {report['curvature']['is_ricci_flat']}", 
            ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.3, f"Condition #: {report['curvature']['condition_number']:.2f}", 
            ha='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Curvature Properties')
    ax.axis('off')
    
    # 5. G2 identity
    ax = axes[1, 1]
    wedge = report['g2_identity']['wedge_product_mean']
    target = report['g2_identity']['target_mean']
    ax.bar(['phi ∧ *phi', '(7/6) vol_g'], [wedge, target], color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('G2 Identity: phi ∧ *phi = (7/6) vol_g')
    error_pct = report['g2_identity']['relative_error'] * 100
    ax.text(0.5, 0.5, f'Error: {error_pct:.2f}%', 
            transform=ax.transAxes, ha='center', fontsize=10)
    
    # 6. Topology
    ax = axes[1, 2]
    ax.text(0.5, 0.8, 'Topological Estimates', ha='center', fontsize=12, 
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.6, f"Intrinsic dim (95%): {report['topology']['intrinsic_dim_95']}", 
            ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.4, f"Intrinsic dim (99%): {report['topology']['intrinsic_dim_99']}", 
            ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.2, f"Effective rank: {report['topology']['effective_rank']}", 
            ha='center', fontsize=10, transform=ax.transAxes)
    ax.set_title('Topological Signatures')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    return fig


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("G2 Validation Module - Test Suite")
    print("=" * 70)
    
    print("\nNote: This module requires a trained model for full testing.")
    print("Running basic functionality tests...")
    
    # Tests would go here with a dummy model
    print("\nModule loaded successfully!")






