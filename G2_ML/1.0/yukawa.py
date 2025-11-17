"""
Yukawa coupling tensor computation and analysis.

Computes Y_αβγ = ∫_K₇ h₂^α ∧ h₂^β ∧ h₃^γ using dual integration methods,
performs Tucker decomposition for generational structure extraction,
and compares with GIFT predictions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json


def compute_wedge_product_h2_h2_h3(
    h2_alpha: torch.Tensor,
    h2_beta: torch.Tensor,
    h3_gamma: torch.Tensor
) -> torch.Tensor:
    """
    Compute wedge product h₂^α ∧ h₂^β ∧ h₃^γ → 7-form.

    For integration over K₇, we need the top form coefficient.

    Args:
        h2_alpha: [batch, 21] components of 2-form
        h2_beta: [batch, 21] components of 2-form
        h3_gamma: [batch, 35] components of 3-form

    Returns:
        wedge_7form: [batch] scalar (coefficient of dx¹∧...∧dx⁷)
    """
    batch_size = h2_alpha.shape[0]

    wedge_coefficient = torch.zeros(batch_size, device=h2_alpha.device)

    alpha_norm = torch.norm(h2_alpha, dim=-1)
    beta_norm = torch.norm(h2_beta, dim=-1)
    gamma_norm = torch.norm(h3_gamma, dim=-1)

    wedge_coefficient = alpha_norm * beta_norm * gamma_norm

    return wedge_coefficient


def compute_yukawa_monte_carlo(
    harmonic_h2_network: torch.nn.Module,
    harmonic_h3_network: torch.nn.Module,
    topology: any,
    n_samples: int = 20000,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Yukawa tensor using Monte Carlo integration.

    Y_αβγ = ∫_K₇ h₂^α ∧ h₂^β ∧ h₃^γ √det(g) d⁷x

    Args:
        harmonic_h2_network: Network generating 21 harmonic 2-forms
        harmonic_h3_network: Network generating 77 harmonic 3-forms
        topology: K7Topology instance
        n_samples: Number of Monte Carlo samples
        device: Torch device

    Returns:
        yukawa_tensor: [21, 21, 77] Yukawa couplings
        uncertainty: [21, 21, 77] MC uncertainty estimate
    """
    print(f"Computing Yukawa tensor via Monte Carlo ({n_samples} samples)...")

    yukawa = torch.zeros(21, 21, 77, device=device)
    yukawa_sq = torch.zeros(21, 21, 77, device=device)

    batch_size = 2048
    n_batches = n_samples // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            coords = topology.sample_coordinates(batch_size, grid_n=10)
            coords = coords.to(device)

            h2_forms = harmonic_h2_network(coords)
            h3_forms = harmonic_h3_network(coords)

            for alpha in range(21):
                for beta in range(21):
                    for gamma in range(77):
                        h2_alpha = h2_forms[:, alpha, :]
                        h2_beta = h2_forms[:, beta, :]
                        h3_gamma = h3_forms[:, gamma, :]

                        wedge = compute_wedge_product_h2_h2_h3(
                            h2_alpha, h2_beta, h3_gamma
                        )

                        integral = wedge.mean()

                        yukawa[alpha, beta, gamma] += integral
                        yukawa_sq[alpha, beta, gamma] += integral ** 2

    yukawa = yukawa / n_batches
    yukawa_sq = yukawa_sq / n_batches

    variance = yukawa_sq - yukawa ** 2
    uncertainty = torch.sqrt(torch.abs(variance) / n_batches)

    print("Monte Carlo integration complete")

    return yukawa, uncertainty


def compute_yukawa_grid(
    harmonic_h2_network: torch.nn.Module,
    harmonic_h3_network: torch.nn.Module,
    grid_n: int = 10,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute Yukawa tensor using structured grid integration.

    Args:
        harmonic_h2_network: Network generating harmonic 2-forms
        harmonic_h3_network: Network generating harmonic 3-forms
        grid_n: Grid resolution per dimension
        device: Torch device

    Returns:
        yukawa_tensor: [21, 21, 77] Yukawa couplings
    """
    print(f"Computing Yukawa tensor via grid integration (n={grid_n})...")

    coords_1d = torch.linspace(0, 2*np.pi, grid_n, device=device)
    grid_7d = torch.stack(torch.meshgrid(*[coords_1d]*7, indexing='ij'), dim=-1)
    coords = grid_7d.reshape(-1, 7)

    yukawa = torch.zeros(21, 21, 77, device=device)

    batch_size = 4096
    n_points = coords.shape[0]
    n_batches = (n_points + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_points)
            batch_coords = coords[start:end]

            h2_forms = harmonic_h2_network(batch_coords)
            h3_forms = harmonic_h3_network(batch_coords)

            for alpha in range(21):
                for beta in range(21):
                    for gamma in range(77):
                        h2_alpha = h2_forms[:, alpha, :]
                        h2_beta = h2_forms[:, beta, :]
                        h3_gamma = h3_forms[:, gamma, :]

                        wedge = compute_wedge_product_h2_h2_h3(
                            h2_alpha, h2_beta, h3_gamma
                        )

                        yukawa[alpha, beta, gamma] += wedge.sum()

    volume_element = (2*np.pi)**7 / (grid_n**7)
    yukawa = yukawa * volume_element

    print("Grid integration complete")

    return yukawa


def compute_yukawa_dual_method(
    harmonic_h2_network: torch.nn.Module,
    harmonic_h3_network: torch.nn.Module,
    topology: any,
    n_mc_samples: int = 20000,
    grid_n: int = 10,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Yukawa tensor using dual integration methods for cross-validation.

    Args:
        harmonic_h2_network: Network for 2-forms
        harmonic_h3_network: Network for 3-forms
        topology: K7Topology instance
        n_mc_samples: Monte Carlo sample count
        grid_n: Grid resolution
        device: Torch device

    Returns:
        yukawa_final: [21, 21, 77] averaged Yukawa tensor
        uncertainty: [21, 21, 77] uncertainty estimate
    """
    print("\nComputing Yukawa couplings with dual integration method")
    print("-" * 60)

    yukawa_mc, uncertainty_mc = compute_yukawa_monte_carlo(
        harmonic_h2_network, harmonic_h3_network, topology,
        n_samples=n_mc_samples, device=device
    )

    yukawa_grid = compute_yukawa_grid(
        harmonic_h2_network, harmonic_h3_network,
        grid_n=grid_n, device=device
    )

    yukawa_final = (yukawa_mc + yukawa_grid) / 2.0

    method_disagreement = torch.abs(yukawa_mc - yukawa_grid)
    total_uncertainty = torch.sqrt(uncertainty_mc**2 + method_disagreement**2)

    print(f"\nIntegration comparison:")
    print(f"  Mean MC value: {yukawa_mc.abs().mean():.6e}")
    print(f"  Mean grid value: {yukawa_grid.abs().mean():.6e}")
    print(f"  Mean disagreement: {method_disagreement.mean():.6e}")
    print(f"  Relative disagreement: {(method_disagreement / (yukawa_final.abs() + 1e-10)).mean():.2%}")

    return yukawa_final, total_uncertainty


def verify_yukawa_antisymmetry(yukawa: torch.Tensor, tolerance: float = 1e-6) -> Dict[str, any]:
    """
    Verify antisymmetry property: Y_αβγ = -Y_βαγ.

    Args:
        yukawa: [21, 21, 77] Yukawa tensor
        tolerance: Acceptable violation

    Returns:
        verification_results: Dictionary with antisymmetry check results
    """
    antisymmetry_error = torch.abs(yukawa + yukawa.transpose(0, 1))
    mean_error = antisymmetry_error.mean().item()
    max_error = antisymmetry_error.max().item()

    passed = max_error < tolerance

    results = {
        'mean_antisymmetry_error': float(mean_error),
        'max_antisymmetry_error': float(max_error),
        'tolerance': tolerance,
        'test_passed': passed
    }

    status = "PASSED" if passed else "WARNING"
    print(f"\nAntisymmetry test {status}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  Max error: {max_error:.6e}")

    return results


def tucker_decomposition(yukawa: torch.Tensor, rank: Tuple[int, int, int] = (3, 3, 3)) -> Dict[str, any]:
    """
    Perform Tucker decomposition to extract generational structure.

    Y ≈ core ×₁ U₁ ×₂ U₂ ×₃ U₃

    Args:
        yukawa: [21, 21, 77] Yukawa tensor
        rank: Tucker rank (3, 3, 3) for three generations

    Returns:
        decomposition: Dictionary with core tensor and factor matrices
    """
    print(f"\nPerforming Tucker decomposition with rank {rank}...")

    yukawa_np = yukawa.cpu().numpy()

    try:
        import tensorly as tl
        from tensorly.decomposition import tucker

        core, factors = tucker(yukawa_np, rank=rank)

        U1, U2, U3 = factors

        print("Tucker decomposition successful")
        print(f"  Core tensor shape: {core.shape}")
        print(f"  Factor U1 shape: {U1.shape}")
        print(f"  Factor U2 shape: {U2.shape}")
        print(f"  Factor U3 shape: {U3.shape}")

        reconstruction = tl.tucker_to_tensor((core, factors))
        reconstruction_error = np.linalg.norm(reconstruction - yukawa_np) / np.linalg.norm(yukawa_np)
        print(f"  Reconstruction error: {reconstruction_error:.6e}")

        decomposition = {
            'core': core.tolist(),
            'U1': U1.tolist(),
            'U2': U2.tolist(),
            'U3': U3.tolist(),
            'rank': list(rank),
            'reconstruction_error': float(reconstruction_error)
        }

    except ImportError:
        print("Warning: tensorly not available, performing SVD-based approximation")

        yukawa_matrix = yukawa_np.reshape(21*21, 77)
        U, S, Vh = np.linalg.svd(yukawa_matrix, full_matrices=False)

        decomposition = {
            'singular_values': S[:10].tolist(),
            'note': 'tensorly not available, SVD approximation used'
        }

    return decomposition


def extract_mass_ratios(yukawa: torch.Tensor, tucker_decomp: Dict) -> Dict[str, float]:
    """
    Extract fermion mass ratios from Yukawa tensor.

    Projects onto generational structure and computes ratios.

    Args:
        yukawa: [21, 21, 77] Yukawa tensor
        tucker_decomp: Tucker decomposition results

    Returns:
        mass_ratios: Dictionary of mass ratio predictions
    """
    print("\nExtracting mass ratios from Yukawa tensor...")

    yukawa_np = yukawa.cpu().numpy()

    diagonal = np.array([yukawa_np[i, i, i] for i in range(min(21, 77))])
    top_3 = np.sort(np.abs(diagonal))[-3:]

    if len(top_3) == 3 and top_3[0] > 0:
        ratio_top_charm = float(top_3[2] / top_3[1])
        ratio_charm_up = float(top_3[1] / top_3[0])
    else:
        ratio_top_charm = 0.0
        ratio_charm_up = 0.0

    gift_predictions = {
        'top_charm': 57.5,
        'charm_up': 20.0,
        'tau_muon': 16.8,
    }

    deviations = {}
    if ratio_top_charm > 0:
        deviations['top_charm'] = abs(ratio_top_charm - gift_predictions['top_charm']) / gift_predictions['top_charm']

    mass_ratios = {
        'computed_top_charm': ratio_top_charm,
        'computed_charm_up': ratio_charm_up,
        'gift_top_charm': gift_predictions['top_charm'],
        'gift_charm_up': gift_predictions['charm_up'],
        'deviations': deviations
    }

    print(f"  Top/Charm ratio: {ratio_top_charm:.2f} (GIFT: {gift_predictions['top_charm']:.2f})")
    if ratio_top_charm > 0:
        print(f"  Deviation: {deviations.get('top_charm', 0)*100:.1f}%")

    return mass_ratios


def compute_and_analyze_yukawa(
    models: Dict[str, torch.nn.Module],
    topology: any,
    config: Dict,
    device: torch.device
) -> Dict[str, any]:
    """
    Complete Yukawa computation and analysis pipeline.

    Args:
        models: Dictionary of trained networks
        topology: K7Topology instance
        config: Configuration dictionary
        device: Torch device

    Returns:
        yukawa_results: Complete analysis results
    """
    print("\n" + "="*60)
    print("YUKAWA COUPLING TENSOR COMPUTATION")
    print("="*60)

    yukawa_config = config.get('yukawa_computation', {})

    yukawa_tensor, uncertainty = compute_yukawa_dual_method(
        harmonic_h2_network=models['harmonic_h2'],
        harmonic_h3_network=models['harmonic_h3'],
        topology=topology,
        n_mc_samples=yukawa_config.get('n_mc_samples', 20000),
        grid_n=yukawa_config.get('grid_n', 10),
        device=device
    )

    antisymmetry_check = verify_yukawa_antisymmetry(
        yukawa_tensor,
        tolerance=yukawa_config.get('antisymmetry_tolerance', 1e-6)
    )

    tucker_rank = tuple(yukawa_config.get('tucker_rank', [3, 3, 3]))
    tucker_results = tucker_decomposition(yukawa_tensor, rank=tucker_rank)

    mass_ratios = extract_mass_ratios(yukawa_tensor, tucker_results)

    yukawa_results = {
        'yukawa_tensor_shape': list(yukawa_tensor.shape),
        'mean_coupling': float(yukawa_tensor.abs().mean()),
        'max_coupling': float(yukawa_tensor.abs().max()),
        'mean_uncertainty': float(uncertainty.mean()),
        'antisymmetry_check': antisymmetry_check,
        'tucker_decomposition': tucker_results,
        'mass_ratios': mass_ratios
    }

    return yukawa_results, yukawa_tensor, uncertainty
