"""
Loss functions for TCS G₂ metric reconstruction.

Implements torsion-free constraints, Gram matrix orthonormalization,
calibration conditions, and auxiliary geometric losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


def torsion_closure_loss(dphi: torch.Tensor) -> torch.Tensor:
    """
    Torsion closure constraint: dφ = 0.

    Args:
        dphi: [batch, 7, 7, 7, 7] exterior derivative of 3-form

    Returns:
        Scalar loss value
    """
    return torch.mean(dphi ** 2)


def torsion_coclosure_loss(dstar_phi: torch.Tensor) -> torch.Tensor:
    """
    Torsion coclosure constraint: d*φ = 0.

    Args:
        dstar_phi: [batch, 7, 7] co-derivative of 3-form

    Returns:
        Scalar loss value
    """
    return torch.mean(dstar_phi ** 2)


def volume_loss(metric: torch.Tensor, target_det: float = 1.0) -> torch.Tensor:
    """
    Volume constraint: det(g) ≈ target_det.

    Args:
        metric: [batch, 7, 7] metric tensor
        target_det: Target determinant value

    Returns:
        Scalar loss value
    """
    det = torch.det(metric)
    return torch.mean((det - target_det) ** 2)


def gram_matrix_loss(harmonic_forms: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Gram matrix orthonormalization loss for harmonic forms.

    Enforces:
    1. Orthonormality: G_ij ≈ δ_ij
    2. Full rank: rank(G) = target_rank
    3. det(G) ≈ 1

    Args:
        harmonic_forms: [batch, n_forms, n_components] harmonic basis
        target_rank: Expected rank (21 for b₂, 77 for b₃)

    Returns:
        loss: Total Gram loss
        det_gram: Determinant of Gram matrix
        rank: Numerical rank
    """
    batch_size, n_forms, n_components = harmonic_forms.shape

    gram = torch.zeros(n_forms, n_forms, device=harmonic_forms.device)
    for i in range(n_forms):
        for j in range(n_forms):
            inner_product = torch.mean(
                torch.sum(harmonic_forms[:, i, :] * harmonic_forms[:, j, :], dim=-1)
            )
            gram[i, j] = inner_product

    identity = torch.eye(n_forms, device=gram.device)

    loss_orthonormality = torch.mean((gram - identity) ** 2)

    det_gram = torch.det(gram + 1e-6 * identity)
    loss_determinant = (det_gram - 1.0) ** 2

    eigenvalues = torch.linalg.eigvalsh(gram)
    rank = (eigenvalues > 1e-4).sum().item()

    loss = loss_orthonormality + 0.1 * loss_determinant

    return loss, det_gram, rank


def boundary_smoothness_loss(phi: torch.Tensor, region_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Boundary smoothness between M₁, Neck, and M₂ regions.

    Penalizes discontinuities at region transitions.

    Args:
        phi: [batch, 7, 7, 7] 3-form
        region_weights: Dictionary of soft region assignments

    Returns:
        Scalar loss value
    """
    w_m1 = region_weights['m1'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    w_neck = region_weights['neck'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    w_m2 = region_weights['m2'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    transition_m1_neck = torch.mean((w_m1 * w_neck).unsqueeze(-1) * phi ** 2)
    transition_neck_m2 = torch.mean((w_neck * w_m2).unsqueeze(-1) * phi ** 2)

    return transition_m1_neck + transition_neck_m2


def calibration_associative_loss(
    phi: torch.Tensor,
    cycles: List[Dict],
    topology,
    n_samples: int = 512
) -> torch.Tensor:
    """
    Calibration constraint for associative 3-cycles: φ|_Σ = vol_Σ.

    Args:
        phi: [batch, 7, 7, 7] 3-form
        cycles: List of associative cycle definitions
        topology: K7Topology instance
        n_samples: Samples per cycle for integration

    Returns:
        Scalar loss value
    """
    total_loss = 0.0
    n_cycles = len(cycles)

    for cycle in cycles:
        samples = topology.sample_on_cycle(cycle, n_samples)
        samples = samples.to(phi.device)

        phi_on_cycle = torch.zeros(samples.shape[0], device=phi.device)

        indices = cycle['indices']
        if len(indices) == 3:
            i, j, k = indices
            phi_on_cycle = torch.abs(phi[:, i, j, k].mean())

        volume_sigma = 1.0

        loss_cycle = (phi_on_cycle - volume_sigma) ** 2
        total_loss += loss_cycle

    return total_loss / max(n_cycles, 1)


def calibration_coassociative_loss(
    star_phi: torch.Tensor,
    cycles: List[Dict],
    topology,
    n_samples: int = 512
) -> torch.Tensor:
    """
    Calibration constraint for coassociative 4-cycles: *φ|_Ω = vol_Ω.

    Args:
        star_phi: [batch, 7, 7, 7, 7] Hodge dual 4-form
        cycles: List of coassociative cycle definitions
        topology: K7Topology instance
        n_samples: Samples per cycle

    Returns:
        Scalar loss value
    """
    total_loss = 0.0
    n_cycles = len(cycles)

    for cycle in cycles:
        samples = topology.sample_on_cycle(cycle, n_samples)
        samples = samples.to(star_phi.device)

        star_phi_on_cycle = torch.zeros(samples.shape[0], device=star_phi.device)

        indices = cycle['indices']
        if len(indices) == 4:
            i, j, k, l = indices
            star_phi_on_cycle = torch.abs(star_phi[:, i, j, k, l].mean())

        volume_omega = 1.0

        loss_cycle = (star_phi_on_cycle - volume_omega) ** 2
        total_loss += loss_cycle

    return total_loss / max(n_cycles, 1)


class AdaptiveLossScheduler:
    """
    Adaptive loss weight scheduler based on training dynamics.

    Monitors torsion component stagnation and dynamically adjusts weights.
    """
    def __init__(self, check_interval: int = 100, plateau_threshold: float = 1e-4):
        self.check_interval = check_interval
        self.plateau_threshold = plateau_threshold
        self.history = {'torsion_closure': [], 'torsion_coclosure': []}
        self.weights = {'torsion_closure': 1.0, 'torsion_coclosure': 1.0}

    def update(self, epoch: int, losses: Dict[str, float]):
        """
        Update loss history and adjust weights if plateau detected.
        """
        for key in ['torsion_closure', 'torsion_coclosure']:
            if key in losses:
                self.history[key].append(losses[key])

        if epoch % self.check_interval == 0 and epoch > 500:
            for key in ['torsion_closure', 'torsion_coclosure']:
                if len(self.history[key]) >= 100:
                    recent = self.history[key][-100:]
                    variance = torch.tensor(recent).var().item()

                    if variance < self.plateau_threshold:
                        self.weights[key] *= 1.5
                        print(f"Epoch {epoch}: Boosting {key} weight to {self.weights[key]:.3f}")

    def get_weights(self) -> Dict[str, float]:
        return self.weights


class CompositeLoss(nn.Module):
    """
    Composite loss function combining all geometric constraints.
    """
    def __init__(self, topology, assoc_cycles, coassoc_cycles):
        super().__init__()
        self.topology = topology
        self.assoc_cycles = assoc_cycles
        self.coassoc_cycles = coassoc_cycles
        self.adaptive_scheduler = AdaptiveLossScheduler()

    def forward(
        self,
        phi: torch.Tensor,
        dphi: torch.Tensor,
        dstar_phi: torch.Tensor,
        star_phi: torch.Tensor,
        metric: torch.Tensor,
        harmonic_h2: torch.Tensor,
        harmonic_h3: torch.Tensor,
        region_weights: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float],
        epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and component breakdown.

        Returns:
            total_loss: Weighted sum of all components
            components: Dictionary of individual loss values
        """
        components = {}

        components['torsion_closure'] = torsion_closure_loss(dphi)
        components['torsion_coclosure'] = torsion_coclosure_loss(dstar_phi)
        components['volume'] = volume_loss(metric)

        gram_h2_loss, det_h2, rank_h2 = gram_matrix_loss(
            harmonic_h2, target_rank=21
        )
        components['gram_h2'] = gram_h2_loss
        components['det_gram_h2'] = det_h2.item()
        components['rank_h2'] = rank_h2

        gram_h3_loss, det_h3, rank_h3 = gram_matrix_loss(
            harmonic_h3, target_rank=77
        )
        components['gram_h3'] = gram_h3_loss
        components['det_gram_h3'] = det_h3.item()
        components['rank_h3'] = rank_h3

        components['boundary'] = boundary_smoothness_loss(phi, region_weights)

        if loss_weights.get('calibration', 0.0) > 0:
            components['calibration_assoc'] = calibration_associative_loss(
                phi, self.assoc_cycles, self.topology
            )
            components['calibration_coassoc'] = calibration_coassociative_loss(
                star_phi, self.coassoc_cycles, self.topology
            )
            components['calibration'] = (
                components['calibration_assoc'] + components['calibration_coassoc']
            ) / 2.0
        else:
            components['calibration'] = torch.tensor(0.0, device=phi.device)

        self.adaptive_scheduler.update(epoch, {
            'torsion_closure': components['torsion_closure'].item(),
            'torsion_coclosure': components['torsion_coclosure'].item()
        })
        adaptive_weights = self.adaptive_scheduler.get_weights()

        total_loss = (
            loss_weights.get('torsion_closure', 1.0) * adaptive_weights['torsion_closure'] * components['torsion_closure'] +
            loss_weights.get('torsion_coclosure', 1.0) * adaptive_weights['torsion_coclosure'] * components['torsion_coclosure'] +
            loss_weights.get('volume', 0.1) * components['volume'] +
            loss_weights.get('gram_h2', 1.0) * components['gram_h2'] +
            loss_weights.get('gram_h3', 1.0) * components['gram_h3'] +
            loss_weights.get('boundary', 1.0) * components['boundary'] +
            loss_weights.get('calibration', 0.0) * components['calibration']
        )

        components_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                          for k, v in components.items()}

        return total_loss, components_dict
