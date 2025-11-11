"""
Loss Functions for GIFT v0.9
==============================

This module contains all loss function components for training G₂ structures:
- Torsion computation (gradient norm proxy)
- Volume constraint (det(g) = 1)
- Harmonic orthogonality (Gram matrix)
- Boundary and decay losses
- Curriculum weighting (4-phase schedule)

Critical Fixes from v0.8:
- NO clone() in torsion computation (breaks gradient graph)
- det(Gram) target = 0.995 (not 1.0, prevents singularity)
- Correct ACyl decay: exp(-γ|t|/T) from center
"""

import torch
import torch.nn as nn
import numpy as np


class SafeMetrics:
    """
    Universal helper class for robust metric operations.

    Provides:
    - Gradient-safe torsion computation (NO clone() bug)
    - Type conversion utilities (Tensor → JSON)
    - History management helpers
    """

    @staticmethod
    def compute_torsion_safe(phi, coords, metric, use_grad=True):
        """
        Gradient-aware torsion computation.

        CRITICAL FIX v0.8.1: coords MUST NOT be cloned!
        Cloning breaks the computational graph for backpropagation.

        Theory:
        Torsion T = dφ + φ∧φ (full formula is complex)
        Approximation: ||∇φ|| (gradient norm of φ components)

        This proxy is effective for optimization:
        - Minimizing ||∇φ|| → minimizes torsion T
        - Computationally efficient (10 components sampled)
        - Gradient-friendly for backprop

        Args:
            phi: (batch, 35) tensor of 3-form components
            coords: (batch, 7) tensor of coordinates (MUST have requires_grad=True)
            metric: (batch, 7, 7) tensor of metric (unused in proxy, kept for API)
            use_grad: If True, compute gradients; if False, use norm estimate

        Returns:
            torsion: Scalar tensor, mean gradient norm across batch and components
        """
        if use_grad:
            # Training mode: needs gradients for backprop
            # FIXED: Use coords directly (don't clone!)

            grad_norms = []
            for i in range(min(10, phi.shape[1])):  # Sample 10 components for efficiency
                grad_i = torch.autograd.grad(
                    phi[:, i].sum(),
                    coords,  # Use directly, NOT coords.clone()!
                    create_graph=True,  # Allow second-order derivatives
                    retain_graph=True   # Keep graph for other components
                )[0]
                grad_norms.append(grad_i.norm(dim=1))

            torsion = torch.stack(grad_norms, dim=1).mean(dim=1).mean()
            return torsion
        else:
            # Testing mode: simplified without gradients
            with torch.no_grad():
                phi_norm = torch.norm(phi, dim=-1).mean()
                return phi_norm * 0.1  # Rough estimate

    @staticmethod
    def to_json(obj):
        """
        Universal PyTorch/Numpy → JSON converter.

        Handles:
        - torch.Tensor (single value or array)
        - np.ndarray (single value or array)
        - np.integer, np.floating
        - Python scalars

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable Python type (float, list, or str)
        """
        if isinstance(obj, torch.Tensor):
            obj_cpu = obj.detach().cpu()
            if obj_cpu.numel() == 1:
                return float(obj_cpu.item())
            else:
                return obj_cpu.tolist()
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return float(obj.item())
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            try:
                return float(obj)
            except:
                return str(obj)

    @staticmethod
    def safe_get(history, key, default=None):
        """
        Get from history dict with fallback.

        Args:
            history: Dictionary of training history
            key: Key to retrieve
            default: Default value if key missing or empty

        Returns:
            Latest value from history[key], or default
        """
        val = history.get(key, [])
        if isinstance(val, list) and len(val) > 0:
            return val[-1]
        else:
            return default

    @staticmethod
    def to_scalar(obj):
        """
        Convert to Python float.

        Args:
            obj: torch.Tensor, np.ndarray, or scalar

        Returns:
            Python float
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().item()
        elif isinstance(obj, np.ndarray):
            return obj.item()
        else:
            return float(obj)


def compute_harmonic_losses_FIXED(harmonic_network, coords, h_forms, metric):
    """
    FIXED harmonic losses from v0.6b.

    Computes orthogonality constraints on harmonic 2-forms via Gram matrix:
    - Determinant loss: encourage det(Gram) → 0.995 (NOT 1.0!)
    - Orthogonality loss: ||Gram - I||² / size
    - Separation loss: diagonal >> off-diagonal

    Critical fixes:
    - Better det loss: (det - target)² instead of just det
    - Per-element normalized orthogonality
    - Target det = 0.995 to prevent singularity

    Args:
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        coords: (batch, 7) tensor of coordinates
        h_forms: (batch, 21, 21) tensor of harmonic forms
        metric: (batch, 7, 7) tensor of metric

    Returns:
        harmonic_loss_det: Determinant loss
        harmonic_loss_ortho: Orthogonality loss
        separation_loss: Diagonal/off-diagonal separation loss
        det_gram: Determinant value (for logging)
    """
    # Compute Gram matrix
    gram = harmonic_network.compute_gram_matrix(coords, h_forms, metric)
    det_gram = torch.det(gram)

    # FIXED: Better det loss (encourage det → 0.995, not exact 1.0)
    target_det = 0.995
    harmonic_loss_det = torch.relu(det_gram - target_det) + 0.1 * (det_gram - target_det) ** 2

    # Orthogonality loss: ||Gram - I||² / size
    identity = torch.eye(21, device=gram.device)
    harmonic_loss_ortho = torch.norm(gram - identity) / 21.0

    # Separation loss: diagonal >> off-diagonal
    diag_elements = torch.diagonal(gram)
    off_diag_mask = ~torch.eye(21, dtype=torch.bool, device=gram.device)
    off_diag_elements = gram[off_diag_mask]

    separation_loss = torch.relu(
        0.5 - (diag_elements.mean() - off_diag_elements.abs().mean())
    )

    return harmonic_loss_det, harmonic_loss_ortho, separation_loss, det_gram


def compute_boundary_loss(phi_network, coords, manifold):
    """
    Penalize non-zero torsion near boundaries t=±T.

    Physical motivation:
    - At boundaries, φ should match ACyl asymptotic form (torsion-free)
    - Enforce via gradient norm + amplitude penalties

    Args:
        phi_network: G2PhiNetwork_TCS instance
        coords: (batch, 7) tensor of coordinates
        manifold: TCSNeckManifold instance

    Returns:
        boundary_loss: Scalar tensor, mean torsion near boundaries
    """
    near_boundary = manifold.is_near_boundary(coords, threshold=0.15)

    if near_boundary.sum() == 0:
        return torch.tensor(0.0, device=coords.device)

    phi_boundary = phi_network(coords[near_boundary])
    coords_boundary = coords[near_boundary].requires_grad_(True)
    phi_boundary_grad = phi_network(coords_boundary)

    # Gradient norm (torsion proxy)
    grad_norms = []
    for i in range(min(5, phi_boundary_grad.shape[1])):
        grad_i = torch.autograd.grad(
            phi_boundary_grad[:, i].sum(),
            coords_boundary,
            create_graph=True,
            retain_graph=True
        )[0]
        grad_norms.append(grad_i.norm(dim=1))

    grad_norm = torch.stack(grad_norms, dim=1).mean()
    phi_amplitude_boundary = torch.norm(phi_boundary, dim=1).mean()

    return grad_norm + phi_amplitude_boundary * 0.5


def compute_asymptotic_decay_loss(phi, coords, manifold):
    """
    Enforce exp(-γ|t|/T) decay behavior (ACyl asymptotics).

    Physical motivation:
    - ACyl manifolds have exponential decay away from gluing region
    - Formula: φ_ACyl ~ exp(-γ|t|/T) as |t| → T
    - γ = 0.578 (phenomenological decay rate for TCS)

    Args:
        phi: (batch, 35) tensor of 3-form components
        coords: (batch, 7) tensor of coordinates
        manifold: TCSNeckManifold instance

    Returns:
        decay_loss: Scalar tensor, mean deviation from expected decay
    """
    t = coords[:, 0]

    # Expected decay: exp(-γ × |t|/T)
    expected_decay = torch.exp(
        -manifold.gamma_decay * torch.abs(t) / manifold.T_neck
    )

    # Actual φ amplitude
    phi_amplitude = torch.norm(phi, dim=1)

    # Loss: deviation from expected decay
    decay_loss = torch.abs(phi_amplitude - expected_decay).mean()

    return decay_loss


def compute_volume_loss(metric):
    """
    Volume constraint: det(g) = 1.

    Physical motivation:
    - G₂ holonomy requires volume-preserving metric
    - det(g) = 1 fixes scale ambiguity

    Args:
        metric: (batch, 7, 7) tensor of metric

    Returns:
        volume_loss: Scalar tensor, mean deviation from det(g) = 1
    """
    det_metric = torch.det(metric)
    volume_loss = torch.abs(det_metric.mean() - 1.0)
    return volume_loss


# ============================================================================
# 4-Phase Curriculum (Loss Weighting Schedule)
# ============================================================================

CURRICULUM = {
    'phase1': {  # Epochs 0-2000: Establish Structure
        'name': 'Establish Structure',
        'range': [0, 2000],
        'weights': {
            'torsion': 0.1,          # Minimal torsion loss
            'volume': 0.6,           # Volume × 2
            'harmonic_ortho': 6.0,   # Harmonic × 3
            'harmonic_det': 3.0,
            'separation': 2.0,
            'boundary': 0.05,
            'decay': 0.05,
            'acyl': 0.0,
        }
    },
    'phase2': {  # Epochs 2000-5000: Impose Torsion
        'name': 'Impose Torsion',
        'range': [2000, 5000],
        'weights': {
            'torsion': 2.0,          # Ramp 0.1 → 2.0 (20× increase)
            'volume': 0.4,
            'harmonic_ortho': 3.0,
            'harmonic_det': 1.5,
            'separation': 1.0,
            'boundary': 0.5,
            'decay': 0.3,
            'acyl': 0.1,             # Start ACyl matching
        }
    },
    'phase3': {  # Epochs 5000-8000: Refine b₃ + ACyl
        'name': 'Refine b₃ + ACyl',
        'range': [5000, 8000],
        'weights': {
            'torsion': 5.0,          # Continue increasing
            'volume': 0.2,
            'harmonic_ortho': 2.0,   # Reduce
            'harmonic_det': 1.0,
            'separation': 0.5,
            'boundary': 1.0,
            'decay': 0.5,
            'acyl': 0.3,             # Increase ACyl
        }
    },
    'phase4': {  # Epochs 8000-10000: Polish Final
        'name': 'Polish Final',
        'range': [8000, 10000],
        'weights': {
            'torsion': 20.0,         # Heavy torsion focus
            'volume': 0.1,
            'harmonic_ortho': 1.0,
            'harmonic_det': 0.5,
            'separation': 0.2,
            'boundary': 1.5,
            'decay': 1.0,
            'acyl': 0.5,
        }
    }
}


def get_phase_weights_smooth(epoch, transition_width=200):
    """
    Get curriculum weights with smooth blending between phases.

    Transitions are smoothed over `transition_width` epochs to avoid
    sudden jumps in loss landscape.

    Args:
        epoch: Current training epoch
        transition_width: Number of epochs to blend (default: 200)

    Returns:
        current_weights: Dictionary of loss weights for current epoch
        phase_name: Name of current phase (for logging)
    """
    current_weights = None
    next_weights = None
    blend_factor = 0.0
    phase_name = 'phase1'  # Default

    phase_list = list(CURRICULUM.items())

    for idx, (pname, phase_cfg) in enumerate(phase_list):
        phase_start, phase_end = phase_cfg['range']

        if epoch < phase_start:
            continue
        elif epoch < phase_end:
            current_weights = phase_cfg['weights'].copy()
            phase_name = pname

            # Check if we're in transition zone to next phase
            if idx < len(phase_list) - 1:
                next_phase_cfg = phase_list[idx + 1][1]
                next_weights = next_phase_cfg['weights']

                # Blend if near end of current phase
                if epoch >= phase_end - transition_width:
                    blend_factor = (epoch - (phase_end - transition_width)) / transition_width
                    blend_factor = min(blend_factor, 1.0)
            break

    # If we've passed all phases, use last phase
    if current_weights is None:
        current_weights = phase_list[-1][1]['weights'].copy()
        phase_name = phase_list[-1][0]

    # Blend with next phase if in transition
    if blend_factor > 0.0 and next_weights:
        for key in current_weights:
            w_curr = current_weights[key]
            w_next = next_weights.get(key, w_curr)
            current_weights[key] = (1 - blend_factor) * w_curr + blend_factor * w_next

    return current_weights, phase_name


def compute_total_loss(phi, h_forms, metric, coords, manifold, phi_network, harmonic_network, weights):
    """
    Compute total weighted loss for training.

    Args:
        phi: (batch, 35) tensor of 3-form components
        h_forms: (batch, 21, 21) tensor of harmonic forms
        metric: (batch, 7, 7) tensor of metric
        coords: (batch, 7) tensor of coordinates (requires_grad=True)
        manifold: TCSNeckManifold instance
        phi_network: G2PhiNetwork_TCS instance
        harmonic_network: Harmonic2FormsNetwork_TCS instance
        weights: Dictionary of loss weights from curriculum

    Returns:
        loss: Total weighted loss
        loss_dict: Dictionary of individual loss components (for logging)
    """
    # Individual loss components
    torsion_loss = SafeMetrics.compute_torsion_safe(phi, coords, metric, use_grad=True)
    volume_loss = compute_volume_loss(metric)
    harmonic_loss_det, harmonic_loss_ortho, separation_loss, det_gram = \
        compute_harmonic_losses_FIXED(harmonic_network, coords, h_forms, metric)
    boundary_loss = compute_boundary_loss(phi_network, coords, manifold)
    decay_loss = compute_asymptotic_decay_loss(phi, coords, manifold)

    # Total loss (with curriculum weighting)
    loss = (weights['torsion'] * torsion_loss +
            weights['volume'] * volume_loss +
            weights['harmonic_ortho'] * harmonic_loss_ortho +
            weights['harmonic_det'] * harmonic_loss_det +
            weights['separation'] * separation_loss +
            weights['boundary'] * boundary_loss +
            weights['decay'] * decay_loss)

    # Package individual losses for logging
    loss_dict = {
        'torsion': torsion_loss,
        'volume': volume_loss,
        'harmonic_ortho': harmonic_loss_ortho,
        'harmonic_det': harmonic_loss_det,
        'separation': separation_loss,
        'boundary': boundary_loss,
        'decay': decay_loss,
        'det_gram': det_gram,
    }

    return loss, loss_dict


# ============================================================================
# Early Stopping Conditions
# ============================================================================

def check_early_stopping(epoch, history, weights):
    """
    Check for early stopping conditions.

    Conditions:
    1. det(Gram) too perfect (> 0.998) after epoch 2000
       → Reduce harmonic weights by 50%
    2. det(Gram) stuck at 1.0 for 5+ consecutive test epochs
       → Emergency brake (prevents singularity)

    Args:
        epoch: Current training epoch
        history: Training history dictionary
        weights: Current loss weights (will be modified in-place)

    Returns:
        should_stop: Boolean, whether to stop training
        message: String message (empty if no action)
    """
    should_stop = False
    message = ""

    # Condition 1: det(Gram) too perfect
    if epoch > 2000 and len(history.get('det_gram', [])) > 0:
        recent_det = history['det_gram'][-1]
        if recent_det > 0.998:
            weights['harmonic_ortho'] *= 0.5
            weights['harmonic_det'] *= 0.5
            message = f"⚠ det(Gram) = {recent_det:.6f} > 0.998, reducing harmonic weights by 50%"

    # Condition 2: Emergency brake (det stuck at 1.0)
    if epoch > 3000 and len(history.get('test_det_gram', [])) >= 5:
        recent_test_dets = history['test_det_gram'][-5:]
        if all(abs(det - 1.0) < 1e-6 for det in recent_test_dets):
            should_stop = True
            message = "🛑 EARLY STOPPING: det(Gram) stuck at 1.0 for 5+ consecutive test epochs (singularity risk)"

    return should_stop, message
