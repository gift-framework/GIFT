"""
Unit tests for G2_ML/2_1 loss module.

Tests the variational loss functions:
- TorsionFunctional
- VariationalLoss
- PhasedLossManager
- Loss logging utilities

This tests the CURRENT production code, not the archived 0.2 version.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add G2_ML/2_1 to path (active production code)
sys.path.insert(0, str(Path(__file__).parent.parent / "2_1"))

try:
    from loss import (
        TorsionFunctional,
        VariationalLoss,
        PhasedLossManager,
        format_loss_dict,
        log_constraints,
    )
    from config import GIFTConfig, default_config
    from g2_geometry import standard_phi_coefficients, MetricFromPhi
    G2_LOSS_AVAILABLE = True
except ImportError as e:
    G2_LOSS_AVAILABLE = False
    G2_LOSS_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not G2_LOSS_AVAILABLE,
    reason=f"G2 v2.1 loss not available: {G2_LOSS_IMPORT_ERROR if not G2_LOSS_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 16


@pytest.fixture
def device():
    """Get appropriate device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def config():
    """Get default GIFT configuration."""
    return default_config


@pytest.fixture
def sample_phi(batch_size, device):
    """Create sample phi near standard G2."""
    phi_0 = standard_phi_coefficients()
    phi = phi_0.unsqueeze(0).repeat(batch_size, 1).to(device)
    phi = phi + torch.randn_like(phi) * 0.01  # Small perturbation
    return phi


@pytest.fixture
def sample_coords(batch_size, device):
    """Create sample coordinates with gradients."""
    return torch.rand(batch_size, 7, device=device, requires_grad=True) * 2 * np.pi


@pytest.fixture
def sample_metric(batch_size, device):
    """Create sample positive definite metric."""
    metric = torch.eye(7, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    # Add small perturbation while keeping positive definite
    perturb = torch.randn(batch_size, 7, 7, device=device) * 0.01
    perturb = 0.5 * (perturb + perturb.transpose(-2, -1))  # Symmetrize
    return metric + perturb


def identity_phi_fn(x):
    """Simple phi function that returns standard phi."""
    phi_0 = standard_phi_coefficients().to(x.device)
    return phi_0.unsqueeze(0).expand(x.shape[0], -1)


# =============================================================================
# TorsionFunctional Tests
# =============================================================================

class TestTorsionFunctional:
    """Test torsion functional F[phi]."""

    def test_output_shape(self, sample_phi, sample_coords, sample_metric, config, device):
        """Test torsion functional output shape."""
        torsion_fn = TorsionFunctional(config).to(device)

        F = torsion_fn(sample_phi, sample_coords, identity_phi_fn, sample_metric)

        assert F.shape == (sample_phi.shape[0],)

    def test_output_positive(self, sample_phi, sample_coords, sample_metric, config, device):
        """Test torsion functional is non-negative."""
        torsion_fn = TorsionFunctional(config).to(device)

        F = torsion_fn(sample_phi, sample_coords, identity_phi_fn, sample_metric)

        assert torch.all(F >= 0)

    def test_output_finite(self, sample_phi, sample_coords, sample_metric, config, device):
        """Test torsion functional produces finite values."""
        torsion_fn = TorsionFunctional(config).to(device)

        F = torsion_fn(sample_phi, sample_coords, identity_phi_fn, sample_metric)

        assert not torch.any(torch.isnan(F))
        assert not torch.any(torch.isinf(F))

    def test_torsion_norm(self, sample_phi, sample_coords, sample_metric, config, device):
        """Test torsion norm computation."""
        torsion_fn = TorsionFunctional(config).to(device)

        T_norm = torsion_fn.torsion_norm(sample_phi, sample_coords, identity_phi_fn, sample_metric)

        assert T_norm.shape == (sample_phi.shape[0],)
        assert torch.all(T_norm >= 0)

    def test_exterior_derivative_norm(self, sample_phi, sample_coords, config, device):
        """Test exterior derivative norm computation."""
        torsion_fn = TorsionFunctional(config).to(device)

        d_phi_sq = torsion_fn.exterior_derivative_norm(sample_phi, sample_coords, identity_phi_fn)

        assert d_phi_sq.shape == (sample_phi.shape[0],)
        assert torch.all(d_phi_sq >= 0)

    def test_codifferential_norm(self, sample_phi, sample_coords, sample_metric, config, device):
        """Test codifferential norm computation."""
        torsion_fn = TorsionFunctional(config).to(device)

        d_star_sq = torsion_fn.codifferential_norm(
            sample_phi, sample_coords, identity_phi_fn, sample_metric
        )

        assert d_star_sq.shape == (sample_phi.shape[0],)
        assert not torch.any(torch.isnan(d_star_sq))


# =============================================================================
# VariationalLoss Tests
# =============================================================================

class TestVariationalLoss:
    """Test combined variational loss."""

    def test_output_format(self, sample_phi, sample_coords, config, device):
        """Test loss returns (tensor, dict)."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}

        total, losses = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights)

        assert isinstance(total, torch.Tensor)
        assert isinstance(losses, dict)

    def test_total_is_scalar(self, sample_phi, sample_coords, config, device):
        """Test total loss is scalar."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}

        total, _ = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights)

        assert total.dim() == 0

    def test_total_positive(self, sample_phi, sample_coords, config, device):
        """Test total loss is non-negative."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}

        total, _ = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights)

        assert total >= 0

    def test_loss_dict_keys(self, sample_phi, sample_coords, config, device):
        """Test loss dictionary contains expected keys."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}

        _, losses = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights)

        assert 'torsion' in losses
        assert 'det' in losses
        assert 'positivity' in losses
        assert 'total' in losses
        assert 'torsion_value' in losses
        assert 'det_value' in losses
        assert 'min_eigenvalue' in losses

    def test_weight_zero_excludes_term(self, sample_phi, sample_coords, config, device):
        """Test that zero weight excludes loss term."""
        loss_fn = VariationalLoss(config).to(device)

        # With torsion weight
        weights_with = {'torsion': 1.0, 'det': 0.0, 'positivity': 0.0}
        total_with, _ = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights_with)

        # Without torsion weight
        weights_without = {'det': 0.0, 'positivity': 0.0}
        total_without, _ = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights_without)

        # total_without should be 0 (only components with non-zero weights)
        assert total_without.item() == 0.0

    def test_differentiable(self, config, device):
        """Test loss is differentiable."""
        loss_fn = VariationalLoss(config).to(device)
        phi = torch.randn(8, 35, device=device, requires_grad=True)
        coords = torch.rand(8, 7, device=device, requires_grad=True)

        def phi_fn(x):
            return phi.expand(x.shape[0], -1)

        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}
        total, _ = loss_fn(phi, coords, phi_fn, weights)

        total.backward()

        assert phi.grad is not None
        assert not torch.any(torch.isnan(phi.grad))

    def test_cohomology_loss_with_h2(self, sample_phi, sample_coords, config, device):
        """Test cohomology loss with H2 forms."""
        loss_fn = VariationalLoss(config).to(device)
        omega = torch.randn(sample_phi.shape[0], config.b2_K7, config.n_2form_components, device=device)
        weights = {'cohomology': 1.0}

        _, losses = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights, omega=omega)

        assert 'h2' in losses

    def test_cohomology_loss_with_h3(self, sample_phi, sample_coords, config, device):
        """Test cohomology loss with H3 forms."""
        loss_fn = VariationalLoss(config).to(device)
        Phi = torch.randn(sample_phi.shape[0], config.b3_K7, config.n_phi_components, device=device)
        weights = {'cohomology': 1.0}

        _, losses = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights, Phi=Phi)

        assert 'h3' in losses


# =============================================================================
# PhasedLossManager Tests
# =============================================================================

class TestPhasedLossManager:
    """Test phased loss manager."""

    def test_initial_state(self, config):
        """Test initial phase is 0."""
        manager = PhasedLossManager(config)

        assert manager.current_phase == 0
        assert manager.epoch_in_phase == 0

    def test_get_weights_returns_dict(self, config):
        """Test weights return type."""
        manager = PhasedLossManager(config)

        weights = manager.get_weights()

        assert isinstance(weights, dict)

    def test_get_phase_name(self, config):
        """Test phase name retrieval."""
        manager = PhasedLossManager(config)

        name = manager.get_phase_name()

        assert isinstance(name, str)
        assert len(name) > 0

    def test_step_increments_epoch(self, config):
        """Test step increments epoch counter."""
        manager = PhasedLossManager(config)
        initial_epoch = manager.epoch_in_phase

        manager.step()

        assert manager.epoch_in_phase == initial_epoch + 1

    def test_step_advances_phase(self, config):
        """Test step advances phase when epochs exhausted."""
        manager = PhasedLossManager(config)
        epochs_in_phase = manager.get_phase_epochs()

        # Step through first phase
        for _ in range(epochs_in_phase):
            manager.step()

        assert manager.current_phase == 1
        assert manager.epoch_in_phase == 0

    def test_step_returns_phase_change(self, config):
        """Test step returns True when phase changes."""
        manager = PhasedLossManager(config)
        epochs_in_phase = manager.get_phase_epochs()

        # Most steps should return False
        for _ in range(epochs_in_phase - 1):
            changed = manager.step()
            assert not changed

        # Last step should trigger phase change
        changed = manager.step()
        assert changed

    def test_reset(self, config):
        """Test reset returns to initial state."""
        manager = PhasedLossManager(config)

        # Advance some
        for _ in range(10):
            manager.step()

        manager.reset()

        assert manager.current_phase == 0
        assert manager.epoch_in_phase == 0

    def test_total_epochs(self, config):
        """Test total epochs calculation."""
        manager = PhasedLossManager(config)

        total = manager.total_epochs

        assert isinstance(total, int)
        assert total > 0


# =============================================================================
# Loss Utilities Tests
# =============================================================================

class TestLossUtilities:
    """Test loss formatting utilities."""

    def test_format_loss_dict_basic(self):
        """Test basic loss formatting."""
        losses = {'total': 1.234, 'torsion': 0.5, 'det': 0.3}

        formatted = format_loss_dict(losses)

        assert isinstance(formatted, str)
        assert 'L=' in formatted
        assert 'L_torsion' in formatted
        assert 'L_det' in formatted

    def test_format_loss_dict_with_values(self):
        """Test formatting with _value entries."""
        losses = {
            'total': 1.0,
            'torsion': 0.5,
            'torsion_value': 0.016,
            'det': 0.3,
            'det_value': 2.03,
        }

        formatted = format_loss_dict(losses)

        assert 'torsion=' in formatted  # From torsion_value
        assert 'det=' in formatted  # From det_value

    def test_format_loss_dict_precision(self):
        """Test precision parameter."""
        losses = {'total': 1.23456789}

        formatted_2 = format_loss_dict(losses, precision=2)
        formatted_6 = format_loss_dict(losses, precision=6)

        # Different precision should give different string lengths
        assert len(formatted_2) < len(formatted_6)

    def test_log_constraints_basic(self, config):
        """Test basic constraint logging."""
        losses = {
            'det_value': 2.03125,  # Exactly 65/32
            'torsion_value': 0.01639,  # Close to 1/61
            'min_eigenvalue': 0.5,
        }

        log = log_constraints(losses, config)

        assert isinstance(log, str)
        assert 'det(g)' in log
        assert 'kappa_T' in log
        assert 'min(eigenvalue)' in log

    def test_log_constraints_status(self, config):
        """Test constraint status in log."""
        # Good values
        losses_good = {
            'det_value': config.det_g_target,
            'torsion_value': config.kappa_T,
            'min_eigenvalue': 0.5,
        }
        log_good = log_constraints(losses_good, config)
        assert 'OK' in log_good

        # Bad values
        losses_bad = {
            'det_value': 0.0,  # Very wrong
            'torsion_value': 10.0,  # Very wrong
            'min_eigenvalue': -0.5,  # Negative
        }
        log_bad = log_constraints(losses_bad, config)
        assert 'FAIL' in log_bad


# =============================================================================
# Edge Cases
# =============================================================================

class TestLossEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_phi(self, sample_coords, config, device):
        """Test with very small phi values."""
        loss_fn = VariationalLoss(config).to(device)
        phi = torch.ones(8, 35, device=device) * 1e-8
        weights = {'det': 1.0, 'positivity': 1.0}

        def phi_fn(x):
            return phi.expand(x.shape[0], -1)

        total, losses = loss_fn(phi, sample_coords, phi_fn, weights)

        assert not torch.isnan(total)
        assert not torch.isinf(total)

    def test_large_phi(self, sample_coords, config, device):
        """Test with large phi values."""
        loss_fn = VariationalLoss(config).to(device)
        phi = torch.ones(8, 35, device=device) * 100
        weights = {'det': 1.0, 'positivity': 1.0}

        def phi_fn(x):
            return phi.expand(x.shape[0], -1)

        total, losses = loss_fn(phi, sample_coords, phi_fn, weights)

        assert not torch.isnan(total)
        assert not torch.isinf(total)

    def test_empty_weights(self, sample_phi, sample_coords, config, device):
        """Test with empty weights dictionary."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {}

        total, losses = loss_fn(sample_phi, sample_coords, identity_phi_fn, weights)

        assert total.item() == 0.0

    def test_batch_size_one(self, config, device):
        """Test with batch size 1."""
        loss_fn = VariationalLoss(config).to(device)
        phi = standard_phi_coefficients().unsqueeze(0).to(device)
        coords = torch.rand(1, 7, device=device, requires_grad=True)
        weights = {'torsion': 1.0, 'det': 1.0}

        def phi_fn(x):
            return phi.expand(x.shape[0], -1)

        total, losses = loss_fn(phi, coords, phi_fn, weights)

        assert total.dim() == 0


@pytest.mark.slow
class TestLossLargeScale:
    """Test loss functions at scale."""

    def test_large_batch(self, config, device):
        """Test with large batch size."""
        loss_fn = VariationalLoss(config).to(device)
        batch = 256
        phi = torch.randn(batch, 35, device=device)
        coords = torch.rand(batch, 7, device=device, requires_grad=True)
        weights = {'torsion': 1.0, 'det': 1.0, 'positivity': 1.0}

        def phi_fn(x):
            return phi[:x.shape[0]]

        total, losses = loss_fn(phi, coords, phi_fn, weights)

        assert not torch.isnan(total)

    def test_many_loss_computations(self, config, device):
        """Test stability over many loss computations."""
        loss_fn = VariationalLoss(config).to(device)
        weights = {'torsion': 1.0, 'det': 1.0}

        for _ in range(50):
            phi = torch.randn(16, 35, device=device)
            coords = torch.rand(16, 7, device=device)

            def phi_fn(x, p=phi):
                return p[:x.shape[0]]

            total, _ = loss_fn(phi, coords, phi_fn, weights)
            assert not torch.isnan(total)
