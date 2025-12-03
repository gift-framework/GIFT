"""
Unit tests for G2 loss functions module.

Tests physics-based loss functions including torsion, volume,
normalization, positivity, and curriculum learning.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add G2_ML/archived/early_development/0.2 to path (legacy version for these tests)
sys.path.insert(0, str(Path(__file__).parent.parent / "archived" / "early_development" / "0.2"))

from G2_losses import (
    torsion_loss,
    volume_loss,
    phi_normalization_loss,
    metric_positivity_loss,
    CurriculumScheduler,
    G2TotalLoss,
)
from G2_geometry import project_spd


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
def sample_phi(batch_size, device):
    """Create normalized sample phi (3-form)."""
    phi = torch.randn(batch_size, 35, device=device)
    # Normalize to ||phi||^2 = 7
    phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)
    return phi


@pytest.fixture
def sample_metric(batch_size, device):
    """Create sample SPD metric."""
    A = torch.randn(batch_size, 7, 7, device=device)
    metric = A @ A.transpose(-2, -1)
    return project_spd(metric)


@pytest.fixture
def sample_coords(batch_size, device):
    """Create sample coordinates with gradients."""
    return torch.rand(batch_size, 7, device=device, requires_grad=True) * 2 * np.pi


# =============================================================================
# Volume Loss Tests
# =============================================================================

class TestVolumeLoss:
    """Test volume normalization loss."""

    def test_volume_loss_output_shape(self, sample_metric):
        """Test that volume loss returns scalar."""
        loss, info = volume_loss(sample_metric)

        assert loss.dim() == 0, "Loss should be scalar"
        assert isinstance(info, dict), "Info should be dict"

    def test_volume_loss_positive(self, sample_metric):
        """Test that volume loss is non-negative."""
        loss, info = volume_loss(sample_metric)

        assert loss >= 0, "Loss must be non-negative"

    def test_volume_loss_zero_for_unit_determinant(self, batch_size, device):
        """Test that loss is zero when det(g) = 1."""
        # Create metric with determinant 1
        metric = torch.eye(7, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        loss, info = volume_loss(metric)

        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)
        assert np.isclose(info['det_g_mean'], 1.0, atol=1e-6)

    def test_volume_loss_scaled_metric(self, device):
        """Test volume loss with scaled metric."""
        metric = 2.0 * torch.eye(7, device=device).unsqueeze(0)

        loss, info = volume_loss(metric)

        # det(2*I) = 2^7 = 128, so loss = (128 - 1)^2
        expected_det = 2.0 ** 7
        assert np.isclose(info['det_g_mean'], expected_det, rtol=1e-4)
        assert loss > 0, "Loss should be positive for non-unit determinant"

    def test_volume_loss_info_keys(self, sample_metric):
        """Test that info dict contains expected keys."""
        loss, info = volume_loss(sample_metric)

        assert 'det_g_mean' in info
        assert 'det_g_std' in info
        assert 'volume_loss' in info

    def test_volume_loss_differentiable(self, batch_size, device):
        """Test that volume loss is differentiable."""
        metric = torch.randn(batch_size, 7, 7, device=device, requires_grad=True)
        metric = 0.5 * (metric + metric.transpose(-2, -1))
        metric = metric + 2.0 * torch.eye(7, device=device)  # Make positive definite

        loss, _ = volume_loss(metric)
        loss.backward()

        assert metric.grad is not None
        assert not torch.any(torch.isnan(metric.grad))


# =============================================================================
# Phi Normalization Loss Tests
# =============================================================================

class TestPhiNormalizationLoss:
    """Test phi normalization loss."""

    def test_phi_norm_loss_output_shape(self, sample_phi):
        """Test output shape."""
        loss, info = phi_normalization_loss(sample_phi)

        assert loss.dim() == 0

    def test_phi_norm_loss_zero_for_normalized(self, sample_phi):
        """Test loss is near zero for properly normalized phi."""
        loss, info = phi_normalization_loss(sample_phi, target=7.0)

        # Should be close to zero since phi is normalized to ||phi||^2 = 7
        assert loss < 0.1, f"Loss too high for normalized phi: {loss}"

    def test_phi_norm_loss_custom_target(self, batch_size, device):
        """Test with custom target norm."""
        phi = torch.ones(batch_size, 35, device=device) * 0.5  # ||phi||^2 = 35*0.25 = 8.75
        target = 8.75

        loss, info = phi_normalization_loss(phi, target=target)

        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-4)

    def test_phi_norm_loss_positive(self, sample_phi):
        """Test loss is non-negative."""
        loss, _ = phi_normalization_loss(sample_phi)

        assert loss >= 0

    def test_phi_norm_loss_info_keys(self, sample_phi):
        """Test info contains expected keys."""
        _, info = phi_normalization_loss(sample_phi)

        assert 'phi_norm_sq_mean' in info
        assert 'phi_norm_sq_std' in info
        assert 'phi_normalization_loss' in info


# =============================================================================
# Metric Positivity Loss Tests
# =============================================================================

class TestMetricPositivityLoss:
    """Test metric positivity enforcement loss."""

    def test_positivity_loss_zero_for_spd(self, sample_metric):
        """Test loss is zero for SPD matrices."""
        loss, info = metric_positivity_loss(sample_metric)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
        assert info['min_eigenvalue'] > 0

    def test_positivity_loss_positive_for_negative_eigenvalues(self, batch_size, device):
        """Test loss is positive when eigenvalues are negative."""
        # Create matrix with negative eigenvalue
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7, device=device))
        eigenvalues = torch.rand(batch_size, 7, device=device) + 0.5
        eigenvalues[:, 0] = -0.5  # Make first eigenvalue negative
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        loss, info = metric_positivity_loss(metric)

        assert loss > 0, "Loss should be positive for negative eigenvalues"
        assert info['min_eigenvalue'] < 0

    def test_positivity_loss_epsilon_threshold(self, batch_size, device):
        """Test epsilon threshold behavior."""
        metric = torch.eye(7, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        metric = metric * 1e-7  # All eigenvalues = 1e-7 < default epsilon

        loss, info = metric_positivity_loss(metric, epsilon=1e-6)

        assert loss > 0, "Loss should penalize eigenvalues below epsilon"

    def test_positivity_loss_info_keys(self, sample_metric):
        """Test info dict keys."""
        _, info = metric_positivity_loss(sample_metric)

        assert 'min_eigenvalue' in info
        assert 'positivity_loss' in info


# =============================================================================
# Curriculum Scheduler Tests
# =============================================================================

class TestCurriculumScheduler:
    """Test curriculum learning scheduler."""

    def test_default_phases(self):
        """Test default phase configuration."""
        scheduler = CurriculumScheduler()

        assert scheduler.get_phase(0) == 0
        assert scheduler.get_phase(250) == 0
        assert scheduler.get_phase(500) == 1
        assert scheduler.get_phase(2000) == 2

    def test_custom_phases(self):
        """Test custom phase boundaries."""
        scheduler = CurriculumScheduler(phase_epochs=[100, 200, 300])

        assert scheduler.get_phase(50) == 0
        assert scheduler.get_phase(150) == 1
        assert scheduler.get_phase(250) == 2
        assert scheduler.get_phase(500) == 2  # Should clamp to last phase

    def test_weight_progression(self):
        """Test that weights change across phases."""
        scheduler = CurriculumScheduler(
            torsion_weights=[0.1, 1.0, 10.0],
            volume_weights=[10.0, 1.0, 0.1]
        )

        weights_phase0 = scheduler.get_weights(0)
        weights_phase2 = scheduler.get_weights(2500)

        # Torsion should increase, volume should decrease
        assert weights_phase2['torsion'] > weights_phase0['torsion']
        assert weights_phase2['volume'] < weights_phase0['volume']

    def test_get_weights_returns_dict(self):
        """Test weights return format."""
        scheduler = CurriculumScheduler()
        weights = scheduler.get_weights(100)

        assert isinstance(weights, dict)
        assert 'torsion' in weights
        assert 'volume' in weights
        assert 'norm' in weights
        assert 'gauge' in weights

    def test_phase_names(self):
        """Test phase name generation."""
        scheduler = CurriculumScheduler()

        name0 = scheduler.get_phase_name(0)
        name1 = scheduler.get_phase_name(1000)
        name2 = scheduler.get_phase_name(2500)

        assert isinstance(name0, str)
        assert len(name0) > 0
        assert name0 != name2  # Different phases should have different names


# =============================================================================
# G2TotalLoss Tests
# =============================================================================

class TestG2TotalLoss:
    """Test combined G2 loss function."""

    def test_total_loss_output_shape(self, sample_phi, sample_metric, sample_coords):
        """Test total loss returns scalar and info dict."""
        loss_fn = G2TotalLoss()

        total_loss, info = loss_fn(sample_phi, sample_metric, sample_coords, epoch=0)

        assert total_loss.dim() == 0
        assert isinstance(info, dict)

    def test_total_loss_positive(self, sample_phi, sample_metric, sample_coords):
        """Test total loss is non-negative."""
        loss_fn = G2TotalLoss()

        total_loss, _ = loss_fn(sample_phi, sample_metric, sample_coords)

        assert total_loss >= 0

    def test_total_loss_info_contains_components(self, sample_phi, sample_metric, sample_coords):
        """Test info dict contains all loss components."""
        loss_fn = G2TotalLoss()

        _, info = loss_fn(sample_phi, sample_metric, sample_coords)

        assert 'total_loss' in info
        assert 'phase' in info
        assert 'weights' in info
        assert 'volume_loss' in info
        assert 'phi_normalization_loss' in info

    def test_total_loss_with_positivity(self, sample_phi, sample_metric, sample_coords):
        """Test loss with positivity term enabled."""
        loss_fn = G2TotalLoss(use_positivity=True)

        _, info = loss_fn(sample_phi, sample_metric, sample_coords)

        assert 'positivity_loss' in info
        assert 'min_eigenvalue' in info

    def test_total_loss_without_positivity(self, sample_phi, sample_metric, sample_coords):
        """Test loss with positivity term disabled."""
        loss_fn = G2TotalLoss(use_positivity=False)

        _, info = loss_fn(sample_phi, sample_metric, sample_coords)

        assert 'positivity_loss' not in info

    def test_total_loss_epoch_affects_weights(self, sample_phi, sample_metric, sample_coords):
        """Test that different epochs produce different weights."""
        loss_fn = G2TotalLoss()

        _, info_early = loss_fn(sample_phi, sample_metric, sample_coords, epoch=0)
        _, info_late = loss_fn(sample_phi, sample_metric, sample_coords, epoch=2500)

        assert info_early['weights'] != info_late['weights']

    def test_total_loss_differentiable(self, batch_size, device):
        """Test that total loss is differentiable."""
        phi = torch.randn(batch_size, 35, device=device, requires_grad=True)
        coords = torch.rand(batch_size, 7, device=device, requires_grad=True)

        A = torch.randn(batch_size, 7, 7, device=device)
        metric = project_spd(A @ A.transpose(-2, -1))

        loss_fn = G2TotalLoss(use_ricci=False)  # Skip Ricci for faster test
        total_loss, _ = loss_fn(phi, metric, coords)

        total_loss.backward()

        assert phi.grad is not None
        assert not torch.any(torch.isnan(phi.grad))


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================

class TestLossEdgeCases:
    """Test edge cases and numerical stability."""

    def test_volume_loss_small_determinant(self, batch_size, device):
        """Test volume loss with very small determinant."""
        metric = torch.eye(7, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 0.01

        loss, info = volume_loss(metric)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_volume_loss_large_determinant(self, batch_size, device):
        """Test volume loss with large determinant."""
        metric = torch.eye(7, device=device).unsqueeze(0).repeat(batch_size, 1, 1) * 10.0

        loss, info = volume_loss(metric)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_phi_norm_loss_zero_phi(self, batch_size, device):
        """Test phi normalization with near-zero phi."""
        phi = torch.zeros(batch_size, 35, device=device) + 1e-10

        loss, info = phi_normalization_loss(phi)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_positivity_loss_near_singular(self, batch_size, device):
        """Test positivity loss with near-singular matrix."""
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7, device=device))
        eigenvalues = torch.rand(batch_size, 7, device=device) + 0.1
        eigenvalues[:, 0] = 1e-10  # Very small but positive
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        loss, info = metric_positivity_loss(metric)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_scheduler_beyond_max_epoch(self):
        """Test scheduler behavior beyond defined phases."""
        scheduler = CurriculumScheduler(phase_epochs=[100, 200, 300])

        weights = scheduler.get_weights(10000)

        # Should return last phase weights without error
        assert isinstance(weights, dict)
        assert all(isinstance(v, (int, float)) for v in weights.values())


@pytest.mark.slow
class TestLossLargeScale:
    """Test loss functions at scale."""

    def test_volume_loss_large_batch(self, device):
        """Test volume loss with large batch."""
        batch_size = 1000
        A = torch.randn(batch_size, 7, 7, device=device)
        metric = project_spd(A @ A.transpose(-2, -1))

        loss, info = volume_loss(metric)

        assert not torch.isnan(loss)
        assert loss.shape == ()

    def test_phi_norm_loss_large_batch(self, device):
        """Test phi normalization with large batch."""
        batch_size = 1000
        phi = torch.randn(batch_size, 35, device=device)

        loss, info = phi_normalization_loss(phi)

        assert not torch.isnan(loss)
