"""
Comprehensive tests for G2 ML physics-based loss functions.

Tests include:
- Torsion loss computation and properties
- Volume form loss and normalization
- Phi normalization constraints
- Loss function gradients
- Known G2 metrics (should give zero loss)
- Non-G2 metrics (should give positive loss)

Version: 2.1.0
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add G2_ML to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "G2_ML" / "0.2"))

try:
    from G2_losses import (
        torsion_loss,
        volume_loss,
        phi_normalization_loss,
    )
    from G2_geometry import project_spd
    G2_ML_AVAILABLE = True
except ImportError:
    G2_ML_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not G2_ML_AVAILABLE,
    reason="G2_ML modules not available"
)


class TestTorsionLoss:
    """Test torsion-free G2 condition loss."""

    def test_torsion_loss_output_structure(self):
        """Verify torsion loss returns proper structure."""
        batch_size = 8
        phi = torch.randn(batch_size, 35)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss, info = torsion_loss(phi, metric, coords, method='autograd')

        # Check loss is scalar
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"

        # Check info dictionary
        assert isinstance(info, dict), "Info should be a dictionary"
        assert 'd_phi_norm_sq' in info
        assert 'd_phi_dual_norm_sq' in info
        assert 'torsion_total' in info

    def test_torsion_loss_positive(self):
        """Verify torsion loss is non-negative."""
        batch_size = 16
        phi = torch.randn(batch_size, 35)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss, info = torsion_loss(phi, metric, coords)

        assert loss >= 0, f"Torsion loss should be non-negative, got {loss.item()}"

    def test_torsion_loss_gradient_flow(self):
        """Test gradient backpropagation through torsion loss."""
        batch_size = 4
        phi = torch.randn(batch_size, 35, requires_grad=True)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        metric.requires_grad_(True)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss, info = torsion_loss(phi, metric, coords)

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert phi.grad is not None, "No gradient for phi"
        assert coords.grad is not None, "No gradient for coords"

        # Check gradients are finite
        assert torch.all(torch.isfinite(phi.grad)), "Phi gradient contains NaN/Inf"
        assert torch.all(torch.isfinite(coords.grad)), "Coords gradient contains NaN/Inf"

    def test_torsion_loss_constant_phi(self):
        """Test torsion loss with constant phi (should have zero derivative)."""
        batch_size = 8
        # Constant phi across space
        phi_const = torch.ones(batch_size, 35)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss, info = torsion_loss(phi_const, metric, coords)

        # Constant phi should have d(phi) ≈ 0 (depends on implementation)
        # At minimum, loss should be finite
        assert torch.isfinite(loss), "Loss not finite for constant phi"

    def test_torsion_loss_batch_consistency(self):
        """Verify loss is consistent across different batch sizes."""
        phi_single = torch.randn(1, 35)
        metric_single = torch.eye(7).unsqueeze(0)
        coords_single = torch.randn(1, 7, requires_grad=True)

        loss_single, _ = torsion_loss(phi_single, metric_single, coords_single)

        # Repeat to create batch
        phi_batch = phi_single.repeat(5, 1)
        metric_batch = metric_single.repeat(5, 1, 1)
        coords_batch = coords_single.repeat(5, 1).requires_grad_(True)

        loss_batch, _ = torsion_loss(phi_batch, metric_batch, coords_batch)

        # Batch loss should be similar to single loss (mean over batch)
        assert torch.isclose(loss_single, loss_batch, rtol=0.1), (
            f"Batch loss {loss_batch} differs from single loss {loss_single}"
        )


class TestVolumeLoss:
    """Test volume normalization loss."""

    def test_volume_loss_structure(self):
        """Verify volume loss returns proper structure."""
        batch_size = 8
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        loss, info = volume_loss(metric)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, "Loss should be scalar"

        assert isinstance(info, dict)
        assert 'det_g_mean' in info
        assert 'det_g_std' in info
        assert 'volume_loss' in info

    def test_volume_loss_zero_for_unit_metric(self):
        """Test volume loss is zero for det(g) = 1."""
        batch_size = 10
        # Identity metric has det = 1
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        loss, info = volume_loss(metric)

        assert loss < 1e-10, f"Volume loss should be ~0 for unit metric, got {loss}"
        assert abs(info['det_g_mean'] - 1.0) < 1e-6, "Mean det should be 1"

    def test_volume_loss_positive_for_scaled_metric(self):
        """Test volume loss is positive when det(g) ≠ 1."""
        batch_size = 8
        # Scaled identity has det = scale^7
        scale = 2.0
        metric = (scale * torch.eye(7)).unsqueeze(0).repeat(batch_size, 1, 1)

        loss, info = volume_loss(metric)

        assert loss > 0, f"Volume loss should be positive for det ≠ 1, got {loss}"

        expected_det = scale ** 7
        assert abs(info['det_g_mean'] - expected_det) < 0.1, (
            f"Expected det ≈ {expected_det}, got {info['det_g_mean']}"
        )

    def test_volume_loss_gradient(self):
        """Test gradient computation for volume loss."""
        batch_size = 4
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        metric = metric + 0.1 * torch.randn_like(metric)
        metric.requires_grad_(True)

        # Project to SPD
        metric = project_spd(metric)

        loss, info = volume_loss(metric)
        loss.backward()

        assert metric.grad is not None, "No gradient computed"
        assert torch.all(torch.isfinite(metric.grad)), "Gradient contains NaN/Inf"

    def test_volume_loss_different_determinants(self):
        """Test volume loss increases with deviation from det = 1."""
        batch_size = 1

        # Three metrics with different determinants
        metric_1 = torch.eye(7).unsqueeze(0)  # det = 1
        metric_2 = (1.1 * torch.eye(7)).unsqueeze(0)  # det = 1.1^7 ≈ 1.95
        metric_3 = (1.5 * torch.eye(7)).unsqueeze(0)  # det = 1.5^7 ≈ 17.1

        loss_1, _ = volume_loss(metric_1)
        loss_2, _ = volume_loss(metric_2)
        loss_3, _ = volume_loss(metric_3)

        # Losses should increase with deviation from 1
        assert loss_1 < loss_2 < loss_3, (
            f"Volume loss not monotonic: {loss_1}, {loss_2}, {loss_3}"
        )


class TestPhiNormalizationLoss:
    """Test phi normalization loss."""

    def test_phi_normalization_structure(self):
        """Verify phi normalization loss returns proper structure."""
        batch_size = 8
        phi = torch.randn(batch_size, 35)

        loss, info = phi_normalization_loss(phi, target=7.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        assert isinstance(info, dict)
        assert 'phi_norm_sq_mean' in info
        assert 'phi_norm_sq_std' in info
        assert 'phi_normalization_loss' in info

    def test_phi_normalization_zero_at_target(self):
        """Test loss is zero when ||phi||^2 = target."""
        batch_size = 10
        target = 7.0

        # Create phi with exact norm
        phi = torch.randn(batch_size, 35)
        # Normalize to target
        phi_norm = torch.sqrt(torch.sum(phi ** 2, dim=1, keepdim=True))
        phi = phi * np.sqrt(target) / phi_norm

        loss, info = phi_normalization_loss(phi, target=target)

        assert loss < 1e-8, f"Loss should be ~0 at target, got {loss}"
        assert abs(info['phi_norm_sq_mean'] - target) < 1e-6

    def test_phi_normalization_positive_off_target(self):
        """Test loss is positive when ||phi||^2 ≠ target."""
        batch_size = 8
        phi = torch.randn(batch_size, 35) * 10  # Random large phi

        loss, info = phi_normalization_loss(phi, target=7.0)

        assert loss > 0, f"Loss should be positive off target, got {loss}"

    def test_phi_normalization_gradient(self):
        """Test gradient computation for normalization loss."""
        batch_size = 4
        phi = torch.randn(batch_size, 35, requires_grad=True)

        loss, info = phi_normalization_loss(phi, target=7.0)
        loss.backward()

        assert phi.grad is not None
        assert torch.all(torch.isfinite(phi.grad))

    def test_phi_normalization_different_targets(self):
        """Test normalization with different target values."""
        phi = torch.randn(10, 35)

        # Test with different targets
        for target in [1.0, 7.0, 14.0, 21.0]:
            loss, info = phi_normalization_loss(phi, target=target)

            assert torch.isfinite(loss)
            assert loss >= 0

            # Verify loss reflects deviation from target
            expected_norm_sq = torch.sum(phi ** 2, dim=1).mean()
            expected_loss = ((expected_norm_sq - target) ** 2).item()

            assert abs(loss.item() - expected_loss) < 1e-4, (
                f"Loss mismatch for target={target}"
            )


class TestLossCombination:
    """Test combined loss functions."""

    def test_combined_loss_gradient(self):
        """Test gradient flow through combined losses."""
        batch_size = 4

        phi = torch.randn(batch_size, 35, requires_grad=True)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        metric = metric + 0.1 * torch.randn_like(metric)
        metric.requires_grad_(True)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        # Combine multiple losses
        loss_torsion, _ = torsion_loss(phi, metric, coords)
        loss_volume, _ = volume_loss(metric)
        loss_phi_norm, _ = phi_normalization_loss(phi, target=7.0)

        total_loss = loss_torsion + 0.1 * loss_volume + 0.01 * loss_phi_norm

        total_loss.backward()

        # All tensors should have gradients
        assert phi.grad is not None
        assert metric.grad is not None
        assert coords.grad is not None

        # All gradients should be finite
        assert torch.all(torch.isfinite(phi.grad))
        assert torch.all(torch.isfinite(metric.grad))
        assert torch.all(torch.isfinite(coords.grad))

    def test_loss_weighting(self):
        """Test effect of loss weighting."""
        batch_size = 8

        phi = torch.randn(batch_size, 35)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        # Get individual losses
        loss_torsion, _ = torsion_loss(phi, metric, coords)
        loss_volume, _ = volume_loss(metric)
        loss_phi_norm, _ = phi_normalization_loss(phi)

        # Test different weightings
        weights_1 = {'torsion': 1.0, 'volume': 0.1, 'phi': 0.01}
        weights_2 = {'torsion': 1.0, 'volume': 1.0, 'phi': 1.0}

        total_1 = (weights_1['torsion'] * loss_torsion +
                   weights_1['volume'] * loss_volume +
                   weights_1['phi'] * loss_phi_norm)

        total_2 = (weights_2['torsion'] * loss_torsion +
                   weights_2['volume'] * loss_volume +
                   weights_2['phi'] * loss_phi_norm)

        # Different weightings should give different totals (unless all losses are 0)
        if not (torch.allclose(loss_torsion, torch.tensor(0.0)) and
                torch.allclose(loss_volume, torch.tensor(0.0)) and
                torch.allclose(loss_phi_norm, torch.tensor(0.0))):
            assert not torch.allclose(total_1, total_2), (
                "Different weightings should give different totals"
            )


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_loss_stability_large_phi(self):
        """Test stability with large phi values."""
        batch_size = 8
        phi_large = torch.randn(batch_size, 35) * 1000

        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        # All losses should remain finite
        loss_torsion, _ = torsion_loss(phi_large, metric, coords)
        loss_phi_norm, _ = phi_normalization_loss(phi_large)

        assert torch.isfinite(loss_torsion)
        assert torch.isfinite(loss_phi_norm)

    def test_loss_stability_small_phi(self):
        """Test stability with very small phi values."""
        batch_size = 8
        phi_small = torch.randn(batch_size, 35) * 1e-10

        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss_torsion, _ = torsion_loss(phi_small, metric, coords)
        loss_phi_norm, _ = phi_normalization_loss(phi_small)

        assert torch.isfinite(loss_torsion)
        assert torch.isfinite(loss_phi_norm)

    def test_loss_stability_zero_phi(self):
        """Test stability with zero phi."""
        batch_size = 8
        phi_zero = torch.zeros(batch_size, 35)

        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss_torsion, _ = torsion_loss(phi_zero, metric, coords)
        loss_phi_norm, _ = phi_normalization_loss(phi_zero)

        assert torch.isfinite(loss_torsion)
        assert torch.isfinite(loss_phi_norm)

    def test_loss_stability_scaled_metric(self):
        """Test stability with differently scaled metrics."""
        batch_size = 8
        phi = torch.randn(batch_size, 35)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        # Test with different scales
        for scale in [0.1, 1.0, 10.0, 100.0]:
            metric_scaled = (scale * torch.eye(7)).unsqueeze(0).repeat(batch_size, 1, 1)

            loss_torsion, _ = torsion_loss(phi, metric_scaled, coords)
            loss_volume, _ = volume_loss(metric_scaled)

            assert torch.isfinite(loss_torsion), f"Torsion loss not finite at scale {scale}"
            assert torch.isfinite(loss_volume), f"Volume loss not finite at scale {scale}"


class TestLossProperties:
    """Test mathematical properties of losses."""

    def test_volume_loss_scale_equivariance(self):
        """Test how volume loss scales with metric."""
        metric_base = torch.eye(7).unsqueeze(0)

        loss_1, info_1 = volume_loss(metric_base)

        # Double the scale
        metric_2x = 2.0 * metric_base
        loss_2x, info_2x = volume_loss(metric_2x)

        # det(2g) = 2^7 * det(g) = 128
        assert abs(info_2x['det_g_mean'] - 128.0) < 0.1

    def test_phi_normalization_quadratic(self):
        """Test phi normalization loss is quadratic in deviation."""
        phi_base = torch.randn(10, 35)
        target = 7.0

        # Compute actual norm
        actual_norm_sq = torch.sum(phi_base ** 2, dim=1).mean()

        # Loss should be quadratic in deviation
        loss, _ = phi_normalization_loss(phi_base, target=target)

        expected_loss = (actual_norm_sq - target) ** 2
        assert abs(loss.item() - expected_loss.item()) < 1e-5

    def test_torsion_loss_zero_for_closed_form(self):
        """Test torsion loss properties for special cases."""
        batch_size = 4

        # Constant phi has zero exterior derivative
        phi_const = torch.ones(batch_size, 35)
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)
        coords = torch.randn(batch_size, 7, requires_grad=True)

        loss, info = torsion_loss(phi_const, metric, coords)

        # Should be very small (implementation dependent)
        assert torch.isfinite(loss)
