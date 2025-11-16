"""
Unit tests for G2 geometry module.

Tests differential geometry operators, metric operations,
SPD projection, and numerical stability.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add G2_ML/0.2 to path (most complete version)
sys.path.insert(0, str(Path(__file__).parent.parent / "0.2"))

from G2_geometry import (
    project_spd,
    volume_form,
    metric_inverse
)


class TestSPDProjection:
    """Test symmetric positive definite projection."""

    def test_project_spd_preserves_positive_definite(self):
        """Test that projection produces positive definite matrices."""
        batch_size = 10
        metric = torch.randn(batch_size, 7, 7)

        spd_metric = project_spd(metric)

        # Check all eigenvalues are positive
        for i in range(batch_size):
            eigenvalues = torch.linalg.eigvalsh(spd_metric[i])
            assert torch.all(eigenvalues > 0), f"Negative eigenvalue at batch {i}"

    def test_project_spd_preserves_symmetry(self):
        """Test that projection preserves symmetry."""
        batch_size = 10
        metric = torch.randn(batch_size, 7, 7)

        spd_metric = project_spd(metric)

        # Check symmetry
        for i in range(batch_size):
            diff = spd_metric[i] - spd_metric[i].T
            assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)

    def test_project_spd_shape_preserved(self):
        """Test that shape is preserved."""
        batch_size = 5
        metric = torch.randn(batch_size, 7, 7)

        spd_metric = project_spd(metric)

        assert spd_metric.shape == metric.shape

    def test_project_spd_already_spd(self):
        """Test projection on already SPD matrix."""
        batch_size = 3

        # Create SPD matrix: A = Q @ diag(positive) @ Q^T
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7))
        eigenvalues = torch.rand(batch_size, 7) + 1.0  # Positive eigenvalues
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        spd_metric = project_spd(metric)

        # Should remain similar (small numerical differences allowed)
        assert torch.allclose(metric, spd_metric, atol=1e-4)

    def test_project_spd_minimum_eigenvalue(self):
        """Test that minimum eigenvalue is above epsilon."""
        batch_size = 10
        epsilon = 1e-6
        metric = torch.randn(batch_size, 7, 7)

        spd_metric = project_spd(metric, epsilon=epsilon)

        for i in range(batch_size):
            eigenvalues = torch.linalg.eigvalsh(spd_metric[i])
            assert torch.min(eigenvalues) >= epsilon - 1e-8


class TestVolumeForm:
    """Test volume form computation."""

    def test_volume_form_positive(self):
        """Test that volume form is positive."""
        batch_size = 10

        # Create SPD matrices
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7))
        eigenvalues = torch.rand(batch_size, 7) + 1.0
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        vol = volume_form(metric)

        assert torch.all(vol > 0)

    def test_volume_form_shape(self):
        """Test volume form has correct shape."""
        batch_size = 5
        metric = torch.randn(batch_size, 7, 7)

        vol = volume_form(metric)

        assert vol.shape == (batch_size,)

    def test_volume_form_identity_matrix(self):
        """Test volume form of identity matrix is 1."""
        batch_size = 3
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        vol = volume_form(metric)

        assert torch.allclose(vol, torch.ones(batch_size), atol=1e-6)

    def test_volume_form_scaled_metric(self):
        """Test volume form scales correctly."""
        metric = torch.eye(7).unsqueeze(0)

        # Scale by factor k: det(k*g) = k^7 * det(g), so vol = k^(7/2)
        k = 4.0
        scaled_metric = k * metric

        vol = volume_form(metric)
        vol_scaled = volume_form(scaled_metric)

        expected_scaling = k ** (7/2)
        assert torch.allclose(vol_scaled / vol, torch.tensor(expected_scaling), atol=1e-5)


class TestMetricInverse:
    """Test metric inverse computation."""

    def test_metric_inverse_identity_recovery(self):
        """Test that g @ g^{-1} = I."""
        batch_size = 5

        # Create SPD matrices
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7))
        eigenvalues = torch.rand(batch_size, 7) + 1.0
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        inv_metric = metric_inverse(metric)

        # Check g @ g^{-1} = I
        identity = metric @ inv_metric
        expected_identity = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        assert torch.allclose(identity, expected_identity, atol=1e-4)

    def test_metric_inverse_shape(self):
        """Test inverse has correct shape."""
        batch_size = 3
        metric = torch.randn(batch_size, 7, 7)

        inv_metric = metric_inverse(metric)

        assert inv_metric.shape == metric.shape

    def test_metric_inverse_identity_matrix(self):
        """Test inverse of identity is identity."""
        batch_size = 2
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        inv_metric = metric_inverse(metric)

        assert torch.allclose(inv_metric, metric, atol=1e-6)

    def test_metric_inverse_numerical_stability(self):
        """Test numerical stability with near-singular matrices."""
        batch_size = 3

        # Create matrix with small eigenvalue
        Q, _ = torch.linalg.qr(torch.randn(batch_size, 7, 7))
        eigenvalues = torch.rand(batch_size, 7) * 0.1 + 0.01  # Small but positive
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        inv_metric = metric_inverse(metric, epsilon=1e-8)

        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(inv_metric))
        assert not torch.any(torch.isinf(inv_metric))


class TestGeometryGradients:
    """Test gradient computations for geometry operations."""

    def test_project_spd_differentiable(self):
        """Test that SPD projection is differentiable."""
        metric = torch.randn(3, 7, 7, requires_grad=True)

        spd_metric = project_spd(metric)
        loss = torch.sum(spd_metric ** 2)

        # Should be able to compute gradients
        loss.backward()
        assert metric.grad is not None
        assert not torch.any(torch.isnan(metric.grad))

    def test_volume_form_differentiable(self):
        """Test that volume form is differentiable."""
        Q, _ = torch.linalg.qr(torch.randn(3, 7, 7))
        eigenvalues = torch.rand(3, 7, requires_grad=True) + 1.0
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        vol = volume_form(metric)
        loss = torch.sum(vol)

        loss.backward()
        assert eigenvalues.grad is not None

    def test_metric_inverse_differentiable(self):
        """Test that metric inverse is differentiable."""
        Q, _ = torch.linalg.qr(torch.randn(2, 7, 7))
        eigenvalues = torch.rand(2, 7, requires_grad=True) + 1.0
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        inv_metric = metric_inverse(metric)
        loss = torch.sum(inv_metric)

        loss.backward()
        assert eigenvalues.grad is not None


class TestNumericalAccuracy:
    """Test numerical accuracy of geometry operations."""

    def test_double_projection_stable(self):
        """Test that projecting twice gives same result."""
        metric = torch.randn(5, 7, 7)

        spd1 = project_spd(metric)
        spd2 = project_spd(spd1)

        assert torch.allclose(spd1, spd2, atol=1e-5)

    def test_inverse_of_inverse(self):
        """Test that (g^{-1})^{-1} = g."""
        Q, _ = torch.linalg.qr(torch.randn(3, 7, 7))
        eigenvalues = torch.rand(3, 7) + 1.0
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        inv_once = metric_inverse(metric)
        inv_twice = metric_inverse(inv_once)

        assert torch.allclose(metric, inv_twice, atol=1e-4)


@pytest.mark.slow
class TestLargeScaleGeometry:
    """Test geometry operations at scale."""

    def test_large_batch_spd_projection(self):
        """Test SPD projection with large batch."""
        batch_size = 1000
        metric = torch.randn(batch_size, 7, 7)

        spd_metric = project_spd(metric)

        # Check all are positive definite
        eigenvalues = torch.linalg.eigvalsh(spd_metric)
        assert torch.all(eigenvalues > 0)

    def test_large_batch_volume_form(self):
        """Test volume form with large batch."""
        batch_size = 1000
        metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

        vol = volume_form(metric)

        assert vol.shape == (batch_size,)
        assert torch.allclose(vol, torch.ones(batch_size), atol=1e-6)
