"""
Unit tests for G2_ML/2_1 model module.

Tests the G2VariationalNet and related network components:
- FourierFeatures encoding
- G2VariationalNet forward pass
- HarmonicFormsNet
- GIFTVariationalModel

This tests the CURRENT production code, not the archived 0.2 version.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add G2_ML/2_1 to path (active production code)
sys.path.insert(0, str(Path(__file__).parent.parent / "2_1"))

try:
    from model import (
        FourierFeatures,
        G2VariationalNet,
        HarmonicFormsNet,
        GIFTVariationalModel,
    )
    from config import GIFTConfig, default_config
    G2_MODEL_AVAILABLE = True
except ImportError as e:
    G2_MODEL_AVAILABLE = False
    G2_MODEL_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not G2_MODEL_AVAILABLE,
    reason=f"G2 v2.1 model not available: {G2_MODEL_IMPORT_ERROR if not G2_MODEL_AVAILABLE else ''}"
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
def sample_coords(batch_size, device):
    """Create sample 7D coordinates."""
    return torch.rand(batch_size, 7, device=device) * 2 * np.pi


# =============================================================================
# Fourier Features Tests
# =============================================================================

class TestFourierFeatures:
    """Test Fourier feature encoding."""

    def test_output_shape(self, sample_coords, device):
        """Test correct output dimensionality."""
        n_features = 64
        fourier = FourierFeatures(in_dim=7, n_features=n_features).to(device)

        features = fourier(sample_coords)

        assert features.shape == (sample_coords.shape[0], 2 * n_features)

    def test_output_bounded(self, sample_coords, device):
        """Test that output is bounded (sin/cos in [-1, 1])."""
        fourier = FourierFeatures(in_dim=7, n_features=64).to(device)

        features = fourier(sample_coords)

        assert torch.all(features >= -1.0 - 1e-5)  # Small tolerance for numerics
        assert torch.all(features <= 1.0 + 1e-5)

    def test_different_feature_counts(self, sample_coords, device):
        """Test different numbers of Fourier features."""
        for n_features in [16, 32, 64, 128]:
            fourier = FourierFeatures(in_dim=7, n_features=n_features).to(device)
            features = fourier(sample_coords)

            assert features.shape[1] == 2 * n_features

    def test_scale_parameter_affects_output(self, sample_coords, device):
        """Test that scale parameter affects output."""
        # Same input, different scales
        torch.manual_seed(42)
        fourier_low = FourierFeatures(in_dim=7, n_features=32, scale=0.1).to(device)
        torch.manual_seed(42)
        fourier_high = FourierFeatures(in_dim=7, n_features=32, scale=10.0).to(device)

        features_low = fourier_low(sample_coords)
        features_high = fourier_high(sample_coords)

        # Different scales should produce different outputs
        assert not torch.allclose(features_low, features_high)

    def test_buffer_not_trainable(self, device):
        """Test that frequency matrix B is not trainable."""
        fourier = FourierFeatures(in_dim=7, n_features=64).to(device)

        assert hasattr(fourier, 'B')
        assert not fourier.B.requires_grad

    def test_differentiable(self, device):
        """Test Fourier features are differentiable."""
        fourier = FourierFeatures(in_dim=7, n_features=64).to(device)
        coords = torch.rand(8, 7, device=device, requires_grad=True)

        features = fourier(coords)
        loss = features.sum()
        loss.backward()

        assert coords.grad is not None
        assert not torch.any(torch.isnan(coords.grad))


# =============================================================================
# G2VariationalNet Tests
# =============================================================================

class TestG2VariationalNet:
    """Test G2 variational network."""

    def test_output_shape(self, sample_coords, config, device):
        """Test phi output has correct shape."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        phi = model(coords)

        assert phi.shape == (coords.shape[0], 35)

    def test_output_finite(self, sample_coords, config, device):
        """Test phi values are finite."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_get_metric_shape(self, sample_coords, config, device):
        """Test metric extraction shape."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        metric = model.get_metric(coords)

        assert metric.shape == (coords.shape[0], 7, 7)

    def test_get_metric_symmetric(self, sample_coords, config, device):
        """Test extracted metric is symmetric."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        metric = model.get_metric(coords)

        assert torch.allclose(metric, metric.transpose(-2, -1), atol=1e-5)

    def test_get_phi_and_metric(self, sample_coords, config, device):
        """Test combined phi and metric extraction."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        phi, metric = model.get_phi_and_metric(coords)

        assert phi.shape == (coords.shape[0], 35)
        assert metric.shape == (coords.shape[0], 7, 7)

    def test_positive_projection(self, sample_coords, config, device):
        """Test G2 positivity projection."""
        model = G2VariationalNet(config).to(device)
        coords = sample_coords

        phi_projected = model(coords, project_positive=True)
        phi_not_projected = model(coords, project_positive=False)

        # Both should be finite but may differ
        assert not torch.any(torch.isnan(phi_projected))
        assert not torch.any(torch.isnan(phi_not_projected))

    def test_standard_phi_buffer(self, config, device):
        """Test standard phi is stored as buffer."""
        model = G2VariationalNet(config).to(device)

        assert hasattr(model, 'phi_0')
        assert model.phi_0.shape == (35,)

    def test_parameter_count(self, config, device):
        """Test network has reasonable parameter count."""
        model = G2VariationalNet(config).to(device)

        n_params = sum(p.numel() for p in model.parameters())

        assert n_params > 0
        assert n_params < 10_000_000  # Less than 10M parameters

    def test_differentiable(self, config, device):
        """Test network is differentiable."""
        model = G2VariationalNet(config).to(device)
        coords = torch.rand(8, 7, device=device, requires_grad=True)

        phi = model(coords)
        loss = phi.sum()
        loss.backward()

        assert coords.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_batch_size_one(self, config, device):
        """Test with batch size 1."""
        model = G2VariationalNet(config).to(device)
        coords = torch.rand(1, 7, device=device)

        phi = model(coords)

        assert phi.shape == (1, 35)

    def test_deterministic_with_seed(self, config, device):
        """Test deterministic output with same seed."""
        torch.manual_seed(42)
        model1 = G2VariationalNet(config).to(device)
        coords = torch.rand(8, 7, device=device)
        phi1 = model1(coords)

        torch.manual_seed(42)
        model2 = G2VariationalNet(config).to(device)
        phi2 = model2(coords)

        assert torch.allclose(phi1, phi2)


# =============================================================================
# HarmonicFormsNet Tests
# =============================================================================

class TestHarmonicFormsNet:
    """Test harmonic forms network."""

    def test_h2_output_shape(self, sample_coords, config, device):
        """Test H2 forms have correct shape."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords

        omega = model.forward_h2(coords)

        # (batch, 21 modes, 21 components)
        assert omega.shape == (coords.shape[0], config.b2_K7, config.n_2form_components)

    def test_h3_output_shape(self, sample_coords, config, device):
        """Test H3 forms have correct shape."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords

        Phi = model.forward_h3(coords)

        # (batch, 77 modes, 35 components)
        assert Phi.shape == (coords.shape[0], config.b3_K7, config.n_phi_components)

    def test_h2_finite(self, sample_coords, config, device):
        """Test H2 values are finite."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords

        omega = model.forward_h2(coords)

        assert not torch.any(torch.isnan(omega))
        assert not torch.any(torch.isinf(omega))

    def test_h3_finite(self, sample_coords, config, device):
        """Test H3 values are finite."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords

        Phi = model.forward_h3(coords)

        assert not torch.any(torch.isnan(Phi))
        assert not torch.any(torch.isinf(Phi))

    def test_gram_matrix_h2_shape(self, sample_coords, config, device):
        """Test H2 Gram matrix shape."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords
        metric = torch.eye(7, device=device).unsqueeze(0).expand(coords.shape[0], 7, 7)

        G = model.gram_matrix_h2(coords, metric)

        assert G.shape == (config.b2_K7, config.b2_K7)

    def test_gram_matrix_h3_shape(self, sample_coords, config, device):
        """Test H3 Gram matrix shape."""
        model = HarmonicFormsNet(config).to(device)
        coords = sample_coords
        metric = torch.eye(7, device=device).unsqueeze(0).expand(coords.shape[0], 7, 7)

        G = model.gram_matrix_h3(coords, metric)

        assert G.shape == (config.b3_K7, config.b3_K7)


# =============================================================================
# GIFTVariationalModel Tests
# =============================================================================

class TestGIFTVariationalModel:
    """Test complete GIFT variational model."""

    def test_forward_phi(self, sample_coords, config, device):
        """Test phi extraction."""
        model = GIFTVariationalModel(config).to(device)
        coords = sample_coords

        phi, metric = model.forward_phi(coords)

        assert phi.shape == (coords.shape[0], 35)
        assert metric.shape == (coords.shape[0], 7, 7)

    def test_forward_harmonic(self, sample_coords, config, device):
        """Test harmonic forms extraction."""
        model = GIFTVariationalModel(config).to(device)
        coords = sample_coords
        metric = torch.eye(7, device=device).unsqueeze(0).expand(coords.shape[0], 7, 7)

        omega, Phi = model.forward_harmonic(coords, metric)

        assert omega.shape == (coords.shape[0], config.b2_K7, config.n_2form_components)
        assert Phi.shape == (coords.shape[0], config.b3_K7, config.n_phi_components)

    def test_phase_management(self, config, device):
        """Test training phase management."""
        model = GIFTVariationalModel(config).to(device)

        assert model.current_phase == 0

        model.set_phase(1)
        assert model.current_phase == 1

        model.set_phase(2)
        assert model.current_phase == 2

    def test_phase_weights(self, config, device):
        """Test phase-dependent weight retrieval."""
        model = GIFTVariationalModel(config).to(device)

        weights_0 = model.get_phase_weights()
        model.set_phase(1)
        weights_1 = model.get_phase_weights()

        assert isinstance(weights_0, dict)
        assert isinstance(weights_1, dict)


# =============================================================================
# Edge Cases
# =============================================================================

class TestModelEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_input(self, config, device):
        """Test with zero input."""
        model = G2VariationalNet(config).to(device)
        coords = torch.zeros(4, 7, device=device)

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_large_input(self, config, device):
        """Test with large input values."""
        model = G2VariationalNet(config).to(device)
        coords = torch.ones(4, 7, device=device) * 1000.0

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_negative_input(self, config, device):
        """Test with negative input."""
        model = G2VariationalNet(config).to(device)
        coords = torch.ones(4, 7, device=device) * -5.0

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_multiple_forward_passes(self, config, device):
        """Test stability over multiple forward passes."""
        model = G2VariationalNet(config).to(device)
        coords = torch.rand(8, 7, device=device)

        for _ in range(50):
            phi = model(coords)
            assert not torch.any(torch.isnan(phi))


@pytest.mark.slow
class TestModelLargeScale:
    """Test models at scale."""

    def test_large_batch(self, config, device):
        """Test with large batch size."""
        model = G2VariationalNet(config).to(device)
        coords = torch.rand(512, 7, device=device)

        phi = model(coords)

        assert phi.shape == (512, 35)
        assert not torch.any(torch.isnan(phi))

    def test_many_training_steps(self, config, device):
        """Test model through simulated training steps."""
        model = G2VariationalNet(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(20):
            coords = torch.rand(16, 7, device=device)
            phi = model(coords)
            loss = phi.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Model should still produce valid output
        test_coords = torch.rand(8, 7, device=device)
        phi = model(test_coords)
        assert not torch.any(torch.isnan(phi))
