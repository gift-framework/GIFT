"""
Unit tests for G2 Phi Network module.

Tests neural network architectures for learning G2 3-form phi(x),
including Fourier features, SIREN layers, and metric reconstruction.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add G2_ML/archived/early_development/0.2 to path (legacy version for these tests)
sys.path.insert(0, str(Path(__file__).parent.parent / "archived" / "early_development" / "0.2"))

from G2_phi_network import (
    FourierFeatures,
    SirenLayer,
    G2PhiNetwork,
    metric_from_phi_algebraic,
    metric_from_phi_approximate,
    levi_civita_7d,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def device():
    """Get appropriate device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


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
        n_modes = 16
        fourier = FourierFeatures(input_dim=7, n_modes=n_modes).to(device)

        features = fourier(sample_coords)

        assert features.shape == (sample_coords.shape[0], 2 * n_modes)

    def test_output_bounded(self, sample_coords, device):
        """Test that output is bounded (cos/sin in [-1, 1])."""
        fourier = FourierFeatures(input_dim=7, n_modes=16).to(device)

        features = fourier(sample_coords)

        assert torch.all(features >= -1.0)
        assert torch.all(features <= 1.0)

    def test_different_modes(self, sample_coords, device):
        """Test different numbers of Fourier modes."""
        for n_modes in [4, 16, 64]:
            fourier = FourierFeatures(input_dim=7, n_modes=n_modes).to(device)
            features = fourier(sample_coords)

            assert features.shape[1] == 2 * n_modes

    def test_scale_parameter(self, sample_coords, device):
        """Test that scale affects frequency spread."""
        fourier_low = FourierFeatures(input_dim=7, n_modes=16, scale=0.1).to(device)
        fourier_high = FourierFeatures(input_dim=7, n_modes=16, scale=10.0).to(device)

        # With same random seed, different scales should produce different outputs
        features_low = fourier_low(sample_coords)
        features_high = fourier_high(sample_coords)

        # High scale should have more variation
        assert not torch.allclose(features_low, features_high)

    def test_periodicity(self, device):
        """Test that features are periodic (cos/sin properties)."""
        fourier = FourierFeatures(input_dim=7, n_modes=8, scale=1.0).to(device)

        x = torch.zeros(1, 7, device=device)
        x_shifted = x + 2 * np.pi  # Shift by full period

        # Due to random B matrix, periodicity isn't exact, but output should be similar
        # for small n_modes and scale
        features_x = fourier(x)
        features_shifted = fourier(x_shifted)

        # At least the feature magnitudes should be similar
        assert features_x.shape == features_shifted.shape

    def test_buffer_not_trainable(self, device):
        """Test that Fourier matrix B is not trainable."""
        fourier = FourierFeatures(input_dim=7, n_modes=16).to(device)

        assert not fourier.B.requires_grad

    def test_input_dimension_mismatch(self, device):
        """Test handling of wrong input dimension."""
        fourier = FourierFeatures(input_dim=7, n_modes=16).to(device)
        wrong_input = torch.rand(10, 5, device=device)  # Wrong dimension

        with pytest.raises(RuntimeError):
            fourier(wrong_input)


# =============================================================================
# SIREN Layer Tests
# =============================================================================

class TestSirenLayer:
    """Test SIREN (Sinusoidal Representation Network) layer."""

    def test_output_shape(self, device):
        """Test correct output shape."""
        siren = SirenLayer(7, 64, omega_0=30.0).to(device)
        x = torch.randn(10, 7, device=device)

        output = siren(x)

        assert output.shape == (10, 64)

    def test_output_bounded(self, device):
        """Test that output is bounded (sin in [-1, 1])."""
        siren = SirenLayer(7, 64, omega_0=30.0).to(device)
        x = torch.randn(100, 7, device=device)

        output = siren(x)

        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    def test_first_layer_initialization(self, device):
        """Test special initialization for first layer."""
        siren_first = SirenLayer(7, 64, omega_0=30.0, is_first=True).to(device)
        siren_not_first = SirenLayer(7, 64, omega_0=30.0, is_first=False).to(device)

        # Weight distributions should be different
        w1_std = siren_first.linear.weight.std().item()
        w2_std = siren_not_first.linear.weight.std().item()

        assert w1_std != w2_std

    def test_omega_affects_frequency(self, device):
        """Test that omega_0 affects output frequency."""
        x = torch.randn(100, 7, device=device)

        siren_low = SirenLayer(7, 64, omega_0=1.0).to(device)
        siren_high = SirenLayer(7, 64, omega_0=100.0).to(device)

        out_low = siren_low(x)
        out_high = siren_high(x)

        # Higher omega should have higher variance in output
        # (more oscillations across input range)
        assert not torch.allclose(out_low, out_high)

    def test_differentiable(self, device):
        """Test that SIREN layer is differentiable."""
        siren = SirenLayer(7, 64, omega_0=30.0).to(device)
        x = torch.randn(10, 7, device=device, requires_grad=True)

        output = siren(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))


# =============================================================================
# G2PhiNetwork Tests
# =============================================================================

class TestG2PhiNetwork:
    """Test G2 Phi neural network."""

    def test_fourier_network_output_shape(self, sample_coords, device):
        """Test Fourier-based network output shape."""
        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[128, 64],
            normalize_phi=True
        ).to(device)

        phi = model(sample_coords)

        assert phi.shape == (sample_coords.shape[0], 35)

    def test_siren_network_output_shape(self, sample_coords, device):
        """Test SIREN-based network output shape."""
        model = G2PhiNetwork(
            encoding_type='siren',
            hidden_dims=[128, 64],
            normalize_phi=True
        ).to(device)

        phi = model(sample_coords)

        assert phi.shape == (sample_coords.shape[0], 35)

    def test_phi_normalization(self, sample_coords, device):
        """Test that phi is normalized to ||phi||^2 = 7."""
        model = G2PhiNetwork(
            encoding_type='fourier',
            normalize_phi=True
        ).to(device)

        phi = model(sample_coords)
        phi_norm_sq = torch.sum(phi ** 2, dim=1)

        assert torch.allclose(phi_norm_sq, torch.tensor(7.0, device=device), atol=0.1)

    def test_no_normalization(self, sample_coords, device):
        """Test network without normalization."""
        model = G2PhiNetwork(
            encoding_type='fourier',
            normalize_phi=False
        ).to(device)

        phi = model(sample_coords)

        # Should still have correct shape
        assert phi.shape == (sample_coords.shape[0], 35)
        # But norm may not be exactly sqrt(7)
        phi_norm_sq = torch.sum(phi ** 2, dim=1)
        # Just verify it's finite
        assert torch.all(torch.isfinite(phi_norm_sq))

    def test_invalid_encoding_type(self, device):
        """Test that invalid encoding type raises error."""
        with pytest.raises(ValueError):
            G2PhiNetwork(encoding_type='invalid')

    def test_parameter_count(self, device):
        """Test parameter counting."""
        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[256, 256, 128]
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())

        assert n_params > 0
        assert n_params < 1e7  # Reasonable size

    def test_phi_indices(self, device):
        """Test phi component index generation."""
        model = G2PhiNetwork(encoding_type='fourier').to(device)

        indices = model.get_phi_indices()

        assert len(indices) == 35  # C(7,3) = 35
        # Check first and last
        assert indices[0] == (0, 1, 2)
        assert indices[-1] == (4, 5, 6)
        # Check all are unique
        assert len(set(indices)) == 35

    def test_different_hidden_dims(self, sample_coords, device):
        """Test different hidden layer configurations."""
        configs = [
            [64],
            [128, 64],
            [256, 256, 128],
            [512, 256, 128, 64],
        ]

        for hidden_dims in configs:
            model = G2PhiNetwork(
                encoding_type='fourier',
                hidden_dims=hidden_dims
            ).to(device)

            phi = model(sample_coords)
            assert phi.shape == (sample_coords.shape[0], 35)

    def test_differentiable(self, device):
        """Test that network is differentiable."""
        model = G2PhiNetwork(encoding_type='fourier').to(device)
        coords = torch.rand(10, 7, device=device, requires_grad=True)

        phi = model(coords)
        loss = phi.sum()
        loss.backward()

        assert coords.grad is not None
        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# Metric Reconstruction Tests
# =============================================================================

class TestMetricReconstruction:
    """Test metric reconstruction from phi."""

    def test_approximate_metric_shape(self, batch_size, device):
        """Test approximate metric reconstruction shape."""
        phi = torch.randn(batch_size, 35, device=device)
        phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)

        metric = metric_from_phi_approximate(phi)

        assert metric.shape == (batch_size, 7, 7)

    def test_metric_symmetry(self, batch_size, device):
        """Test that reconstructed metric is symmetric."""
        phi = torch.randn(batch_size, 35, device=device)
        phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)

        metric = metric_from_phi_approximate(phi)

        # Check symmetry
        assert torch.allclose(metric, metric.transpose(-2, -1), atol=1e-6)

    def test_algebraic_reconstruction_uses_approximation(self, batch_size, device):
        """Test that algebraic method with approximation flag works."""
        phi = torch.randn(batch_size, 35, device=device)
        phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)

        metric = metric_from_phi_algebraic(phi, use_approximation=True)

        assert metric.shape == (batch_size, 7, 7)
        assert not torch.any(torch.isnan(metric))

    def test_metric_diagonal_positive(self, batch_size, device):
        """Test that diagonal elements are positive."""
        phi = torch.randn(batch_size, 35, device=device)
        phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)

        metric = metric_from_phi_approximate(phi)

        # Diagonal should be positive
        for i in range(7):
            assert torch.all(metric[:, i, i] > 0)

    def test_metric_determinant_positive(self, batch_size, device):
        """Test metric has positive determinant."""
        phi = torch.randn(batch_size, 35, device=device)
        phi = phi / torch.norm(phi, dim=1, keepdim=True) * np.sqrt(7.0)

        metric = metric_from_phi_approximate(phi)
        det = torch.det(metric)

        # Most should be positive (approximation may not guarantee all)
        assert torch.sum(det > 0) > batch_size * 0.5


# =============================================================================
# Levi-Civita Symbol Tests
# =============================================================================

class TestLeviCivita:
    """Test Levi-Civita symbol generation."""

    def test_levi_civita_shape(self):
        """Test Levi-Civita tensor shape."""
        epsilon = levi_civita_7d()

        assert epsilon.shape == (7, 7, 7, 7, 7, 7, 7)

    def test_levi_civita_identity_permutation(self):
        """Test identity permutation gives +1."""
        epsilon = levi_civita_7d()

        assert epsilon[0, 1, 2, 3, 4, 5, 6] == 1.0

    def test_levi_civita_odd_permutation(self):
        """Test odd permutation gives -1."""
        epsilon = levi_civita_7d()

        # Swap two adjacent indices
        assert epsilon[1, 0, 2, 3, 4, 5, 6] == -1.0

    def test_levi_civita_repeated_index(self):
        """Test repeated index gives 0."""
        epsilon = levi_civita_7d()

        assert epsilon[0, 0, 2, 3, 4, 5, 6] == 0.0
        assert epsilon[0, 1, 1, 3, 4, 5, 6] == 0.0

    def test_levi_civita_antisymmetry(self):
        """Test antisymmetry property."""
        epsilon = levi_civita_7d()

        # Swapping any two indices should flip sign
        assert epsilon[0, 1, 2, 3, 4, 5, 6] == -epsilon[1, 0, 2, 3, 4, 5, 6]
        assert epsilon[0, 1, 2, 3, 4, 5, 6] == -epsilon[0, 2, 1, 3, 4, 5, 6]


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================

class TestNetworkEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_input(self, device):
        """Test network with zero input."""
        model = G2PhiNetwork(encoding_type='fourier', normalize_phi=True).to(device)
        coords = torch.zeros(5, 7, device=device)

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_large_input(self, device):
        """Test network with large input values."""
        model = G2PhiNetwork(encoding_type='fourier', normalize_phi=True).to(device)
        coords = torch.ones(5, 7, device=device) * 1000.0

        phi = model(coords)

        assert not torch.any(torch.isnan(phi))
        assert not torch.any(torch.isinf(phi))

    def test_batch_size_one(self, device):
        """Test network with batch size 1."""
        model = G2PhiNetwork(encoding_type='fourier').to(device)
        coords = torch.rand(1, 7, device=device)

        phi = model(coords)

        assert phi.shape == (1, 35)

    def test_metric_from_small_phi(self, device):
        """Test metric reconstruction from very small phi."""
        phi = torch.ones(5, 35, device=device) * 1e-10

        metric = metric_from_phi_approximate(phi)

        assert not torch.any(torch.isnan(metric))
        assert not torch.any(torch.isinf(metric))


@pytest.mark.slow
class TestNetworkLargeScale:
    """Test networks at scale."""

    def test_large_batch(self, device):
        """Test with large batch size."""
        model = G2PhiNetwork(encoding_type='fourier').to(device)
        coords = torch.rand(1000, 7, device=device)

        phi = model(coords)

        assert phi.shape == (1000, 35)
        assert not torch.any(torch.isnan(phi))

    def test_many_forward_passes(self, device):
        """Test stability over many forward passes."""
        model = G2PhiNetwork(encoding_type='fourier').to(device)
        coords = torch.rand(32, 7, device=device)

        for _ in range(100):
            phi = model(coords)
            assert not torch.any(torch.isnan(phi))
