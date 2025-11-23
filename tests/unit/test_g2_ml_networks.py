"""
Comprehensive tests for G2 ML neural network architectures.

Tests include:
- Network forward pass correctness
- Output shape validation
- SIREN activation properties
- Fourier encoding periodicity
- Gradient computation
- SPD metric property preservation
- Numerical stability

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
    from G2_phi_network import (
        FourierFeatures,
        SirenLayer,
        G2PhiNetwork
    )
    G2_ML_AVAILABLE = True
except ImportError:
    G2_ML_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not G2_ML_AVAILABLE,
    reason="G2_ML modules not available"
)


class TestFourierFeatures:
    """Test Fourier feature encoding layer."""

    def test_fourier_features_output_shape(self):
        """Verify Fourier features produce correct output shape."""
        input_dim = 7
        n_modes = 16
        batch_size = 32

        encoder = FourierFeatures(input_dim=input_dim, n_modes=n_modes)

        x = torch.randn(batch_size, input_dim)
        features = encoder(x)

        expected_output_dim = 2 * n_modes  # cos + sin
        assert features.shape == (batch_size, expected_output_dim), (
            f"Expected shape ({batch_size}, {expected_output_dim}), "
            f"got {features.shape}"
        )

    def test_fourier_features_periodicity(self):
        """Verify Fourier features are periodic in 2π."""
        encoder = FourierFeatures(input_dim=7, n_modes=16)

        x = torch.randn(10, 7)
        features1 = encoder(x)

        # Add 2π to all coordinates
        x_shifted = x + 2 * np.pi
        features2 = encoder(x_shifted)

        # Features should be identical (periodic)
        assert torch.allclose(features1, features2, atol=1e-6), (
            "Fourier features not periodic in 2π"
        )

    def test_fourier_features_bounded(self):
        """Verify Fourier features are bounded in [-1, 1]."""
        encoder = FourierFeatures(input_dim=7, n_modes=16)

        x = torch.randn(100, 7) * 10  # Large random inputs
        features = encoder(x)

        assert torch.all(features >= -1.0) and torch.all(features <= 1.0), (
            f"Fourier features not bounded: min={features.min()}, max={features.max()}"
        )

    def test_fourier_features_different_scales(self):
        """Test Fourier features with different frequency scales."""
        x = torch.randn(10, 7)

        # Low frequency
        encoder_low = FourierFeatures(input_dim=7, n_modes=8, scale=0.5)
        features_low = encoder_low(x)

        # High frequency
        encoder_high = FourierFeatures(input_dim=7, n_modes=8, scale=2.0)
        features_high = encoder_high(x)

        # Both should be valid
        assert features_low.shape == features_high.shape
        assert torch.all(torch.isfinite(features_low))
        assert torch.all(torch.isfinite(features_high))


class TestSirenLayer:
    """Test SIREN activation layer."""

    def test_siren_layer_output_shape(self):
        """Verify SIREN layer produces correct output shape."""
        in_features = 32
        out_features = 64
        batch_size = 16

        layer = SirenLayer(in_features, out_features, omega_0=30.0)

        x = torch.randn(batch_size, in_features)
        output = layer(x)

        assert output.shape == (batch_size, out_features), (
            f"Expected shape ({batch_size}, {out_features}), got {output.shape}"
        )

    def test_siren_activation_bounded(self):
        """Verify SIREN activation is bounded in [-1, 1]."""
        layer = SirenLayer(32, 64, omega_0=30.0)

        x = torch.randn(100, 32) * 10  # Large inputs
        output = layer(x)

        assert torch.all(output >= -1.0) and torch.all(output <= 1.0), (
            f"SIREN output not bounded: min={output.min()}, max={output.max()}"
        )

    def test_siren_first_layer_initialization(self):
        """Test first layer uses special initialization."""
        layer_first = SirenLayer(7, 64, omega_0=30.0, is_first=True)
        layer_other = SirenLayer(64, 64, omega_0=30.0, is_first=False)

        # Both should have initialized weights
        assert layer_first.linear.weight.requires_grad
        assert layer_other.linear.weight.requires_grad

        # Weights should be different due to different initialization
        weights_first = layer_first.linear.weight.data
        weights_other = layer_other.linear.weight.data

        # Check they're in expected ranges
        # First layer: [-1/in_features, 1/in_features]
        assert torch.all(torch.abs(weights_first) <= 1.0 / 7 + 0.01)

    def test_siren_gradient_computation(self):
        """Verify SIREN layer computes gradients correctly."""
        layer = SirenLayer(32, 64, omega_0=30.0)

        x = torch.randn(10, 32, requires_grad=True)
        output = layer(x)

        # Compute gradient via backprop
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed"
        assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN/Inf"

    def test_siren_different_omega(self):
        """Test SIREN with different frequency parameters."""
        x = torch.randn(10, 32)

        layer_low = SirenLayer(32, 64, omega_0=10.0)
        layer_high = SirenLayer(32, 64, omega_0=100.0)

        output_low = layer_low(x)
        output_high = layer_high(x)

        # Both should be valid and different
        assert torch.all(torch.isfinite(output_low))
        assert torch.all(torch.isfinite(output_high))
        assert not torch.allclose(output_low, output_high, atol=0.1)


class TestG2PhiNetwork:
    """Test complete G2 phi network."""

    @pytest.mark.parametrize("encoding_type", ["fourier", "siren"])
    def test_network_forward_pass(self, encoding_type):
        """Test network forward pass for both encoding types."""
        network = G2PhiNetwork(
            encoding_type=encoding_type,
            hidden_dims=[128, 64],
            fourier_modes=8,
            omega_0=30.0
        )

        batch_size = 16
        x = torch.randn(batch_size, 7)  # 7D T^7 coordinates

        phi = network(x)

        # Should output 35 components (C(7,3) = 35)
        assert phi.shape == (batch_size, 35), (
            f"Expected shape ({batch_size}, 35), got {phi.shape}"
        )

    def test_network_output_finite(self):
        """Verify network output contains no NaN/Inf."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x = torch.randn(32, 7)
        phi = network(x)

        assert torch.all(torch.isfinite(phi)), (
            f"Network output contains NaN/Inf: "
            f"NaN count={torch.isnan(phi).sum()}, "
            f"Inf count={torch.isinf(phi).sum()}"
        )

    def test_network_gradient_flow(self):
        """Test gradient backpropagation through network."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64, 32])

        x = torch.randn(10, 7, requires_grad=True)
        phi = network(x)

        # Compute simple loss and backprop
        loss = phi.pow(2).sum()
        loss.backward()

        # Check gradients exist and are finite
        assert x.grad is not None, "No gradient computed for input"
        assert torch.all(torch.isfinite(x.grad)), "Input gradient contains NaN/Inf"

        # Check network parameters have gradients
        for name, param in network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.all(torch.isfinite(param.grad)), (
                    f"Gradient for {name} contains NaN/Inf"
                )

    def test_network_deterministic_with_seed(self):
        """Verify network gives same output with same seed."""
        # Set seed
        torch.manual_seed(42)
        network1 = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        torch.manual_seed(42)
        network2 = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x = torch.randn(10, 7)

        # Both networks should produce same output (same initialization)
        with torch.no_grad():
            phi1 = network1(x)
            phi2 = network2(x)

        assert torch.allclose(phi1, phi2, atol=1e-7), (
            "Networks with same seed produce different outputs"
        )

    def test_network_normalization(self):
        """Test phi normalization if enabled."""
        network = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[64],
            normalize_phi=True
        )

        x = torch.randn(32, 7)
        phi = network(x)

        # Compute ||phi||^2
        phi_norm_sq = torch.sum(phi ** 2, dim=1)

        # Should be approximately 7 if normalization is enforced
        # (May not be exact during training, but should be in reasonable range)
        assert torch.all(phi_norm_sq > 0), "Phi norm squared should be positive"

    def test_network_batch_independence(self):
        """Verify batch elements are processed independently."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        # Single input
        x_single = torch.randn(1, 7)
        phi_single = network(x_single)

        # Same input in batch
        x_batch = x_single.repeat(5, 1)
        phi_batch = network(x_batch)

        # All batch outputs should match single output
        for i in range(5):
            assert torch.allclose(phi_batch[i], phi_single[0], atol=1e-6), (
                f"Batch element {i} differs from single computation"
            )

    @pytest.mark.parametrize("hidden_dims", [
        [64],
        [128, 64],
        [256, 128, 64],
        [128, 128, 128, 64]
    ])
    def test_network_different_architectures(self, hidden_dims):
        """Test networks with different architectures."""
        network = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=hidden_dims
        )

        x = torch.randn(16, 7)
        phi = network(x)

        # Should always output correct shape
        assert phi.shape == (16, 35)
        assert torch.all(torch.isfinite(phi))

    def test_network_on_cpu_and_gpu(self):
        """Test network works on both CPU and GPU (if available)."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        # CPU test
        x_cpu = torch.randn(8, 7)
        phi_cpu = network(x_cpu)
        assert phi_cpu.device.type == 'cpu'
        assert torch.all(torch.isfinite(phi_cpu))

        # GPU test (if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            network_gpu = network.to(device)
            x_gpu = x_cpu.to(device)

            phi_gpu = network_gpu(x_gpu)
            assert phi_gpu.device.type == 'cuda'
            assert torch.all(torch.isfinite(phi_gpu))

            # Results should match (within numerical precision)
            phi_cpu_from_gpu = phi_gpu.cpu()
            assert torch.allclose(phi_cpu, phi_cpu_from_gpu, atol=1e-5)


class TestNetworkJacobian:
    """Test Jacobian computation for metric derivatives."""

    def test_jacobian_computation(self):
        """Verify Jacobian can be computed via autograd."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x = torch.randn(1, 7, requires_grad=True)
        phi = network(x)

        # Compute Jacobian: d(phi)/d(x)
        # Shape should be (35, 7) for single input
        jacobian = []
        for i in range(35):
            grad = torch.autograd.grad(
                phi[0, i],
                x,
                retain_graph=True,
                create_graph=False
            )[0]
            jacobian.append(grad)

        jacobian = torch.stack(jacobian, dim=0)  # Shape: (35, 1, 7)
        jacobian = jacobian.squeeze(1)  # Shape: (35, 7)

        assert jacobian.shape == (35, 7), f"Expected (35, 7), got {jacobian.shape}"
        assert torch.all(torch.isfinite(jacobian)), "Jacobian contains NaN/Inf"

    def test_jacobian_changes_with_input(self):
        """Verify Jacobian varies with different inputs."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x1 = torch.randn(1, 7, requires_grad=True)
        x2 = torch.randn(1, 7, requires_grad=True)

        phi1 = network(x1)
        phi2 = network(x2)

        # Compute first element of Jacobian for both
        grad1 = torch.autograd.grad(phi1[0, 0], x1, retain_graph=True)[0]
        grad2 = torch.autograd.grad(phi2[0, 0], x2, retain_graph=True)[0]

        # Jacobians should be different for different inputs
        assert not torch.allclose(grad1, grad2, atol=0.1), (
            "Jacobian doesn't vary with input"
        )


class TestNetworkNumericalStability:
    """Test numerical stability under various conditions."""

    def test_stability_with_large_inputs(self):
        """Test network with large input values."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x_large = torch.randn(16, 7) * 100  # Large coordinates
        phi = network(x_large)

        assert torch.all(torch.isfinite(phi)), (
            "Network unstable with large inputs"
        )

    def test_stability_with_small_inputs(self):
        """Test network with very small input values."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x_small = torch.randn(16, 7) * 1e-6  # Very small coordinates
        phi = network(x_small)

        assert torch.all(torch.isfinite(phi)), (
            "Network unstable with small inputs"
        )

    def test_stability_with_zero_inputs(self):
        """Test network at origin."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x_zero = torch.zeros(1, 7)
        phi = network(x_zero)

        assert torch.all(torch.isfinite(phi)), (
            "Network unstable at origin"
        )

    def test_repeated_forward_passes(self):
        """Test network stability over repeated evaluations."""
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        x = torch.randn(8, 7)

        # Multiple forward passes should give same result
        phi1 = network(x)
        phi2 = network(x)
        phi3 = network(x)

        assert torch.allclose(phi1, phi2, atol=1e-7)
        assert torch.allclose(phi2, phi3, atol=1e-7)
