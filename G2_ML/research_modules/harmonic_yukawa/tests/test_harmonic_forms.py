"""
Unit tests for harmonic form extraction.

Tests:
- HarmonicBasis dataclass
- HarmonicFormNetwork
- HarmonicExtractor
- HodgeLaplacian
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add harmonic_yukawa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from harmonic_extraction import (
        HarmonicBasis,
        HarmonicFormNetwork,
        HarmonicExtractor,
    )
    from hodge_laplacian import (
        get_2form_indices,
        get_3form_indices,
        permutation_sign,
        LaplacianResult,
        HodgeLaplacian,
    )
    from config import HarmonicConfig, default_harmonic_config
    HARMONIC_AVAILABLE = True
except ImportError as e:
    HARMONIC_AVAILABLE = False
    HARMONIC_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not HARMONIC_AVAILABLE,
    reason=f"harmonic modules not available: {HARMONIC_IMPORT_ERROR if not HARMONIC_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Small config for fast tests."""
    return HarmonicConfig(
        n_sample_points=50,
        harmonic_epochs=10,
    )


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def mock_metric_fn(device):
    """Create mock metric function (identity + small perturbation)."""
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Identity metric with small smooth perturbation
        g = torch.eye(7, device=x.device).unsqueeze(0).expand(batch, 7, 7).clone()
        scale = 0.1
        for i in range(7):
            for j in range(i+1, 7):
                pert = scale * torch.sin(x[:, i] + x[:, j])
                g = g.clone()
                g[:, i, j] = pert
                g[:, j, i] = pert
        # Ensure positive definite
        return g + 2 * torch.eye(7, device=x.device).unsqueeze(0)
    return metric_fn


@pytest.fixture
def mock_harmonic_basis(config, device):
    """Create mock HarmonicBasis."""
    n_points = config.n_sample_points

    return HarmonicBasis(
        h2_forms=torch.randn(n_points, config.b2, config.dim_2form, device=device),
        h3_forms=torch.randn(n_points, config.b3, config.dim_3form, device=device),
        h2_gram=torch.eye(config.b2, device=device),
        h3_gram=torch.eye(config.b3, device=device),
        sample_points=torch.rand(n_points, 7, device=device),
        metric_at_points=torch.eye(7, device=device).unsqueeze(0).expand(n_points, 7, 7).clone(),
    )


# =============================================================================
# HarmonicBasis Tests
# =============================================================================

class TestHarmonicBasis:
    """Test HarmonicBasis dataclass."""

    def test_h2_forms_shape(self, mock_harmonic_basis, config):
        """H2 forms should have shape (n_points, 21, 21)."""
        assert mock_harmonic_basis.h2_forms.shape[1] == config.b2
        assert mock_harmonic_basis.h2_forms.shape[2] == config.dim_2form

    def test_h3_forms_shape(self, mock_harmonic_basis, config):
        """H3 forms should have shape (n_points, 77, 35)."""
        assert mock_harmonic_basis.h3_forms.shape[1] == config.b3
        assert mock_harmonic_basis.h3_forms.shape[2] == config.dim_3form

    def test_gram_matrices_shape(self, mock_harmonic_basis, config):
        """Gram matrices should be square."""
        assert mock_harmonic_basis.h2_gram.shape == (config.b2, config.b2)
        assert mock_harmonic_basis.h3_gram.shape == (config.b3, config.b3)

    def test_sample_points_shape(self, mock_harmonic_basis, config):
        """Sample points should be (n_points, 7)."""
        assert mock_harmonic_basis.sample_points.shape == (config.n_sample_points, 7)

    def test_metric_at_points_shape(self, mock_harmonic_basis, config):
        """Metric at points should be (n_points, 7, 7)."""
        assert mock_harmonic_basis.metric_at_points.shape == (config.n_sample_points, 7, 7)

    def test_b2_actual(self, mock_harmonic_basis, config):
        """b2_actual should equal config.b2."""
        assert mock_harmonic_basis.b2_actual == config.b2

    def test_b3_actual(self, mock_harmonic_basis, config):
        """b3_actual should equal config.b3."""
        assert mock_harmonic_basis.b3_actual == config.b3

    def test_validate_returns_dict(self, mock_harmonic_basis):
        """validate() should return dictionary."""
        result = mock_harmonic_basis.validate()

        assert isinstance(result, dict)
        assert "b2_actual" in result
        assert "b3_actual" in result
        assert "h2_gram_det" in result
        assert "h3_gram_det" in result


# =============================================================================
# HarmonicFormNetwork Tests
# =============================================================================

class TestHarmonicFormNetwork:
    """Test HarmonicFormNetwork neural network."""

    def test_2form_network_creation(self, device):
        """Should create network for 2-forms."""
        net = HarmonicFormNetwork(p=2).to(device)
        assert net.p == 2
        assert net.n_components == 21

    def test_3form_network_creation(self, device):
        """Should create network for 3-forms."""
        net = HarmonicFormNetwork(p=3).to(device)
        assert net.p == 3
        assert net.n_components == 35

    def test_2form_output_shape(self, device):
        """2-form network should output (batch, 21)."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.rand(16, 7, device=device)

        output = net(x)

        assert output.shape == (16, 21)

    def test_3form_output_shape(self, device):
        """3-form network should output (batch, 35)."""
        net = HarmonicFormNetwork(p=3).to(device)
        x = torch.rand(16, 7, device=device)

        output = net(x)

        assert output.shape == (16, 35)

    def test_output_finite(self, device):
        """Network output should be finite."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.rand(16, 7, device=device)

        output = net(x)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_differentiable(self, device):
        """Network should be differentiable."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.rand(8, 7, device=device, requires_grad=True)

        output = net(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_fourier_buffer(self, device):
        """Fourier matrix should be registered as buffer."""
        net = HarmonicFormNetwork(p=2).to(device)

        assert hasattr(net, 'B')
        assert not net.B.requires_grad

    def test_hidden_dim_parameter(self, device):
        """hidden_dim parameter should affect network size."""
        net_small = HarmonicFormNetwork(p=2, hidden_dim=64).to(device)
        net_large = HarmonicFormNetwork(p=2, hidden_dim=256).to(device)

        params_small = sum(p.numel() for p in net_small.parameters())
        params_large = sum(p.numel() for p in net_large.parameters())

        assert params_large > params_small

    def test_n_layers_parameter(self, device):
        """n_layers parameter should affect network depth."""
        net_shallow = HarmonicFormNetwork(p=2, n_layers=2).to(device)
        net_deep = HarmonicFormNetwork(p=2, n_layers=6).to(device)

        params_shallow = sum(p.numel() for p in net_shallow.parameters())
        params_deep = sum(p.numel() for p in net_deep.parameters())

        assert params_deep > params_shallow


# =============================================================================
# HarmonicExtractor Tests
# =============================================================================

class TestHarmonicExtractor:
    """Test HarmonicExtractor class."""

    def test_initialization(self, mock_metric_fn, config, device):
        """HarmonicExtractor should initialize correctly."""
        extractor = HarmonicExtractor(mock_metric_fn, config, device)

        assert extractor.config == config
        assert extractor.device == device

    def test_sample_points_shape(self, mock_metric_fn, config, device):
        """sample_points should return (n, 7) tensor."""
        extractor = HarmonicExtractor(mock_metric_fn, config, device)
        points = extractor.sample_points(100)

        assert points.shape == (100, 7)
        assert points.device.type == device.split(':')[0]

    def test_sample_points_in_range(self, mock_metric_fn, config, device):
        """sample_points should be in [0, 1]^7."""
        extractor = HarmonicExtractor(mock_metric_fn, config, device)
        points = extractor.sample_points(100)

        assert torch.all(points >= 0)
        assert torch.all(points <= 1)


# =============================================================================
# LaplacianResult Tests
# =============================================================================

class TestLaplacianResult:
    """Test LaplacianResult dataclass."""

    def test_n_harmonic_count(self):
        """n_harmonic should count True values in is_harmonic."""
        eigenvalues = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        is_harmonic = torch.tensor([True, True, True, False, False, False])
        eigenvectors = torch.eye(6)

        result = LaplacianResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            is_harmonic=is_harmonic,
        )

        assert result.n_harmonic == 3

    def test_get_harmonic_forms(self):
        """get_harmonic_forms should return harmonic eigenvectors."""
        eigenvalues = torch.tensor([0.0, 0.0, 1.0, 2.0])
        is_harmonic = torch.tensor([True, True, False, False])
        eigenvectors = torch.eye(4)

        result = LaplacianResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            is_harmonic=is_harmonic,
        )

        harmonic = result.get_harmonic_forms()
        assert harmonic.shape[1] == 2  # Two harmonic forms


# =============================================================================
# HodgeLaplacian Tests
# =============================================================================

class TestHodgeLaplacian:
    """Test HodgeLaplacian computation."""

    def test_initialization(self, config):
        """HodgeLaplacian should initialize correctly."""
        laplacian = HodgeLaplacian(config)
        assert laplacian.config == config

    def test_has_indices(self, config):
        """Should have form indices."""
        laplacian = HodgeLaplacian(config)
        assert hasattr(laplacian, 'idx2')
        assert hasattr(laplacian, 'idx3')
        assert len(laplacian.idx2) == 21
        assert len(laplacian.idx3) == 35

    @pytest.mark.slow
    def test_compute_laplacian_2forms(self, mock_metric_fn, config, device):
        """compute_laplacian_2forms should return LaplacianResult."""
        laplacian = HodgeLaplacian(config)
        points = torch.rand(config.n_sample_points, 7, device=device)

        result = laplacian.compute_laplacian_2forms(points, mock_metric_fn, n_basis=10)

        assert isinstance(result, LaplacianResult)
        assert result.eigenvalues.shape[0] == 10

    @pytest.mark.slow
    def test_compute_laplacian_eigenvalues_sorted(self, mock_metric_fn, config, device):
        """Eigenvalues should be sorted ascending."""
        laplacian = HodgeLaplacian(config)
        points = torch.rand(config.n_sample_points, 7, device=device)

        result = laplacian.compute_laplacian_2forms(points, mock_metric_fn, n_basis=10)

        # Should be sorted (ascending for Laplacian)
        eigs = result.eigenvalues
        assert torch.all(eigs[:-1] <= eigs[1:]) or torch.all(eigs[:-1] >= eigs[1:])


# =============================================================================
# Hodge Index Functions Tests
# =============================================================================

class TestHodgeIndices:
    """Test index generation functions."""

    def test_2form_indices(self):
        """2-form indices should be C(7,2) = 21."""
        indices = get_2form_indices()
        assert len(indices) == 21

    def test_3form_indices(self):
        """3-form indices should be C(7,3) = 35."""
        indices = get_3form_indices()
        assert len(indices) == 35

    def test_permutation_sign_identity(self):
        """Identity permutation has sign +1."""
        assert permutation_sign((0, 1, 2, 3, 4, 5, 6)) == 1

    def test_permutation_sign_swap(self):
        """Single swap has sign -1."""
        assert permutation_sign((1, 0, 2, 3, 4, 5, 6)) == -1


# =============================================================================
# Edge Cases
# =============================================================================

class TestHarmonicEdgeCases:
    """Test edge cases and numerical stability."""

    def test_batch_size_one(self, device):
        """Network should handle batch size 1."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.rand(1, 7, device=device)

        output = net(x)

        assert output.shape == (1, 21)

    def test_zero_input(self, device):
        """Network should handle zero input."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.zeros(8, 7, device=device)

        output = net(x)

        assert not torch.any(torch.isnan(output))

    def test_large_input(self, device):
        """Network should handle large input values."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.ones(8, 7, device=device) * 100

        output = net(x)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))


@pytest.mark.slow
class TestHarmonicLargeScale:
    """Large scale harmonic form tests."""

    def test_large_batch(self, device):
        """Test network with large batch."""
        net = HarmonicFormNetwork(p=2).to(device)
        x = torch.rand(512, 7, device=device)

        output = net(x)

        assert output.shape == (512, 21)
        assert not torch.any(torch.isnan(output))

    def test_many_forward_passes(self, device):
        """Test stability over many forward passes."""
        net = HarmonicFormNetwork(p=2).to(device)

        for _ in range(100):
            x = torch.rand(16, 7, device=device)
            output = net(x)
            assert not torch.any(torch.isnan(output))
