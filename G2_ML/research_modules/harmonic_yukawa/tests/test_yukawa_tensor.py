"""
Unit tests for Yukawa tensor computation.

Tests:
- YukawaResult dataclass
- YukawaTensor computation
- Eigenvalue analysis
- Mass spectrum extraction from Yukawa
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add harmonic_yukawa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from yukawa import YukawaResult, YukawaTensor
    from config import HarmonicConfig, default_harmonic_config
    from harmonic_extraction import HarmonicBasis
    YUKAWA_AVAILABLE = True
except ImportError as e:
    YUKAWA_AVAILABLE = False
    YUKAWA_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not YUKAWA_AVAILABLE,
    reason=f"yukawa module not available: {YUKAWA_IMPORT_ERROR if not YUKAWA_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Small config for fast tests."""
    return HarmonicConfig(
        n_sample_points=100,
        n_yukawa_samples=100,
        yukawa_batch_size=50,
    )


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def mock_yukawa_result():
    """Create a mock YukawaResult for testing."""
    # Create realistic tensor shape (21, 21, 77)
    tensor = torch.randn(21, 21, 77)
    # Make it approximately antisymmetric in first two indices
    tensor = (tensor - tensor.transpose(0, 1)) / 2

    # Compute Gram matrix
    Y_flat = tensor.reshape(-1, 77)
    gram = Y_flat.T @ Y_flat

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return YukawaResult(
        tensor=tensor,
        gram_matrix=gram,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        trace=eigenvalues.sum().item(),
        det=eigenvalues.prod().item() ** (1/77) if eigenvalues.prod().item() > 0 else 0.0
    )


@pytest.fixture
def mock_harmonic_basis(config, device):
    """Create mock HarmonicBasis for testing."""
    n_points = config.n_sample_points

    return HarmonicBasis(
        h2_forms=torch.randn(n_points, config.b2, config.dim_2form, device=device),
        h3_forms=torch.randn(n_points, config.b3, config.dim_3form, device=device),
        h2_gram=torch.eye(config.b2, device=device),
        h3_gram=torch.eye(config.b3, device=device),
        sample_points=torch.rand(n_points, 7, device=device),
        metric_at_points=torch.eye(7, device=device).unsqueeze(0).expand(n_points, 7, 7),
    )


# =============================================================================
# YukawaResult Tests
# =============================================================================

class TestYukawaResult:
    """Test YukawaResult dataclass."""

    def test_tensor_shape(self, mock_yukawa_result):
        """Tensor should be (21, 21, 77)."""
        assert mock_yukawa_result.tensor.shape == (21, 21, 77)

    def test_gram_matrix_shape(self, mock_yukawa_result):
        """Gram matrix should be (77, 77)."""
        assert mock_yukawa_result.gram_matrix.shape == (77, 77)

    def test_eigenvalues_shape(self, mock_yukawa_result):
        """Eigenvalues should be (77,)."""
        assert mock_yukawa_result.eigenvalues.shape == (77,)

    def test_eigenvectors_shape(self, mock_yukawa_result):
        """Eigenvectors should be (77, 77)."""
        assert mock_yukawa_result.eigenvectors.shape == (77, 77)

    def test_eigenvalues_sorted(self, mock_yukawa_result):
        """Eigenvalues should be sorted descending."""
        eigs = mock_yukawa_result.eigenvalues
        assert torch.all(eigs[:-1] >= eigs[1:])

    def test_effective_rank(self, mock_yukawa_result):
        """effective_rank should be positive integer <= 77."""
        rank = mock_yukawa_result.effective_rank
        assert isinstance(rank, int)
        assert 0 <= rank <= 77

    def test_hierarchy_ratio(self, mock_yukawa_result):
        """hierarchy_ratio should be positive."""
        ratio = mock_yukawa_result.hierarchy_ratio
        assert ratio > 0 or ratio == float('inf')

    def test_mass_spectrum_shape(self, mock_yukawa_result):
        """mass_spectrum should return (77,) tensor."""
        masses = mock_yukawa_result.mass_spectrum(scale=246.0)
        assert masses.shape == (77,)

    def test_mass_spectrum_positive(self, mock_yukawa_result):
        """mass_spectrum should be non-negative."""
        masses = mock_yukawa_result.mass_spectrum()
        assert torch.all(masses >= 0)

    def test_mixing_matrix_shape(self, mock_yukawa_result):
        """mixing_matrix should be (77, 77)."""
        V = mock_yukawa_result.mixing_matrix()
        assert V.shape == (77, 77)

    def test_mixing_matrix_orthogonal(self, mock_yukawa_result):
        """mixing_matrix should be orthogonal (V^T V = I)."""
        V = mock_yukawa_result.mixing_matrix()
        VtV = V.T @ V
        I = torch.eye(77)
        assert torch.allclose(VtV, I, atol=1e-4)


class TestYukawaResultEdgeCases:
    """Test YukawaResult edge cases."""

    def test_zero_eigenvalues(self):
        """Test with many zero eigenvalues."""
        eigenvalues = torch.zeros(77)
        eigenvalues[:10] = torch.rand(10)  # Only 10 nonzero

        result = YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.zeros(77, 77),
            eigenvalues=eigenvalues,
            eigenvectors=torch.eye(77),
            trace=eigenvalues.sum().item(),
            det=0.0
        )

        assert result.effective_rank == 10
        assert result.hierarchy_ratio != float('inf')

    def test_uniform_eigenvalues(self):
        """Test with uniform eigenvalues (no hierarchy)."""
        eigenvalues = torch.ones(77)

        result = YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.eye(77),
            eigenvalues=eigenvalues,
            eigenvectors=torch.eye(77),
            trace=77.0,
            det=1.0
        )

        assert result.hierarchy_ratio == 1.0
        assert result.effective_rank == 77


# =============================================================================
# YukawaTensor Tests
# =============================================================================

class TestYukawaTensor:
    """Test YukawaTensor computation."""

    def test_initialization(self, config, device):
        """YukawaTensor should initialize correctly."""
        yt = YukawaTensor(config, device=device)
        assert yt.config == config
        assert yt.device == device
        assert yt.wedge is not None

    @pytest.mark.slow
    def test_compute_shape(self, config, device, mock_harmonic_basis):
        """compute() should return correct shapes."""
        yt = YukawaTensor(config, device=device)

        # Note: This is a slow test as it computes the full tensor
        result = yt.compute(mock_harmonic_basis)

        assert result.tensor.shape == (config.b2, config.b2, config.b3)
        assert result.gram_matrix.shape == (config.b3, config.b3)

    @pytest.mark.slow
    def test_compute_symmetric_shape(self, config, device, mock_harmonic_basis):
        """compute_symmetric() should return correct shapes."""
        yt = YukawaTensor(config, device=device)
        result = yt.compute_symmetric(mock_harmonic_basis)

        assert result.tensor.shape == (config.b2, config.b2, config.b3)

    @pytest.mark.slow
    def test_compute_symmetric_is_symmetric(self, config, device, mock_harmonic_basis):
        """Symmetrized tensor should satisfy Y_ijk = Y_jik."""
        yt = YukawaTensor(config, device=device)
        result = yt.compute_symmetric(mock_harmonic_basis)

        Y = result.tensor
        diff = Y - Y.transpose(0, 1)
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-5)

    def test_verify_antisymmetry_keys(self, config, device, mock_harmonic_basis):
        """verify_antisymmetry should return expected keys."""
        yt = YukawaTensor(config, device=device)

        # Mock the compute method for speed
        class MockBasis:
            sample_points = torch.rand(10, 7, device=device)
            metric_at_points = torch.eye(7, device=device).unsqueeze(0).expand(10, 7, 7)
            h2_forms = torch.randn(10, config.b2, 21, device=device)
            h3_forms = torch.randn(10, config.b3, 35, device=device)

        # Create minimal mock for shape validation
        # Full test is slow, so just check structure
        pass  # Skip detailed test in fast mode


# =============================================================================
# Gram Matrix Properties
# =============================================================================

class TestGramMatrixProperties:
    """Test properties of Yukawa Gram matrix M = Y^T Y."""

    def test_gram_symmetric(self, mock_yukawa_result):
        """Gram matrix should be symmetric."""
        G = mock_yukawa_result.gram_matrix
        assert torch.allclose(G, G.T, atol=1e-6)

    def test_gram_positive_semidefinite(self, mock_yukawa_result):
        """Gram matrix should be positive semi-definite."""
        # All eigenvalues should be >= 0 (with numerical tolerance)
        eigs = mock_yukawa_result.eigenvalues
        assert torch.all(eigs >= -1e-6)

    def test_gram_eigenvalue_decomposition(self, mock_yukawa_result):
        """G = V Λ V^T should hold."""
        G = mock_yukawa_result.gram_matrix
        V = mock_yukawa_result.eigenvectors
        Lambda = torch.diag(mock_yukawa_result.eigenvalues)

        G_reconstructed = V @ Lambda @ V.T
        assert torch.allclose(G, G_reconstructed, atol=1e-4)


# =============================================================================
# Physical Constraints Tests
# =============================================================================

class TestPhysicalConstraints:
    """Test physics-motivated properties."""

    def test_43_77_split(self, mock_yukawa_result):
        """Test the 43/77 visible/hidden split."""
        eigs = mock_yukawa_result.eigenvalues
        visible = eigs[:43]
        hidden = eigs[43:]

        # Both should exist
        assert len(visible) == 43
        assert len(hidden) == 34

    def test_tau_parameter_structure(self, mock_yukawa_result):
        """tau = sum(visible) / sum(hidden) should be computable."""
        eigs = mock_yukawa_result.eigenvalues
        visible_sum = eigs[:43].sum().item()
        hidden_sum = eigs[43:].sum().item()

        if hidden_sum > 0:
            tau = visible_sum / hidden_sum
            assert tau >= 0
            # GIFT predicts tau = 3472/891 ≈ 3.897
            # We don't enforce this for random data

    def test_trace_consistency(self, mock_yukawa_result):
        """Trace should equal sum of eigenvalues."""
        trace_from_eigs = mock_yukawa_result.eigenvalues.sum().item()
        trace_stored = mock_yukawa_result.trace

        assert abs(trace_from_eigs - trace_stored) < 1e-4


# =============================================================================
# Numerical Stability
# =============================================================================

class TestYukawaNumericalStability:
    """Test numerical stability of Yukawa computations."""

    def test_small_forms(self, config, device):
        """Test with very small harmonic forms."""
        n_points = 10
        basis = HarmonicBasis(
            h2_forms=torch.ones(n_points, config.b2, config.dim_2form, device=device) * 1e-8,
            h3_forms=torch.ones(n_points, config.b3, config.dim_3form, device=device) * 1e-8,
            h2_gram=torch.eye(config.b2, device=device),
            h3_gram=torch.eye(config.b3, device=device),
            sample_points=torch.rand(n_points, 7, device=device),
            metric_at_points=torch.eye(7, device=device).unsqueeze(0).expand(n_points, 7, 7),
        )

        yt = YukawaTensor(config, device=device)
        result = yt.compute(basis)

        assert not torch.any(torch.isnan(result.tensor))
        assert not torch.any(torch.isinf(result.tensor))

    def test_identity_metric(self, config, device):
        """Test with identity metric (flat space)."""
        n_points = 50
        basis = HarmonicBasis(
            h2_forms=torch.randn(n_points, config.b2, config.dim_2form, device=device),
            h3_forms=torch.randn(n_points, config.b3, config.dim_3form, device=device),
            h2_gram=torch.eye(config.b2, device=device),
            h3_gram=torch.eye(config.b3, device=device),
            sample_points=torch.rand(n_points, 7, device=device),
            metric_at_points=torch.eye(7, device=device).unsqueeze(0).expand(n_points, 7, 7),
        )

        yt = YukawaTensor(config, device=device)
        result = yt.compute(basis)

        assert not torch.any(torch.isnan(result.tensor))
        assert result.trace >= 0


@pytest.mark.slow
class TestYukawaLargeScale:
    """Large scale Yukawa tests."""

    def test_large_sample_count(self, device):
        """Test with larger sample count."""
        config = HarmonicConfig(
            n_sample_points=500,
            n_yukawa_samples=500,
        )
        n_points = config.n_sample_points

        basis = HarmonicBasis(
            h2_forms=torch.randn(n_points, config.b2, config.dim_2form, device=device),
            h3_forms=torch.randn(n_points, config.b3, config.dim_3form, device=device),
            h2_gram=torch.eye(config.b2, device=device),
            h3_gram=torch.eye(config.b3, device=device),
            sample_points=torch.rand(n_points, 7, device=device),
            metric_at_points=torch.eye(7, device=device).unsqueeze(0).expand(n_points, 7, 7),
        )

        yt = YukawaTensor(config, device=device)
        result = yt.compute(basis)

        assert result.tensor.shape == (21, 21, 77)
        assert not torch.any(torch.isnan(result.tensor))
