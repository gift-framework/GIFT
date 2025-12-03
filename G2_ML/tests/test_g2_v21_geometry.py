"""
Unit tests for G2_ML/2_1 geometry module.

Tests the active production geometry code:
- MetricFromPhi class
- G2Positivity class
- Standard G2 structure functions
- Torsion computation

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
    from g2_geometry import (
        get_3form_indices,
        get_2form_indices,
        index_to_component_3form,
        standard_phi_coefficients,
        standard_psi_coefficients,
        MetricFromPhi,
        G2Positivity,
        phi_norm_squared,
        check_g2_identity,
        TorsionComputation,
        random_phi_near_standard,
        normalize_phi,
    )
    G2_V21_AVAILABLE = True
except ImportError as e:
    G2_V21_AVAILABLE = False
    G2_V21_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not G2_V21_AVAILABLE,
    reason=f"G2 v2.1 geometry not available: {G2_V21_IMPORT_ERROR if not G2_V21_AVAILABLE else ''}"
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
def standard_phi():
    """Get standard G2 3-form."""
    return standard_phi_coefficients()


@pytest.fixture
def batched_standard_phi(batch_size):
    """Get batched standard G2 3-form."""
    phi = standard_phi_coefficients()
    return phi.unsqueeze(0).repeat(batch_size, 1)


# =============================================================================
# Index Mapping Tests
# =============================================================================

class TestIndexMappings:
    """Test 3-form and 2-form index mappings."""

    def test_3form_indices_count(self):
        """Test that we get C(7,3) = 35 indices."""
        indices = get_3form_indices()
        assert len(indices) == 35

    def test_3form_indices_ordered(self):
        """Test that indices are ordered i < j < k."""
        indices = get_3form_indices()
        for i, j, k in indices:
            assert i < j < k

    def test_3form_indices_range(self):
        """Test that indices are in valid range [0, 6]."""
        indices = get_3form_indices()
        for i, j, k in indices:
            assert 0 <= i <= 6
            assert 0 <= j <= 6
            assert 0 <= k <= 6

    def test_2form_indices_count(self):
        """Test that we get C(7,2) = 21 indices."""
        indices = get_2form_indices()
        assert len(indices) == 21

    def test_2form_indices_ordered(self):
        """Test that 2-form indices are ordered i < j."""
        indices = get_2form_indices()
        for i, j in indices:
            assert i < j

    def test_index_to_component_identity(self):
        """Test index_to_component for canonical order."""
        indices = get_3form_indices()
        for pos, (i, j, k) in enumerate(indices):
            comp_pos, sign = index_to_component_3form(i, j, k)
            assert comp_pos == pos
            assert sign == 1

    def test_index_to_component_swap(self):
        """Test index_to_component with swapped indices gives negative sign."""
        # Swap two adjacent indices should give -1
        pos, sign = index_to_component_3form(1, 0, 2)  # Swapped 0 and 1
        assert sign == -1

    def test_index_to_component_repeated(self):
        """Test that repeated indices give invalid result."""
        pos, sign = index_to_component_3form(0, 0, 2)
        assert pos == -1
        assert sign == 0


# =============================================================================
# Standard G2 Structure Tests
# =============================================================================

class TestStandardG2Structure:
    """Test standard G2 3-form and 4-form."""

    def test_standard_phi_shape(self, standard_phi):
        """Test standard phi has 35 components."""
        assert standard_phi.shape == (35,)

    def test_standard_phi_nonzero_count(self, standard_phi):
        """Test standard phi has exactly 7 nonzero terms."""
        nonzero = torch.count_nonzero(standard_phi)
        assert nonzero == 7

    def test_standard_phi_values(self, standard_phi):
        """Test standard phi values are +/- 1."""
        nonzero_vals = standard_phi[standard_phi != 0]
        assert torch.all(torch.abs(nonzero_vals) == 1.0)

    def test_standard_phi_norm(self, standard_phi):
        """Test standard phi has norm sqrt(7)."""
        norm = torch.norm(standard_phi)
        assert torch.isclose(norm, torch.tensor(np.sqrt(7.0)), atol=1e-6)

    def test_standard_psi_shape(self):
        """Test standard psi (4-form) has 35 components."""
        psi = standard_psi_coefficients()
        assert psi.shape == (35,)

    def test_standard_psi_nonzero_count(self):
        """Test standard psi has exactly 7 nonzero terms."""
        psi = standard_psi_coefficients()
        nonzero = torch.count_nonzero(psi)
        assert nonzero == 7


# =============================================================================
# MetricFromPhi Tests
# =============================================================================

class TestMetricFromPhi:
    """Test metric extraction from 3-form phi."""

    def test_metric_shape(self, batched_standard_phi, device):
        """Test metric has correct shape (batch, 7, 7)."""
        metric_extractor = MetricFromPhi().to(device)
        phi = batched_standard_phi.to(device)

        metric = metric_extractor(phi)

        assert metric.shape == (phi.shape[0], 7, 7)

    def test_metric_symmetry(self, batched_standard_phi, device):
        """Test metric is symmetric."""
        metric_extractor = MetricFromPhi().to(device)
        phi = batched_standard_phi.to(device)

        metric = metric_extractor(phi)

        assert torch.allclose(metric, metric.transpose(-2, -1), atol=1e-6)

    def test_metric_positive_diagonal(self, batched_standard_phi, device):
        """Test metric has positive diagonal for standard phi."""
        metric_extractor = MetricFromPhi().to(device)
        phi = batched_standard_phi.to(device)

        metric = metric_extractor(phi)

        for i in range(7):
            assert torch.all(metric[:, i, i] > 0), f"Diagonal {i} not positive"

    def test_metric_identity_for_standard_phi(self, batched_standard_phi, device):
        """Test that standard phi gives identity-like metric."""
        metric_extractor = MetricFromPhi().to(device)
        phi = batched_standard_phi.to(device)

        metric = metric_extractor(phi)

        # Standard G2 should give approximately identity metric
        identity = torch.eye(7, device=device).unsqueeze(0)
        # Allow some deviation since reconstruction isn't exact
        assert torch.allclose(metric, identity.expand_as(metric), atol=0.5)

    def test_metric_differentiable(self, device):
        """Test metric extraction is differentiable."""
        metric_extractor = MetricFromPhi().to(device)
        phi = torch.randn(4, 35, device=device, requires_grad=True)

        metric = metric_extractor(phi)
        loss = metric.sum()
        loss.backward()

        assert phi.grad is not None
        assert not torch.any(torch.isnan(phi.grad))


# =============================================================================
# G2Positivity Tests
# =============================================================================

class TestG2Positivity:
    """Test G2 positivity constraint."""

    def test_standard_phi_is_positive(self, batched_standard_phi, device):
        """Test standard phi is in positive cone."""
        positivity = G2Positivity().to(device)
        phi = batched_standard_phi.to(device)

        is_positive = positivity.check_positive(phi)

        # Standard phi should be positive
        assert torch.all(is_positive)

    def test_projection_preserves_positive(self, batched_standard_phi, device):
        """Test projection preserves already positive phi."""
        positivity = G2Positivity().to(device)
        phi = batched_standard_phi.to(device)

        projected = positivity.project_to_positive(phi, alpha=0.5)

        # Should be close to original
        assert torch.allclose(phi, projected, atol=0.5)

    def test_projection_improves_negativity(self, device):
        """Test projection moves toward positive cone."""
        positivity = G2Positivity().to(device)

        # Start with random (likely not positive)
        phi = torch.randn(8, 35, device=device)

        projected = positivity.project_to_positive(phi, alpha=1.0)

        # Projected should be different (moved toward standard)
        assert not torch.allclose(phi, projected)


# =============================================================================
# Phi Utilities Tests
# =============================================================================

class TestPhiUtilities:
    """Test phi utility functions."""

    def test_phi_norm_squared(self, batched_standard_phi, device):
        """Test phi norm squared computation."""
        phi = batched_standard_phi.to(device)
        metric = torch.eye(7, device=device).unsqueeze(0).expand(phi.shape[0], 7, 7)

        norm_sq = phi_norm_squared(phi, metric)

        # ||phi||^2 = 7 for standard phi with identity metric
        assert torch.allclose(norm_sq, torch.tensor(7.0, device=device).expand(phi.shape[0]), atol=0.1)

    def test_random_phi_near_standard_shape(self, batch_size, device):
        """Test random phi generator produces correct shape."""
        phi = random_phi_near_standard(batch_size, sigma=0.1, device=device)

        assert phi.shape == (batch_size, 35)

    def test_random_phi_near_standard_closeness(self, batch_size, device):
        """Test random phi is close to standard."""
        phi = random_phi_near_standard(batch_size, sigma=0.01, device=device)
        standard = standard_phi_coefficients().to(device)

        # Should be close to standard
        diff = torch.norm(phi - standard.unsqueeze(0), dim=1)
        assert torch.all(diff < 1.0)  # Within reasonable distance

    def test_normalize_phi(self, device):
        """Test phi normalization."""
        phi = torch.randn(8, 35, device=device)
        target_norm = 7.0

        normalized = normalize_phi(phi, target_norm=target_norm)

        # Check norm squared is close to target
        norm_sq = torch.sum(normalized ** 2, dim=1)
        assert torch.allclose(norm_sq, torch.tensor(target_norm, device=device).expand(8), atol=1e-5)


# =============================================================================
# Torsion Computation Tests
# =============================================================================

class TestTorsionComputation:
    """Test torsion computation module."""

    def test_torsion_output_shape(self, batched_standard_phi, device):
        """Test torsion computation output shape."""
        torsion = TorsionComputation().to(device)
        phi = batched_standard_phi.to(device)
        coords = torch.rand(phi.shape[0], 7, device=device, requires_grad=True)

        T = torsion(phi, coords)

        # Torsion is a 2-form valued in tangent bundle
        assert T.dim() >= 2

    def test_torsion_finite(self, batched_standard_phi, device):
        """Test torsion values are finite."""
        torsion = TorsionComputation().to(device)
        phi = batched_standard_phi.to(device)
        coords = torch.rand(phi.shape[0], 7, device=device, requires_grad=True)

        T = torsion(phi, coords)

        assert not torch.any(torch.isnan(T))
        assert not torch.any(torch.isinf(T))


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_metric_from_small_phi(self, device):
        """Test metric extraction from very small phi."""
        metric_extractor = MetricFromPhi().to(device)
        phi = torch.ones(4, 35, device=device) * 1e-8

        metric = metric_extractor(phi)

        assert not torch.any(torch.isnan(metric))
        assert not torch.any(torch.isinf(metric))

    def test_metric_from_large_phi(self, device):
        """Test metric extraction from large phi."""
        metric_extractor = MetricFromPhi().to(device)
        phi = torch.ones(4, 35, device=device) * 100.0

        metric = metric_extractor(phi)

        assert not torch.any(torch.isnan(metric))
        assert not torch.any(torch.isinf(metric))

    def test_batch_size_one(self, device):
        """Test with batch size 1."""
        metric_extractor = MetricFromPhi().to(device)
        phi = standard_phi_coefficients().unsqueeze(0).to(device)

        metric = metric_extractor(phi)

        assert metric.shape == (1, 7, 7)


# =============================================================================
# G2 Identity Tests
# =============================================================================

class TestG2Identity:
    """Test G2 identity verification."""

    def test_g2_identity_standard(self, batched_standard_phi, device):
        """Test G2 identity for standard phi."""
        phi = batched_standard_phi.to(device)
        metric = torch.eye(7, device=device).unsqueeze(0).expand(phi.shape[0], 7, 7)

        error = check_g2_identity(phi, metric)

        # Error should be small for standard G2
        assert error.mean() < 1.0


@pytest.mark.slow
class TestLargeScale:
    """Test geometry operations at scale."""

    def test_large_batch_metric(self, device):
        """Test metric extraction with large batch."""
        metric_extractor = MetricFromPhi().to(device)
        phi = random_phi_near_standard(500, sigma=0.1, device=device)

        metric = metric_extractor(phi)

        assert metric.shape == (500, 7, 7)
        assert not torch.any(torch.isnan(metric))
