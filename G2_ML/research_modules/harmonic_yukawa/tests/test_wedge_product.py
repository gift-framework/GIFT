"""
Unit tests for wedge product computations.

Tests the combinatorial structure of differential form wedge products:
- 2-form ∧ 2-form → 4-form
- 4-form ∧ 3-form → 7-form (scalar)
- Full Yukawa integrand computation
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add harmonic_yukawa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from wedge_product import (
        get_2form_indices,
        get_3form_indices,
        permutation_sign,
        WedgeProduct,
        wedge_2_2_3,
        wedge_3_3_to_6,
    )
    WEDGE_AVAILABLE = True
except ImportError as e:
    WEDGE_AVAILABLE = False
    WEDGE_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not WEDGE_AVAILABLE,
    reason=f"wedge_product module not available: {WEDGE_IMPORT_ERROR if not WEDGE_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def batch_size():
    return 16


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def wedge():
    """Pre-built WedgeProduct instance."""
    return WedgeProduct()


@pytest.fixture
def random_2form(batch_size, device):
    """Random 2-form (21 components)."""
    return torch.randn(batch_size, 21, device=device)


@pytest.fixture
def random_3form(batch_size, device):
    """Random 3-form (35 components)."""
    return torch.randn(batch_size, 35, device=device)


# =============================================================================
# Index Tests
# =============================================================================

class TestIndices:
    """Test form index generation."""

    def test_2form_indices_count(self):
        """2-form has C(7,2) = 21 components."""
        indices = get_2form_indices()
        assert len(indices) == 21

    def test_3form_indices_count(self):
        """3-form has C(7,3) = 35 components."""
        indices = get_3form_indices()
        assert len(indices) == 35

    def test_2form_indices_ordered(self):
        """2-form indices should be strictly increasing pairs."""
        indices = get_2form_indices()
        for i, j in indices:
            assert i < j
            assert 0 <= i < 7
            assert 0 <= j < 7

    def test_3form_indices_ordered(self):
        """3-form indices should be strictly increasing triples."""
        indices = get_3form_indices()
        for i, j, k in indices:
            assert i < j < k
            assert 0 <= i < 7
            assert 0 <= k < 7

    def test_indices_unique(self):
        """All index tuples should be unique."""
        idx2 = get_2form_indices()
        idx3 = get_3form_indices()
        assert len(set(idx2)) == 21
        assert len(set(idx3)) == 35

    def test_indices_cached(self):
        """Indices should be cached (same object on repeated calls)."""
        idx1 = get_2form_indices()
        idx2 = get_2form_indices()
        assert idx1 is idx2  # Same object due to lru_cache


# =============================================================================
# Permutation Sign Tests
# =============================================================================

class TestPermutationSign:
    """Test permutation sign computation."""

    def test_identity_permutation(self):
        """Identity permutation has sign +1."""
        assert permutation_sign((0, 1, 2)) == 1
        assert permutation_sign((0, 1, 2, 3)) == 1
        assert permutation_sign((0, 1, 2, 3, 4, 5, 6)) == 1

    def test_single_swap(self):
        """Single swap gives sign -1."""
        assert permutation_sign((1, 0, 2)) == -1
        assert permutation_sign((0, 2, 1)) == -1
        assert permutation_sign((1, 0, 2, 3, 4, 5, 6)) == -1

    def test_double_swap(self):
        """Two swaps gives sign +1."""
        # (0,1,2) -> (1,0,2) -> (1,2,0)
        assert permutation_sign((1, 2, 0)) == 1

    def test_cyclic_permutations(self):
        """Test cyclic permutations."""
        # 3-cycle has sign +1 (even permutation)
        assert permutation_sign((1, 2, 0)) == 1
        assert permutation_sign((2, 0, 1)) == 1

    def test_reverse_permutation(self):
        """Test reversing order."""
        # (6,5,4,3,2,1,0) requires 7*6/2 = 21 swaps -> odd -> -1
        assert permutation_sign((2, 1, 0)) == -1  # 3 elements reversed


# =============================================================================
# WedgeProduct Class Tests
# =============================================================================

class TestWedgeProductClass:
    """Test WedgeProduct class initialization and tables."""

    def test_initialization(self, wedge):
        """WedgeProduct should initialize without error."""
        assert wedge is not None
        assert hasattr(wedge, 'idx2')
        assert hasattr(wedge, 'idx3')

    def test_wedge_tables_exist(self, wedge):
        """Precomputed wedge tables should exist."""
        assert hasattr(wedge, 'wedge_22_table')
        assert hasattr(wedge, 'wedge_43_table')
        assert len(wedge.wedge_22_table) > 0
        assert len(wedge.wedge_43_table) > 0

    def test_wedge_22_table_structure(self, wedge):
        """wedge_22_table entries should have (idx_a, idx_b, idx_4, sign)."""
        for entry in wedge.wedge_22_table[:10]:  # Check first 10
            assert len(entry) == 4
            idx_a, idx_b, idx_4, sign = entry
            assert 0 <= idx_a < 21
            assert 0 <= idx_b < 21
            assert 0 <= idx_4 < 35
            assert sign in (-1, 1)

    def test_wedge_43_table_structure(self, wedge):
        """wedge_43_table entries should have (idx_4, idx_3, sign)."""
        for entry in wedge.wedge_43_table[:10]:
            assert len(entry) == 3
            idx_4, idx_3, sign = entry
            assert 0 <= idx_4 < 35
            assert 0 <= idx_3 < 35
            assert sign in (-1, 1)


# =============================================================================
# wedge_2_2 Tests
# =============================================================================

class TestWedge22:
    """Test 2-form ∧ 2-form → 4-form."""

    def test_output_shape(self, wedge, random_2form):
        """Output should be (batch, 35) for 4-form."""
        omega_a = random_2form
        omega_b = torch.randn_like(omega_a)

        result = wedge.wedge_2_2(omega_a, omega_b)

        assert result.shape == (omega_a.shape[0], 35)

    def test_output_finite(self, wedge, random_2form):
        """Output should be finite."""
        omega_a = random_2form
        omega_b = torch.randn_like(omega_a)

        result = wedge.wedge_2_2(omega_a, omega_b)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_antisymmetry(self, wedge, random_2form):
        """ω ∧ η = -η ∧ ω for 2-forms."""
        omega_a = random_2form
        omega_b = torch.randn_like(omega_a)

        ab = wedge.wedge_2_2(omega_a, omega_b)
        ba = wedge.wedge_2_2(omega_b, omega_a)

        # Should be antisymmetric
        assert torch.allclose(ab, -ba, atol=1e-5)

    def test_self_wedge_zero(self, wedge, random_2form):
        """ω ∧ ω = 0 for a 2-form."""
        omega = random_2form

        result = wedge.wedge_2_2(omega, omega)

        # Should be zero (antisymmetry)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)

    def test_bilinearity_left(self, wedge, batch_size, device):
        """(a*ω + b*η) ∧ φ = a*(ω ∧ φ) + b*(η ∧ φ)"""
        omega = torch.randn(batch_size, 21, device=device)
        eta = torch.randn(batch_size, 21, device=device)
        phi = torch.randn(batch_size, 21, device=device)
        a, b = 2.5, -1.3

        lhs = wedge.wedge_2_2(a * omega + b * eta, phi)
        rhs = a * wedge.wedge_2_2(omega, phi) + b * wedge.wedge_2_2(eta, phi)

        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_zero_input(self, wedge, batch_size, device):
        """Zero input should give zero output."""
        omega = torch.zeros(batch_size, 21, device=device)
        eta = torch.randn(batch_size, 21, device=device)

        result = wedge.wedge_2_2(omega, eta)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


# =============================================================================
# wedge_4_3 Tests
# =============================================================================

class TestWedge43:
    """Test 4-form ∧ 3-form → 7-form (scalar)."""

    def test_output_shape(self, wedge, batch_size, device):
        """Output should be (batch,) scalar."""
        eta = torch.randn(batch_size, 35, device=device)  # 4-form
        Phi = torch.randn(batch_size, 35, device=device)  # 3-form

        result = wedge.wedge_4_3(eta, Phi)

        assert result.shape == (batch_size,)

    def test_output_finite(self, wedge, batch_size, device):
        """Output should be finite."""
        eta = torch.randn(batch_size, 35, device=device)
        Phi = torch.randn(batch_size, 35, device=device)

        result = wedge.wedge_4_3(eta, Phi)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_bilinearity(self, wedge, batch_size, device):
        """Wedge product should be bilinear."""
        eta = torch.randn(batch_size, 35, device=device)
        Phi1 = torch.randn(batch_size, 35, device=device)
        Phi2 = torch.randn(batch_size, 35, device=device)
        a, b = 3.0, -2.0

        lhs = wedge.wedge_4_3(eta, a * Phi1 + b * Phi2)
        rhs = a * wedge.wedge_4_3(eta, Phi1) + b * wedge.wedge_4_3(eta, Phi2)

        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_zero_input(self, wedge, batch_size, device):
        """Zero input gives zero output."""
        eta = torch.zeros(batch_size, 35, device=device)
        Phi = torch.randn(batch_size, 35, device=device)

        result = wedge.wedge_4_3(eta, Phi)

        assert torch.allclose(result, torch.zeros(batch_size, device=device), atol=1e-6)


# =============================================================================
# wedge_2_2_3 Tests
# =============================================================================

class TestWedge223:
    """Test full Yukawa integrand ω_i ∧ ω_j ∧ Φ_k."""

    def test_output_shape(self, random_2form, random_3form):
        """Output should be (batch,) scalar."""
        result = wedge_2_2_3(random_2form, random_2form, random_3form)
        assert result.shape == (random_2form.shape[0],)

    def test_output_finite(self, random_2form, random_3form):
        """Output should be finite."""
        result = wedge_2_2_3(random_2form, random_2form, random_3form)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_antisymmetry_in_2forms(self, random_2form, random_3form):
        """ω_i ∧ ω_j ∧ Φ = -ω_j ∧ ω_i ∧ Φ."""
        omega_i = random_2form
        omega_j = torch.randn_like(omega_i)
        Phi = random_3form

        ij = wedge_2_2_3(omega_i, omega_j, Phi)
        ji = wedge_2_2_3(omega_j, omega_i, Phi)

        assert torch.allclose(ij, -ji, atol=1e-5)

    def test_with_prebuilt_wedge(self, random_2form, random_3form, wedge):
        """Should work with prebuilt WedgeProduct."""
        result = wedge_2_2_3(random_2form, random_2form, random_3form, wedge=wedge)
        assert result.shape == (random_2form.shape[0],)


# =============================================================================
# wedge_3_3_to_6 Tests
# =============================================================================

class TestWedge33to6:
    """Test 3-form ∧ 3-form → 6-form."""

    def test_output_shape(self, random_3form):
        """Output should be (batch, 7) for 6-form."""
        result = wedge_3_3_to_6(random_3form, random_3form)
        assert result.shape == (random_3form.shape[0], 7)

    def test_output_finite(self, random_3form):
        """Output should be finite."""
        Phi_a = random_3form
        Phi_b = torch.randn_like(Phi_a)
        result = wedge_3_3_to_6(Phi_a, Phi_b)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_antisymmetry(self, random_3form):
        """Φ_a ∧ Φ_b = -Φ_b ∧ Φ_a for 3-forms."""
        Phi_a = random_3form
        Phi_b = torch.randn_like(Phi_a)

        ab = wedge_3_3_to_6(Phi_a, Phi_b)
        ba = wedge_3_3_to_6(Phi_b, Phi_a)

        assert torch.allclose(ab, -ba, atol=1e-4)

    def test_self_wedge_zero(self, random_3form):
        """Φ ∧ Φ = 0 for odd-degree form."""
        result = wedge_3_3_to_6(random_3form, random_3form)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-4)


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_small_inputs(self, wedge, batch_size, device):
        """Test with very small values."""
        omega = torch.ones(batch_size, 21, device=device) * 1e-10
        result = wedge.wedge_2_2(omega, omega)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_large_inputs(self, wedge, batch_size, device):
        """Test with large values."""
        omega = torch.ones(batch_size, 21, device=device) * 1e6
        eta = torch.randn(batch_size, 21, device=device)
        result = wedge.wedge_2_2(omega, eta)

        assert not torch.any(torch.isnan(result))

    def test_batch_size_one(self, wedge, device):
        """Test with batch size 1."""
        omega = torch.randn(1, 21, device=device)
        eta = torch.randn(1, 21, device=device)
        result = wedge.wedge_2_2(omega, eta)

        assert result.shape == (1, 35)


@pytest.mark.slow
class TestWedgeLargeScale:
    """Large scale wedge product tests."""

    def test_large_batch(self, wedge, device):
        """Test with large batch."""
        batch = 1000
        omega_a = torch.randn(batch, 21, device=device)
        omega_b = torch.randn(batch, 21, device=device)

        result = wedge.wedge_2_2(omega_a, omega_b)

        assert result.shape == (batch, 35)
        assert not torch.any(torch.isnan(result))

    def test_many_computations(self, wedge, device):
        """Test stability over many computations."""
        for _ in range(100):
            omega_a = torch.randn(16, 21, device=device)
            omega_b = torch.randn(16, 21, device=device)
            Phi = torch.randn(16, 35, device=device)

            eta = wedge.wedge_2_2(omega_a, omega_b)
            result = wedge.wedge_4_3(eta, Phi)

            assert not torch.any(torch.isnan(result))
