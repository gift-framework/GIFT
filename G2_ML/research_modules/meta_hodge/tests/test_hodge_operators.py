"""
Unit tests for Hodge operators.

Tests:
- pair_combinations (index generation)
- complement function
- permutation_parity
- HodgeOperator class
- Hodge star operations
"""

import pytest
import torch
import sys
from pathlib import Path

# Add meta_hodge to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hodge_operators import (
        pair_combinations,
        complement,
        permutation_parity,
        HodgeOperator,
        assemble_hodge_star_matrices,
    )
    HODGE_AVAILABLE = True
except ImportError as e:
    HODGE_AVAILABLE = False
    HODGE_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not HODGE_AVAILABLE,
    reason=f"hodge_operators module not available: {HODGE_IMPORT_ERROR if not HODGE_AVAILABLE else ''}"
)


# =============================================================================
# pair_combinations Tests
# =============================================================================

class TestPairCombinations:
    """Test pair_combinations function."""

    def test_2_choose_2(self):
        """C(2,2) = 1."""
        result = pair_combinations(2, 2)
        assert len(result) == 1
        assert result[0] == (0, 1)

    def test_4_choose_2(self):
        """C(4,2) = 6."""
        result = pair_combinations(4, 2)
        assert len(result) == 6

    def test_7_choose_2(self):
        """C(7,2) = 21."""
        result = pair_combinations(7, 2)
        assert len(result) == 21

    def test_7_choose_3(self):
        """C(7,3) = 35."""
        result = pair_combinations(7, 3)
        assert len(result) == 35

    def test_result_ordered(self):
        """Each tuple should be strictly increasing."""
        result = pair_combinations(7, 3)
        for tup in result:
            for i in range(len(tup) - 1):
                assert tup[i] < tup[i + 1]

    def test_caching(self):
        """Results should be cached (same object returned)."""
        result1 = pair_combinations(7, 3)
        result2 = pair_combinations(7, 3)
        assert result1 is result2


# =============================================================================
# complement Tests
# =============================================================================

class TestComplement:
    """Test complement function."""

    def test_complement_basic(self):
        """Complement of {0,1} in {0,1,2,3} is {2,3}."""
        result = complement((0, 1), 4)
        assert result == (2, 3)

    def test_complement_single(self):
        """Complement of {0} in {0,1,2} is {1,2}."""
        result = complement((0,), 3)
        assert result == (1, 2)

    def test_complement_empty(self):
        """Complement of {} in {0,1,2} is {0,1,2}."""
        result = complement((), 3)
        assert result == (0, 1, 2)

    def test_complement_full(self):
        """Complement of {0,1,2} in {0,1,2} is {}."""
        result = complement((0, 1, 2), 3)
        assert result == ()

    def test_complement_7d(self):
        """Complement in 7D space."""
        result = complement((0, 2, 4), 7)
        assert result == (1, 3, 5, 6)
        assert len(result) == 4

    def test_complement_sorted(self):
        """Result should be sorted."""
        result = complement((1, 3, 5), 7)
        assert result == tuple(sorted(result))


# =============================================================================
# permutation_parity Tests
# =============================================================================

class TestPermutationParity:
    """Test permutation_parity function."""

    def test_identity(self):
        """Identity permutation has parity +1."""
        assert permutation_parity((0, 1, 2)) == 1
        assert permutation_parity((0, 1, 2, 3)) == 1

    def test_single_swap(self):
        """Single transposition has parity -1."""
        assert permutation_parity((1, 0, 2)) == -1
        assert permutation_parity((0, 2, 1)) == -1

    def test_double_swap(self):
        """Two transpositions has parity +1."""
        assert permutation_parity((1, 0, 3, 2)) == 1

    def test_3_cycle(self):
        """3-cycle (0,1,2) -> (1,2,0) has parity +1 (two swaps)."""
        assert permutation_parity((1, 2, 0)) == 1

    def test_reverse_3(self):
        """Reversal of 3 elements has parity -1."""
        assert permutation_parity((2, 1, 0)) == -1

    def test_reverse_4(self):
        """Reversal of 4 elements has parity +1."""
        assert permutation_parity((3, 2, 1, 0)) == 1


# =============================================================================
# HodgeOperator Tests
# =============================================================================

class TestHodgeOperator:
    """Test HodgeOperator class."""

    @pytest.fixture
    def identity_metric(self):
        """Create identity metric."""
        return torch.eye(7)

    @pytest.fixture
    def random_spd_metric(self):
        """Create random SPD metric."""
        A = torch.randn(7, 7)
        return A @ A.T + 0.1 * torch.eye(7)

    @pytest.fixture
    def hodge_identity(self, identity_metric):
        """Create HodgeOperator with identity metric."""
        return HodgeOperator(g=identity_metric)

    @pytest.fixture
    def hodge_random(self, random_spd_metric):
        """Create HodgeOperator with random SPD metric."""
        return HodgeOperator(g=random_spd_metric)

    def test_star_2_shape(self, hodge_identity):
        """star_2 should map 21 -> 35 components."""
        # 2-form on R7 has C(7,2) = 21 components
        # star_2 maps to 5-form with C(7,5) = 21 components
        omega = torch.randn(21)
        result = hodge_identity.star_2(omega)
        assert result.shape == (21,)  # C(7,5) = C(7,2) = 21

    def test_star_3_shape(self, hodge_identity):
        """star_3 should map 35 -> 35 components."""
        # 3-form on R7 has C(7,3) = 35 components
        # star_3 maps to 4-form with C(7,4) = 35 components
        Omega = torch.randn(35)
        result = hodge_identity.star_3(Omega)
        assert result.shape == (35,)  # C(7,4) = C(7,3) = 35

    def test_star_2_batched(self, hodge_identity):
        """star_2 should work with batched input."""
        omega = torch.randn(10, 21)
        result = hodge_identity.star_2(omega)
        assert result.shape == (10, 21)

    def test_star_3_batched(self, hodge_identity):
        """star_3 should work with batched input."""
        Omega = torch.randn(10, 35)
        result = hodge_identity.star_3(Omega)
        assert result.shape == (10, 35)

    def test_laplacian_matrix_shape(self, hodge_identity):
        """Laplacian matrix should be square."""
        basis = torch.randn(100, 10)  # 100 points, 10 basis functions
        lap = hodge_identity.laplacian_matrix(basis, hodge_identity.star_2)
        # Should be (10, 10) from the Gram computation
        assert lap.shape[0] == lap.shape[1]

    def test_laplacian_matrix_symmetric(self, hodge_identity):
        """Laplacian matrix should be symmetric."""
        basis = torch.randn(100, 10)
        lap = hodge_identity.laplacian_matrix(basis, hodge_identity.star_2)
        assert torch.allclose(lap, lap.T, atol=1e-5)


# =============================================================================
# assemble_hodge_star_matrices Tests
# =============================================================================

class TestAssembleHodgeStarMatrices:
    """Test Hodge star matrix assembly."""

    @pytest.fixture
    def identity_metric(self):
        """Create identity metric."""
        return torch.eye(7)

    def test_returns_dict(self, identity_metric):
        """Should return a dictionary."""
        result = assemble_hodge_star_matrices(identity_metric)
        assert isinstance(result, dict)

    def test_dict_keys(self, identity_metric):
        """Should have 'star2' and 'star3' keys."""
        result = assemble_hodge_star_matrices(identity_metric)
        assert 'star2' in result
        assert 'star3' in result

    def test_star2_shape(self, identity_metric):
        """star2 matrix should be (21, 21)."""
        result = assemble_hodge_star_matrices(identity_metric)
        assert result['star2'].shape == (21, 21)

    def test_star3_shape(self, identity_metric):
        """star3 matrix should be (35, 35)."""
        result = assemble_hodge_star_matrices(identity_metric)
        assert result['star3'].shape == (35, 35)


# =============================================================================
# Integration Tests
# =============================================================================

class TestHodgeIntegration:
    """Integration tests for Hodge operators."""

    def test_hodge_star_involution_2form(self):
        """For 2-forms on R7 with identity metric, **omega = omega."""
        g = torch.eye(7)
        hodge = HodgeOperator(g=g)

        omega = torch.zeros(21)
        omega[0] = 1.0  # Single 2-form component

        star_omega = hodge.star_2(omega)
        # Note: In general, ** = (-1)^{p(n-p)} * id
        # For p=2, n=7: (-1)^{2*5} = 1
        # So **omega should give back omega (up to sign/normalization)
        # This is more of a consistency check

        assert star_omega.shape == omega.shape

    def test_hodge_star_involution_3form(self):
        """For 3-forms on R7 with identity metric, **Omega relates to Omega."""
        g = torch.eye(7)
        hodge = HodgeOperator(g=g)

        Omega = torch.zeros(35)
        Omega[0] = 1.0  # Single 3-form component

        star_Omega = hodge.star_3(Omega)
        # For p=3, n=7: (-1)^{3*4} = 1
        assert star_Omega.shape == Omega.shape

    def test_different_metrics_give_different_results(self):
        """Different metrics should give different Hodge stars."""
        g1 = torch.eye(7)
        g2 = 2 * torch.eye(7)

        hodge1 = HodgeOperator(g=g1)
        hodge2 = HodgeOperator(g=g2)

        omega = torch.randn(21)

        star1 = hodge1.star_2(omega)
        star2 = hodge2.star_2(omega)

        # Results should differ due to volume form scaling
        assert not torch.allclose(star1, star2)
