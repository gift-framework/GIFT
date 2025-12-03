"""
Unit tests for G2 constraint functions.

Tests:
- 3-form index generation
- Antisymmetric tensor expansion
- Metric extraction from phi
- Determinant constraint
- G2 positivity check
- Standard G2 3-form
"""

import pytest
import torch
import sys
from pathlib import Path

# Add variational_g2/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from constraints import (
        generate_3form_indices,
        expand_to_antisymmetric,
        metric_from_phi,
        det_constraint_loss,
        g2_positivity_check,
        standard_g2_phi,
        permutation_sign,
        G2_STANDARD_INDICES,
        G2_STANDARD_SIGNS,
    )
    CONSTRAINTS_AVAILABLE = True
except ImportError as e:
    CONSTRAINTS_AVAILABLE = False
    CONSTRAINTS_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not CONSTRAINTS_AVAILABLE,
    reason=f"constraints module not available: {CONSTRAINTS_IMPORT_ERROR if not CONSTRAINTS_AVAILABLE else ''}"
)


# =============================================================================
# Index Generation Tests
# =============================================================================

class TestIndexGeneration:
    """Test 3-form index generation."""

    def test_3form_indices_count(self):
        """Should generate exactly 35 indices (C(7,3))."""
        indices = generate_3form_indices()
        assert indices.shape == (35, 3)

    def test_3form_indices_ordered(self):
        """Indices should be strictly increasing (i < j < k)."""
        indices = generate_3form_indices()
        for row in indices:
            assert row[0] < row[1] < row[2]

    def test_3form_indices_range(self):
        """All indices should be in [0, 6]."""
        indices = generate_3form_indices()
        assert indices.min() >= 0
        assert indices.max() <= 6


# =============================================================================
# Antisymmetric Expansion Tests
# =============================================================================

class TestAntisymmetricExpansion:
    """Test antisymmetric tensor expansion."""

    def test_expansion_shape(self):
        """Expansion should produce (7, 7, 7) tensor."""
        phi_comp = torch.randn(35)
        phi_full = expand_to_antisymmetric(phi_comp)
        assert phi_full.shape == (7, 7, 7)

    def test_expansion_batched_shape(self):
        """Batched expansion should work."""
        phi_comp = torch.randn(10, 35)
        phi_full = expand_to_antisymmetric(phi_comp)
        assert phi_full.shape == (10, 7, 7, 7)

    def test_expansion_antisymmetry_ijk(self):
        """phi[i,j,k] = -phi[i,k,j]."""
        phi_comp = torch.randn(35)
        phi_full = expand_to_antisymmetric(phi_comp)
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if i != j and j != k and i != k:
                        assert torch.isclose(
                            phi_full[i, j, k],
                            -phi_full[i, k, j],
                            atol=1e-6
                        )

    def test_expansion_antisymmetry_ijk_jik(self):
        """phi[i,j,k] = -phi[j,i,k]."""
        phi_comp = torch.randn(35)
        phi_full = expand_to_antisymmetric(phi_comp)
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    if i != j and j != k and i != k:
                        assert torch.isclose(
                            phi_full[i, j, k],
                            -phi_full[j, i, k],
                            atol=1e-6
                        )

    def test_expansion_cyclic_symmetry(self):
        """phi[i,j,k] = phi[j,k,i] = phi[k,i,j]."""
        phi_comp = torch.randn(35)
        phi_full = expand_to_antisymmetric(phi_comp)
        for i in range(7):
            for j in range(i + 1, 7):
                for k in range(j + 1, 7):
                    # Cyclic permutations have same sign
                    assert torch.isclose(
                        phi_full[i, j, k],
                        phi_full[j, k, i],
                        atol=1e-6
                    )
                    assert torch.isclose(
                        phi_full[i, j, k],
                        phi_full[k, i, j],
                        atol=1e-6
                    )


# =============================================================================
# Metric Extraction Tests
# =============================================================================

class TestMetricFromPhi:
    """Test metric extraction from 3-form."""

    def test_metric_shape(self):
        """Metric should be 7x7."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        g = metric_from_phi(phi_full)
        assert g.shape == (7, 7)

    def test_metric_symmetric(self):
        """Metric should be symmetric."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        g = metric_from_phi(phi_full)
        assert torch.allclose(g, g.T, atol=1e-6)

    def test_metric_from_components(self):
        """Should work with 35-component input."""
        phi = standard_g2_phi()
        g = metric_from_phi(phi)
        assert g.shape == (7, 7)

    def test_metric_batched(self):
        """Should work with batched input."""
        phi = torch.randn(10, 7, 7, 7)
        # Make it antisymmetric
        phi = phi - phi.transpose(-2, -1)
        g = metric_from_phi(phi)
        assert g.shape == (10, 7, 7)

    def test_standard_g2_metric_identity(self):
        """Standard G2 form should give identity-like metric."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        g = metric_from_phi(phi_full)
        # The metric from standard G2 should be proportional to identity
        # (it's the flat metric on R7)
        diag = torch.diag(g)
        off_diag = g - torch.diag(diag)
        # Check diagonal dominance
        assert diag.min() > 0
        assert off_diag.abs().max() < diag.mean() * 0.5


# =============================================================================
# Determinant Constraint Tests
# =============================================================================

class TestDetConstraint:
    """Test determinant constraint loss."""

    def test_det_loss_scalar(self):
        """Loss should be scalar."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        loss = det_constraint_loss(phi_full)
        assert loss.dim() == 0

    def test_det_loss_non_negative(self):
        """Loss should be non-negative."""
        phi = torch.randn(35)
        phi_full = expand_to_antisymmetric(phi)
        loss = det_constraint_loss(phi_full)
        assert loss >= 0

    def test_det_loss_target_value(self):
        """Default target should be 65/32."""
        # The default target_det is 65/32 = 2.03125
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        g = metric_from_phi(phi_full)
        det_g = torch.det(g)
        # Loss = (det_g - 65/32)^2
        expected_loss = (det_g - 65.0 / 32.0) ** 2
        actual_loss = det_constraint_loss(phi_full)
        assert torch.isclose(actual_loss, expected_loss, atol=1e-6)

    def test_det_loss_reduction_none(self):
        """With reduction='none', should return per-sample loss."""
        phi = torch.randn(10, 35)
        loss = det_constraint_loss(phi, reduction='none')
        assert loss.shape == (10,)


# =============================================================================
# G2 Positivity Tests
# =============================================================================

class TestG2Positivity:
    """Test G2 cone positivity check."""

    def test_standard_g2_positive(self):
        """Standard G2 form should be in G2 cone."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        violation = g2_positivity_check(phi_full)
        assert violation < 1e-5

    def test_positivity_returns_eigenvalues(self):
        """Should return eigenvalues when requested."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        violation, eigenvalues = g2_positivity_check(phi_full, return_eigenvalues=True)
        assert eigenvalues is not None
        assert eigenvalues.shape == (7,)

    def test_standard_g2_positive_eigenvalues(self):
        """Standard G2 should have positive eigenvalues."""
        phi = standard_g2_phi()
        phi_full = expand_to_antisymmetric(phi)
        _, eigenvalues = g2_positivity_check(phi_full, return_eigenvalues=True)
        assert eigenvalues.min() > -1e-5


# =============================================================================
# Standard G2 3-form Tests
# =============================================================================

class TestStandardG2Phi:
    """Test standard G2 3-form."""

    def test_standard_g2_shape(self):
        """Should have shape (35,)."""
        phi = standard_g2_phi()
        assert phi.shape == (35,)

    def test_standard_g2_sparsity(self):
        """Should have exactly 7 non-zero components."""
        phi = standard_g2_phi()
        nonzero = (phi != 0).sum().item()
        assert nonzero == 7

    def test_standard_g2_values(self):
        """Non-zero values should be +/-1."""
        phi = standard_g2_phi()
        nonzero_vals = phi[phi != 0]
        assert all(abs(v) == 1.0 for v in nonzero_vals)

    def test_standard_g2_device(self):
        """Should respect device parameter."""
        phi = standard_g2_phi(device=torch.device('cpu'))
        assert phi.device == torch.device('cpu')

    def test_standard_g2_dtype(self):
        """Should respect dtype parameter."""
        phi = standard_g2_phi(dtype=torch.float64)
        assert phi.dtype == torch.float64


# =============================================================================
# Permutation Sign Tests
# =============================================================================

class TestPermutationSign:
    """Test permutation sign calculation."""

    def test_identity_permutation(self):
        """Identity permutation has sign +1."""
        assert permutation_sign([0, 1, 2, 3]) == 1

    def test_single_swap(self):
        """Single swap has sign -1."""
        assert permutation_sign([1, 0, 2, 3]) == -1
        assert permutation_sign([0, 2, 1, 3]) == -1
        assert permutation_sign([0, 1, 3, 2]) == -1

    def test_double_swap(self):
        """Double swap has sign +1."""
        assert permutation_sign([1, 0, 3, 2]) == 1

    def test_cyclic_3(self):
        """3-cycle has sign +1."""
        assert permutation_sign([1, 2, 0, 3]) == 1

    def test_anticyclic_3(self):
        """Anti-3-cycle has sign -1."""
        assert permutation_sign([2, 0, 1, 3]) == 1  # This is actually a 3-cycle too


# =============================================================================
# G2 Constants Tests
# =============================================================================

class TestG2Constants:
    """Test G2 structure constants."""

    def test_g2_indices_count(self):
        """Should have 7 index triples."""
        assert len(G2_STANDARD_INDICES) == 7

    def test_g2_signs_count(self):
        """Should have 7 signs."""
        assert len(G2_STANDARD_SIGNS) == 7

    def test_g2_signs_values(self):
        """Signs should be +/-1."""
        assert all(s in [-1, 1] for s in G2_STANDARD_SIGNS)

    def test_g2_signs_sum(self):
        """4 positive, 3 negative."""
        assert sum(G2_STANDARD_SIGNS) == 1  # 4 - 3 = 1
