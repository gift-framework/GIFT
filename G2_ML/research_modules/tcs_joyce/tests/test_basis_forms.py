"""
Unit tests for G2 3-form basis functions.

Tests:
- Index conversion functions
- Canonical G2 3-form
- Orthogonal basis generation
- G2BasisLibrary class
"""

import pytest
import torch
import sys
from pathlib import Path

# Add tcs_joyce to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from basis_forms import (
        all_3form_indices,
        triple_to_index,
        index_to_triple,
        canonical_g2_indices,
        canonical_g2_coefficients,
        canonical_g2_3form_components,
        generate_g2_orthogonal_basis,
        G2BasisLibrary,
        get_g2_representation_decomposition,
    )
    BASIS_AVAILABLE = True
except ImportError as e:
    BASIS_AVAILABLE = False
    BASIS_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not BASIS_AVAILABLE,
    reason=f"basis_forms module not available: {BASIS_IMPORT_ERROR if not BASIS_AVAILABLE else ''}"
)


# =============================================================================
# Index Functions Tests
# =============================================================================

class TestIndexFunctions:
    """Test index conversion functions."""

    def test_all_3form_indices_count(self):
        """Should have exactly 35 indices (C(7,3))."""
        indices = all_3form_indices()
        assert len(indices) == 35

    def test_all_3form_indices_ordered(self):
        """Each triple should be strictly increasing."""
        indices = all_3form_indices()
        for i, j, k in indices:
            assert i < j < k

    def test_all_3form_indices_range(self):
        """All indices should be in [0, 6]."""
        indices = all_3form_indices()
        for i, j, k in indices:
            assert 0 <= i < 7
            assert 0 <= j < 7
            assert 0 <= k < 7

    def test_triple_to_index_roundtrip(self):
        """triple_to_index and index_to_triple should be inverses."""
        for idx in range(35):
            triple = index_to_triple(idx)
            assert triple_to_index(*triple) == idx

    def test_triple_to_index_unsorted(self):
        """Should work with unsorted triples."""
        # (2, 1, 0) should map same as (0, 1, 2)
        idx_sorted = triple_to_index(0, 1, 2)
        idx_unsorted = triple_to_index(2, 1, 0)
        assert idx_sorted == idx_unsorted

    def test_index_to_triple_range(self):
        """All indices 0-34 should have valid triples."""
        for idx in range(35):
            triple = index_to_triple(idx)
            assert len(triple) == 3
            assert triple[0] < triple[1] < triple[2]


# =============================================================================
# Canonical G2 3-form Tests
# =============================================================================

class TestCanonicalG2:
    """Test canonical G2 3-form."""

    def test_canonical_g2_indices_count(self):
        """Canonical G2 form has exactly 7 terms."""
        indices = canonical_g2_indices()
        assert len(indices) == 7

    def test_canonical_g2_indices_signs(self):
        """Should have 4 positive and 3 negative signs."""
        indices = canonical_g2_indices()
        signs = [s for _, s in indices]
        assert signs.count(1) == 4
        assert signs.count(-1) == 3

    def test_canonical_g2_coefficients_shape(self):
        """Coefficients should have shape (35,)."""
        coeffs = canonical_g2_coefficients()
        assert coeffs.shape == (35,)

    def test_canonical_g2_coefficients_sparsity(self):
        """Only 7 non-zero coefficients."""
        coeffs = canonical_g2_coefficients()
        nonzero = (coeffs != 0).sum().item()
        assert nonzero == 7

    def test_canonical_g2_coefficients_values(self):
        """Non-zero values should be +/-1."""
        coeffs = canonical_g2_coefficients()
        nonzero_vals = coeffs[coeffs != 0]
        assert all(abs(v) == 1.0 for v in nonzero_vals)

    def test_canonical_g2_3form_components_shape(self):
        """Components without batch should have shape (35,)."""
        comps = canonical_g2_3form_components()
        assert comps.shape == (35,)

    def test_canonical_g2_3form_components_batched(self):
        """Components with batch should have shape (batch, 35)."""
        comps = canonical_g2_3form_components(batch_size=10)
        assert comps.shape == (10, 35)

    def test_canonical_g2_3form_device(self):
        """Should respect device parameter."""
        comps = canonical_g2_3form_components(device=torch.device('cpu'))
        assert comps.device == torch.device('cpu')

    def test_canonical_g2_3form_dtype(self):
        """Should respect dtype parameter."""
        comps = canonical_g2_3form_components(dtype=torch.float64)
        assert comps.dtype == torch.float64


# =============================================================================
# Orthogonal Basis Tests
# =============================================================================

class TestOrthogonalBasis:
    """Test orthogonal basis generation."""

    def test_basis_shape(self):
        """Basis should have shape (35, 35)."""
        basis = generate_g2_orthogonal_basis()
        assert basis.shape == (35, 35)

    def test_basis_orthogonality(self):
        """Basis vectors should be pairwise orthogonal."""
        basis = generate_g2_orthogonal_basis()
        gram = basis @ basis.T
        # Off-diagonal elements should be ~0
        off_diag = gram - torch.diag(torch.diag(gram))
        assert off_diag.abs().max() < 1e-5

    def test_basis_normality(self):
        """Basis vectors should have unit norm."""
        basis = generate_g2_orthogonal_basis()
        norms = torch.norm(basis, dim=1)
        assert torch.allclose(norms, torch.ones(35), atol=1e-5)

    def test_basis_includes_canonical(self):
        """First basis element should be canonical G2 form (normalized)."""
        basis = generate_g2_orthogonal_basis(include_canonical=True)
        canonical = canonical_g2_coefficients()
        canonical = canonical / torch.norm(canonical)
        # First row should be parallel to canonical
        dot = torch.dot(basis[0], canonical).abs()
        assert dot > 0.99

    def test_basis_device(self):
        """Should respect device parameter."""
        basis = generate_g2_orthogonal_basis(device=torch.device('cpu'))
        assert basis.device == torch.device('cpu')

    def test_basis_dtype(self):
        """Should respect dtype parameter."""
        basis = generate_g2_orthogonal_basis(dtype=torch.float64)
        assert basis.dtype == torch.float64


# =============================================================================
# G2BasisLibrary Tests
# =============================================================================

class TestG2BasisLibrary:
    """Test G2BasisLibrary class."""

    @pytest.fixture
    def library(self):
        """Create a G2BasisLibrary instance."""
        return G2BasisLibrary()

    def test_local_basis_shape(self, library):
        """Local basis should have shape (35, 35)."""
        assert library.local_basis.shape == (35, 35)

    def test_global_left_basis_shape(self, library):
        """Global left basis should have shape (14, 35)."""
        assert library.global_left_basis.shape == (14, 35)

    def test_global_right_basis_shape(self, library):
        """Global right basis should have shape (14, 35)."""
        assert library.global_right_basis.shape == (14, 35)

    def test_global_neck_basis_shape(self, library):
        """Global neck basis should have shape (14, 35)."""
        assert library.global_neck_basis.shape == (14, 35)

    def test_get_local_forms_shape(self, library):
        """get_local_forms should return correct shape."""
        forms = library.get_local_forms(batch_size=100)
        assert forms.shape == (100, 35)

    def test_get_global_templates_keys(self, library):
        """get_global_templates should return dict with correct keys."""
        templates = library.get_global_templates()
        assert set(templates.keys()) == {'left', 'right', 'neck'}

    def test_get_global_templates_shapes(self, library):
        """Global templates should have shape (14, 35)."""
        templates = library.get_global_templates()
        assert templates['left'].shape == (14, 35)
        assert templates['right'].shape == (14, 35)
        assert templates['neck'].shape == (14, 35)

    def test_orthonormalize_against_local(self, library):
        """Orthonormalization should produce orthogonal vectors."""
        # Use global left basis as test input
        result = library.orthonormalize_against_local(library.global_left_basis)
        if result.shape[0] > 0:
            # Check orthogonality
            gram = result @ result.T
            off_diag = gram - torch.diag(torch.diag(gram))
            assert off_diag.abs().max() < 1e-4


# =============================================================================
# Representation Decomposition Tests
# =============================================================================

class TestRepresentationDecomposition:
    """Test G2 representation decomposition."""

    def test_decomposition_keys(self):
        """Should have expected representation keys."""
        decomp = get_g2_representation_decomposition()
        assert 'trivial_1' in decomp
        assert 'fundamental_7' in decomp
        assert 'adjoint_27' in decomp

    def test_decomposition_dimensions(self):
        """Dimensions should sum to 35."""
        decomp = get_g2_representation_decomposition()
        total = (
            len(decomp['trivial_1']) +
            len(decomp['fundamental_7']) +
            len(decomp['adjoint_27'])
        )
        assert total == 35

    def test_trivial_representation(self):
        """Trivial representation should have dimension 1."""
        decomp = get_g2_representation_decomposition()
        assert len(decomp['trivial_1']) == 1

    def test_fundamental_representation(self):
        """Fundamental representation should have dimension 7."""
        decomp = get_g2_representation_decomposition()
        assert len(decomp['fundamental_7']) == 7

    def test_adjoint_representation(self):
        """Adjoint representation should have dimension 27."""
        decomp = get_g2_representation_decomposition()
        assert len(decomp['adjoint_27']) == 27
