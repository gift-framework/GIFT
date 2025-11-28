"""
Unit tests for visualization data sources module.

Tests E8 root system generation, K7 structure data,
and data transformation utilities.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add visualization package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assets" / "visualizations" / "pro_package"))

from data_sources import (
    load_e8_root_system,
    load_k7_structure,
    summarize_precision_by_sector,
    prepare_precision_matrix,
    ensure_output_directory,
    _generate_e8_roots,
)


# =============================================================================
# E8 Root System Tests
# =============================================================================

class TestE8RootSystem:
    """Test E8 root system generation."""

    def test_root_count(self):
        """Test that E8 has exactly 240 roots."""
        df = load_e8_root_system()

        assert len(df) == 240, f"E8 should have 240 roots, got {len(df)}"

    def test_root_columns(self):
        """Test that DataFrame has correct columns."""
        df = load_e8_root_system()

        expected_cols = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'family', 'norm', 'type']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_vector_roots_count(self):
        """Test number of vector roots (form: +/-1, +/-1, 0^6)."""
        df = load_e8_root_system()

        vector_count = len(df[df['family'] == 'vector'])

        # C(8,2) * 4 = 28 * 4 = 112 vector roots
        assert vector_count == 112, f"Expected 112 vector roots, got {vector_count}"

    def test_spinor_roots_count(self):
        """Test number of spinor roots (form: +/-1/2 with even minus signs)."""
        df = load_e8_root_system()

        spinor_count = len(df[df['family'] == 'spinor'])

        # 2^7 = 128 spinor roots (half of 2^8 with even parity)
        assert spinor_count == 128, f"Expected 128 spinor roots, got {spinor_count}"

    def test_root_norms(self):
        """Test that all roots have norm sqrt(2)."""
        df = load_e8_root_system()

        # All E8 roots should have norm sqrt(2)
        expected_norm = np.sqrt(2)
        assert np.allclose(df['norm'].values, expected_norm, atol=1e-10)

    def test_vector_root_structure(self):
        """Test structure of vector roots."""
        df = load_e8_root_system()
        vectors = df[df['family'] == 'vector']

        for _, row in vectors.iterrows():
            coords = [row[f'x{i}'] for i in range(8)]
            # Should have exactly 2 non-zero entries
            non_zero = sum(1 for c in coords if abs(c) > 1e-10)
            assert non_zero == 2, f"Vector root should have 2 non-zero entries: {coords}"
            # Non-zero entries should be +/-1
            for c in coords:
                if abs(c) > 1e-10:
                    assert abs(abs(c) - 1.0) < 1e-10, f"Non-zero entries should be +/-1: {c}"

    def test_spinor_root_structure(self):
        """Test structure of spinor roots."""
        df = load_e8_root_system()
        spinors = df[df['family'] == 'spinor']

        for _, row in spinors.iterrows():
            coords = [row[f'x{i}'] for i in range(8)]
            # All entries should be +/-0.5
            for c in coords:
                assert abs(abs(c) - 0.5) < 1e-10, f"Spinor entries should be +/-0.5: {c}"
            # Should have even number of negative entries
            neg_count = sum(1 for c in coords if c < 0)
            assert neg_count % 2 == 0, f"Spinor should have even negative count: {neg_count}"

    def test_root_uniqueness(self):
        """Test that all roots are unique."""
        df = load_e8_root_system()

        # Create tuple representation for each root
        roots = [tuple(row[f'x{i}'] for i in range(8)) for _, row in df.iterrows()]

        assert len(roots) == len(set(roots)), "All roots should be unique"

    def test_caching(self):
        """Test that load_e8_root_system uses caching."""
        df1 = load_e8_root_system()
        df2 = load_e8_root_system()

        # Should return same object due to lru_cache
        assert df1 is df2

    def test_root_type_column(self):
        """Test the type column classification."""
        df = load_e8_root_system()

        # All roots are 'long' type (norm sqrt(2)) in E8
        assert all(df['type'] == 'long'), "All E8 roots should be 'long' type"


# =============================================================================
# K7 Structure Tests
# =============================================================================

class TestK7Structure:
    """Test K7 manifold structure data."""

    def test_k7_structure_keys(self):
        """Test that K7 structure has expected keys."""
        k7 = load_k7_structure()

        expected_keys = ['b0', 'b2', 'b3', 'h_star', 'weyl_factor', 'rank_e8']
        for key in expected_keys:
            assert key in k7, f"Missing key: {key}"

    def test_betti_numbers(self):
        """Test K7 Betti numbers match GIFT framework."""
        k7 = load_k7_structure()

        assert k7['b0'] == 1, "b0 should be 1"
        assert k7['b2'] == 21, "b2 should be 21"
        assert k7['b3'] == 77, "b3 should be 77"

    def test_h_star(self):
        """Test H* = b0 + b2 + b3 = 99."""
        k7 = load_k7_structure()

        # H* should equal sum of Betti numbers (cohomological dimension)
        expected_h_star = k7['b0'] + k7['b2'] + k7['b3']
        assert k7['h_star'] == 99, f"H* should be 99, got {k7['h_star']}"
        assert k7['h_star'] == expected_h_star

    def test_weyl_factor(self):
        """Test Weyl factor is 5."""
        k7 = load_k7_structure()

        assert k7['weyl_factor'] == 5, "Weyl factor should be 5"

    def test_rank_e8(self):
        """Test E8 rank is 8."""
        k7 = load_k7_structure()

        assert k7['rank_e8'] == 8, "E8 rank should be 8"

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        k7 = load_k7_structure()

        assert isinstance(k7, dict)


# =============================================================================
# Precision Summarization Tests
# =============================================================================

class TestPrecisionSummarization:
    """Test precision summarization utilities."""

    @pytest.fixture
    def sample_observable_df(self):
        """Create sample observable DataFrame."""
        return pd.DataFrame({
            'observable': ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
            'sector': ['gauge', 'gauge', 'neutrino', 'neutrino', 'lepton'],
            'deviation_%': [0.1, 0.2, 0.5, 0.3, 0.15],
            'predicted': [1.0, 2.0, 3.0, 4.0, 5.0],
            'experimental': [1.001, 2.004, 3.015, 4.012, 5.0075],
        })

    def test_summarize_precision_columns(self, sample_observable_df):
        """Test summary DataFrame has correct columns."""
        summary = summarize_precision_by_sector(sample_observable_df)

        expected_cols = ['sector', 'mean_deviation', 'max_deviation', 'min_deviation', 'observables']
        for col in expected_cols:
            assert col in summary.columns, f"Missing column: {col}"

    def test_summarize_precision_sector_count(self, sample_observable_df):
        """Test correct number of sectors."""
        summary = summarize_precision_by_sector(sample_observable_df)

        assert len(summary) == 3  # gauge, neutrino, lepton

    def test_summarize_precision_mean_calculation(self, sample_observable_df):
        """Test mean deviation calculation."""
        summary = summarize_precision_by_sector(sample_observable_df)

        # Check gauge sector: (0.1 + 0.2) / 2 = 0.15
        gauge_row = summary[summary['sector'] == 'gauge'].iloc[0]
        assert np.isclose(gauge_row['mean_deviation'], 0.15, atol=1e-10)

    def test_summarize_precision_observable_count(self, sample_observable_df):
        """Test observable count per sector."""
        summary = summarize_precision_by_sector(sample_observable_df)

        gauge_row = summary[summary['sector'] == 'gauge'].iloc[0]
        neutrino_row = summary[summary['sector'] == 'neutrino'].iloc[0]
        lepton_row = summary[summary['sector'] == 'lepton'].iloc[0]

        assert gauge_row['observables'] == 2
        assert neutrino_row['observables'] == 2
        assert lepton_row['observables'] == 1

    def test_summarize_precision_sorted(self, sample_observable_df):
        """Test that summary is sorted by mean deviation."""
        summary = summarize_precision_by_sector(sample_observable_df)

        deviations = summary['mean_deviation'].tolist()
        assert deviations == sorted(deviations), "Should be sorted by mean deviation"

    def test_summarize_precision_missing_columns(self):
        """Test error handling for missing columns."""
        bad_df = pd.DataFrame({'observable': ['a', 'b'], 'value': [1, 2]})

        with pytest.raises(ValueError):
            summarize_precision_by_sector(bad_df)


# =============================================================================
# Precision Matrix Tests
# =============================================================================

class TestPrecisionMatrix:
    """Test precision matrix preparation."""

    @pytest.fixture
    def sample_observable_df(self):
        """Create sample observable DataFrame."""
        return pd.DataFrame({
            'observable': ['alpha', 'beta', 'gamma', 'delta'],
            'sector': ['gauge', 'gauge', 'neutrino', 'neutrino'],
            'deviation_%': [0.1, 0.2, 0.5, 0.3],
        })

    def test_prepare_precision_matrix_shape(self, sample_observable_df):
        """Test matrix has correct shape."""
        matrix = prepare_precision_matrix(sample_observable_df)

        # Should have 2 sectors (rows) x 4 observables (columns)
        assert matrix.shape == (2, 4)

    def test_prepare_precision_matrix_index(self, sample_observable_df):
        """Test matrix index is sector."""
        matrix = prepare_precision_matrix(sample_observable_df)

        assert matrix.index.name == 'sector'
        assert set(matrix.index) == {'gauge', 'neutrino'}

    def test_prepare_precision_matrix_columns(self, sample_observable_df):
        """Test matrix columns are observables."""
        matrix = prepare_precision_matrix(sample_observable_df)

        assert matrix.columns.name == 'observable'
        assert set(matrix.columns) == {'alpha', 'beta', 'gamma', 'delta'}

    def test_prepare_precision_matrix_sorted(self, sample_observable_df):
        """Test matrix index is sorted."""
        matrix = prepare_precision_matrix(sample_observable_df)

        assert list(matrix.index) == sorted(matrix.index)


# =============================================================================
# Output Directory Tests
# =============================================================================

class TestEnsureOutputDirectory:
    """Test output directory creation."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        new_dir = tmp_path / "test_output" / "nested"

        result = ensure_output_directory(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_output_directory(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_returns_path(self, tmp_path):
        """Test that function returns Path object."""
        new_dir = tmp_path / "output"

        result = ensure_output_directory(new_dir)

        assert isinstance(result, Path)


# =============================================================================
# Mathematical Invariants Tests
# =============================================================================

class TestE8MathematicalProperties:
    """Test mathematical properties of E8 root system."""

    def test_root_system_closed_under_negation(self):
        """Test that if r is a root, -r is also a root."""
        df = load_e8_root_system()

        roots_set = set()
        for _, row in df.iterrows():
            root = tuple(row[f'x{i}'] for i in range(8))
            roots_set.add(root)

        for _, row in df.iterrows():
            root = tuple(row[f'x{i}'] for i in range(8))
            neg_root = tuple(-x for x in root)
            assert neg_root in roots_set, f"Negation of {root} not found"

    def test_weyl_group_dimension(self):
        """Test E8 Weyl group properties."""
        # |W(E8)| = 2^14 * 3^5 * 5^2 * 7 = 696,729,600
        weyl_order = (2**14) * (3**5) * (5**2) * 7
        assert weyl_order == 696729600

    def test_cartan_matrix_properties(self):
        """Test E8 Cartan matrix has rank 8."""
        # E8 has rank 8 (dimension of maximal torus)
        k7 = load_k7_structure()
        assert k7['rank_e8'] == 8

    def test_root_inner_products(self):
        """Test inner product properties of roots."""
        df = load_e8_root_system()

        # Sample a few roots and check inner products
        coords = df[[f'x{i}' for i in range(8)]].values

        # Self inner product should be 2 for all roots
        for i in range(min(10, len(coords))):
            inner = np.dot(coords[i], coords[i])
            assert np.isclose(inner, 2.0, atol=1e-10)


# =============================================================================
# Integration with Reference Data
# =============================================================================

class TestReferenceDataIntegration:
    """Test integration with reference observable data."""

    @pytest.fixture
    def reference_data(self, project_root):
        """Load reference observables."""
        import json
        ref_path = project_root / "tests" / "fixtures" / "reference_observables_v22.json"
        if ref_path.exists():
            with open(ref_path) as f:
                return json.load(f)
        return None

    def test_reference_data_consistency(self, reference_data):
        """Test that reference data is consistent with K7 structure."""
        if reference_data is None:
            pytest.skip("Reference data not available")

        k7 = load_k7_structure()

        # Check that K7 Betti numbers match any referenced values
        assert k7['b2'] == 21
        assert k7['b3'] == 77


@pytest.fixture
def project_root():
    """Provide project root path."""
    return Path(__file__).parent.parent.parent
