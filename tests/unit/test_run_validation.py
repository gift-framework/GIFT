"""
Tests for run_validation_v21.py - Statistical Validation Runner

These tests cover:
- GIFTFrameworkV21 class initialization and parameter handling
- Observable computation methods
- Experimental data loading
- Monte Carlo orchestration
- Result serialization and output format
- Edge cases and error handling

Author: GIFT Framework
Date: 2025-11-27
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add statistical_validation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation_v21 import GIFTFrameworkV21


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def framework():
    """Create a default GIFTFrameworkV21 instance."""
    return GIFTFrameworkV21()


@pytest.fixture
def framework_custom():
    """Create a custom-parameterized GIFTFrameworkV21 instance."""
    return GIFTFrameworkV21(
        p2=2.0,
        beta0=np.pi / 8,
        weyl_factor=5.0,
        det_g=2.031,
        torsion_magnitude=0.0164
    )


# =============================================================================
# Test: Framework Initialization
# =============================================================================

class TestFrameworkInitialization:
    """Tests for GIFTFrameworkV21 initialization."""

    def test_default_initialization(self, framework):
        """Test framework initializes with correct default values."""
        assert framework.p2 == 2.0
        assert np.isclose(framework.beta0, np.pi / 8)
        assert framework.weyl_factor == 5.0
        assert framework.det_g == 2.031
        assert framework.torsion_magnitude == 0.0164

    def test_topological_invariants(self, framework):
        """Test topological invariants are correctly set."""
        assert framework.b2_K7 == 21
        assert framework.b3_K7 == 77
        assert framework.H_star == 99
        assert framework.dim_E8 == 248
        assert framework.rank_E8 == 8
        assert framework.dim_G2 == 14
        assert framework.dim_K7 == 7
        assert framework.D_bulk == 11
        assert framework.dim_J3O == 27

    def test_derived_parameters(self, framework):
        """Test derived parameters are computed correctly."""
        # xi = (weyl_factor / p2) * beta0 = 5/2 * pi/8 = 5*pi/16
        expected_xi = (5.0 / 2.0) * (np.pi / 8)
        assert np.isclose(framework.xi, expected_xi, rtol=1e-10)

        # tau = (496 * 21) / (27 * 99) = 3.89675...
        expected_tau = (496 * 21) / (27 * 99)
        assert np.isclose(framework.tau, expected_tau, rtol=1e-10)

    def test_mathematical_constants(self, framework):
        """Test mathematical constants are correct."""
        assert np.isclose(framework.phi, (1 + np.sqrt(5)) / 2)
        assert np.isclose(framework.zeta3, 1.2020569031595942)
        assert np.isclose(framework.gamma_EM, 0.5772156649)

    def test_torsion_tensor_components(self, framework):
        """Test torsion tensor components are set."""
        assert hasattr(framework, 'T_ephi_pi')
        assert hasattr(framework, 'T_piphi_e')
        assert hasattr(framework, 'T_epi_phi')
        assert framework.T_ephi_pi == -4.89
        assert framework.T_piphi_e == -0.45
        assert framework.T_epi_phi == 3.1e-5

    def test_custom_beta0_initialization(self):
        """Test custom beta0 parameter is used when provided."""
        custom_beta0 = np.pi / 4
        framework = GIFTFrameworkV21(beta0=custom_beta0)
        assert np.isclose(framework.beta0, custom_beta0)

        # xi should update accordingly
        expected_xi = (framework.weyl_factor / framework.p2) * custom_beta0
        assert np.isclose(framework.xi, expected_xi)

    def test_parameter_independence(self, framework):
        """Test that parameters are independent."""
        fw1 = GIFTFrameworkV21(p2=1.5)
        fw2 = GIFTFrameworkV21(p2=2.5)

        # p2 should differ
        assert fw1.p2 != fw2.p2

        # Topological invariants should be same
        assert fw1.b2_K7 == fw2.b2_K7
        assert fw1.H_star == fw2.H_star


# =============================================================================
# Test: Experimental Data Loading
# =============================================================================

class TestExperimentalData:
    """Tests for experimental data loading."""

    def test_experimental_data_loaded(self, framework):
        """Test experimental data is loaded."""
        assert hasattr(framework, 'experimental_data')
        assert isinstance(framework.experimental_data, dict)
        assert len(framework.experimental_data) > 0

    def test_experimental_data_structure(self, framework):
        """Test experimental data has correct structure (value, uncertainty)."""
        for key, data in framework.experimental_data.items():
            assert isinstance(data, tuple), f"{key} should be a tuple"
            assert len(data) == 2, f"{key} should have (value, uncertainty)"
            assert isinstance(data[0], (int, float)), f"{key} value should be numeric"
            assert isinstance(data[1], (int, float)), f"{key} uncertainty should be numeric"
            assert data[1] >= 0, f"{key} uncertainty should be non-negative"

    def test_key_experimental_values(self, framework):
        """Test key experimental values are present."""
        required_keys = [
            'alpha_inv', 'sin2_theta_W', 'alpha_s_MZ',
            'theta_12', 'theta_13', 'theta_23', 'delta_CP',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e',
            'm_s_m_d', 'Omega_DE', 'H0'
        ]
        for key in required_keys:
            assert key in framework.experimental_data, f"Missing key: {key}"

    def test_experimental_values_reasonable(self, framework):
        """Test experimental values are within reasonable ranges."""
        # Fine structure constant inverse
        alpha_inv = framework.experimental_data['alpha_inv'][0]
        assert 136 < alpha_inv < 138

        # Weak mixing angle
        sin2_theta_W = framework.experimental_data['sin2_theta_W'][0]
        assert 0.20 < sin2_theta_W < 0.25

        # Neutrino mixing angles (degrees)
        theta_12 = framework.experimental_data['theta_12'][0]
        assert 30 < theta_12 < 40

        # Dark energy fraction
        omega_de = framework.experimental_data['Omega_DE'][0]
        assert 0.6 < omega_de < 0.8


# =============================================================================
# Test: Gauge Sector Computations
# =============================================================================

class TestGaugeSectorComputations:
    """Tests for gauge sector observable computations."""

    def test_alpha_inverse_computation(self, framework):
        """Test fine structure constant inverse calculation."""
        alpha_inv = framework.compute_alpha_inverse()

        # Should be close to experimental value ~137.036
        assert 136 < alpha_inv < 138

        # Check it's computed from three components
        algebraic = (framework.dim_E8 + framework.rank_E8) / 2  # = 128
        bulk_impedance = framework.H_star / framework.D_bulk     # = 9
        torsional = framework.det_g * framework.torsion_magnitude

        expected = algebraic + bulk_impedance + torsional
        assert np.isclose(alpha_inv, expected, rtol=1e-10)

    def test_alpha_inverse_components(self, framework):
        """Test individual components of alpha inverse."""
        # Algebraic: (248 + 8) / 2 = 128
        algebraic = (framework.dim_E8 + framework.rank_E8) / 2
        assert np.isclose(algebraic, 128.0)

        # Bulk impedance: 99 / 11 = 9
        bulk = framework.H_star / framework.D_bulk
        assert np.isclose(bulk, 9.0)

    def test_sin2_theta_W_computation(self, framework):
        """Test weak mixing angle calculation."""
        sin2_theta_W = framework.compute_sin2_theta_W()

        # Should be close to experimental value ~0.231
        assert 0.20 < sin2_theta_W < 0.25

        # Verify formula: zeta3 * gamma_EM / 3
        expected = framework.zeta3 * framework.gamma_EM / 3.0
        assert np.isclose(sin2_theta_W, expected)

    def test_alpha_s_computation(self, framework):
        """Test strong coupling constant calculation."""
        alpha_s = framework.compute_alpha_s()

        # Should be close to experimental value ~0.118
        assert 0.10 < alpha_s < 0.15

        # Verify formula: sqrt(2) / 12
        expected = np.sqrt(2) / 12
        assert np.isclose(alpha_s, expected, rtol=1e-10)


# =============================================================================
# Test: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of computations."""

    def test_no_nan_in_observables(self, framework):
        """Test that no NaN values are produced."""
        methods = [
            'compute_alpha_inverse',
            'compute_sin2_theta_W',
            'compute_alpha_s',
        ]

        for method_name in methods:
            if hasattr(framework, method_name):
                result = getattr(framework, method_name)()
                assert not np.isnan(result), f"{method_name} produced NaN"

    def test_no_inf_in_observables(self, framework):
        """Test that no Inf values are produced."""
        methods = [
            'compute_alpha_inverse',
            'compute_sin2_theta_W',
            'compute_alpha_s',
        ]

        for method_name in methods:
            if hasattr(framework, method_name):
                result = getattr(framework, method_name)()
                assert not np.isinf(result), f"{method_name} produced Inf"

    def test_positive_coupling_constants(self, framework):
        """Test that coupling constants are positive."""
        alpha_inv = framework.compute_alpha_inverse()
        sin2_theta_W = framework.compute_sin2_theta_W()
        alpha_s = framework.compute_alpha_s()

        assert alpha_inv > 0, "alpha_inverse should be positive"
        assert sin2_theta_W > 0, "sin2_theta_W should be positive"
        assert alpha_s > 0, "alpha_s should be positive"

    def test_reproducibility(self, framework):
        """Test that results are reproducible."""
        # Compute twice
        result1 = framework.compute_alpha_inverse()
        result2 = framework.compute_alpha_inverse()

        assert result1 == result2, "Results should be exactly reproducible"

    def test_extreme_parameter_handling(self):
        """Test framework handles extreme parameters gracefully."""
        # Very small torsion
        fw = GIFTFrameworkV21(torsion_magnitude=1e-15)
        alpha_inv = fw.compute_alpha_inverse()
        assert np.isfinite(alpha_inv)

        # Very large det_g
        fw = GIFTFrameworkV21(det_g=1e6)
        alpha_inv = fw.compute_alpha_inverse()
        assert np.isfinite(alpha_inv)


# =============================================================================
# Test: Parameter Variations
# =============================================================================

class TestParameterVariations:
    """Tests for parameter sensitivity and variations."""

    @pytest.mark.parametrize("p2", [1.5, 2.0, 2.5, 3.0])
    def test_p2_variations(self, p2):
        """Test framework with different p2 values."""
        fw = GIFTFrameworkV21(p2=p2)
        assert fw.p2 == p2

        # xi should scale with p2
        expected_xi = (fw.weyl_factor / p2) * fw.beta0
        assert np.isclose(fw.xi, expected_xi)

    @pytest.mark.parametrize("weyl", [3.0, 4.0, 5.0, 6.0, 7.0])
    def test_weyl_factor_variations(self, weyl):
        """Test framework with different Weyl factor values."""
        fw = GIFTFrameworkV21(weyl_factor=weyl)
        assert fw.weyl_factor == weyl

        # xi should scale with weyl_factor
        expected_xi = (weyl / fw.p2) * fw.beta0
        assert np.isclose(fw.xi, expected_xi)

    @pytest.mark.parametrize("det_g,torsion", [
        (1.0, 0.01),
        (2.0, 0.02),
        (3.0, 0.015),
    ])
    def test_metric_parameters(self, det_g, torsion):
        """Test framework with different metric parameters."""
        fw = GIFTFrameworkV21(det_g=det_g, torsion_magnitude=torsion)

        alpha_inv = fw.compute_alpha_inverse()

        # Torsional contribution should scale with det_g * torsion
        expected_torsional = det_g * torsion
        algebraic = (fw.dim_E8 + fw.rank_E8) / 2
        bulk = fw.H_star / fw.D_bulk
        expected = algebraic + bulk + expected_torsional

        assert np.isclose(alpha_inv, expected)


# =============================================================================
# Test: Topological Invariance
# =============================================================================

class TestTopologicalInvariance:
    """Tests for topological invariants remaining constant."""

    @pytest.mark.parametrize("p2,weyl,det_g", [
        (1.0, 3.0, 1.5),
        (2.0, 5.0, 2.0),
        (3.0, 7.0, 3.0),
        (2.5, 4.0, 2.5),
    ])
    def test_topological_constants_invariant(self, p2, weyl, det_g):
        """Test topological invariants don't change with parameters."""
        fw = GIFTFrameworkV21(p2=p2, weyl_factor=weyl, det_g=det_g)

        # These should always be the same
        assert fw.b2_K7 == 21
        assert fw.b3_K7 == 77
        assert fw.H_star == 99
        assert fw.dim_E8 == 248
        assert fw.rank_E8 == 8
        assert fw.dim_G2 == 14
        assert fw.dim_K7 == 7

    def test_h_star_formula(self, framework):
        """Test H* = b2 + b3 + 1."""
        assert framework.H_star == framework.b2_K7 + framework.b3_K7 + 1

    def test_tau_formula(self, framework):
        """Test tau = (496 * 21) / (27 * 99)."""
        # 496 = dim(E8 x E8)
        # 21 = b2_K7
        # 27 = dim(J3O)
        # 99 = H_star
        expected = (496 * framework.b2_K7) / (framework.dim_J3O * framework.H_star)
        assert np.isclose(framework.tau, expected, rtol=1e-10)


# =============================================================================
# Test: JSON Serialization
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON output format validation."""

    def test_experimental_data_serializable(self, framework):
        """Test experimental data can be serialized to JSON."""
        # Convert tuples to lists for JSON
        data_for_json = {k: list(v) for k, v in framework.experimental_data.items()}

        # Should not raise
        json_str = json.dumps(data_for_json)

        # Should be parseable
        parsed = json.loads(json_str)
        assert len(parsed) == len(framework.experimental_data)

    def test_observable_results_serializable(self, framework):
        """Test observable results can be serialized to JSON."""
        results = {
            'alpha_inv': framework.compute_alpha_inverse(),
            'sin2_theta_W': framework.compute_sin2_theta_W(),
            'alpha_s': framework.compute_alpha_s(),
        }

        # Should not raise
        json_str = json.dumps(results)

        # Should be parseable and values preserved
        parsed = json.loads(json_str)
        for key, value in results.items():
            assert np.isclose(parsed[key], value)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_torsion(self):
        """Test framework with zero torsion magnitude."""
        fw = GIFTFrameworkV21(torsion_magnitude=0.0)

        alpha_inv = fw.compute_alpha_inverse()

        # Should be exactly algebraic + bulk (137.0)
        expected = 128.0 + 9.0  # No torsional contribution
        assert np.isclose(alpha_inv, expected)

    def test_zero_det_g(self):
        """Test framework with zero metric determinant."""
        fw = GIFTFrameworkV21(det_g=0.0)

        alpha_inv = fw.compute_alpha_inverse()

        # Torsional contribution should be zero
        expected = 128.0 + 9.0
        assert np.isclose(alpha_inv, expected)

    def test_negative_parameters_warning(self):
        """Test that negative parameters still work (may be unphysical)."""
        # Framework should not crash with negative values
        fw = GIFTFrameworkV21(torsion_magnitude=-0.01)
        alpha_inv = fw.compute_alpha_inverse()
        assert np.isfinite(alpha_inv)


# =============================================================================
# Test: Consistency Checks
# =============================================================================

class TestConsistencyChecks:
    """Tests for internal consistency of the framework."""

    def test_dimension_consistency(self, framework):
        """Test dimension-related constants are consistent."""
        # E8 dimension
        assert framework.dim_E8 == 248

        # E8 rank
        assert framework.rank_E8 == 8

        # G2 holonomy group dimension
        assert framework.dim_G2 == 14

        # K7 manifold dimension
        assert framework.dim_K7 == 7

        # p2 = dim(G2) / dim(K7) = 14/7 = 2
        assert framework.p2 == framework.dim_G2 / framework.dim_K7

    def test_beta0_definition(self, framework):
        """Test beta0 = pi / rank(E8)."""
        expected_beta0 = np.pi / framework.rank_E8
        assert np.isclose(framework.beta0, expected_beta0)

    def test_golden_ratio_properties(self, framework):
        """Test golden ratio satisfies phi^2 = phi + 1."""
        phi = framework.phi
        assert np.isclose(phi * phi, phi + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
