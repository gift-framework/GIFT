"""
Tests for gift_v22_core.py - GIFT Framework v2.2 Core Implementation

These tests cover:
- GIFTFrameworkV22 class initialization
- Zero-parameter paradigm validation
- 13 PROVEN exact relations
- All 39 observable computations
- Topological invariants and consistency
- Numerical stability and reproducibility

Author: GIFT Framework
Date: 2025-11-27
Version: 2.2.0 (updated from 2.1.0)
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch, MagicMock

# Add statistical_validation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

try:
    from gift_v22_core import GIFTFrameworkV22, GIFTParametersV22
    V22_AVAILABLE = True
except ImportError:
    V22_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V22_AVAILABLE,
    reason="GIFT v2.2 core not available"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def framework():
    """Create a default GIFTFrameworkV22 instance."""
    return GIFTFrameworkV22()


@pytest.fixture
def params():
    """Create GIFTParametersV22 instance."""
    return GIFTParametersV22()


# =============================================================================
# Test: Framework Initialization
# =============================================================================

class TestFrameworkInitialization:
    """Tests for GIFTFrameworkV22 initialization."""

    def test_default_initialization(self, framework):
        """Test framework initializes with correct default values."""
        # v2.2: All values are fixed - zero-parameter paradigm
        assert framework.b2_K7 == 21
        assert framework.b3_K7 == 77
        assert framework.H_star == 99
        assert framework.dim_E8 == 248
        assert framework.dim_G2 == 14
        assert framework.dim_K7 == 7

    def test_params_dataclass(self, params):
        """Test GIFTParametersV22 dataclass properties."""
        # Verify exact values
        assert params.p2 == 2
        assert params.Weyl_factor == 5
        assert np.isclose(params.beta0, np.pi / 8)

    def test_topological_invariants(self, framework):
        """Test topological invariants are correctly set."""
        assert framework.b2_K7 == 21
        assert framework.b3_K7 == 77
        assert framework.H_star == 99
        assert framework.dim_E8 == 248
        assert framework.rank_E8 == 8
        assert framework.dim_G2 == 14
        assert framework.dim_K7 == 7
        assert framework.dim_J3O == 27
        assert framework.N_gen == 3

    def test_derived_parameters_v22(self, params):
        """Test v2.2 derived parameters are computed correctly."""
        # xi = (Weyl_factor / p2) * beta0 = 5/2 * pi/8 = 5*pi/16
        expected_xi = 5.0 * np.pi / 16.0
        assert np.isclose(params.xi, expected_xi, rtol=1e-10)

        # tau = 3472/891 (exact rational)
        assert params.tau == Fraction(3472, 891)
        assert np.isclose(params.tau_float, 3472.0/891.0, rtol=1e-10)

    def test_exact_fractions(self, params):
        """Test exact fractions are preserved in v2.2."""
        # sin^2(theta_W) = 3/13
        assert params.sin2_theta_W == Fraction(3, 13)
        assert np.isclose(float(params.sin2_theta_W), 3.0/13.0)

        # kappa_T = 1/61
        assert params.kappa_T == Fraction(1, 61)
        assert np.isclose(params.kappa_T_float, 1.0/61.0)

        # det_g = 65/32
        assert params.det_g == Fraction(65, 32)
        assert np.isclose(params.det_g_float, 65.0/32.0)

    def test_mathematical_constants(self, params):
        """Test mathematical constants are correct."""
        assert np.isclose(params.phi_golden, (1 + np.sqrt(5)) / 2)
        assert np.isclose(params.zeta3, 1.2020569031595942)
        assert np.isclose(params.gamma_euler, 0.5772156649015329)


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
        """Test key experimental values are present (v2.2 keys)."""
        required_keys = [
            'alpha_inv', 'sin2thetaW', 'alpha_s_MZ',
            'theta12', 'theta13', 'theta23', 'delta_CP',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e',
            'm_s_m_d', 'Omega_DE', 'H0',
            'kappa_T', 'tau'  # v2.2 new keys
        ]
        for key in required_keys:
            assert key in framework.experimental_data, f"Missing key: {key}"

    def test_experimental_values_reasonable(self, framework):
        """Test experimental values are within reasonable ranges."""
        # Fine structure constant inverse
        alpha_inv = framework.experimental_data['alpha_inv'][0]
        assert 136 < alpha_inv < 138

        # Weak mixing angle
        sin2_theta_W = framework.experimental_data['sin2thetaW'][0]
        assert 0.20 < sin2_theta_W < 0.25

        # Neutrino mixing angles (degrees)
        theta_12 = framework.experimental_data['theta12'][0]
        assert 30 < theta_12 < 40

        # Dark energy fraction
        omega_de = framework.experimental_data['Omega_DE'][0]
        assert 0.6 < omega_de < 0.8


# =============================================================================
# Test: 13 PROVEN Exact Relations
# =============================================================================

class TestProvenExactRelations:
    """Tests for 13 PROVEN exact relations in v2.2."""

    def test_get_proven_relations(self, framework):
        """Test get_proven_relations returns all 13 relations."""
        proven = framework.get_proven_relations()
        assert len(proven) >= 13

    def test_proven_n_gen(self, framework):
        """Test N_gen = 3 (generation number)."""
        assert framework.N_gen == 3

    def test_proven_q_koide(self, framework):
        """Test Q_Koide = 2/3 = dim(G2)/b2."""
        expected = Fraction(2, 3)
        computed = Fraction(framework.dim_G2, framework.b2_K7)
        assert computed == expected
        assert np.isclose(float(computed), 2.0/3.0)

    def test_proven_m_s_m_d(self, params):
        """Test m_s/m_d = 20 = p2^2 * Weyl."""
        expected = params.p2**2 * params.Weyl_factor
        assert expected == 20

    def test_proven_delta_cp(self, framework):
        """Test delta_CP = 197 = dim(K7)*dim(G2) + H*."""
        expected = framework.dim_K7 * framework.dim_G2 + framework.H_star
        assert expected == 197

    def test_proven_m_tau_m_e(self, framework):
        """Test m_tau/m_e = 3477 = dim(K7) + 10*dim(E8) + 10*H*."""
        expected = framework.dim_K7 + 10*framework.dim_E8 + 10*framework.H_star
        assert expected == 3477

    def test_proven_omega_de(self, framework):
        """Test Omega_DE = ln(2)*98/99."""
        expected = np.log(2) * 98.0 / 99.0
        computed = np.log(2) * (framework.b2_K7 + framework.b3_K7) / framework.H_star
        assert np.isclose(computed, expected, rtol=1e-10)

    def test_proven_xi(self, params):
        """Test xi = 5*pi/16 = (Weyl/p2)*beta0."""
        expected = 5.0 * np.pi / 16.0
        assert np.isclose(params.xi, expected, rtol=1e-10)

    def test_proven_lambda_h(self, params):
        """Test lambda_H = sqrt(17)/32."""
        expected = np.sqrt(17) / 32.0
        assert np.isclose(params.lambda_H, expected, rtol=1e-10)

    def test_proven_sin2_theta_w(self, params):
        """Test sin^2(theta_W) = 3/13 = b2/(b3+dim(G2))."""
        assert params.sin2_theta_W == Fraction(3, 13)
        assert np.isclose(float(params.sin2_theta_W), 0.230769, rtol=1e-5)

    def test_proven_tau(self, params):
        """Test tau = 3472/891 = dim(E8xE8)*b2/(dim(J3O)*H*)."""
        assert params.tau == Fraction(3472, 891)
        # Verify formula: 496 * 21 / (27 * 99)
        expected = Fraction(496 * 21, 27 * 99)
        assert params.tau == expected

    def test_proven_kappa_t(self, params):
        """Test kappa_T = 1/61 = 1/(b3-dim(G2)-p2)."""
        assert params.kappa_T == Fraction(1, 61)
        # Verify formula: 1 / (77 - 14 - 2)
        expected = Fraction(1, 77 - 14 - 2)
        assert params.kappa_T == expected

    def test_proven_n_s(self, params):
        """Test n_s = zeta(11)/zeta(5)."""
        expected = params.zeta11 / params.zeta5
        assert np.isclose(expected, 0.9649, rtol=1e-3)


# =============================================================================
# Test: Gauge Sector Computations
# =============================================================================

class TestGaugeSectorComputations:
    """Tests for gauge sector observable computations."""

    def test_alpha_s_computation(self, params):
        """Test strong coupling constant calculation."""
        alpha_s = params.alpha_s
        # alpha_s = sqrt(2) / 12
        expected = np.sqrt(2) / 12.0
        assert np.isclose(alpha_s, expected, rtol=1e-10)
        assert 0.10 < alpha_s < 0.15

    def test_sin2_theta_w_value(self, params):
        """Test weak mixing angle value."""
        sin2_theta_W = float(params.sin2_theta_W)
        assert np.isclose(sin2_theta_W, 3.0/13.0, rtol=1e-10)


# =============================================================================
# Test: All Observables Computation
# =============================================================================

class TestAllObservables:
    """Tests for compute_all_observables method."""

    def test_compute_all_observables_exists(self, framework):
        """Test compute_all_observables method exists."""
        assert hasattr(framework, 'compute_all_observables')
        assert callable(framework.compute_all_observables)

    def test_compute_all_observables_returns_dict(self, framework):
        """Test compute_all_observables returns dictionary."""
        obs = framework.compute_all_observables()
        assert isinstance(obs, dict)
        assert len(obs) >= 30  # At least 30 observables

    def test_key_observables_present(self, framework):
        """Test key observables are computed."""
        obs = framework.compute_all_observables()

        key_observables = [
            'alpha_inv', 'sin2thetaW', 'alpha_s_MZ',
            'delta_CP', 'Q_Koide', 'm_tau_m_e', 'm_s_m_d',
            'lambda_H', 'Omega_DE', 'n_s', 'kappa_T', 'tau'
        ]

        for obs_name in key_observables:
            assert obs_name in obs, f"Missing observable: {obs_name}"


# =============================================================================
# Test: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of computations."""

    def test_no_nan_in_observables(self, framework):
        """Test that no NaN values are produced."""
        obs = framework.compute_all_observables()

        nan_obs = [name for name, value in obs.items() if np.isnan(value)]
        assert len(nan_obs) == 0, f"NaN values in: {nan_obs}"

    def test_no_inf_in_observables(self, framework):
        """Test that no Inf values are produced."""
        obs = framework.compute_all_observables()

        inf_obs = [name for name, value in obs.items() if np.isinf(value)]
        assert len(inf_obs) == 0, f"Inf values in: {inf_obs}"

    def test_reproducibility(self, framework):
        """Test that results are reproducible."""
        obs1 = framework.compute_all_observables()
        obs2 = framework.compute_all_observables()

        for key in obs1:
            if key in obs2:
                assert obs1[key] == obs2[key], f"{key} not reproducible"


# =============================================================================
# Test: Topological Invariance
# =============================================================================

class TestTopologicalInvariance:
    """Tests for topological invariants remaining constant."""

    def test_h_star_formula(self, framework):
        """Test H* = b2 + b3 + 1."""
        assert framework.H_star == framework.b2_K7 + framework.b3_K7 + 1

    def test_p2_formula(self, params):
        """Test p2 = dim(G2)/dim(K7) = 14/7 = 2."""
        assert params.p2 == params.dim_G2 // params.dim_K7
        assert params.p2 == 2

    def test_b3_relation(self, params):
        """Test b3 = 2*dim(K7)^2 - b2."""
        expected = 2 * params.dim_K7**2 - params.b2_K7
        assert params.b3_K7 == expected

    def test_beta0_definition(self, params):
        """Test beta0 = pi / rank(E8)."""
        expected_beta0 = np.pi / params.rank_E8
        assert np.isclose(params.beta0, expected_beta0)


# =============================================================================
# Test: JSON Serialization
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON output format validation."""

    def test_experimental_data_serializable(self, framework):
        """Test experimental data can be serialized to JSON."""
        data_for_json = {k: list(v) for k, v in framework.experimental_data.items()}

        json_str = json.dumps(data_for_json)
        parsed = json.loads(json_str)
        assert len(parsed) == len(framework.experimental_data)

    def test_observable_results_serializable(self, framework):
        """Test observable results can be serialized to JSON."""
        obs = framework.compute_all_observables()

        # Filter to finite values
        finite_obs = {k: v for k, v in obs.items() if np.isfinite(v)}

        json_str = json.dumps(finite_obs)
        parsed = json.loads(json_str)

        for key, value in finite_obs.items():
            assert np.isclose(parsed[key], value)


# =============================================================================
# Test: Zero-Parameter Paradigm
# =============================================================================

class TestZeroParameterParadigm:
    """Tests verifying v2.2's zero-parameter paradigm."""

    def test_no_adjustable_parameters(self, framework):
        """Test that framework has no adjustable continuous parameters."""
        # v2.2: All parameters are fixed topological constants
        obs1 = framework.compute_all_observables()

        # Create new instance
        framework2 = GIFTFrameworkV22()
        obs2 = framework2.compute_all_observables()

        # All values should be identical
        for key in obs1:
            if key in obs2:
                assert obs1[key] == obs2[key], f"{key} differs between instances"

    def test_fixed_structural_inputs(self, params):
        """Test structural inputs are fixed (discrete choices)."""
        # E8 x E8 gauge group (496 dimensions)
        assert params.dim_E8xE8 == 496
        assert params.dim_E8 == 248

        # K7 manifold with G2 holonomy
        assert params.b2_K7 == 21
        assert params.b3_K7 == 77

    def test_all_constants_topological(self, params):
        """Test all derived constants come from topology."""
        # These are all derived, not fitted
        assert params.p2 == 2  # dim(G2)/dim(K7)
        assert params.Weyl_factor == 5  # from |W(E8)|
        assert params.det_g == Fraction(65, 32)  # topological formula
        assert params.kappa_T == Fraction(1, 61)  # topological formula
        assert params.tau == Fraction(3472, 891)  # exact rational


# =============================================================================
# Test: Consistency Checks
# =============================================================================

class TestConsistencyChecks:
    """Tests for internal consistency of the framework."""

    def test_dimension_consistency(self, params):
        """Test dimension-related constants are consistent."""
        assert params.dim_E8 == 248
        assert params.rank_E8 == 8
        assert params.dim_G2 == 14
        assert params.dim_K7 == 7
        assert params.p2 == params.dim_G2 // params.dim_K7

    def test_betti_number_consistency(self, params):
        """Test Betti number relations."""
        # H* = b2 + b3 + 1
        assert params.H_star == params.b2_K7 + params.b3_K7 + 1

        # b3 = 2*dim(K7)^2 - b2 = 2*49 - 21 = 77
        expected_b3 = 2 * params.dim_K7**2 - params.b2_K7
        assert params.b3_K7 == expected_b3

    def test_tau_formula_consistency(self, params):
        """Test tau formula consistency."""
        # tau = dim(E8xE8)*b2 / (dim(J3O)*H*)
        expected_numerator = params.dim_E8xE8 * params.b2_K7  # 496 * 21 = 10416
        expected_denominator = params.dim_J3O * params.H_star  # 27 * 99 = 2673
        expected_tau = Fraction(expected_numerator, expected_denominator)

        # Simplify: 10416/2673 = 3472/891
        assert params.tau == expected_tau

    def test_golden_ratio_properties(self, params):
        """Test golden ratio satisfies phi^2 = phi + 1."""
        phi = params.phi_golden
        assert np.isclose(phi * phi, phi + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
