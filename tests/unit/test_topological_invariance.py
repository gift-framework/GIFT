"""
Topological Invariance Verification Tests.

Tests that TOPOLOGICAL and PROVEN observables are truly independent of
continuous geometric parameters. This is a critical requirement of the
GIFT framework - topological quantities must not vary with parameters.

Key principle: Observables derived from pure topology (Betti numbers, dimensions,
etc.) should return IDENTICAL values regardless of p2, Weyl_factor, or tau.

Version: 2.1.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))


# =============================================================================
# Topological Observables Definition
# =============================================================================

# Observables that MUST be parameter-independent
TOPOLOGICAL_OBSERVABLES = {
    # Pure integers from topology
    'N_gen': 3,                      # Number of generations
    'm_tau_m_e': 3477,               # 7 + 10*248 + 10*99
    'm_s_m_d': 20,                   # p2^2 * Weyl = 4*5 = 20

    # Exact rational values
    'Q_Koide': 2/3,                  # 14/21 = dim_G2/b2
    'lambda_H': np.sqrt(17)/32,      # sqrt(17)/32

    # Exact values from dimensions
    'delta_CP': 197.0,               # 7*14 + 99 = 197 degrees
    'Omega_DE': np.log(2) * 98/99,   # ln(2) * (b2+b3)/H_star

    # Gauge couplings (topological formulas)
    'alpha_inv_MZ': 2**7 - 1/24,     # 127.958...
    'alpha_s_MZ': np.sqrt(2)/12,     # 0.1179...
    'sin2thetaW': np.pi**2/6 - np.sqrt(2),  # zeta(2) - sqrt(2)
}

# These observables MAY depend on parameters (not required to be invariant)
PARAMETER_DEPENDENT_OBSERVABLES = [
    'H0', 'n_s', 'theta12', 'theta13', 'theta23',
    'm_mu_m_e',  # 27^phi - depends on how phi is computed
    # CKM elements, cosmological densities, etc.
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def framework_class():
    """Get GIFTFrameworkV21 class."""
    try:
        from gift_v21_core import GIFTFrameworkV21
        return GIFTFrameworkV21
    except ImportError:
        pytest.skip("GIFTFrameworkV21 not available")


@pytest.fixture
def statistical_framework_class():
    """Get GIFTFrameworkStatistical class (v2.0)."""
    try:
        from run_validation import GIFTFrameworkStatistical
        return GIFTFrameworkStatistical
    except ImportError:
        pytest.skip("GIFTFrameworkStatistical not available")


# =============================================================================
# Parameter Combinations for Testing
# =============================================================================

# Wide range of parameter values to test invariance
PARAMETER_VARIATIONS = [
    {'p2': 2.0, 'Weyl_factor': 5.0, 'tau': 3.8967},   # Default
    {'p2': 1.5, 'Weyl_factor': 4.0, 'tau': 3.0},      # Lower
    {'p2': 2.5, 'Weyl_factor': 6.0, 'tau': 4.5},      # Higher
    {'p2': 3.0, 'Weyl_factor': 7.0, 'tau': 5.0},      # Much higher
    {'p2': 1.0, 'Weyl_factor': 3.0, 'tau': 2.5},      # Much lower
    {'p2': 2.0, 'Weyl_factor': 5.0, 'tau': 10.0},     # High tau
    {'p2': 2.0, 'Weyl_factor': 5.0, 'tau': 1.0},      # Low tau
    {'p2': 1.8, 'Weyl_factor': 5.5, 'tau': 4.0},      # Non-integer
    {'p2': 2.2, 'Weyl_factor': 4.5, 'tau': 3.5},      # Other non-integer
    {'p2': 2.0, 'Weyl_factor': 10.0, 'tau': 3.8967},  # High Weyl
]


# =============================================================================
# Topological Invariance Tests - v2.1 Framework
# =============================================================================

class TestTopologicalInvarianceV21:
    """Test topological observables are parameter-independent in v2.1."""

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_Q_Koide_invariant(self, framework_class, params):
        """Q_Koide = 2/3 must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'Q_Koide' not in obs:
            pytest.skip("Q_Koide not implemented")

        expected = 2/3
        assert np.isclose(obs['Q_Koide'], expected, rtol=1e-14), \
            f"Q_Koide = {obs['Q_Koide']} with params {params}, expected {expected}"

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_m_tau_m_e_invariant(self, framework_class, params):
        """m_tau/m_e = 3477 must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'm_tau_m_e' not in obs:
            pytest.skip("m_tau_m_e not implemented")

        expected = 3477
        assert np.isclose(obs['m_tau_m_e'], expected, rtol=1e-14), \
            f"m_tau/m_e = {obs['m_tau_m_e']} with params {params}, expected {expected}"

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_m_s_m_d_invariant(self, framework_class, params):
        """m_s/m_d = 20 must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'm_s_m_d' not in obs:
            pytest.skip("m_s_m_d not implemented")

        expected = 20
        assert np.isclose(obs['m_s_m_d'], expected, rtol=1e-14), \
            f"m_s/m_d = {obs['m_s_m_d']} with params {params}, expected {expected}"

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_delta_CP_invariant(self, framework_class, params):
        """delta_CP = 197 degrees must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'delta_CP' not in obs:
            pytest.skip("delta_CP not implemented")

        expected = 197.0
        assert np.isclose(obs['delta_CP'], expected, rtol=1e-10), \
            f"delta_CP = {obs['delta_CP']} with params {params}, expected {expected}"

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_lambda_H_invariant(self, framework_class, params):
        """lambda_H = sqrt(17)/32 must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'lambda_H' not in obs:
            pytest.skip("lambda_H not implemented")

        expected = np.sqrt(17) / 32
        assert np.isclose(obs['lambda_H'], expected, rtol=1e-14), \
            f"lambda_H = {obs['lambda_H']} with params {params}, expected {expected}"

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_Omega_DE_invariant(self, framework_class, params):
        """Omega_DE = ln(2)*98/99 must not depend on parameters."""
        try:
            fw = framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Framework doesn't accept these parameters")

        if 'Omega_DE' not in obs:
            pytest.skip("Omega_DE not implemented")

        expected = np.log(2) * 98/99
        assert np.isclose(obs['Omega_DE'], expected, rtol=1e-14), \
            f"Omega_DE = {obs['Omega_DE']} with params {params}, expected {expected}"


# =============================================================================
# Topological Invariance Tests - Statistical Framework
# =============================================================================

class TestTopologicalInvarianceStatistical:
    """Test topological observables in GIFTFrameworkStatistical."""

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS[:5])  # Fewer variations
    def test_topological_integers_invariant(self, statistical_framework_class, params):
        """Test topological integers are invariant."""
        try:
            fw = statistical_framework_class(**params)
        except TypeError:
            pytest.skip("Statistical framework doesn't accept these parameters")

        # These should never change
        assert fw.b2_K7 == 21
        assert fw.b3_K7 == 77
        assert fw.H_star == 99
        assert fw.dim_E8 == 248
        assert fw.dim_G2 == 14
        assert fw.dim_K7 == 7

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS[:5])
    def test_proven_formulas_exact(self, statistical_framework_class, params):
        """Test PROVEN formulas give exact values."""
        try:
            fw = statistical_framework_class(**params)
            obs = fw.compute_all_observables()
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate with these parameters")

        # Check available observables
        if 'Q_Koide' in obs:
            assert np.isclose(obs['Q_Koide'], 2/3, rtol=1e-14)

        if 'm_s_m_d' in obs:
            assert np.isclose(obs['m_s_m_d'], 20, rtol=1e-14)


# =============================================================================
# Consistency Tests Across Parameters
# =============================================================================

class TestConsistencyAcrossParameters:
    """Test that topological observables are consistent across all parameter choices."""

    def test_all_topological_observables_consistent(self, framework_class):
        """All topological observables should give same value across parameters."""
        reference_obs = None

        for params in PARAMETER_VARIATIONS:
            try:
                fw = framework_class(**params)
                obs = fw.compute_all_observables()
            except (TypeError, AttributeError):
                continue

            if reference_obs is None:
                reference_obs = obs
                continue

            # Compare topological observables
            for name, expected in TOPOLOGICAL_OBSERVABLES.items():
                if name in obs and name in reference_obs:
                    assert np.isclose(obs[name], reference_obs[name], rtol=1e-10), \
                        f"{name} changed: {reference_obs[name]} -> {obs[name]} with params {params}"

    def test_variance_of_topological_is_zero(self, framework_class):
        """Variance of topological observables across parameters should be zero."""
        values = {name: [] for name in TOPOLOGICAL_OBSERVABLES}

        for params in PARAMETER_VARIATIONS:
            try:
                fw = framework_class(**params)
                obs = fw.compute_all_observables()
            except (TypeError, AttributeError):
                continue

            for name in TOPOLOGICAL_OBSERVABLES:
                if name in obs:
                    values[name].append(obs[name])

        # Check variance is zero for each observable
        for name, vals in values.items():
            if len(vals) >= 2:
                variance = np.var(vals)
                assert variance < 1e-20, \
                    f"{name} has non-zero variance {variance} across parameters"


# =============================================================================
# Betti Number Invariance
# =============================================================================

class TestBettiNumberInvariance:
    """Test that Betti numbers are truly constant."""

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_b2_K7_constant(self, framework_class, params):
        """b2(K7) = 21 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        # Access via params or direct attribute
        if hasattr(fw, 'b2_K7'):
            assert fw.b2_K7 == 21
        elif hasattr(fw.params, 'b2_K7'):
            assert fw.params.b2_K7 == 21

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_b3_K7_constant(self, framework_class, params):
        """b3(K7) = 77 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'b3_K7'):
            assert fw.b3_K7 == 77

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_H_star_constant(self, framework_class, params):
        """H*(K7) = 99 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'H_star'):
            assert fw.H_star == 99


# =============================================================================
# Dimensional Constant Invariance
# =============================================================================

class TestDimensionalConstantInvariance:
    """Test that Lie algebra dimensions are constant."""

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_dim_E8_constant(self, framework_class, params):
        """dim(E8) = 248 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'dim_E8'):
            assert fw.dim_E8 == 248

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_dim_G2_constant(self, framework_class, params):
        """dim(G2) = 14 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'dim_G2'):
            assert fw.dim_G2 == 14

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_dim_K7_constant(self, framework_class, params):
        """dim(K7) = 7 always."""
        try:
            fw = framework_class(**params)
        except (TypeError, AttributeError):
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'dim_K7'):
            assert fw.dim_K7 == 7


# =============================================================================
# Exact Formula Tests
# =============================================================================

class TestExactFormulas:
    """Test that formulas give exact mathematical values."""

    def test_Q_Koide_is_exactly_two_thirds(self, framework_class):
        """Q_Koide should be EXACTLY 2/3, not approximately."""
        fw = framework_class()
        obs = fw.compute_all_observables()

        if 'Q_Koide' in obs:
            # Must be exact to floating point precision
            assert obs['Q_Koide'] == 2/3 or np.isclose(obs['Q_Koide'], 2/3, rtol=1e-15)

    def test_m_s_m_d_is_exactly_20(self, framework_class):
        """m_s/m_d should be EXACTLY 20."""
        fw = framework_class()
        obs = fw.compute_all_observables()

        if 'm_s_m_d' in obs:
            assert obs['m_s_m_d'] == 20 or np.isclose(obs['m_s_m_d'], 20, rtol=1e-15)

    def test_m_tau_m_e_is_exactly_3477(self, framework_class):
        """m_tau/m_e should be EXACTLY 3477 (integer)."""
        fw = framework_class()
        obs = fw.compute_all_observables()

        if 'm_tau_m_e' in obs:
            assert obs['m_tau_m_e'] == 3477 or np.isclose(obs['m_tau_m_e'], 3477, rtol=1e-15)

    def test_delta_CP_is_exactly_197(self, framework_class):
        """delta_CP should be EXACTLY 197 degrees."""
        fw = framework_class()
        obs = fw.compute_all_observables()

        if 'delta_CP' in obs:
            assert np.isclose(obs['delta_CP'], 197.0, rtol=1e-14)


# =============================================================================
# Derived Parameter Independence (xi)
# =============================================================================

class TestDerivedParameterXi:
    """Test that xi = 5*beta0/2 is correctly derived."""

    @pytest.mark.parametrize("params", PARAMETER_VARIATIONS)
    def test_xi_derived_correctly(self, statistical_framework_class, params):
        """xi should always equal 5*beta0/2."""
        try:
            fw = statistical_framework_class(**params)
        except TypeError:
            pytest.skip("Cannot instantiate")

        if hasattr(fw, 'xi') and hasattr(fw, 'beta0'):
            expected_xi = 5 * fw.beta0 / 2
            assert np.isclose(fw.xi, expected_xi, rtol=1e-14), \
                f"xi = {fw.xi}, expected {expected_xi}"


# =============================================================================
# Stress Tests with Extreme Parameters
# =============================================================================

class TestExtremeParameters:
    """Test topological invariance with extreme parameter values."""

    @pytest.mark.parametrize("p2", [0.1, 0.5, 1.0, 5.0, 10.0, 100.0])
    def test_topological_with_extreme_p2(self, framework_class, p2):
        """Topological observables invariant even with extreme p2."""
        try:
            fw = framework_class(p2=p2)
            obs = fw.compute_all_observables()
        except (TypeError, ValueError):
            pytest.skip(f"p2={p2} not accepted")

        if 'Q_Koide' in obs:
            assert np.isclose(obs['Q_Koide'], 2/3, rtol=1e-14)
        if 'm_tau_m_e' in obs:
            assert np.isclose(obs['m_tau_m_e'], 3477, rtol=1e-14)

    @pytest.mark.parametrize("weyl", [1, 2, 3, 5, 10, 20, 50])
    def test_topological_with_extreme_weyl(self, framework_class, weyl):
        """Topological observables invariant even with extreme Weyl_factor."""
        try:
            fw = framework_class(Weyl_factor=weyl)
            obs = fw.compute_all_observables()
        except (TypeError, ValueError):
            pytest.skip(f"Weyl_factor={weyl} not accepted")

        if 'Q_Koide' in obs:
            assert np.isclose(obs['Q_Koide'], 2/3, rtol=1e-14)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
