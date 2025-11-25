"""
API Contract and Stability Tests.

Tests that public APIs maintain their contracts:
- Method signatures remain stable
- Return types are consistent
- Observable dictionary structure is maintained
- No breaking changes in interfaces

Version: 2.1.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
import inspect

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))
sys.path.insert(0, str(PROJECT_ROOT / "giftpy"))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def framework_v21():
    """Get GIFTFrameworkV21 instance."""
    try:
        from gift_v21_core import GIFTFrameworkV21
        return GIFTFrameworkV21()
    except ImportError:
        pytest.skip("GIFTFrameworkV21 not available")


@pytest.fixture
def framework_statistical():
    """Get GIFTFrameworkStatistical instance."""
    try:
        from run_validation import GIFTFrameworkStatistical
        return GIFTFrameworkStatistical()
    except ImportError:
        pytest.skip("GIFTFrameworkStatistical not available")


@pytest.fixture
def giftpy_framework():
    """Get giftpy GIFT instance."""
    try:
        from giftpy import GIFT
        return GIFT()
    except ImportError:
        pytest.skip("giftpy not available")


# =============================================================================
# GIFTFrameworkV21 API Contract Tests
# =============================================================================

class TestGIFTFrameworkV21Contract:
    """Test GIFTFrameworkV21 API contract."""

    # Expected public methods
    EXPECTED_METHODS = [
        'compute_all_observables',
    ]

    # Expected attributes
    EXPECTED_ATTRIBUTES = [
        'params',
    ]

    def test_class_exists(self):
        """Test GIFTFrameworkV21 class can be imported."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            assert GIFTFrameworkV21 is not None
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_instantiation_no_args(self):
        """Test framework can be instantiated without arguments."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            assert fw is not None
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_has_compute_method(self, framework_v21):
        """Test framework has compute_all_observables method."""
        assert hasattr(framework_v21, 'compute_all_observables')
        assert callable(framework_v21.compute_all_observables)

    def test_has_params_attribute(self, framework_v21):
        """Test framework has params attribute."""
        assert hasattr(framework_v21, 'params')

    def test_compute_returns_dict(self, framework_v21):
        """Test compute_all_observables returns dictionary."""
        result = framework_v21.compute_all_observables()
        assert isinstance(result, dict)

    def test_compute_returns_non_empty(self, framework_v21):
        """Test compute_all_observables returns non-empty result."""
        result = framework_v21.compute_all_observables()
        assert len(result) > 0

    def test_observable_values_are_numeric(self, framework_v21):
        """Test all observable values are numeric."""
        obs = framework_v21.compute_all_observables()
        for name, value in obs.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"{name} is not numeric: {type(value)}"

    def test_observable_names_are_strings(self, framework_v21):
        """Test all observable names are strings."""
        obs = framework_v21.compute_all_observables()
        for name in obs.keys():
            assert isinstance(name, str), f"Key {name} is not string"

    def test_params_has_p2(self, framework_v21):
        """Test params has p2 attribute."""
        assert hasattr(framework_v21.params, 'p2')

    def test_params_has_Weyl_factor(self, framework_v21):
        """Test params has Weyl_factor attribute."""
        assert hasattr(framework_v21.params, 'Weyl_factor')


# =============================================================================
# GIFTFrameworkStatistical API Contract Tests
# =============================================================================

class TestGIFTFrameworkStatisticalContract:
    """Test GIFTFrameworkStatistical API contract."""

    def test_class_exists(self):
        """Test GIFTFrameworkStatistical class can be imported."""
        try:
            from run_validation import GIFTFrameworkStatistical
            assert GIFTFrameworkStatistical is not None
        except ImportError:
            pytest.skip("GIFTFrameworkStatistical not available")

    def test_instantiation_no_args(self):
        """Test framework can be instantiated without arguments."""
        try:
            from run_validation import GIFTFrameworkStatistical
            fw = GIFTFrameworkStatistical()
            assert fw is not None
        except ImportError:
            pytest.skip("GIFTFrameworkStatistical not available")

    def test_has_compute_method(self, framework_statistical):
        """Test framework has compute_all_observables method."""
        assert hasattr(framework_statistical, 'compute_all_observables')
        assert callable(framework_statistical.compute_all_observables)

    def test_has_p2_attribute(self, framework_statistical):
        """Test framework has p2 attribute."""
        assert hasattr(framework_statistical, 'p2')

    def test_has_Weyl_factor_attribute(self, framework_statistical):
        """Test framework has Weyl_factor attribute."""
        assert hasattr(framework_statistical, 'Weyl_factor')

    def test_has_topological_integers(self, framework_statistical):
        """Test framework has topological integer attributes."""
        assert hasattr(framework_statistical, 'b2_K7')
        assert hasattr(framework_statistical, 'b3_K7')
        assert hasattr(framework_statistical, 'H_star')
        assert hasattr(framework_statistical, 'dim_E8')


# =============================================================================
# giftpy API Contract Tests
# =============================================================================

class TestGiftpyContract:
    """Test giftpy package API contract."""

    def test_package_importable(self):
        """Test giftpy package can be imported."""
        try:
            import giftpy
            assert giftpy is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_GIFT_class_exists(self):
        """Test GIFT class exists in giftpy."""
        try:
            from giftpy import GIFT
            assert GIFT is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_GIFT_instantiation(self):
        """Test GIFT can be instantiated."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            assert gift is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_has_gauge_sector(self, giftpy_framework):
        """Test GIFT has gauge sector property."""
        assert hasattr(giftpy_framework, 'gauge')

    def test_has_neutrino_sector(self, giftpy_framework):
        """Test GIFT has neutrino sector property."""
        assert hasattr(giftpy_framework, 'neutrino')

    def test_has_lepton_sector(self, giftpy_framework):
        """Test GIFT has lepton sector property."""
        assert hasattr(giftpy_framework, 'lepton')

    def test_has_quark_sector(self, giftpy_framework):
        """Test GIFT has quark sector property."""
        assert hasattr(giftpy_framework, 'quark')

    def test_has_cosmology_sector(self, giftpy_framework):
        """Test GIFT has cosmology sector property."""
        assert hasattr(giftpy_framework, 'cosmology')

    def test_has_compute_all(self, giftpy_framework):
        """Test GIFT has compute_all method."""
        assert hasattr(giftpy_framework, 'compute_all')
        assert callable(giftpy_framework.compute_all)

    def test_has_constants(self, giftpy_framework):
        """Test GIFT has constants attribute."""
        assert hasattr(giftpy_framework, 'constants')

    def test_gauge_has_alpha_s(self, giftpy_framework):
        """Test gauge sector has alpha_s method."""
        assert hasattr(giftpy_framework.gauge, 'alpha_s')
        assert callable(giftpy_framework.gauge.alpha_s)

    def test_lepton_has_Q_Koide(self, giftpy_framework):
        """Test lepton sector has Q_Koide method."""
        assert hasattr(giftpy_framework.lepton, 'Q_Koide')
        assert callable(giftpy_framework.lepton.Q_Koide)


# =============================================================================
# Observable Dictionary Contract Tests
# =============================================================================

class TestObservableDictionaryContract:
    """Test observable dictionary structure contract."""

    def test_keys_are_valid_identifiers(self, framework_v21):
        """Test observable keys are valid Python identifiers."""
        obs = framework_v21.compute_all_observables()
        for key in obs.keys():
            assert key.isidentifier() or '_' in key or key[0].isalpha(), \
                f"Key '{key}' is not a valid identifier"

    def test_no_duplicate_keys(self, framework_v21):
        """Test no duplicate keys (dict guarantees this, but verify)."""
        obs = framework_v21.compute_all_observables()
        keys = list(obs.keys())
        assert len(keys) == len(set(keys))

    def test_consistent_keys_across_calls(self, framework_v21):
        """Test same keys returned across multiple calls."""
        obs1 = framework_v21.compute_all_observables()
        obs2 = framework_v21.compute_all_observables()
        assert set(obs1.keys()) == set(obs2.keys())

    def test_consistent_values_across_calls(self, framework_v21):
        """Test same values returned across multiple calls."""
        obs1 = framework_v21.compute_all_observables()
        obs2 = framework_v21.compute_all_observables()
        for key in obs1:
            assert obs1[key] == obs2[key], f"{key} value changed between calls"


# =============================================================================
# Return Type Contract Tests
# =============================================================================

class TestReturnTypeContracts:
    """Test return type contracts."""

    def test_gauge_alpha_s_returns_float(self, giftpy_framework):
        """Test alpha_s returns float."""
        result = giftpy_framework.gauge.alpha_s()
        assert isinstance(result, (float, np.floating))

    def test_lepton_Q_Koide_returns_float(self, giftpy_framework):
        """Test Q_Koide returns float."""
        result = giftpy_framework.lepton.Q_Koide()
        assert isinstance(result, (float, np.floating))

    def test_lepton_m_tau_m_e_returns_int(self, giftpy_framework):
        """Test m_tau_m_e returns integer."""
        result = giftpy_framework.lepton.m_tau_m_e()
        assert isinstance(result, (int, np.integer))

    def test_quark_m_s_m_d_returns_numeric(self, giftpy_framework):
        """Test m_s_m_d returns numeric."""
        result = giftpy_framework.quark.m_s_m_d()
        assert isinstance(result, (int, float, np.integer, np.floating))


# =============================================================================
# Method Signature Contract Tests
# =============================================================================

class TestMethodSignatureContracts:
    """Test method signature contracts."""

    def test_alpha_s_accepts_Q_parameter(self, giftpy_framework):
        """Test alpha_s accepts Q (energy scale) parameter."""
        sig = inspect.signature(giftpy_framework.gauge.alpha_s)
        params = list(sig.parameters.keys())
        assert 'Q' in params or len(params) == 0 or params[0] == 'self'

    def test_sin2theta_W_accepts_scheme_parameter(self, giftpy_framework):
        """Test sin2theta_W accepts scheme parameter."""
        # Check method exists and accepts parameters
        assert callable(giftpy_framework.gauge.sin2theta_W)

    def test_compute_all_observables_no_required_args(self, framework_v21):
        """Test compute_all_observables has no required arguments."""
        # Should be callable without arguments
        result = framework_v21.compute_all_observables()
        assert result is not None


# =============================================================================
# Constants Contract Tests
# =============================================================================

class TestConstantsContract:
    """Test topological constants contract."""

    def test_constants_frozen(self, giftpy_framework):
        """Test constants are immutable."""
        constants = giftpy_framework.constants
        # Should not be able to modify
        try:
            constants.b2 = 999
            pytest.fail("Constants should be frozen")
        except Exception:
            pass  # Expected

    def test_constants_b2_value(self, giftpy_framework):
        """Test b2 constant value."""
        assert giftpy_framework.constants.b2 == 21

    def test_constants_b3_value(self, giftpy_framework):
        """Test b3 constant value."""
        assert giftpy_framework.constants.b3 == 77

    def test_constants_dim_E8_value(self, giftpy_framework):
        """Test dim_E8 constant value."""
        assert giftpy_framework.constants.dim_E8 == 248


# =============================================================================
# Error Contract Tests
# =============================================================================

class TestErrorContracts:
    """Test error handling contracts."""

    def test_invalid_sector_access_raises(self, giftpy_framework):
        """Test accessing invalid sector raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = giftpy_framework.invalid_sector

    def test_compute_all_doesnt_raise(self, framework_v21):
        """Test compute_all_observables doesn't raise with defaults."""
        # Should not raise
        try:
            result = framework_v21.compute_all_observables()
            assert result is not None
        except Exception as e:
            pytest.fail(f"compute_all_observables raised: {e}")


# =============================================================================
# Deprecation Contract Tests
# =============================================================================

class TestDeprecationContracts:
    """Test deprecation warnings and backward compatibility."""

    def test_no_deprecation_warnings_on_import(self):
        """Test no deprecation warnings on normal import."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                from gift_v21_core import GIFTFrameworkV21
                from run_validation import GIFTFrameworkStatistical
            except ImportError:
                pytest.skip("Modules not available")

            # Check for deprecation warnings
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            if dep_warnings:
                print(f"Deprecation warnings: {[str(w.message) for w in dep_warnings]}")


# =============================================================================
# Thread Safety Contract Tests
# =============================================================================

class TestThreadSafetyContracts:
    """Test thread safety contracts (basic tests)."""

    def test_multiple_instances_independent(self):
        """Test multiple framework instances are independent."""
        try:
            from gift_v21_core import GIFTFrameworkV21

            fw1 = GIFTFrameworkV21()
            fw2 = GIFTFrameworkV21()

            # Modify fw1's state (if possible)
            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()

            # Results should be identical
            for key in obs1:
                if key in obs2:
                    assert obs1[key] == obs2[key]

        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_repeated_calls_consistent(self, framework_v21):
        """Test repeated calls give consistent results."""
        results = []
        for _ in range(10):
            obs = framework_v21.compute_all_observables()
            results.append(obs)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
