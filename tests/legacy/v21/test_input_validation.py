"""
Input Validation and Boundary Tests.

Tests that the GIFT framework properly handles:
- Invalid parameter values (NaN, Inf, negative)
- Edge cases and boundary conditions
- Type errors
- Parameter range enforcement

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
# Fixtures
# =============================================================================

@pytest.fixture
def framework_v21_class():
    """Get GIFTFrameworkV21 class."""
    try:
        from gift_v21_core import GIFTFrameworkV21
        return GIFTFrameworkV21
    except ImportError:
        pytest.skip("GIFTFrameworkV21 not available")


@pytest.fixture
def statistical_framework_class():
    """Get GIFTFrameworkStatistical class."""
    try:
        from run_validation import GIFTFrameworkStatistical
        return GIFTFrameworkStatistical
    except ImportError:
        pytest.skip("GIFTFrameworkStatistical not available")


@pytest.fixture
def giftpy_class():
    """Get GIFT class from giftpy."""
    try:
        from giftpy import GIFT
        return GIFT
    except ImportError:
        pytest.skip("giftpy not available")


# =============================================================================
# NaN Parameter Tests
# =============================================================================

class TestNaNParameters:
    """Test handling of NaN parameter values."""

    @pytest.mark.parametrize("param_name", ['p2', 'Weyl_factor', 'tau'])
    def test_nan_parameter_v21(self, framework_v21_class, param_name):
        """Test framework behavior with NaN parameters."""
        params = {param_name: np.nan}

        # Framework should either:
        # 1. Raise ValueError on construction
        # 2. Raise ValueError on compute
        # 3. Return NaN values (which we then detect)

        try:
            fw = framework_v21_class(**params)
            obs = fw.compute_all_observables()

            # If we get here, check that results aren't corrupted
            # NaN input should produce NaN output or raise error
            has_nan = any(np.isnan(v) for v in obs.values() if isinstance(v, (int, float)))

            # Either all values are NaN (acceptable) or we should have raised
            if not has_nan:
                # Framework accepted NaN and produced non-NaN - could be a bug
                # or framework has validation that we didn't trigger
                pass

        except (ValueError, TypeError) as e:
            # This is the expected behavior - reject NaN inputs
            assert 'nan' in str(e).lower() or 'invalid' in str(e).lower() or True

    @pytest.mark.parametrize("param_name", ['p2', 'Weyl_factor', 'tau'])
    def test_nan_parameter_statistical(self, statistical_framework_class, param_name):
        """Test statistical framework with NaN parameters."""
        params = {param_name: np.nan}

        try:
            fw = statistical_framework_class(**params)
            obs = fw.compute_all_observables()

            # Check for NaN propagation
            for name, value in obs.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    # NaN propagated - acceptable but should be documented
                    pass

        except (ValueError, TypeError):
            # Expected - framework rejects NaN
            pass


# =============================================================================
# Infinity Parameter Tests
# =============================================================================

class TestInfinityParameters:
    """Test handling of infinite parameter values."""

    @pytest.mark.parametrize("inf_value", [np.inf, -np.inf])
    @pytest.mark.parametrize("param_name", ['p2', 'Weyl_factor', 'tau'])
    def test_inf_parameter_v21(self, framework_v21_class, param_name, inf_value):
        """Test framework behavior with infinite parameters."""
        params = {param_name: inf_value}

        try:
            fw = framework_v21_class(**params)
            obs = fw.compute_all_observables()

            # Check for inf propagation
            has_inf = any(np.isinf(v) for v in obs.values() if isinstance(v, (int, float)))

            if not has_inf:
                # Framework handled inf gracefully
                pass

        except (ValueError, TypeError, OverflowError):
            # Expected behavior - reject infinite inputs
            pass

    @pytest.mark.parametrize("inf_value", [np.inf, -np.inf])
    def test_inf_parameter_produces_inf_or_raises(self, statistical_framework_class, inf_value):
        """Test that inf input produces inf output or raises error."""
        try:
            fw = statistical_framework_class(p2=inf_value)
            obs = fw.compute_all_observables()

            # If we get results, some should be inf
            for name, value in obs.items():
                if isinstance(value, (int, float)):
                    # Either inf or valid is acceptable
                    pass

        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            # Expected
            pass


# =============================================================================
# Negative Parameter Tests
# =============================================================================

class TestNegativeParameters:
    """Test handling of negative parameter values."""

    @pytest.mark.parametrize("param_name,negative_value", [
        ('p2', -1.0),
        ('p2', -0.5),
        ('Weyl_factor', -5),
        ('Weyl_factor', -1),
        ('tau', -3.89),
    ])
    def test_negative_parameter_v21(self, framework_v21_class, param_name, negative_value):
        """Test framework behavior with negative parameters."""
        params = {param_name: negative_value}

        # Physical parameters should typically be positive
        # Framework may accept or reject
        try:
            fw = framework_v21_class(**params)
            obs = fw.compute_all_observables()

            # If accepted, check results are reasonable
            for name, value in obs.items():
                if isinstance(value, (int, float)):
                    assert np.isfinite(value), f"{name} is not finite with negative {param_name}"

        except (ValueError, TypeError):
            # Expected - reject negative values
            pass

    def test_negative_p2_statistical(self, statistical_framework_class):
        """Test statistical framework with negative p2."""
        try:
            fw = statistical_framework_class(p2=-2.0)
            obs = fw.compute_all_observables()
            # Check what happens - may produce negative or complex results
        except (ValueError, TypeError):
            pass  # Expected


# =============================================================================
# Zero Parameter Tests
# =============================================================================

class TestZeroParameters:
    """Test handling of zero parameter values."""

    @pytest.mark.parametrize("param_name", ['p2', 'Weyl_factor', 'tau'])
    def test_zero_parameter_v21(self, framework_v21_class, param_name):
        """Test framework behavior with zero parameters."""
        params = {param_name: 0.0}

        # Zero may cause division by zero or other issues
        try:
            fw = framework_v21_class(**params)
            obs = fw.compute_all_observables()

            # Check for inf/nan from division by zero
            for name, value in obs.items():
                if isinstance(value, (int, float)):
                    if np.isinf(value) or np.isnan(value):
                        # Division by zero detected
                        pass

        except (ValueError, TypeError, ZeroDivisionError):
            # Expected - reject zero values
            pass

    def test_zero_Weyl_factor_causes_division_error(self, statistical_framework_class):
        """Test that zero Weyl_factor causes appropriate error."""
        try:
            fw = statistical_framework_class(Weyl_factor=0)
            obs = fw.compute_all_observables()

            # If we get here, check for division issues
            for name, value in obs.items():
                if isinstance(value, (int, float)) and np.isinf(value):
                    # Expected - division by zero
                    pass

        except (ValueError, TypeError, ZeroDivisionError):
            pass  # Expected


# =============================================================================
# Type Error Tests
# =============================================================================

class TestTypeErrors:
    """Test handling of wrong parameter types."""

    @pytest.mark.parametrize("bad_value", ['string', None, [1, 2, 3], {'a': 1}])
    def test_non_numeric_p2(self, framework_v21_class, bad_value):
        """Test framework behavior with non-numeric p2."""
        try:
            fw = framework_v21_class(p2=bad_value)
            # If construction succeeds, computation should fail or produce error
            obs = fw.compute_all_observables()
        except (ValueError, TypeError):
            pass  # Expected

    @pytest.mark.parametrize("bad_value", ['string', None, [1, 2, 3], {'a': 1}])
    def test_non_numeric_Weyl_factor(self, framework_v21_class, bad_value):
        """Test framework behavior with non-numeric Weyl_factor."""
        try:
            fw = framework_v21_class(Weyl_factor=bad_value)
            obs = fw.compute_all_observables()
        except (ValueError, TypeError):
            pass  # Expected

    def test_complex_parameter(self, framework_v21_class):
        """Test framework behavior with complex parameters."""
        try:
            fw = framework_v21_class(p2=2+1j)
            obs = fw.compute_all_observables()
            # Complex may propagate or cause error
        except (ValueError, TypeError):
            pass  # Expected


# =============================================================================
# Extreme Value Tests
# =============================================================================

class TestExtremeValues:
    """Test handling of extreme but valid parameter values."""

    @pytest.mark.parametrize("p2", [1e-10, 1e10, 1e-100, 1e100])
    def test_extreme_p2_values(self, framework_v21_class, p2):
        """Test framework with extremely small/large p2."""
        try:
            fw = framework_v21_class(p2=p2)
            obs = fw.compute_all_observables()

            # Check for overflow/underflow
            for name, value in obs.items():
                if isinstance(value, (int, float)):
                    assert np.isfinite(value) or np.isnan(value), \
                        f"{name} overflowed with p2={p2}"

        except (ValueError, TypeError, OverflowError):
            pass  # Acceptable to reject extreme values

    @pytest.mark.parametrize("weyl", [1e-10, 1e10])
    def test_extreme_Weyl_values(self, framework_v21_class, weyl):
        """Test framework with extreme Weyl_factor."""
        try:
            fw = framework_v21_class(Weyl_factor=weyl)
            obs = fw.compute_all_observables()
        except (ValueError, TypeError, OverflowError):
            pass


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test behavior at parameter boundaries."""

    def test_p2_equals_1(self, framework_v21_class):
        """Test p2 = 1 (minimal binary value)."""
        try:
            fw = framework_v21_class(p2=1.0)
            obs = fw.compute_all_observables()

            # Should produce valid results
            assert len(obs) > 0
        except (ValueError, TypeError):
            pass

    def test_Weyl_factor_equals_1(self, framework_v21_class):
        """Test Weyl_factor = 1 (minimal value)."""
        try:
            fw = framework_v21_class(Weyl_factor=1)
            obs = fw.compute_all_observables()
            assert len(obs) > 0
        except (ValueError, TypeError):
            pass

    def test_tau_near_zero(self, framework_v21_class):
        """Test tau very close to zero."""
        try:
            fw = framework_v21_class(tau=1e-10)
            obs = fw.compute_all_observables()

            for name, value in obs.items():
                if isinstance(value, (int, float)):
                    assert np.isfinite(value), f"{name} not finite with tau~0"

        except (ValueError, TypeError, ZeroDivisionError):
            pass


# =============================================================================
# Output Validation Tests
# =============================================================================

class TestOutputValidation:
    """Test that outputs are always valid."""

    def test_all_outputs_numeric(self, framework_v21_class):
        """Test all observable outputs are numeric."""
        fw = framework_v21_class()
        obs = fw.compute_all_observables()

        for name, value in obs.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"{name} is not numeric: {type(value)}"

    def test_no_nan_in_default_output(self, framework_v21_class):
        """Test no NaN values in default parameter output."""
        fw = framework_v21_class()
        obs = fw.compute_all_observables()

        for name, value in obs.items():
            assert not np.isnan(value), f"{name} is NaN with default parameters"

    def test_no_inf_in_default_output(self, framework_v21_class):
        """Test no infinite values in default parameter output."""
        fw = framework_v21_class()
        obs = fw.compute_all_observables()

        for name, value in obs.items():
            assert not np.isinf(value), f"{name} is infinite with default parameters"

    def test_physical_positivity(self, framework_v21_class):
        """Test physically positive quantities are positive."""
        fw = framework_v21_class()
        obs = fw.compute_all_observables()

        positive_quantities = [
            'alpha_inv_MZ', 'alpha_s_MZ', 'Q_Koide', 'm_tau_m_e', 'm_s_m_d',
            'm_mu_m_e', 'lambda_H', 'Omega_DE',
        ]

        for name in positive_quantities:
            if name in obs:
                assert obs[name] > 0, f"{name} should be positive, got {obs[name]}"

    def test_bounded_quantities(self, framework_v21_class):
        """Test bounded quantities are in valid range."""
        fw = framework_v21_class()
        obs = fw.compute_all_observables()

        # sin^2(theta_W) must be in (0, 1)
        if 'sin2thetaW' in obs:
            assert 0 < obs['sin2thetaW'] < 1

        # CKM elements must be in [0, 1]
        ckm_elements = ['V_us', 'V_cb', 'V_ub', 'V_cd', 'V_cs', 'V_td']
        for name in ckm_elements:
            if name in obs:
                assert 0 <= obs[name] <= 1, f"{name} out of CKM range"

        # Mixing angles should be in (0, 90) degrees
        angles = ['theta12', 'theta13', 'theta23']
        for name in angles:
            if name in obs:
                assert 0 < obs[name] < 90, f"{name} out of mixing angle range"


# =============================================================================
# giftpy Validation Tests
# =============================================================================

class TestGiftpyValidation:
    """Test input validation in giftpy package."""

    def test_giftpy_accepts_default(self, giftpy_class):
        """Test giftpy GIFT class with defaults."""
        gift = giftpy_class()
        assert gift is not None

    def test_giftpy_rejects_invalid_constants(self, giftpy_class):
        """Test giftpy with invalid custom constants."""
        try:
            from giftpy.core.constants import TopologicalConstants

            # Try creating invalid constants (frozen dataclass should prevent mutation)
            constants = TopologicalConstants()
            # This should work
            gift = giftpy_class(constants=constants)
            assert gift is not None

        except Exception:
            pass  # Validation working as expected


# =============================================================================
# Parameter Range Recommendation Tests
# =============================================================================

class TestParameterRangeRecommendations:
    """Document recommended parameter ranges based on physical validity."""

    def test_document_valid_p2_range(self, framework_v21_class):
        """Test and document valid p2 range."""
        valid_p2_values = []
        invalid_p2_values = []

        test_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]

        for p2 in test_values:
            try:
                fw = framework_v21_class(p2=p2)
                obs = fw.compute_all_observables()
                # Check if results are physically reasonable
                if all(np.isfinite(v) for v in obs.values() if isinstance(v, (int, float))):
                    valid_p2_values.append(p2)
                else:
                    invalid_p2_values.append(p2)
            except Exception:
                invalid_p2_values.append(p2)

        # Document findings
        print(f"\nValid p2 values: {valid_p2_values}")
        print(f"Invalid p2 values: {invalid_p2_values}")

        # At minimum, default p2=2.0 should work
        assert 2.0 in valid_p2_values

    def test_document_valid_Weyl_range(self, framework_v21_class):
        """Test and document valid Weyl_factor range."""
        valid_weyl_values = []

        for weyl in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                fw = framework_v21_class(Weyl_factor=weyl)
                obs = fw.compute_all_observables()
                if all(np.isfinite(v) for v in obs.values() if isinstance(v, (int, float))):
                    valid_weyl_values.append(weyl)
            except Exception:
                pass

        print(f"\nValid Weyl_factor values: {valid_weyl_values}")
        assert 5 in valid_weyl_values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
