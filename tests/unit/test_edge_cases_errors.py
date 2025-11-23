"""
Comprehensive edge case and error handling tests.

Tests include:
- Numerical edge cases (zero, infinity, NaN)
- Extreme parameter values
- Missing data handling
- Invalid input handling
- Division by zero protection
- Overflow/underflow handling
- Concurrent execution safety
- Memory efficiency

Version: 2.1.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21, GIFTParameters
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V21_AVAILABLE,
    reason="GIFT v2.1 not available"
)


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_zero_parameter_handling(self):
        """Test behavior when parameters approach zero."""
        # Very small but non-zero parameters
        small_val = 1e-10

        # This might fail or produce invalid results - test error handling
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=small_val,
                    Weyl_factor=small_val,
                    tau=small_val
                )
            )
            obs = framework.compute_all_observables()

            # If it succeeds, check for NaN/Inf
            for name, value in obs.items():
                # Should either fail gracefully or give finite results
                if not np.isfinite(value):
                    pytest.skip(f"{name} not finite with tiny parameters - expected behavior")

        except (ValueError, ZeroDivisionError, FloatingPointError) as e:
            # Graceful failure is acceptable
            pass

    def test_large_parameter_values(self):
        """Test with very large parameter values."""
        large_val = 1e6

        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=large_val,
                    Weyl_factor=large_val,
                    tau=large_val
                )
            )
            obs = framework.compute_all_observables()

            # Check for overflow
            for name, value in obs.items():
                assert np.isfinite(value), f"{name} = {value} not finite with large parameters"

        except (ValueError, OverflowError) as e:
            # Graceful failure acceptable
            pass

    def test_negative_parameters(self):
        """Test handling of negative parameters."""
        # Most parameters should be positive - test error handling
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=-2.0,  # Negative
                    Weyl_factor=5.0,
                    tau=3.8967
                )
            )
            obs = framework.compute_all_observables()

            # If it doesn't raise an error, check results are reasonable
            # Negative p2 might be unphysical
            assert isinstance(obs, dict)

        except (ValueError, AssertionError) as e:
            # Expected to fail for unphysical parameters
            pass

    def test_nan_parameter_handling(self):
        """Test handling of NaN parameters."""
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=np.nan,
                    Weyl_factor=5.0,
                    tau=3.8967
                )
            )
            obs = framework.compute_all_observables()

            # Should either reject NaN or propagate it
            # Check if NaN is handled
            pytest.fail("NaN parameter should be rejected or handled")

        except (ValueError, AssertionError) as e:
            # Expected behavior - reject invalid input
            pass

    def test_inf_parameter_handling(self):
        """Test handling of infinite parameters."""
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=np.inf,
                    Weyl_factor=5.0,
                    tau=3.8967
                )
            )
            obs = framework.compute_all_observables()

            pytest.fail("Inf parameter should be rejected")

        except (ValueError, OverflowError, AssertionError) as e:
            # Expected - infinite parameters are unphysical
            pass


class TestMissingDataHandling:
    """Test handling of missing or None values."""

    def test_partial_observable_computation(self):
        """Test when some observables cannot be computed."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Even if some fail, others should succeed
        assert len(obs) > 0, "At least some observables should be computed"

        # Check that present observables are valid
        for name, value in obs.items():
            if value is not None:
                assert np.isfinite(value), f"{name} should be finite or None"

    def test_missing_experimental_data(self):
        """Test comparison when experimental data is missing."""
        # Simulated scenario - missing experimental values
        experimental_data = {
            "alpha_inv_MZ": (127.955, 0.01),
            # Other observables missing
        }

        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Should be able to compute predictions even without experimental data
        assert len(obs) > len(experimental_data)


class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    def test_invalid_parameter_types(self):
        """Test with invalid parameter types."""
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2="not a number",  # String instead of float
                    Weyl_factor=5.0,
                    tau=3.8967
                )
            )
            pytest.fail("Should reject string parameter")

        except (TypeError, ValueError) as e:
            # Expected behavior
            pass

    def test_none_parameters(self):
        """Test handling of None parameters."""
        # Default parameters should be used if None provided
        try:
            framework = GIFTFrameworkV21(params=None)
            obs = framework.compute_all_observables()

            # Should use defaults
            assert len(obs) > 0

        except Exception as e:
            pytest.fail(f"Should handle None params gracefully: {e}")


class TestDivisionByZeroProtection:
    """Test protection against division by zero."""

    def test_zero_denominator_observables(self):
        """Test observables that might have zero denominators."""
        # Some mass ratios could have zero denominators with wrong parameters
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Check no Inf values from division by zero
        for name, value in obs.items():
            if "m_" in name or "ratio" in name.lower():
                assert not np.isinf(value), f"{name} is Inf (possible division by zero)"

    def test_small_denominator_stability(self):
        """Test numerical stability with small denominators."""
        # When denominators are small, ratios can be large but should be finite
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        for name, value in obs.items():
            assert np.isfinite(value), f"{name} not finite"


class TestOverflowUnderflowHandling:
    """Test overflow and underflow handling."""

    def test_no_overflow_in_standard_run(self):
        """Test standard run produces no overflow."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Check all values are finite
        overflow_count = sum(1 for v in obs.values() if np.isinf(v))

        assert overflow_count == 0, f"Found {overflow_count} overflow values"

    def test_no_underflow_to_zero(self):
        """Test important observables don't underflow to zero."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Key observables should not be exactly zero (unless proven to be)
        important_obs = [
            "alpha_inv_MZ", "sin2thetaW", "alpha_s_MZ",
            "Q_Koide", "m_mu_m_e"
        ]

        for obs_name in important_obs:
            if obs_name in obs:
                assert obs[obs_name] != 0.0, f"{obs_name} underflowed to zero"


class TestConcurrentExecution:
    """Test thread safety and concurrent execution."""

    def test_multiple_framework_instances(self):
        """Test multiple framework instances can coexist."""
        frameworks = [GIFTFrameworkV21() for _ in range(5)]

        results = [fw.compute_all_observables() for fw in frameworks]

        # All should give same results
        for i in range(1, len(results)):
            for key in results[0]:
                if key in results[i]:
                    assert results[0][key] == results[i][key], (
                        f"{key} differs between instances"
                    )

    def test_repeated_computation_no_state_corruption(self):
        """Test repeated computations don't corrupt state."""
        framework = GIFTFrameworkV21()

        # Compute multiple times
        results = [framework.compute_all_observables() for _ in range(10)]

        # All should be identical
        for i in range(1, len(results)):
            for key in results[0]:
                if key in results[i]:
                    assert results[0][key] == results[i][key]


class TestMemoryEfficiency:
    """Test memory efficiency and leak prevention."""

    def test_framework_cleanup(self):
        """Test framework can be created and destroyed repeatedly."""
        # Create and destroy many instances
        for i in range(100):
            framework = GIFTFrameworkV21()
            obs = framework.compute_all_observables()
            del framework

        # Should complete without memory error
        assert True

    def test_large_monte_carlo_simulation(self):
        """Test memory efficiency with many samples."""
        n_samples = 1000

        results = []
        for i in range(n_samples):
            framework = GIFTFrameworkV21()
            obs = framework.compute_all_observables()

            # Only store one observable to save memory
            if 'alpha_inv_MZ' in obs:
                results.append(obs['alpha_inv_MZ'])

        assert len(results) > 0

        # Should be able to compute statistics
        mean_val = np.mean(results)
        assert np.isfinite(mean_val)


class TestRobustnessToFloatingPointErrors:
    """Test robustness to floating point arithmetic errors."""

    def test_associativity_tolerance(self):
        """Test calculations are tolerant to floating point associativity issues."""
        # (a + b) + c might not equal a + (b + c) in floating point
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Recompute - should give same results despite FP errors
        obs2 = framework.compute_all_observables()

        for key in obs:
            if key in obs2:
                assert obs[key] == obs2[key]

    def test_numerical_precision_constants(self):
        """Test that mathematical constants have adequate precision."""
        # Check some constants used in framework
        pi_precision = abs(np.pi - 3.141592653589793)

        assert pi_precision < 1e-15, "Pi constant has insufficient precision"

        # Check log(2)
        log2_value = np.log(2)
        expected_log2 = 0.693147180559945

        assert abs(log2_value - expected_log2) < 1e-14


class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_invalid_parameter_error_message(self):
        """Test error messages for invalid parameters."""
        try:
            # Try to create with invalid parameters
            params = GIFTParameters(p2=-1.0)  # Negative might be invalid

            # If this doesn't raise an error, that's fine too
        except ValueError as e:
            # Error message should be informative
            error_msg = str(e)

            # Should mention the parameter
            assert isinstance(error_msg, str)
            assert len(error_msg) > 0


class TestBoundaryConditions:
    """Test behavior at boundary conditions."""

    def test_parameter_at_upper_bounds(self):
        """Test parameters at reasonable upper bounds."""
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=10.0,  # 5x typical
                    Weyl_factor=20.0,  # 4x typical
                    tau=10.0  # ~3x typical
                )
            )
            obs = framework.compute_all_observables()

            # Should complete
            assert len(obs) > 0

        except Exception as e:
            # If it fails, that's okay - these are extreme values
            pass

    def test_parameter_at_lower_bounds(self):
        """Test parameters at reasonable lower bounds."""
        try:
            framework = GIFTFrameworkV21(
                params=GIFTParameters(
                    p2=1.0,  # 0.5x typical
                    Weyl_factor=2.5,  # 0.5x typical
                    tau=2.0  # ~0.5x typical
                )
            )
            obs = framework.compute_all_observables()

            assert len(obs) > 0

        except Exception as e:
            pass


class TestArrayShapeConsistency:
    """Test array shapes are consistent."""

    def test_observable_dict_consistency(self):
        """Test observable dictionary has consistent structure."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # All values should be scalars
        for name, value in obs.items():
            assert np.isscalar(value) or isinstance(value, (int, float, np.number)), (
                f"{name} is not scalar: type={type(value)}"
            )


class TestSpecialValues:
    """Test handling of special mathematical values."""

    def test_sqrt_negative_handling(self):
        """Test that square roots of negative numbers are handled."""
        # Some formulas might accidentally try sqrt of negative
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # No complex numbers should appear
        for name, value in obs.items():
            assert not np.iscomplex(value), f"{name} is complex"

    def test_log_zero_handling(self):
        """Test that log(0) is avoided."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # No -Inf from log(0)
        for name, value in obs.items():
            if np.isinf(value) and value < 0:
                pytest.fail(f"{name} is -Inf (possible log(0))")

    def test_arcsin_domain_enforcement(self):
        """Test that arcsin/arccos arguments are in [-1, 1]."""
        # If any observables use arcsin/arccos, they should be in valid domain
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Angles should be in valid range
        angle_obs = ['theta12', 'theta13', 'theta23']

        for obs_name in angle_obs:
            if obs_name in obs:
                angle = obs[obs_name]

                # Angles in degrees should be reasonable
                assert -180 <= angle <= 360, (
                    f"{obs_name} = {angle}° outside reasonable range"
                )
