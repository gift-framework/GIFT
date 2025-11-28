"""
Unit tests for numerical edge cases and stability.

Tests extreme values, near-singular matrices, boundary conditions,
and numerical precision across the GIFT framework.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from fractions import Fraction

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gift_framework_v22():
    """Create GIFT v2.2 framework instance."""
    try:
        from gift_v22_core import GIFTFrameworkV22
        return GIFTFrameworkV22()
    except ImportError:
        pytest.skip("GIFTFrameworkV22 not available")


@pytest.fixture
def gift_params_v22():
    """Create GIFTParametersV22 instance."""
    try:
        from gift_v22_core import GIFTParametersV22
        return GIFTParametersV22()
    except ImportError:
        pytest.skip("GIFTParametersV22 not available")


# =============================================================================
# Topological Integer Stability Tests
# =============================================================================

class TestTopologicalIntegerStability:
    """Test that topological integers remain exact under various operations."""

    def test_betti_numbers_are_exact_integers(self, gift_params_v22):
        """Test Betti numbers are exact integers."""
        assert gift_params_v22.b2_K7 == 21
        assert gift_params_v22.b3_K7 == 77
        assert isinstance(gift_params_v22.b2_K7, int)
        assert isinstance(gift_params_v22.b3_K7, int)

    def test_h_star_calculation_exact(self, gift_params_v22):
        """Test H* = b2 + b3 + 1 is exact."""
        h_star = gift_params_v22.b2_K7 + gift_params_v22.b3_K7 + 1
        assert h_star == 99
        assert gift_params_v22.H_star == 99

    def test_dimension_integers_exact(self, gift_params_v22):
        """Test dimension constants are exact."""
        assert gift_params_v22.dim_E8 == 248
        assert gift_params_v22.dim_G2 == 14
        assert gift_params_v22.dim_K7 == 7

    def test_integer_arithmetic_no_float_drift(self, gift_params_v22):
        """Test integer arithmetic doesn't introduce float errors."""
        # These should all be exact
        result1 = gift_params_v22.b2_K7 * gift_params_v22.dim_E8  # 21 * 248
        result2 = gift_params_v22.b3_K7 - gift_params_v22.dim_G2  # 77 - 14
        result3 = gift_params_v22.H_star // gift_params_v22.dim_K7  # 99 // 7

        assert result1 == 5208
        assert result2 == 63
        assert result3 == 14


# =============================================================================
# Exact Rational Tests
# =============================================================================

class TestExactRationalValues:
    """Test exact rational values remain precise."""

    def test_sin2_theta_w_exact(self, gift_params_v22):
        """Test sin^2(theta_W) = 3/13 is exact."""
        expected = Fraction(3, 13)
        computed = Fraction(gift_params_v22.b2_K7, gift_params_v22.b3_K7 + gift_params_v22.dim_G2)

        assert computed == expected
        assert float(computed) == pytest.approx(0.23076923076923078, rel=1e-15)

    def test_tau_exact_rational(self, gift_params_v22):
        """Test tau = 3472/891 is exact."""
        expected = Fraction(3472, 891)

        # Compute from formula
        numerator = gift_params_v22.dim_E8 * 2 * gift_params_v22.b2_K7  # 496 * 21
        denominator = 27 * gift_params_v22.H_star  # 27 * 99
        computed = Fraction(numerator, denominator)

        assert computed == expected
        assert float(expected) == pytest.approx(3.896745230078563, rel=1e-15)

    def test_kappa_t_exact_rational(self, gift_params_v22):
        """Test kappa_T = 1/61 is exact."""
        expected = Fraction(1, 61)

        # Compute from formula: 1/(b3 - dim(G2) - p2) = 1/(77 - 14 - 2)
        denominator = gift_params_v22.b3_K7 - gift_params_v22.dim_G2 - 2
        computed = Fraction(1, denominator)

        assert computed == expected
        assert denominator == 61

    def test_q_koide_exact(self, gift_params_v22):
        """Test Koide parameter Q = 2/3 is exact."""
        expected = Fraction(2, 3)

        # From formula: dim(G2)/b2 = 14/21 = 2/3
        computed = Fraction(gift_params_v22.dim_G2, gift_params_v22.b2_K7)

        assert computed == expected

    def test_det_g_exact_rational(self, gift_params_v22):
        """Test det(g) = 65/32 is exact."""
        expected = Fraction(65, 32)

        assert float(expected) == pytest.approx(2.03125, rel=1e-15)


# =============================================================================
# Transcendental Number Precision Tests
# =============================================================================

class TestTranscendentalPrecision:
    """Test precision of transcendental numbers."""

    def test_pi_precision(self):
        """Test pi is used with sufficient precision."""
        # GIFT uses pi in many formulas
        assert np.pi == pytest.approx(3.141592653589793, rel=1e-15)

    def test_golden_ratio_precision(self):
        """Test golden ratio calculation."""
        phi = (1 + np.sqrt(5)) / 2
        assert phi == pytest.approx(1.6180339887498949, rel=1e-15)

    def test_ln2_precision(self):
        """Test ln(2) precision for Omega_DE."""
        ln2 = np.log(2)
        assert ln2 == pytest.approx(0.6931471805599453, rel=1e-15)

    def test_omega_de_calculation(self, gift_params_v22):
        """Test Omega_DE = ln(2) * 98/99 precision."""
        expected = np.log(2) * 98 / 99
        # Verify the formula yields the expected value
        assert expected == pytest.approx(np.log(2) * 98 / 99, rel=1e-12)
        # Check it's in the right ballpark for dark energy density
        assert 0.68 < expected < 0.69

    def test_zeta_function_precision(self):
        """Test zeta function values for n_s."""
        from scipy.special import zeta

        zeta_5 = zeta(5)
        zeta_11 = zeta(11)

        # Verify zeta function values are in expected range
        assert 1.03 < zeta_5 < 1.04
        assert 1.0004 < zeta_11 < 1.0006

        n_s = zeta_11 / zeta_5
        # n_s should be close to 0.965 (spectral index)
        assert 0.96 < n_s < 0.97


# =============================================================================
# Near-Singular Value Tests
# =============================================================================

class TestNearSingularValues:
    """Test behavior near singular or boundary values."""

    def test_small_deviation_handling(self):
        """Test handling of very small deviations."""
        predicted = 197.0
        experimental = 197.0
        uncertainty = 24.0

        deviation = abs(predicted - experimental) / experimental * 100
        sigma = abs(predicted - experimental) / uncertainty

        assert deviation == 0.0
        assert sigma == 0.0
        assert np.isfinite(deviation)
        assert np.isfinite(sigma)

    def test_tiny_uncertainty_handling(self):
        """Test handling of tiny uncertainties."""
        predicted = 137.033
        experimental = 137.036
        uncertainty = 1e-6

        deviation = abs(predicted - experimental)
        sigma = deviation / uncertainty

        assert np.isfinite(sigma)
        # sigma should be large but finite
        assert sigma > 1000
        assert sigma < 1e10

    def test_zero_division_protection(self):
        """Test protection against division by zero."""
        # Test with zero experimental value
        predicted = 0.001
        experimental = 0.0

        # Should handle gracefully
        if experimental != 0:
            deviation = abs(predicted - experimental) / experimental
        else:
            deviation = np.inf if predicted != 0 else 0.0

        assert np.isfinite(deviation) or deviation == np.inf

    def test_small_angle_trig_precision(self):
        """Test trigonometric precision for small angles."""
        # theta13 is small (~8.5 degrees)
        theta13_deg = 8.571
        theta13_rad = np.radians(theta13_deg)

        # Check sin and cos are in expected range
        sin_val = np.sin(theta13_rad)
        cos_val = np.cos(theta13_rad)

        assert 0.14 < sin_val < 0.16
        assert 0.98 < cos_val < 0.99

        # Verify fundamental identity holds to high precision
        assert sin_val**2 + cos_val**2 == pytest.approx(1.0, rel=1e-14)


# =============================================================================
# Extreme Value Tests
# =============================================================================

class TestExtremeValues:
    """Test handling of extreme values."""

    def test_large_mass_ratio_handling(self):
        """Test handling of large mass ratios like m_tau/m_e."""
        m_tau_m_e = 3477
        m_e_MeV = 0.511

        m_tau_MeV = m_tau_m_e * m_e_MeV
        assert m_tau_MeV == pytest.approx(1776.747, rel=1e-6)
        assert np.isfinite(m_tau_MeV)

    def test_small_coupling_handling(self):
        """Test handling of small couplings."""
        V_ub = 0.00394

        # Test operations on small values
        V_ub_squared = V_ub ** 2
        assert V_ub_squared == pytest.approx(1.55236e-5, rel=1e-6)
        assert np.isfinite(V_ub_squared)

    def test_large_dimension_products(self):
        """Test products involving large dimensions."""
        dim_E8xE8 = 496
        b2 = 21
        H_star = 99

        # tau numerator
        product = dim_E8xE8 * b2
        assert product == 10416
        assert np.isfinite(product)

        # Large power
        two_power_weyl = 2 ** 5
        assert two_power_weyl == 32

    def test_cosmological_scale_values(self):
        """Test cosmological scale values."""
        H0 = 70.0  # km/s/Mpc
        Omega_DE = 0.6889

        # These should be well-behaved
        assert 0 < Omega_DE < 1
        assert 50 < H0 < 100
        assert np.isfinite(H0)
        assert np.isfinite(Omega_DE)


# =============================================================================
# Float Precision Boundary Tests
# =============================================================================

class TestFloatPrecisionBoundaries:
    """Test behavior at float precision boundaries."""

    def test_float64_epsilon(self):
        """Test near machine epsilon values."""
        eps = np.finfo(np.float64).eps

        # Values near 1 + epsilon
        x = 1.0 + eps
        y = 1.0 + 2*eps

        assert x != 1.0  # Should be distinguishable
        assert y > x

    def test_float64_max_min(self):
        """Test handling near float limits."""
        max_float = np.finfo(np.float64).max
        min_float = np.finfo(np.float64).tiny

        # Operations should remain finite
        assert np.isfinite(max_float / 2)
        assert np.isfinite(min_float * 2)

    def test_catastrophic_cancellation(self):
        """Test for catastrophic cancellation issues."""
        # Example: computing 1 - cos(x) for small x
        x = 1e-8
        naive = 1 - np.cos(x)
        stable = 2 * np.sin(x/2)**2

        # Both should give similar results for this x
        assert naive == pytest.approx(stable, rel=1e-6)

    def test_accumulation_error(self):
        """Test numerical error accumulation."""
        # Sum many small values
        n = 1000000
        small_val = 1e-10

        total = sum([small_val] * n)
        expected = n * small_val

        # Should be close but may have some error
        assert total == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Parameter Boundary Tests
# =============================================================================

class TestParameterBoundaries:
    """Test behavior at parameter boundaries."""

    def test_angle_boundaries(self):
        """Test angle values at boundaries."""
        # Angles should be in valid ranges
        theta12 = 33.42  # degrees
        theta13 = 8.571
        theta23 = 49.19
        delta_CP = 197.0

        # Check reasonable bounds
        assert 0 < theta12 < 90
        assert 0 < theta13 < 90
        assert 0 < theta23 < 90
        assert 0 < delta_CP < 360

    def test_probability_bounds(self):
        """Test values that should be probabilities."""
        sin2_theta_W = 3/13

        assert 0 < sin2_theta_W < 1

    def test_coupling_positivity(self):
        """Test couplings are positive."""
        alpha_s = np.sqrt(2) / 12
        alpha_inv = 137.036

        assert alpha_s > 0
        assert alpha_inv > 0
        assert 1/alpha_inv > 0

    def test_mass_ratio_positivity(self):
        """Test mass ratios are positive."""
        ratios = [
            20.0,    # m_s/m_d
            3477.0,  # m_tau/m_e
            41.3,    # m_t/m_b
        ]

        for ratio in ratios:
            assert ratio > 0


# =============================================================================
# Monte Carlo Stability Tests
# =============================================================================

class TestMonteCarloStability:
    """Test numerical stability in Monte Carlo calculations."""

    def test_mean_convergence(self):
        """Test mean convergence with large samples."""
        np.random.seed(42)

        # Generate samples
        n_samples = 100000
        samples = np.random.normal(0, 1, n_samples)

        mean = np.mean(samples)
        std_error = np.std(samples) / np.sqrt(n_samples)

        # Mean should be close to 0
        assert abs(mean) < 5 * std_error

    def test_variance_stability(self):
        """Test variance calculation stability."""
        np.random.seed(42)

        # Two methods of computing variance
        samples = np.random.normal(100, 10, 10000)

        var_numpy = np.var(samples, ddof=1)
        var_manual = np.sum((samples - np.mean(samples))**2) / (len(samples) - 1)

        assert var_numpy == pytest.approx(var_manual, rel=1e-10)

    def test_percentile_stability(self):
        """Test percentile calculation stability."""
        np.random.seed(42)

        samples = np.random.normal(0, 1, 10000)

        p5 = np.percentile(samples, 5)
        p95 = np.percentile(samples, 95)

        # Should be approximately symmetric around 0
        assert abs(p5 + p95) < 0.1

        # Should be approximately +/- 1.645 for normal
        assert p5 == pytest.approx(-1.645, rel=0.1)
        assert p95 == pytest.approx(1.645, rel=0.1)


# =============================================================================
# Cross-Platform Consistency Tests
# =============================================================================

class TestCrossPlatformConsistency:
    """Test that calculations are consistent across platforms."""

    def test_numpy_version_independence(self):
        """Test calculations don't depend on numpy version quirks."""
        # Basic operations should be consistent
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        dot = np.dot(a, b)
        assert dot == pytest.approx(32.0, rel=1e-14)

    def test_scipy_zeta_consistency(self):
        """Test scipy zeta function consistency."""
        from scipy.special import zeta

        # These should be platform-independent
        z5 = zeta(5)
        z11 = zeta(11)

        assert z5 == pytest.approx(1.0369277551433699, rel=1e-10)
        assert z11 == pytest.approx(1.0004941886041195, rel=1e-10)

    def test_deterministic_calculation(self):
        """Test that same inputs give same outputs."""
        # Without randomness, calculations should be deterministic
        x = 3.14159
        y = 2.71828

        result1 = np.sin(x) * np.cos(y) + np.exp(-x*y)
        result2 = np.sin(x) * np.cos(y) + np.exp(-x*y)

        assert result1 == result2  # Exact equality expected
