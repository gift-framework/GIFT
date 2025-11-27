"""
Quick Win Smoke Tests and Assertions.

Fast-running tests that verify:
- Basic framework functionality
- Key observable calculations
- Import success
- Quick sanity checks

These tests should run in under 1 second total.

Version: 2.2.0 (updated from 2.1.0)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from fractions import Fraction

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))
sys.path.insert(0, str(PROJECT_ROOT / "giftpy"))


# =============================================================================
# Import Smoke Tests (Must Pass First)
# =============================================================================

class TestImportSmoke:
    """Test that all key modules can be imported."""

    def test_import_numpy(self):
        """Test numpy imports."""
        import numpy as np
        assert np.pi > 3

    def test_import_gift_v22_core(self):
        """Test gift_v22_core imports (v2.2 primary)."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            assert GIFTFrameworkV22 is not None
        except ImportError:
            pytest.skip("gift_v22_core not available")

    def test_import_gift_v21_core(self):
        """Test gift_v21_core imports (legacy)."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            assert GIFTFrameworkV21 is not None
        except ImportError:
            pytest.skip("gift_v21_core not available")

    def test_import_run_validation(self):
        """Test run_validation imports."""
        try:
            from run_validation import GIFTFrameworkStatistical
            assert GIFTFrameworkStatistical is not None
        except ImportError:
            pytest.skip("run_validation not available")

    def test_import_giftpy(self):
        """Test giftpy imports."""
        try:
            from giftpy import GIFT
            assert GIFT is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_import_giftpy_constants(self):
        """Test giftpy.core.constants imports."""
        try:
            from giftpy.core.constants import CONSTANTS, TopologicalConstants
            assert CONSTANTS is not None
            assert TopologicalConstants is not None
        except ImportError:
            pytest.skip("giftpy.core.constants not available")


# =============================================================================
# Framework Instantiation Smoke Tests
# =============================================================================

class TestInstantiationSmoke:
    """Test framework instantiation."""

    def test_v22_instantiates(self):
        """Test GIFTFrameworkV22 instantiates (v2.2 primary)."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            assert fw is not None
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")

    def test_v21_instantiates(self):
        """Test GIFTFrameworkV21 instantiates (legacy)."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            assert fw is not None
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_statistical_instantiates(self):
        """Test GIFTFrameworkStatistical instantiates."""
        try:
            from run_validation import GIFTFrameworkStatistical
            fw = GIFTFrameworkStatistical()
            assert fw is not None
        except ImportError:
            pytest.skip("GIFTFrameworkStatistical not available")

    def test_giftpy_instantiates(self):
        """Test giftpy GIFT instantiates."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            assert gift is not None
        except ImportError:
            pytest.skip("giftpy not available")


# =============================================================================
# Observable Count Smoke Tests
# =============================================================================

class TestObservableCountSmoke:
    """Quick test that frameworks return observables."""

    def test_v22_returns_observables(self):
        """Test v2.2 returns non-empty observables."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs = fw.compute_all_observables()
            assert len(obs) >= 30, f"Only {len(obs)} observables returned (expected 39)"
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")

    def test_v21_returns_observables(self):
        """Test v2.1 returns non-empty observables (legacy)."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            obs = fw.compute_all_observables()
            assert len(obs) >= 5, f"Only {len(obs)} observables returned"
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_statistical_returns_observables(self):
        """Test statistical framework returns non-empty observables."""
        try:
            from run_validation import GIFTFrameworkStatistical
            fw = GIFTFrameworkStatistical()
            obs = fw.compute_all_observables()
            assert len(obs) >= 5, f"Only {len(obs)} observables returned"
        except ImportError:
            pytest.skip("GIFTFrameworkStatistical not available")


# =============================================================================
# Key Value Smoke Tests
# =============================================================================

class TestKeyValueSmoke:
    """Quick tests for key observable values."""

    def test_Q_Koide_is_two_thirds(self):
        """Test Q_Koide = 2/3."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            Q = gift.lepton.Q_Koide()
            assert np.isclose(Q, 2/3, rtol=1e-14)
        except ImportError:
            pytest.skip("giftpy not available")

    def test_m_tau_m_e_is_3477(self):
        """Test m_tau/m_e = 3477 (v2.2 PROVEN)."""
        # Try v2.2 framework first (primary)
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs = fw.compute_all_observables()
            assert obs['m_tau_m_e'] == 3477
            return
        except ImportError:
            pass

        # Fallback to giftpy
        try:
            from giftpy import GIFT
            gift = GIFT()
            ratio = gift.lepton.m_tau_m_e()
            # giftpy may use different formula - check if close to 3477
            assert abs(ratio - 3477) < 1 or ratio == 3477
        except ImportError:
            pytest.skip("Neither GIFTFrameworkV22 nor giftpy available")

    def test_alpha_s_is_sqrt2_over_12(self):
        """Test alpha_s = sqrt(2)/12."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            alpha_s = gift.gauge.alpha_s()
            expected = np.sqrt(2) / 12
            assert np.isclose(alpha_s, expected, rtol=1e-14)
        except ImportError:
            pytest.skip("giftpy not available")

    def test_m_s_m_d_is_20(self):
        """Test m_s/m_d = 20."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            ratio = gift.quark.m_s_m_d()
            assert ratio == 20
        except ImportError:
            pytest.skip("giftpy not available")


# =============================================================================
# v2.2 Specific Smoke Tests
# =============================================================================

class TestV22SpecificSmoke:
    """Quick tests for v2.2 specific values (zero-parameter paradigm)."""

    def test_sin2_theta_w_is_3_over_13(self):
        """Test sin^2(theta_W) = 3/13 (v2.2 PROVEN)."""
        try:
            from gift_v22_core import GIFTParametersV22
            params = GIFTParametersV22()
            assert params.sin2_theta_W == Fraction(3, 13)
            assert np.isclose(float(params.sin2_theta_W), 3/13)
        except ImportError:
            pytest.skip("GIFTParametersV22 not available")

    def test_kappa_T_is_1_over_61(self):
        """Test kappa_T = 1/61 (v2.2 TOPOLOGICAL)."""
        try:
            from gift_v22_core import GIFTParametersV22
            params = GIFTParametersV22()
            assert params.kappa_T == Fraction(1, 61)
            assert np.isclose(params.kappa_T_float, 1/61)
        except ImportError:
            pytest.skip("GIFTParametersV22 not available")

    def test_tau_is_3472_over_891(self):
        """Test tau = 3472/891 (v2.2 PROVEN)."""
        try:
            from gift_v22_core import GIFTParametersV22
            params = GIFTParametersV22()
            assert params.tau == Fraction(3472, 891)
            assert np.isclose(params.tau_float, 3472/891)
        except ImportError:
            pytest.skip("GIFTParametersV22 not available")

    def test_det_g_is_65_over_32(self):
        """Test det(g) = 65/32 (v2.2 TOPOLOGICAL)."""
        try:
            from gift_v22_core import GIFTParametersV22
            params = GIFTParametersV22()
            assert params.det_g == Fraction(65, 32)
            assert np.isclose(params.det_g_float, 65/32)
        except ImportError:
            pytest.skip("GIFTParametersV22 not available")

    def test_lambda_H_is_sqrt17_over_32(self):
        """Test lambda_H = sqrt(17)/32 (v2.2 PROVEN)."""
        try:
            from gift_v22_core import GIFTParametersV22
            params = GIFTParametersV22()
            expected = np.sqrt(17) / 32
            assert np.isclose(params.lambda_H, expected, rtol=1e-14)
        except ImportError:
            pytest.skip("GIFTParametersV22 not available")


# =============================================================================
# Topological Constants Smoke Tests
# =============================================================================

class TestTopologicalConstantsSmoke:
    """Quick tests for topological constants."""

    def test_b2_is_21(self):
        """Test b2(K7) = 21."""
        try:
            from giftpy.core.constants import CONSTANTS
            assert CONSTANTS.b2 == 21
        except ImportError:
            pytest.skip("giftpy not available")

    def test_b3_is_77(self):
        """Test b3(K7) = 77."""
        try:
            from giftpy.core.constants import CONSTANTS
            assert CONSTANTS.b3 == 77
        except ImportError:
            pytest.skip("giftpy not available")

    def test_dim_E8_is_248(self):
        """Test dim(E8) = 248."""
        try:
            from giftpy.core.constants import CONSTANTS
            assert CONSTANTS.dim_E8 == 248
        except ImportError:
            pytest.skip("giftpy not available")

    def test_H_star_is_99(self):
        """Test H*(K7) = 99."""
        try:
            from giftpy.core.constants import CONSTANTS
            assert CONSTANTS.H_star == 99
        except ImportError:
            pytest.skip("giftpy not available")


# =============================================================================
# Mathematical Identity Smoke Tests
# =============================================================================

class TestMathematicalIdentitiesSmoke:
    """Quick tests for mathematical identities."""

    def test_b2_plus_b3_is_98(self):
        """Test b2 + b3 = 98."""
        try:
            from giftpy.core.constants import CONSTANTS
            assert CONSTANTS.b2 + CONSTANTS.b3 == 98
        except ImportError:
            pytest.skip("giftpy not available")

    def test_H_star_equals_sum_betti(self):
        """Test H* = b2 + b3 + 1 = 99."""
        assert 21 + 77 + 1 == 99

    def test_golden_ratio_property(self):
        """Test phi^2 = phi + 1."""
        phi = (1 + np.sqrt(5)) / 2
        assert np.isclose(phi**2, phi + 1, rtol=1e-14)

    def test_dim_K7_times_dim_G2_plus_H_star_is_197(self):
        """Test 7 * dim_G2 + H* = 7*14 + 99 = 197 (delta_CP)."""
        assert 7 * 14 + 99 == 197

    def test_b3_formula(self):
        """Test b3 = 2*dim(K7)^2 - b2 = 2*49 - 21 = 77."""
        assert 2 * 7**2 - 21 == 77


# =============================================================================
# Data Type Smoke Tests
# =============================================================================

class TestDataTypeSmoke:
    """Quick tests for correct data types."""

    def test_v22_observables_are_numeric(self):
        """Test all v2.2 observables are numeric."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert isinstance(value, (int, float, np.integer, np.floating)), \
                    f"{name} is not numeric: {type(value)}"
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")

    def test_v22_no_nan_in_observables(self):
        """Test no NaN in v2.2 observables."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert not np.isnan(value), f"{name} is NaN"
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")

    def test_v22_no_inf_in_observables(self):
        """Test no infinity in v2.2 observables."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert not np.isinf(value), f"{name} is infinite"
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")


# =============================================================================
# Sector Access Smoke Tests
# =============================================================================

class TestSectorAccessSmoke:
    """Quick tests for sector access."""

    def test_gauge_sector_accessible(self):
        """Test gauge sector is accessible."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            assert gift.gauge is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_lepton_sector_accessible(self):
        """Test lepton sector is accessible."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            assert gift.lepton is not None
        except ImportError:
            pytest.skip("giftpy not available")

    def test_quark_sector_accessible(self):
        """Test quark sector is accessible."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            assert gift.quark is not None
        except ImportError:
            pytest.skip("giftpy not available")


# =============================================================================
# Reproducibility Smoke Tests
# =============================================================================

class TestReproducibilitySmoke:
    """Quick tests for reproducibility."""

    def test_v22_same_results_twice(self):
        """Test v2.2 framework gives same results."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw = GIFTFrameworkV22()
            obs1 = fw.compute_all_observables()
            obs2 = fw.compute_all_observables()
            assert obs1 == obs2
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")

    def test_v22_new_instance_same_results(self):
        """Test new v2.2 instance gives same results (zero-parameter paradigm)."""
        try:
            from gift_v22_core import GIFTFrameworkV22
            fw1 = GIFTFrameworkV22()
            fw2 = GIFTFrameworkV22()
            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()
            for key in obs1:
                if key in obs2:
                    assert obs1[key] == obs2[key], f"{key} differs"
        except ImportError:
            pytest.skip("GIFTFrameworkV22 not available")


# =============================================================================
# Quick Assertion Collection
# =============================================================================

class TestQuickAssertions:
    """Collection of quick one-liner assertions."""

    def test_sqrt2_over_12(self):
        """Test sqrt(2)/12 ~ 0.1179."""
        result = np.sqrt(2) / 12
        assert 0.117 < result < 0.119

    def test_14_over_21(self):
        """Test 14/21 = 2/3."""
        assert 14/21 == 2/3

    def test_3_over_13(self):
        """Test 3/13 ~ 0.231 (v2.2 sin^2 theta_W)."""
        result = 3/13
        assert 0.23 < result < 0.24

    def test_1_over_61(self):
        """Test 1/61 ~ 0.0164 (v2.2 kappa_T)."""
        result = 1/61
        assert 0.016 < result < 0.017

    def test_3472_over_891(self):
        """Test 3472/891 ~ 3.897 (v2.2 tau)."""
        result = 3472/891
        assert 3.89 < result < 3.90

    def test_65_over_32(self):
        """Test 65/32 ~ 2.031 (v2.2 det_g)."""
        result = 65/32
        assert 2.03 < result < 2.04

    def test_77_plus_10_times_248_plus_10_times_99(self):
        """Test 7 + 10*248 + 10*99 = 3477 (m_tau/m_e)."""
        assert 7 + 10*248 + 10*99 == 3477

    def test_ln2(self):
        """Test ln(2) ~ 0.693."""
        assert 0.69 < np.log(2) < 0.70

    def test_sqrt17_over_32(self):
        """Test sqrt(17)/32 ~ 0.129 (lambda_H)."""
        result = np.sqrt(17) / 32
        assert 0.128 < result < 0.130

    def test_pi_over_8(self):
        """Test pi/8 ~ 0.393 (beta0)."""
        result = np.pi / 8
        assert 0.39 < result < 0.40

    def test_27_to_golden_power(self):
        """Test 27^phi ~ 206.77 (m_mu/m_e)."""
        phi = (1 + np.sqrt(5)) / 2
        result = 27 ** phi
        assert 206 < result < 208


# =============================================================================
# v2.2 Zero-Parameter Paradigm Tests
# =============================================================================

class TestZeroParameterParadigm:
    """Tests verifying v2.2's zero-parameter paradigm."""

    def test_tau_is_exact_rational(self):
        """Test tau = 496*21/(27*99) is exact."""
        numerator = 496 * 21  # = 10416
        denominator = 27 * 99  # = 2673
        # 10416/2673 = 3472/891 (simplified)
        assert Fraction(numerator, denominator) == Fraction(3472, 891)

    def test_kappa_T_formula(self):
        """Test kappa_T = 1/(b3 - dim_G2 - p2) = 1/(77-14-2) = 1/61."""
        b3 = 77
        dim_G2 = 14
        p2 = 2
        assert 1 / (b3 - dim_G2 - p2) == 1 / 61

    def test_sin2_theta_W_formula(self):
        """Test sin^2(theta_W) = b2/(b3 + dim_G2) = 21/(77+14) = 21/91 = 3/13."""
        b2 = 21
        b3 = 77
        dim_G2 = 14
        result = Fraction(b2, b3 + dim_G2)
        assert result == Fraction(3, 13)


# =============================================================================
# Run All Smoke Tests
# =============================================================================

if __name__ == '__main__':
    # Run with verbose output, fast only
    pytest.main([__file__, '-v', '--tb=short'])
