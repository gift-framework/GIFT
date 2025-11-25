"""
Quick Win Smoke Tests and Assertions.

Fast-running tests that verify:
- Basic framework functionality
- Key observable calculations
- Import success
- Quick sanity checks

These tests should run in under 1 second total.

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

    def test_import_gift_v21_core(self):
        """Test gift_v21_core imports."""
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

    def test_v21_instantiates(self):
        """Test GIFTFrameworkV21 instantiates."""
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

    def test_v21_returns_observables(self):
        """Test v2.1 returns non-empty observables."""
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
        """Test m_tau/m_e = 3477."""
        try:
            from giftpy import GIFT
            gift = GIFT()
            ratio = gift.lepton.m_tau_m_e()
            assert ratio == 3477
        except ImportError:
            pytest.skip("giftpy not available")

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
        """Test H* = 1 + 0 + 21 + 77 + 77 + 21 + 0 + 1 = 99."""
        expected = 1 + 0 + 21 + 77 + 77 + 21 + 0 + 1
        assert expected == 99

    def test_golden_ratio_property(self):
        """Test phi^2 = phi + 1."""
        phi = (1 + np.sqrt(5)) / 2
        assert np.isclose(phi**2, phi + 1, rtol=1e-14)

    def test_dim_G2_times_7_plus_99_is_197(self):
        """Test 7 * dim_G2 + 99 = 7*14 + 99 = 197 (delta_CP)."""
        assert 7 * 14 + 99 == 197


# =============================================================================
# Data Type Smoke Tests
# =============================================================================

class TestDataTypeSmoke:
    """Quick tests for correct data types."""

    def test_observables_are_numeric(self):
        """Test all observables are numeric."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert isinstance(value, (int, float, np.integer, np.floating)), \
                    f"{name} is not numeric: {type(value)}"
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_no_nan_in_observables(self):
        """Test no NaN in observables."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert not np.isnan(value), f"{name} is NaN"
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_no_inf_in_observables(self):
        """Test no infinity in observables."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            obs = fw.compute_all_observables()
            for name, value in obs.items():
                assert not np.isinf(value), f"{name} is infinite"
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")


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

    def test_same_results_twice(self):
        """Test same framework gives same results."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw = GIFTFrameworkV21()
            obs1 = fw.compute_all_observables()
            obs2 = fw.compute_all_observables()
            assert obs1 == obs2
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")

    def test_new_instance_same_results(self):
        """Test new instance gives same results."""
        try:
            from gift_v21_core import GIFTFrameworkV21
            fw1 = GIFTFrameworkV21()
            fw2 = GIFTFrameworkV21()
            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()
            for key in obs1:
                if key in obs2:
                    assert obs1[key] == obs2[key], f"{key} differs"
        except ImportError:
            pytest.skip("GIFTFrameworkV21 not available")


# =============================================================================
# Quick Assertion Collection
# =============================================================================

class TestQuickAssertions:
    """Collection of quick one-liner assertions."""

    def test_2_to_7_minus_1_over_24(self):
        """Test 2^7 - 1/24 ~ 127.958."""
        result = 2**7 - 1/24
        assert 127.9 < result < 128.0

    def test_sqrt2_over_12(self):
        """Test sqrt(2)/12 ~ 0.1179."""
        result = np.sqrt(2) / 12
        assert 0.117 < result < 0.119

    def test_14_over_21(self):
        """Test 14/21 = 2/3."""
        assert 14/21 == 2/3

    def test_77_plus_10_times_248_plus_10_times_99(self):
        """Test 77 + 10*248 + 10*99 = 3477."""
        assert 77 + 10*248 + 10*99 == 3477

    def test_ln2(self):
        """Test ln(2) ~ 0.693."""
        assert 0.69 < np.log(2) < 0.70

    def test_sqrt17_over_32(self):
        """Test sqrt(17)/32 ~ 0.129."""
        result = np.sqrt(17) / 32
        assert 0.128 < result < 0.130

    def test_3_over_13(self):
        """Test 3/13 ~ 0.231."""
        result = 3/13
        assert 0.23 < result < 0.24

    def test_pi_over_9_degrees(self):
        """Test pi/9 radians ~ 20 degrees."""
        result = np.degrees(np.pi / 9)
        assert 19 < result < 21

    def test_27_to_golden_power(self):
        """Test 27^phi ~ 206.77."""
        phi = (1 + np.sqrt(5)) / 2
        result = 27 ** phi
        assert 206 < result < 207


# =============================================================================
# Run All Smoke Tests
# =============================================================================

if __name__ == '__main__':
    # Run with verbose output, fast only
    pytest.main([__file__, '-v', '--tb=short'])
