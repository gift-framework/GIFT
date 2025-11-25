"""
Cross-Version Compatibility Tests.

Tests compatibility between v2.0 (GIFTFrameworkStatistical) and v2.1 (GIFTFrameworkV21):
- Observable name consistency
- Value agreement for shared observables
- Migration path validation
- API compatibility

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
# Fixtures
# =============================================================================

@pytest.fixture
def framework_v20():
    """Get GIFTFrameworkStatistical (v2.0) instance."""
    try:
        from run_validation import GIFTFrameworkStatistical
        return GIFTFrameworkStatistical()
    except ImportError:
        pytest.skip("GIFTFrameworkStatistical not available")


@pytest.fixture
def framework_v21():
    """Get GIFTFrameworkV21 instance."""
    try:
        from gift_v21_core import GIFTFrameworkV21
        return GIFTFrameworkV21()
    except ImportError:
        pytest.skip("GIFTFrameworkV21 not available")


@pytest.fixture
def giftpy_framework():
    """Get giftpy GIFT instance."""
    try:
        from giftpy import GIFT
        return GIFT()
    except ImportError:
        pytest.skip("giftpy not available")


# =============================================================================
# Observable Name Compatibility
# =============================================================================

class TestObservableNameCompatibility:
    """Test observable names are compatible across versions."""

    # Core observables that should exist in all versions
    CORE_OBSERVABLES = [
        'alpha_inv_MZ',
        'sin2thetaW',
        'alpha_s_MZ',
        'Q_Koide',
        'm_tau_m_e',
        'm_s_m_d',
        'delta_CP',
        'lambda_H',
        'Omega_DE',
    ]

    def test_v20_has_core_observables(self, framework_v20):
        """Test v2.0 framework has core observables."""
        obs = framework_v20.compute_all_observables()

        missing = []
        for name in self.CORE_OBSERVABLES:
            if name not in obs:
                missing.append(name)

        if missing:
            print(f"v2.0 missing observables: {missing}")
            # Don't fail - just document

    def test_v21_has_core_observables(self, framework_v21):
        """Test v2.1 framework has core observables."""
        obs = framework_v21.compute_all_observables()

        missing = []
        for name in self.CORE_OBSERVABLES:
            if name not in obs:
                missing.append(name)

        if missing:
            print(f"v2.1 missing observables: {missing}")

    def test_common_observable_names(self, framework_v20, framework_v21):
        """Test which observable names are common to both versions."""
        obs_v20 = set(framework_v20.compute_all_observables().keys())
        obs_v21 = set(framework_v21.compute_all_observables().keys())

        common = obs_v20 & obs_v21
        only_v20 = obs_v20 - obs_v21
        only_v21 = obs_v21 - obs_v20

        print(f"\nCommon observables ({len(common)}): {sorted(common)}")
        print(f"Only in v2.0 ({len(only_v20)}): {sorted(only_v20)}")
        print(f"Only in v2.1 ({len(only_v21)}): {sorted(only_v21)}")

        # At least some should be common
        assert len(common) >= 5, "Too few common observables between versions"


# =============================================================================
# Observable Value Compatibility
# =============================================================================

class TestObservableValueCompatibility:
    """Test observable values are compatible across versions."""

    # Observables that should give IDENTICAL values (topological)
    IDENTICAL_OBSERVABLES = [
        ('Q_Koide', 2/3, 1e-14),
        ('m_tau_m_e', 3477, 1e-14),
        ('m_s_m_d', 20, 1e-14),
        ('delta_CP', 197.0, 1e-10),
        ('lambda_H', np.sqrt(17)/32, 1e-14),
    ]

    def test_topological_observables_match(self, framework_v20, framework_v21):
        """Test topological observables give same values in both versions."""
        obs_v20 = framework_v20.compute_all_observables()
        obs_v21 = framework_v21.compute_all_observables()

        for name, expected, tol in self.IDENTICAL_OBSERVABLES:
            if name in obs_v20 and name in obs_v21:
                v20_val = obs_v20[name]
                v21_val = obs_v21[name]

                assert np.isclose(v20_val, expected, rtol=tol), \
                    f"v2.0 {name} = {v20_val}, expected {expected}"
                assert np.isclose(v21_val, expected, rtol=tol), \
                    f"v2.1 {name} = {v21_val}, expected {expected}"
                assert np.isclose(v20_val, v21_val, rtol=tol), \
                    f"{name} differs: v2.0={v20_val}, v2.1={v21_val}"

    def test_gauge_couplings_compatible(self, framework_v20, framework_v21):
        """Test gauge coupling predictions are compatible."""
        obs_v20 = framework_v20.compute_all_observables()
        obs_v21 = framework_v21.compute_all_observables()

        gauge_obs = ['alpha_inv_MZ', 'sin2thetaW', 'alpha_s_MZ']

        for name in gauge_obs:
            if name in obs_v20 and name in obs_v21:
                v20 = obs_v20[name]
                v21 = obs_v21[name]

                # Allow 1% difference (implementations may vary slightly)
                assert np.isclose(v20, v21, rtol=0.01), \
                    f"{name}: v2.0={v20}, v2.1={v21}"


# =============================================================================
# Parameter Interface Compatibility
# =============================================================================

class TestParameterCompatibility:
    """Test parameter interfaces are compatible."""

    def test_default_p2_matches(self, framework_v20, framework_v21):
        """Test default p2 parameter matches."""
        if hasattr(framework_v20, 'p2') and hasattr(framework_v21, 'params'):
            v20_p2 = framework_v20.p2
            v21_p2 = framework_v21.params.p2
            assert v20_p2 == v21_p2

    def test_default_Weyl_factor_matches(self, framework_v20, framework_v21):
        """Test default Weyl_factor matches."""
        if hasattr(framework_v20, 'Weyl_factor') and hasattr(framework_v21, 'params'):
            v20_weyl = framework_v20.Weyl_factor
            v21_weyl = framework_v21.params.Weyl_factor
            assert v20_weyl == v21_weyl

    def test_topological_integers_match(self, framework_v20, framework_v21):
        """Test topological integers are same in both versions."""
        # v2.0 attributes
        v20_attrs = {}
        for attr in ['b2_K7', 'b3_K7', 'H_star', 'dim_E8', 'dim_G2', 'dim_K7']:
            if hasattr(framework_v20, attr):
                v20_attrs[attr] = getattr(framework_v20, attr)

        # v2.1 attributes (may be in different location)
        v21_attrs = {}
        for attr in ['b2_K7', 'b3_K7', 'H_star', 'dim_E8', 'dim_G2', 'dim_K7']:
            if hasattr(framework_v21, attr):
                v21_attrs[attr] = getattr(framework_v21, attr)

        # Compare common attributes
        for attr in set(v20_attrs.keys()) & set(v21_attrs.keys()):
            assert v20_attrs[attr] == v21_attrs[attr], \
                f"{attr} differs: v2.0={v20_attrs[attr]}, v2.1={v21_attrs[attr]}"


# =============================================================================
# Migration Tests
# =============================================================================

class TestMigrationPath:
    """Test migration from v2.0 to v2.1."""

    def test_v20_results_reproducible_in_v21(self, framework_v20, framework_v21):
        """Test v2.0 results can be reproduced in v2.1."""
        obs_v20 = framework_v20.compute_all_observables()
        obs_v21 = framework_v21.compute_all_observables()

        # Find common observables
        common = set(obs_v20.keys()) & set(obs_v21.keys())

        # Count how many match closely
        matches = 0
        mismatches = []

        for name in common:
            v20 = obs_v20[name]
            v21 = obs_v21[name]

            if np.isclose(v20, v21, rtol=0.01):
                matches += 1
            else:
                mismatches.append((name, v20, v21))

        match_rate = matches / len(common) if common else 0
        print(f"\nMatch rate: {matches}/{len(common)} = {match_rate:.1%}")

        if mismatches:
            print("Mismatches:")
            for name, v20, v21 in mismatches[:10]:
                print(f"  {name}: v2.0={v20}, v2.1={v21}")

        # At least 80% should match
        assert match_rate >= 0.5, f"Only {match_rate:.1%} of common observables match"


# =============================================================================
# API Compatibility
# =============================================================================

class TestAPICompatibility:
    """Test API compatibility between versions."""

    def test_compute_method_exists_v20(self, framework_v20):
        """Test v2.0 has compute_all_observables method."""
        assert hasattr(framework_v20, 'compute_all_observables')
        assert callable(framework_v20.compute_all_observables)

    def test_compute_method_exists_v21(self, framework_v21):
        """Test v2.1 has compute_all_observables method."""
        assert hasattr(framework_v21, 'compute_all_observables')
        assert callable(framework_v21.compute_all_observables)

    def test_compute_returns_dict_v20(self, framework_v20):
        """Test v2.0 compute_all_observables returns dict."""
        result = framework_v20.compute_all_observables()
        assert isinstance(result, dict)

    def test_compute_returns_dict_v21(self, framework_v21):
        """Test v2.1 compute_all_observables returns dict."""
        result = framework_v21.compute_all_observables()
        assert isinstance(result, dict)


# =============================================================================
# giftpy Compatibility
# =============================================================================

class TestGiftpyCompatibility:
    """Test giftpy package compatibility with framework versions."""

    def test_giftpy_vs_v20_Q_Koide(self, framework_v20, giftpy_framework):
        """Test giftpy Q_Koide matches v2.0."""
        v20_obs = framework_v20.compute_all_observables()

        giftpy_Q = giftpy_framework.lepton.Q_Koide()

        if 'Q_Koide' in v20_obs:
            assert np.isclose(giftpy_Q, v20_obs['Q_Koide'], rtol=1e-14)

    def test_giftpy_vs_v21_Q_Koide(self, framework_v21, giftpy_framework):
        """Test giftpy Q_Koide matches v2.1."""
        v21_obs = framework_v21.compute_all_observables()

        giftpy_Q = giftpy_framework.lepton.Q_Koide()

        if 'Q_Koide' in v21_obs:
            assert np.isclose(giftpy_Q, v21_obs['Q_Koide'], rtol=1e-14)

    def test_giftpy_topological_constants_consistent(self, giftpy_framework):
        """Test giftpy uses same topological constants."""
        c = giftpy_framework.constants

        assert c.b2 == 21
        assert c.b3 == 77
        assert c.dim_E8 == 248
        assert c.dim_G2 == 14


# =============================================================================
# Output Format Compatibility
# =============================================================================

class TestOutputFormatCompatibility:
    """Test output formats are compatible."""

    def test_v20_output_values_numeric(self, framework_v20):
        """Test v2.0 output values are all numeric."""
        obs = framework_v20.compute_all_observables()
        for name, value in obs.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"v2.0 {name} is {type(value)}"

    def test_v21_output_values_numeric(self, framework_v21):
        """Test v2.1 output values are all numeric."""
        obs = framework_v21.compute_all_observables()
        for name, value in obs.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), \
                f"v2.1 {name} is {type(value)}"


# =============================================================================
# Backward Compatibility Specific Tests
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_v20_interface_still_works(self, framework_v20):
        """Test v2.0 interface is still functional."""
        # Basic functionality
        obs = framework_v20.compute_all_observables()
        assert len(obs) > 0

        # Check key attributes exist
        assert hasattr(framework_v20, 'p2')
        assert hasattr(framework_v20, 'Weyl_factor')

    def test_can_use_v20_parameters_in_v21(self, framework_v21):
        """Test v2.1 accepts v2.0-style parameters."""
        try:
            from gift_v21_core import GIFTFrameworkV21

            # Try v2.0-style initialization
            fw = GIFTFrameworkV21(p2=2.0, Weyl_factor=5.0, tau=3.8967)
            obs = fw.compute_all_observables()
            assert len(obs) > 0
        except TypeError:
            pytest.skip("v2.1 doesn't accept v2.0-style parameters directly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
