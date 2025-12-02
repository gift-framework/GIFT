"""
Dimensional Analysis Tests

Tests for units consistency and dimensional correctness of GIFT predictions.
Verifies that mass dimensions, angular units, and derived quantities are consistent.

Author: GIFT Framework Team
"""

import pytest
import numpy as np
import os
import sys

# Add statistical_validation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../statistical_validation'))
from gift_v22_core import GIFTFrameworkV22, GIFTParametersV22


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gift_framework():
    """Create GIFT v2.3a framework."""
    return GIFTFrameworkV22()


@pytest.fixture
def observables(gift_framework):
    """Compute all observables."""
    return gift_framework.compute_all_observables()


@pytest.fixture
def params():
    """Get GIFT parameters."""
    return GIFTParametersV22()


# ============================================================================
# Mass Unit Consistency Tests
# ============================================================================

class TestMassUnits:
    """Tests for mass unit consistency."""

    def test_quark_masses_in_mev(self, observables):
        """Light quark masses should be in MeV."""
        # u, d, s quarks are in MeV
        assert 1 < observables['m_u_MeV'] < 10, "m_u should be a few MeV"
        assert 1 < observables['m_d_MeV'] < 10, "m_d should be a few MeV"
        assert 50 < observables['m_s_MeV'] < 150, "m_s should be ~100 MeV"

    def test_charm_bottom_in_mev(self, observables):
        """c, b quark masses should be in MeV."""
        assert 1000 < observables['m_c_MeV'] < 2000, "m_c should be ~1.3 GeV = 1300 MeV"
        assert 3000 < observables['m_b_MeV'] < 5000, "m_b should be ~4.2 GeV = 4200 MeV"

    def test_top_quark_in_gev(self, observables):
        """Top quark mass should be in GeV."""
        assert 150 < observables['m_t_GeV'] < 200, "m_t should be ~173 GeV"

    def test_electroweak_scale_in_gev(self, observables):
        """Electroweak scale should be in GeV."""
        assert 200 < observables['v_EW'] < 300, "v_EW should be ~246 GeV"
        assert 70 < observables['M_W'] < 90, "M_W should be ~80 GeV"
        assert 85 < observables['M_Z'] < 95, "M_Z should be ~91 GeV"

    def test_mass_hierarchy_quark_sector(self, observables):
        """Quark masses should follow expected hierarchy."""
        # Convert all to same units (MeV)
        m_u = observables['m_u_MeV']
        m_d = observables['m_d_MeV']
        m_s = observables['m_s_MeV']
        m_c = observables['m_c_MeV']
        m_b = observables['m_b_MeV']
        m_t = observables['m_t_GeV'] * 1000  # Convert to MeV

        # Hierarchy: m_u < m_d < m_s < m_c < m_b < m_t
        assert m_u < m_d, "m_u < m_d"
        assert m_d < m_s, "m_d < m_s"
        assert m_s < m_c, "m_s < m_c"
        assert m_c < m_b, "m_c < m_b"
        assert m_b < m_t, "m_b < m_t"


# ============================================================================
# Angular Unit Consistency Tests
# ============================================================================

class TestAngularUnits:
    """Tests for angular unit consistency."""

    def test_neutrino_mixing_angles_in_degrees(self, observables):
        """Neutrino mixing angles should be in degrees."""
        # All should be between 0 and 90 degrees
        assert 0 < observables['theta12'] < 90, "theta12 should be in degrees"
        assert 0 < observables['theta13'] < 90, "theta13 should be in degrees"
        assert 0 < observables['theta23'] < 90, "theta23 should be in degrees"

    def test_cp_phase_in_degrees(self, observables):
        """CP violation phase should be in degrees."""
        # delta_CP can be 0-360 degrees
        assert 0 <= observables['delta_CP'] <= 360, "delta_CP should be in degrees"

    def test_angle_values_reasonable(self, observables):
        """Mixing angles should have reasonable values."""
        # theta12 (solar): ~33-34 degrees
        assert 30 < observables['theta12'] < 40, "theta12 should be ~33 degrees"

        # theta13 (reactor): ~8-9 degrees
        assert 7 < observables['theta13'] < 10, "theta13 should be ~8.5 degrees"

        # theta23 (atmospheric): ~45-50 degrees
        assert 40 < observables['theta23'] < 55, "theta23 should be ~49 degrees"

    def test_internal_angle_consistency(self, gift_framework, params):
        """Internal angular calculations should be consistent."""
        # beta0 = pi/8 radians
        beta0_rad = params.beta0
        assert np.isclose(beta0_rad, np.pi / 8), "beta0 should be pi/8"

        # xi = 5*pi/16 radians
        xi_rad = params.xi
        assert np.isclose(xi_rad, 5 * np.pi / 16), "xi should be 5*pi/16"


# ============================================================================
# Dimensionless Ratio Tests
# ============================================================================

class TestDimensionlessRatios:
    """Tests for dimensionless quantity consistency."""

    def test_mass_ratios_dimensionless(self, observables):
        """Mass ratios should be pure numbers."""
        dimensionless_ratios = [
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e',
            'm_s_m_d', 'm_c_m_s', 'm_b_m_u', 'm_t_m_b',
            'm_c_m_d', 'm_b_m_d', 'm_t_m_c', 'm_t_m_s', 'm_d_m_u'
        ]

        for ratio in dimensionless_ratios:
            if ratio in observables:
                value = observables[ratio]
                # Should be finite positive number
                assert value > 0, f"{ratio} should be positive"
                assert np.isfinite(value), f"{ratio} should be finite"

    def test_ckm_elements_dimensionless(self, observables):
        """CKM matrix elements should be between 0 and 1."""
        ckm_elements = ['V_us', 'V_cb', 'V_ub', 'V_td', 'V_ts', 'V_tb']

        for elem in ckm_elements:
            value = observables[elem]
            assert 0 <= value <= 1, f"{elem} should be in [0, 1]"

    def test_gauge_couplings_dimensionless(self, observables):
        """Gauge couplings should be dimensionless."""
        # alpha^-1 is dimensionless
        assert observables['alpha_inv'] > 100, "alpha^-1 should be ~137"

        # sin^2(theta_W) is dimensionless, between 0 and 1
        assert 0 < observables['sin2thetaW'] < 1

        # alpha_s is dimensionless, ~0.12
        assert 0 < observables['alpha_s_MZ'] < 1

    def test_cosmological_dimensionless(self, observables):
        """Cosmological density parameters should be dimensionless."""
        # Omega_DE should be ~0.7
        assert 0 < observables['Omega_DE'] < 1

        # n_s (spectral index) should be ~1
        assert 0.9 < observables['n_s'] < 1.1


# ============================================================================
# Hubble Constant Units Tests
# ============================================================================

class TestHubbleUnits:
    """Tests for Hubble constant units."""

    def test_h0_in_km_s_mpc(self, observables):
        """H0 should be in km/s/Mpc."""
        H0 = observables['H0']

        # Should be 60-80 km/s/Mpc
        assert 60 < H0 < 80, f"H0 = {H0} should be ~70 km/s/Mpc"

    def test_h0_reasonable_age_universe(self, observables):
        """H0 should give reasonable age of universe."""
        H0 = observables['H0']  # km/s/Mpc

        # Convert to 1/seconds
        # 1 Mpc = 3.086e19 km
        # H0 in 1/s = H0 / 3.086e19
        # Age ~ 1/H0

        # Hubble time in Gyr
        # t_H = 1/H0 * (3.086e19 km/Mpc) / (3.156e16 s/Gyr) / (km/s)
        # t_H = 3.086e19 / (H0 * 3.156e16) Gyr
        # t_H = 977.8 / H0 Gyr

        t_H = 977.8 / H0  # Hubble time in Gyr

        # Should be ~14 Gyr (actual age is ~13.8 Gyr)
        assert 10 < t_H < 20, f"Hubble time = {t_H} Gyr unreasonable"


# ============================================================================
# Mass Ratio Self-Consistency Tests
# ============================================================================

class TestMassRatioSelfConsistency:
    """Tests that mass ratios are self-consistent with absolute masses."""

    def test_strange_down_ratio_consistent(self, observables):
        """m_s/m_d from ratio should match absolute masses."""
        m_s = observables['m_s_MeV']
        m_d = observables['m_d_MeV']
        ratio = observables['m_s_m_d']

        computed_ratio = m_s / m_d
        # Allow some tolerance since formulas may differ
        assert np.isclose(computed_ratio, ratio, rtol=0.1), \
            f"m_s/m_d = {computed_ratio} != {ratio}"

    def test_charm_strange_ratio_consistent(self, observables):
        """m_c/m_s from ratio should match absolute masses."""
        m_c = observables['m_c_MeV']
        m_s = observables['m_s_MeV']
        ratio = observables['m_c_m_s']

        computed_ratio = m_c / m_s
        assert np.isclose(computed_ratio, ratio, rtol=0.1), \
            f"m_c/m_s = {computed_ratio} != {ratio}"

    def test_top_bottom_ratio_direction(self, observables):
        """m_t/m_b should match hierarchy m_t > m_b."""
        ratio = observables['m_t_m_b']

        # m_t > m_b means ratio > 1
        assert ratio > 1, "m_t/m_b should be > 1"


# ============================================================================
# Coupling Constant Running Tests
# ============================================================================

class TestCouplingRunning:
    """Tests related to coupling constant scale dependence."""

    def test_alpha_s_at_mz(self, observables):
        """alpha_s(M_Z) should be ~0.118."""
        alpha_s = observables['alpha_s_MZ']

        # PDG 2024: alpha_s(M_Z) = 0.1180 +/- 0.0009
        assert 0.10 < alpha_s < 0.13, f"alpha_s(M_Z) = {alpha_s} unreasonable"

    def test_sin2_theta_w_at_mz(self, observables):
        """sin^2(theta_W)(M_Z) should be ~0.231."""
        sin2 = observables['sin2thetaW']

        # PDG: 0.23122 +/- 0.00003
        assert 0.20 < sin2 < 0.25, f"sin^2(theta_W) = {sin2} unreasonable"


# ============================================================================
# Topological Integer Tests
# ============================================================================

class TestTopologicalIntegers:
    """Tests that topological integers are exact integers."""

    def test_dim_e8_integer(self, params):
        """dim(E8) = 248 should be exact integer."""
        assert params.dim_E8 == 248
        assert isinstance(params.dim_E8, int)

    def test_dim_g2_integer(self, params):
        """dim(G2) = 14 should be exact integer."""
        assert params.dim_G2 == 14
        assert isinstance(params.dim_G2, int)

    def test_dim_k7_integer(self, params):
        """dim(K7) = 7 should be exact integer."""
        assert params.dim_K7 == 7
        assert isinstance(params.dim_K7, int)

    def test_betti_numbers_integer(self, params):
        """Betti numbers should be exact integers."""
        assert params.b2_K7 == 21
        assert params.b3_K7 == 77
        assert isinstance(params.b2_K7, int)
        assert isinstance(params.b3_K7, int)

    def test_h_star_integer(self, params):
        """H* = 99 should be exact integer."""
        assert params.H_star == 99
        assert isinstance(params.H_star, int)

    def test_n_gen_integer(self, params):
        """N_gen = 3 should be exact integer."""
        assert params.N_gen == 3
        assert isinstance(params.N_gen, int)


# ============================================================================
# Exact Rational Tests
# ============================================================================

class TestExactRationals:
    """Tests for quantities that should be exact rationals."""

    def test_tau_exact_rational(self, params):
        """tau = 3472/891 should be exact."""
        from fractions import Fraction

        tau_frac = params.tau
        expected = Fraction(3472, 891)

        assert tau_frac == expected

    def test_det_g_exact_rational(self, params):
        """det(g) = 65/32 should be exact."""
        from fractions import Fraction

        det_g = params.det_g
        expected = Fraction(65, 32)

        assert det_g == expected

    def test_kappa_t_exact_rational(self, params):
        """kappa_T = 1/61 should be exact."""
        from fractions import Fraction

        kappa_T = params.kappa_T
        expected = Fraction(1, 61)

        assert kappa_T == expected

    def test_sin2_theta_w_exact_rational(self, params):
        """sin^2(theta_W) = 3/13 should be exact."""
        from fractions import Fraction

        sin2 = params.sin2_theta_W
        expected = Fraction(3, 13)

        assert sin2 == expected


# ============================================================================
# Derived Quantity Dimension Tests
# ============================================================================

class TestDerivedQuantityDimensions:
    """Tests for dimensions of derived quantities."""

    def test_lambda_h_dimensionless(self, observables):
        """Higgs quartic coupling should be dimensionless."""
        lambda_H = observables['lambda_H']

        # Should be O(0.1)
        assert 0 < lambda_H < 1, f"lambda_H = {lambda_H} should be O(0.1)"

    def test_kappa_t_dimensionless(self, observables):
        """Torsion magnitude should be dimensionless."""
        kappa_T = observables['kappa_T']

        # Should be small (~0.016)
        assert 0 < kappa_T < 1, f"kappa_T = {kappa_T} should be small"

    def test_tau_dimensionless(self, observables):
        """Hierarchy parameter should be dimensionless."""
        tau = observables['tau']

        # Should be ~3.9
        assert 1 < tau < 10, f"tau = {tau} should be O(1-10)"


# ============================================================================
# Unit Conversion Consistency Tests
# ============================================================================

class TestUnitConversions:
    """Tests for consistency under unit conversions."""

    def test_gev_mev_consistency(self, observables):
        """1 GeV = 1000 MeV consistency."""
        m_t_gev = observables['m_t_GeV']
        m_t_mev = m_t_gev * 1000

        # Should be ~173000 MeV
        assert 150000 < m_t_mev < 200000

    def test_w_z_mass_relation(self, observables):
        """M_W and M_Z should satisfy cos(theta_W) relation."""
        M_W = observables['M_W']
        M_Z = observables['M_Z']
        sin2_theta_W = observables['sin2thetaW']

        # M_W = M_Z * cos(theta_W)
        # M_W/M_Z = sqrt(1 - sin^2(theta_W))
        expected_ratio = np.sqrt(1 - sin2_theta_W)
        actual_ratio = M_W / M_Z

        # Should be close (tree-level relation)
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.02), \
            f"M_W/M_Z = {actual_ratio} != {expected_ratio}"


# ============================================================================
# Sign Convention Tests
# ============================================================================

class TestSignConventions:
    """Tests for sign conventions."""

    def test_all_masses_positive(self, observables):
        """All masses should be positive."""
        mass_keys = ['m_u_MeV', 'm_d_MeV', 'm_s_MeV', 'm_c_MeV',
                     'm_b_MeV', 'm_t_GeV', 'v_EW', 'M_W', 'M_Z']

        for key in mass_keys:
            assert observables[key] > 0, f"{key} should be positive"

    def test_angles_non_negative(self, observables):
        """All angles should be non-negative."""
        angle_keys = ['theta12', 'theta13', 'theta23', 'delta_CP']

        for key in angle_keys:
            assert observables[key] >= 0, f"{key} should be non-negative"

    def test_couplings_positive(self, observables):
        """Coupling constants should be positive."""
        coupling_keys = ['alpha_inv', 'sin2thetaW', 'alpha_s_MZ', 'lambda_H']

        for key in coupling_keys:
            assert observables[key] > 0, f"{key} should be positive"
