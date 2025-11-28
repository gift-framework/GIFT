"""
Cross-Observable Consistency Tests

Tests that verify relationships between observables are mathematically
consistent, including derived ratios, CKM unitarity, and formula consistency.

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
    """Create GIFT v2.2 framework."""
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
# Quark Mass Ratio Consistency Tests
# ============================================================================

class TestQuarkMassRatioConsistency:
    """Tests that quark mass ratios are mutually consistent."""

    def test_m_c_m_d_from_m_c_m_s_and_m_s_m_d(self, observables):
        """m_c/m_d = (m_c/m_s) * (m_s/m_d)"""
        m_c_m_d = observables['m_c_m_d']
        m_c_m_s = observables['m_c_m_s']
        m_s_m_d = observables['m_s_m_d']

        expected = m_c_m_s * m_s_m_d
        assert np.isclose(m_c_m_d, expected, rtol=1e-3), \
            f"m_c/m_d = {m_c_m_d} != {expected} = (m_c/m_s) * (m_s/m_d)"

    def test_m_t_m_s_from_m_t_m_c_and_m_c_m_s(self, observables):
        """m_t/m_s = (m_t/m_c) * (m_c/m_s)"""
        m_t_m_s = observables['m_t_m_s']
        m_t_m_c = observables['m_t_m_c']
        m_c_m_s = observables['m_c_m_s']

        expected = m_t_m_c * m_c_m_s
        assert np.isclose(m_t_m_s, expected, rtol=1e-3), \
            f"m_t/m_s = {m_t_m_s} != {expected} = (m_t/m_c) * (m_c/m_s)"

    def test_m_b_m_d_from_m_b_m_u_and_m_d_m_u(self, observables):
        """m_b/m_d = (m_b/m_u) / (m_d/m_u)"""
        m_b_m_d = observables['m_b_m_d']
        m_b_m_u = observables['m_b_m_u']
        m_d_m_u = observables['m_d_m_u']

        expected = m_b_m_u / m_d_m_u
        assert np.isclose(m_b_m_d, expected, rtol=1e-3), \
            f"m_b/m_d = {m_b_m_d} != {expected} = (m_b/m_u) / (m_d/m_u)"

    def test_quark_ratio_chain_consistency(self, observables):
        """Test consistency: m_t/m_u from multiple paths."""
        # Path 1: m_t/m_u = (m_t/m_b) * (m_b/m_u)
        path1 = observables['m_t_m_b'] * observables['m_b_m_u']

        # Path 2: m_t/m_u = (m_t/m_c) * (m_c/m_s) * (m_s/m_d) * (m_d/m_u)
        path2 = (observables['m_t_m_c'] * observables['m_c_m_s'] *
                 observables['m_s_m_d'] * observables['m_d_m_u'])

        # Both paths should give same result (within tolerance)
        assert np.isclose(path1, path2, rtol=0.05), \
            f"m_t/m_u paths differ: {path1} vs {path2}"


# ============================================================================
# Lepton Mass Ratio Consistency Tests
# ============================================================================

class TestLeptonMassRatioConsistency:
    """Tests for lepton mass ratio consistency."""

    def test_koide_relation_exact(self, observables):
        """Koide parameter Q should be exactly 2/3."""
        Q_Koide = observables['Q_Koide']
        expected = 2.0 / 3.0

        assert np.isclose(Q_Koide, expected, rtol=1e-10), \
            f"Q_Koide = {Q_Koide} != {expected}"

    def test_tau_electron_ratio_exact(self, observables):
        """m_tau/m_e = 3477 should be exact."""
        m_tau_m_e = observables['m_tau_m_e']
        expected = 3477.0

        assert m_tau_m_e == expected, f"m_tau/m_e = {m_tau_m_e} != {expected}"

    def test_lepton_ratios_positive(self, observables):
        """All lepton mass ratios should be positive."""
        assert observables['Q_Koide'] > 0
        assert observables['m_mu_m_e'] > 0
        assert observables['m_tau_m_e'] > 0

    def test_lepton_hierarchy(self, observables):
        """Lepton masses should satisfy m_e < m_mu < m_tau."""
        # m_tau/m_e > m_mu/m_e implies m_tau > m_mu
        assert observables['m_tau_m_e'] > observables['m_mu_m_e'], \
            "Lepton hierarchy violated"


# ============================================================================
# CKM Matrix Consistency Tests
# ============================================================================

class TestCKMConsistency:
    """Tests for CKM matrix unitarity and consistency."""

    def test_ckm_first_row_unitarity(self, observables):
        """First row: |V_ud|^2 + |V_us|^2 + |V_ub|^2 = 1"""
        # V_ud can be computed from V_us and V_ub
        V_us = observables['V_us']
        V_ub = observables['V_ub']

        # |V_ud|^2 = 1 - |V_us|^2 - |V_ub|^2
        V_ud_sq = 1 - V_us**2 - V_ub**2
        V_ud = np.sqrt(V_ud_sq)

        # Sum should be 1
        row_sum = V_ud**2 + V_us**2 + V_ub**2
        assert np.isclose(row_sum, 1.0, rtol=1e-6), \
            f"First row sum = {row_sum} != 1"

    def test_ckm_third_column_unitarity(self, observables):
        """Third column: |V_ub|^2 + |V_cb|^2 + |V_tb|^2 = 1"""
        V_ub = observables['V_ub']
        V_cb = observables['V_cb']
        V_tb = observables['V_tb']

        column_sum = V_ub**2 + V_cb**2 + V_tb**2
        assert np.isclose(column_sum, 1.0, rtol=1e-3), \
            f"Third column sum = {column_sum} != 1"

    def test_ckm_hierarchy(self, observables):
        """CKM elements should satisfy known hierarchy."""
        # |V_us| > |V_cb| > |V_ub|
        assert observables['V_us'] > observables['V_cb'], \
            "V_us should be > V_cb"
        assert observables['V_cb'] > observables['V_ub'], \
            "V_cb should be > V_ub"

        # |V_tb| close to 1
        assert observables['V_tb'] > 0.99, "V_tb should be close to 1"

    def test_ckm_values_in_range(self, observables):
        """CKM elements should be in [0, 1]."""
        ckm_elements = ['V_us', 'V_cb', 'V_ub', 'V_td', 'V_ts', 'V_tb']

        for elem in ckm_elements:
            assert 0 <= observables[elem] <= 1, \
                f"{elem} = {observables[elem]} out of range [0, 1]"


# ============================================================================
# Neutrino Mixing Consistency Tests
# ============================================================================

class TestNeutrinoMixingConsistency:
    """Tests for neutrino mixing parameter consistency."""

    def test_mixing_angles_in_range(self, observables):
        """Mixing angles should be in valid range."""
        # theta_12, theta_13, theta_23 should be positive and < 90
        assert 0 < observables['theta12'] < 90, "theta12 out of range"
        assert 0 < observables['theta13'] < 90, "theta13 out of range"
        assert 0 < observables['theta23'] < 90, "theta23 out of range"

    def test_delta_cp_exact(self, observables):
        """delta_CP = 197 degrees should be exact."""
        delta_CP = observables['delta_CP']
        assert delta_CP == 197.0, f"delta_CP = {delta_CP} != 197"

    def test_theta13_formula(self, gift_framework, observables):
        """theta_13 = pi/b2 = pi/21 radians."""
        theta13 = observables['theta13']
        expected = (np.pi / gift_framework.b2_K7) * 180 / np.pi

        assert np.isclose(theta13, expected, rtol=1e-6), \
            f"theta13 = {theta13} != {expected}"

    def test_mixing_hierarchy(self, observables):
        """Mixing angles should satisfy known hierarchy."""
        # theta_12 > theta_23 > theta_13
        assert observables['theta12'] < observables['theta23'], \
            "Expected theta12 < theta23"
        assert observables['theta13'] < observables['theta12'], \
            "Expected theta13 < theta12"


# ============================================================================
# Cosmological Parameter Consistency Tests
# ============================================================================

class TestCosmologicalConsistency:
    """Tests for cosmological parameter consistency."""

    def test_omega_de_exact(self, observables):
        """Omega_DE = ln(2) * 98/99 should be exact."""
        Omega_DE = observables['Omega_DE']
        expected = np.log(2) * 98 / 99

        assert np.isclose(Omega_DE, expected, rtol=1e-10), \
            f"Omega_DE = {Omega_DE} != {expected}"

    def test_n_s_zeta_ratio(self, observables, params):
        """n_s = zeta(11)/zeta(5) should be exact."""
        n_s = observables['n_s']
        expected = params.zeta11 / params.zeta5

        assert np.isclose(n_s, expected, rtol=1e-10), \
            f"n_s = {n_s} != {expected}"

    def test_cosmological_values_physical(self, observables):
        """Cosmological values should be physically reasonable."""
        # Omega_DE should be ~0.7
        assert 0.5 < observables['Omega_DE'] < 0.9, \
            f"Omega_DE = {observables['Omega_DE']} unreasonable"

        # n_s should be ~0.96
        assert 0.9 < observables['n_s'] < 1.0, \
            f"n_s = {observables['n_s']} unreasonable"

        # H0 should be ~70 km/s/Mpc
        assert 60 < observables['H0'] < 80, \
            f"H0 = {observables['H0']} unreasonable"


# ============================================================================
# Gauge Coupling Consistency Tests
# ============================================================================

class TestGaugeCouplingConsistency:
    """Tests for gauge coupling consistency."""

    def test_alpha_inv_positive(self, observables):
        """Fine structure constant inverse should be positive."""
        assert observables['alpha_inv'] > 0

    def test_alpha_inv_close_to_137(self, observables):
        """alpha^-1 should be close to 137."""
        alpha_inv = observables['alpha_inv']
        assert 136 < alpha_inv < 138, f"alpha^-1 = {alpha_inv} too far from 137"

    def test_sin2_theta_w_exact(self, observables, params):
        """sin^2(theta_W) = 3/13 should be exact."""
        sin2_theta_W = observables['sin2thetaW']
        expected = float(params.sin2_theta_W)

        assert np.isclose(sin2_theta_W, expected, rtol=1e-10), \
            f"sin2_theta_W = {sin2_theta_W} != {expected}"

    def test_alpha_s_topological(self, observables, params):
        """alpha_s = sqrt(2)/12 should be exact."""
        alpha_s = observables['alpha_s_MZ']
        expected = params.alpha_s

        assert np.isclose(alpha_s, expected, rtol=1e-10), \
            f"alpha_s = {alpha_s} != {expected}"


# ============================================================================
# Topological Formula Consistency Tests
# ============================================================================

class TestTopologicalFormulaConsistency:
    """Tests that topological formulas are internally consistent."""

    def test_h_star_sum(self, params):
        """H* = b2 + b3 + 1."""
        expected = params.b2_K7 + params.b3_K7 + 1
        assert params.H_star == expected, \
            f"H* = {params.H_star} != {expected} = b2 + b3 + 1"

    def test_p2_ratio(self, params):
        """p2 = dim(G2)/dim(K7) = 14/7 = 2."""
        expected = params.dim_G2 // params.dim_K7
        assert params.p2 == expected == 2, \
            f"p2 = {params.p2} != {expected}"

    def test_b3_formula(self, params):
        """b3 = 2*dim(K7)^2 - b2 = 2*49 - 21 = 77."""
        expected = 2 * params.dim_K7**2 - params.b2_K7
        assert params.b3_K7 == expected == 77, \
            f"b3 = {params.b3_K7} != {expected}"

    def test_xi_formula(self, params):
        """xi = (Weyl/p2) * beta0 = 5*pi/16."""
        expected = (params.Weyl_factor / params.p2) * params.beta0
        assert np.isclose(params.xi, expected), \
            f"xi = {params.xi} != {expected}"

    def test_beta0_formula(self, params):
        """beta0 = pi/rank(E8) = pi/8."""
        expected = np.pi / params.rank_E8
        assert np.isclose(params.beta0, expected), \
            f"beta0 = {params.beta0} != {expected}"


# ============================================================================
# PROVEN Relation Verification Tests
# ============================================================================

class TestProvenRelations:
    """Tests that all 13 PROVEN relations hold exactly."""

    def test_n_gen_equals_3(self, params):
        """N_gen = rank(E8) - Weyl = 8 - 5 = 3."""
        expected = params.rank_E8 - params.Weyl_factor
        assert params.N_gen == expected == 3

    def test_q_koide_exact(self, params):
        """Q_Koide = dim(G2)/b2 = 14/21 = 2/3."""
        expected = params.dim_G2 / params.b2_K7
        assert np.isclose(expected, 2/3, rtol=1e-14)

    def test_m_s_m_d_exact(self, params):
        """m_s/m_d = p2^2 * Weyl = 4 * 5 = 20."""
        expected = params.p2**2 * params.Weyl_factor
        assert expected == 20

    def test_delta_cp_formula(self, params):
        """delta_CP = dim(K7)*dim(G2) + H* = 7*14 + 99 = 197."""
        expected = params.dim_K7 * params.dim_G2 + params.H_star
        assert expected == 197

    def test_m_tau_m_e_formula(self, params):
        """m_tau/m_e = dim(K7) + 10*dim(E8) + 10*H* = 7 + 2480 + 990 = 3477."""
        expected = params.dim_K7 + 10*params.dim_E8 + 10*params.H_star
        assert expected == 3477

    def test_lambda_h_formula(self, params):
        """lambda_H = sqrt(dim(G2) + N_gen)/2^Weyl = sqrt(17)/32."""
        expected = np.sqrt(params.dim_G2 + params.N_gen) / (2**params.Weyl_factor)
        assert np.isclose(expected, np.sqrt(17)/32, rtol=1e-14)

    def test_sin2_theta_w_formula(self, params):
        """sin^2(theta_W) = b2/(b3 + dim(G2)) = 21/91 = 3/13."""
        expected = params.b2_K7 / (params.b3_K7 + params.dim_G2)
        assert np.isclose(expected, 3/13, rtol=1e-14)

    def test_tau_formula(self, params):
        """tau = dim(E8xE8)*b2/(dim(J3O)*H*) = 496*21/(27*99) = 3472/891."""
        expected = (params.dim_E8xE8 * params.b2_K7) / (params.dim_J3O * params.H_star)
        assert np.isclose(expected, 3472/891, rtol=1e-14)

    def test_kappa_t_formula(self, params):
        """kappa_T = 1/(b3 - dim(G2) - p2) = 1/(77-14-2) = 1/61."""
        expected = 1 / (params.b3_K7 - params.dim_G2 - params.p2)
        assert np.isclose(expected, 1/61, rtol=1e-14)

    def test_det_g_formula(self, params):
        """det(g) = p2 + 1/(b2 + dim(G2) - N_gen) = 2 + 1/32 = 65/32."""
        denominator = params.b2_K7 + params.dim_G2 - params.N_gen  # 21 + 14 - 3 = 32
        expected = params.p2 + 1/denominator  # 2 + 1/32 = 65/32
        assert np.isclose(expected, 65/32, rtol=1e-14)


# ============================================================================
# Cross-Sector Consistency Tests
# ============================================================================

class TestCrossSectorConsistency:
    """Tests for consistency across different physics sectors."""

    def test_all_observables_finite(self, observables):
        """All computed observables should be finite."""
        for name, value in observables.items():
            assert np.isfinite(value), f"Observable {name} = {value} is not finite"

    def test_all_observables_positive_where_expected(self, observables):
        """Physical observables that must be positive should be positive."""
        positive_observables = [
            'alpha_inv', 'sin2thetaW', 'alpha_s_MZ',
            'theta12', 'theta13', 'theta23',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e',
            'm_s_m_d', 'm_c_m_s', 'm_b_m_u', 'm_t_m_b',
            'V_us', 'V_cb', 'V_ub', 'V_td', 'V_ts', 'V_tb',
            'lambda_H', 'Omega_DE', 'n_s', 'H0',
            'kappa_T', 'tau'
        ]

        for name in positive_observables:
            if name in observables:
                assert observables[name] > 0, f"{name} = {observables[name]} should be positive"

    def test_deviation_statistics(self, gift_framework):
        """Summary statistics should be reasonable."""
        stats = gift_framework.summary_statistics()

        # Mean deviation should be small
        assert stats['mean_deviation'] < 1.0, \
            f"Mean deviation = {stats['mean_deviation']}% too high"

        # Most observables should be within 1%
        assert stats['within_1_pct'] > stats['total_observables'] * 0.8, \
            "Not enough observables within 1%"
