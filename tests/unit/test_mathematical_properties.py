"""
Property-based tests for mathematical identities in GIFT framework.

Tests mathematical invariants that must hold across parameter ranges.
Uses hypothesis for property-based testing when available.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation import GIFTFrameworkStatistical

# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


class TestMathematicalIdentities:
    """Test mathematical identities that must always hold."""

    def test_xi_equals_5_beta0_over_2(self):
        """Identity: xi = 5*beta0/2 when Weyl_factor=5, p2=2."""
        gift = GIFTFrameworkStatistical(p2=2.0, Weyl_factor=5)
        expected = 5 * gift.beta0 / 2
        assert np.isclose(gift.xi, expected, rtol=1e-14)

    def test_xi_general_formula(self):
        """Identity: xi = (Weyl_factor/p2) * beta0 for any parameters."""
        test_cases = [
            (2.0, 5), (2.5, 6), (1.5, 4), (3.0, 7), (1.0, 10)
        ]
        for p2, weyl in test_cases:
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl)
            expected = (weyl / p2) * gift.beta0
            assert np.isclose(gift.xi, expected, rtol=1e-14), (
                f"xi identity failed for p2={p2}, Weyl={weyl}"
            )

    def test_beta0_equals_pi_over_rank(self):
        """Identity: beta0 = pi / rank_E8 = pi/8."""
        gift = GIFTFrameworkStatistical()
        assert np.isclose(gift.beta0, np.pi / 8, rtol=1e-14)
        assert np.isclose(gift.beta0, np.pi / gift.rank_E8, rtol=1e-14)

    def test_H_star_equals_b2_plus_b3_plus_1(self):
        """Identity: H* = b2(K7) + b3(K7) + 1."""
        gift = GIFTFrameworkStatistical()
        assert gift.H_star == gift.b2_K7 + gift.b3_K7 + 1
        assert gift.H_star == 21 + 77 + 1
        assert gift.H_star == 99

    def test_Q_Koide_equals_2_thirds(self):
        """Identity: Q_Koide = dim_G2 / b2_K7 = 14/21 = 2/3."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        assert np.isclose(obs["Q_Koide"], 2/3, rtol=1e-14)
        assert np.isclose(obs["Q_Koide"], gift.dim_G2 / gift.b2_K7, rtol=1e-14)

    def test_delta_CP_formula(self):
        """Identity: delta_CP = 7*dim_G2 + H* = 7*14 + 99 = 197."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = 7 * gift.dim_G2 + gift.H_star
        assert obs["delta_CP"] == expected
        assert obs["delta_CP"] == 197

    def test_m_tau_m_e_formula(self):
        """Identity: m_tau/m_e = dim_K7 + 10*dim_E8 + 10*H*."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = gift.dim_K7 + 10 * gift.dim_E8 + 10 * gift.H_star
        assert obs["m_tau_m_e"] == expected
        assert obs["m_tau_m_e"] == 7 + 2480 + 990
        assert obs["m_tau_m_e"] == 3477

    def test_lambda_H_formula(self):
        """Identity: lambda_H = sqrt(17)/32."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = np.sqrt(17) / 32
        assert np.isclose(obs["lambda_H"], expected, rtol=1e-14)

    def test_Omega_DE_formula(self):
        """Identity: Omega_DE = ln(2) * 98/99."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = np.log(2) * 98 / 99
        assert np.isclose(obs["Omega_DE"], expected, rtol=1e-14)

    def test_alpha_inv_formula(self):
        """Identity: alpha_inv = 2^(rank_E8-1) - 1/24 = 128 - 1/24."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = 2**(gift.rank_E8 - 1) - 1/24
        assert np.isclose(obs["alpha_inv_MZ"], expected, rtol=1e-14)
        assert np.isclose(obs["alpha_inv_MZ"], 127 + 23/24, rtol=1e-14)

    def test_m_s_m_d_formula(self):
        """Identity: m_s/m_d = p2^2 * Weyl_factor."""
        gift = GIFTFrameworkStatistical(p2=2.0, Weyl_factor=5)
        obs = gift.compute_all_observables()
        expected = gift.p2**2 * gift.Weyl_factor
        assert np.isclose(obs["m_s_m_d"], expected, rtol=1e-14)
        assert obs["m_s_m_d"] == 20


class TestTopologicalInvariants:
    """Test topological quantities that must be integers."""

    def test_b2_K7_is_21(self):
        """b2(K7) = 21 (second Betti number)."""
        gift = GIFTFrameworkStatistical()
        assert gift.b2_K7 == 21
        assert isinstance(gift.b2_K7, int)

    def test_b3_K7_is_77(self):
        """b3(K7) = 77 (third Betti number)."""
        gift = GIFTFrameworkStatistical()
        assert gift.b3_K7 == 77
        assert isinstance(gift.b3_K7, int)

    def test_dim_E8_is_248(self):
        """dim(E8) = 248."""
        gift = GIFTFrameworkStatistical()
        assert gift.dim_E8 == 248
        assert isinstance(gift.dim_E8, int)

    def test_dim_G2_is_14(self):
        """dim(G2) = 14."""
        gift = GIFTFrameworkStatistical()
        assert gift.dim_G2 == 14
        assert isinstance(gift.dim_G2, int)

    def test_dim_K7_is_7(self):
        """dim(K7) = 7."""
        gift = GIFTFrameworkStatistical()
        assert gift.dim_K7 == 7
        assert isinstance(gift.dim_K7, int)

    def test_rank_E8_is_8(self):
        """rank(E8) = 8."""
        gift = GIFTFrameworkStatistical()
        assert gift.rank_E8 == 8
        assert isinstance(gift.rank_E8, int)

    def test_dim_J3O_is_27(self):
        """dim(J3O) = 27 (Jordan algebra)."""
        gift = GIFTFrameworkStatistical()
        assert gift.dim_J3O == 27
        assert isinstance(gift.dim_J3O, int)


class TestZetaValues:
    """Test Riemann zeta function values used in the framework."""

    def test_zeta2(self):
        """zeta(2) = pi^2/6."""
        gift = GIFTFrameworkStatistical()
        expected = np.pi**2 / 6
        assert np.isclose(gift.zeta2, expected, rtol=1e-14)

    def test_zeta3(self):
        """zeta(3) = Apery's constant."""
        gift = GIFTFrameworkStatistical()
        # Known value of zeta(3)
        assert np.isclose(gift.zeta3, 1.2020569031595942, rtol=1e-10)

    def test_zeta5(self):
        """zeta(5) value check."""
        gift = GIFTFrameworkStatistical()
        assert np.isclose(gift.zeta5, 1.0369277551433699, rtol=1e-10)

    def test_zeta11(self):
        """zeta(11) value check."""
        gift = GIFTFrameworkStatistical()
        assert np.isclose(gift.zeta11, 1.0004941886041195, rtol=1e-10)


class TestGoldenRatio:
    """Test golden ratio phi used in the framework."""

    def test_phi_definition(self):
        """phi = (1 + sqrt(5))/2."""
        gift = GIFTFrameworkStatistical()
        expected = (1 + np.sqrt(5)) / 2
        assert np.isclose(gift.phi, expected, rtol=1e-14)

    def test_phi_property(self):
        """phi satisfies phi^2 = phi + 1."""
        gift = GIFTFrameworkStatistical()
        assert np.isclose(gift.phi**2, gift.phi + 1, rtol=1e-14)


class TestParameterIndependence:
    """Test which observables are parameter-independent (topological)."""

    @pytest.mark.parametrize("p2,weyl,tau", [
        (2.0, 5, 3.9),
        (2.5, 6, 4.0),
        (1.5, 4, 3.5),
        (3.0, 7, 4.5),
        (1.0, 10, 5.0),
    ])
    def test_topological_observables_invariant(self, p2, weyl, tau):
        """Topological observables don't depend on parameters."""
        gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl, tau=tau)
        obs = gift.compute_all_observables()

        # These must be exactly constant
        assert obs["delta_CP"] == 197
        assert np.isclose(obs["Q_Koide"], 2/3, rtol=1e-14)
        assert obs["m_tau_m_e"] == 3477
        assert np.isclose(obs["lambda_H"], np.sqrt(17)/32, rtol=1e-14)
        assert np.isclose(obs["Omega_DE"], np.log(2)*98/99, rtol=1e-14)
        assert np.isclose(obs["alpha_inv_MZ"], 127 + 23/24, rtol=1e-14)


# Property-based tests with hypothesis (if available)
if HAS_HYPOTHESIS:

    class TestHypothesisProperties:
        """Property-based tests using hypothesis."""

        @given(
            p2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            weyl=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
        )
        @settings(max_examples=100)
        def test_xi_formula_property(self, p2, weyl):
            """xi = (Weyl/p2) * beta0 for all valid parameters."""
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl)
            expected = (weyl / p2) * gift.beta0
            assert np.isclose(gift.xi, expected, rtol=1e-12)

        @given(
            p2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            weyl=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
        )
        @settings(max_examples=100)
        def test_m_s_m_d_property(self, p2, weyl):
            """m_s/m_d = p2^2 * Weyl for all parameters."""
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl)
            obs = gift.compute_all_observables()
            expected = p2**2 * weyl
            assert np.isclose(obs["m_s_m_d"], expected, rtol=1e-12)

        @given(
            p2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            weyl=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
            tau=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def test_topological_invariance_property(self, p2, weyl, tau):
            """Topological observables invariant under parameter changes."""
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl, tau=tau)
            obs = gift.compute_all_observables()

            assert obs["delta_CP"] == 197
            assert np.isclose(obs["Q_Koide"], 2/3, rtol=1e-14)
            assert obs["m_tau_m_e"] == 3477

        @given(
            p2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            weyl=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
        )
        @settings(max_examples=100)
        def test_all_observables_finite(self, p2, weyl):
            """All observables should be finite for valid parameters."""
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl)
            obs = gift.compute_all_observables()
            for name, value in obs.items():
                assert np.isfinite(value), f"{name} not finite for p2={p2}, weyl={weyl}"

        @given(
            p2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            weyl=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
        )
        @settings(max_examples=100)
        def test_all_observables_positive(self, p2, weyl):
            """All observables should be positive for valid parameters."""
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl)
            obs = gift.compute_all_observables()
            for name, value in obs.items():
                assert value > 0, f"{name} not positive for p2={p2}, weyl={weyl}"


class TestDerivedRelations:
    """Test relations derived from fundamental identities."""

    def test_delta_decomposition(self):
        """delta = 2*pi / Weyl^2."""
        gift = GIFTFrameworkStatistical(Weyl_factor=5)
        expected = 2 * np.pi / 25
        assert np.isclose(gift.delta, expected, rtol=1e-14)

    def test_delta_varies_with_weyl(self):
        """delta depends on Weyl_factor."""
        for weyl in [3, 4, 5, 6, 7]:
            gift = GIFTFrameworkStatistical(Weyl_factor=weyl)
            expected = 2 * np.pi / (weyl**2)
            assert np.isclose(gift.delta, expected, rtol=1e-14)

    def test_sin2thetaW_formula(self):
        """sin2(theta_W) = zeta(2) - sqrt(2)."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = gift.zeta2 - np.sqrt(2)
        assert np.isclose(obs["sin2thetaW"], expected, rtol=1e-14)

    def test_alpha_s_formula(self):
        """alpha_s = sqrt(2)/12."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = np.sqrt(2) / 12
        assert np.isclose(obs["alpha_s_MZ"], expected, rtol=1e-14)

    def test_n_s_formula(self):
        """n_s = zeta(11) / zeta(5)."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected = gift.zeta11 / gift.zeta5
        assert np.isclose(obs["n_s"], expected, rtol=1e-14)
