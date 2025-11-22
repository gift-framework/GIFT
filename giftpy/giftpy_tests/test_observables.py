"""Tests for observable sectors."""
import pytest
import numpy as np
from giftpy import GIFT


class TestGaugeSector:
    """Test gauge sector computations."""

    @pytest.fixture
    def gift(self):
        return GIFT()

    def test_alpha_s(self, gift):
        """Test strong coupling α_s(M_Z) = √2/12."""
        alpha_s = gift.gauge.alpha_s()

        # Check formula
        expected = np.sqrt(2) / 12
        assert abs(alpha_s - expected) < 1e-12

        # Check value range
        assert 0.11 < alpha_s < 0.13

        # Check experimental agreement
        experimental = 0.1179
        deviation = abs(alpha_s - experimental) / experimental * 100
        assert deviation < 1.0  # Within 1%

    def test_sin2theta_W(self, gift):
        """Test weak mixing angle sin²θ_W = 3/13."""
        sin2 = gift.gauge.sin2theta_W()

        # Check exact formula
        assert abs(sin2 - 3 / 13) < 1e-12

        # Check experimental agreement
        experimental = 0.23122
        deviation = abs(sin2 - experimental) / experimental * 100
        assert deviation < 1.0

    def test_alpha_inv(self, gift):
        """Test fine structure constant α⁻¹ = 2⁷ - 1/24."""
        alpha_inv = gift.gauge.alpha_inv()

        # Check exact formula
        expected = 2**7 - 1 / 24
        assert abs(alpha_inv - expected) < 1e-12

        # Check experimental agreement (very precise!)
        experimental = 127.952
        deviation = abs(alpha_inv - experimental) / experimental * 100
        assert deviation < 0.01  # < 0.01%!

    def test_gauge_compute_all(self, gift):
        """Test computing all gauge observables."""
        results = gift.gauge.compute_all()

        assert len(results) == 3
        observables = [r["observable"] for r in results]
        assert "alpha_s" in observables
        assert "sin2theta_W" in observables
        assert "alpha_inv" in observables


class TestLeptonSector:
    """Test lepton sector computations."""

    @pytest.fixture
    def gift(self):
        return GIFT()

    def test_m_mu_m_e(self, gift):
        """Test muon/electron ratio = 27^φ."""
        ratio = gift.lepton.m_mu_m_e()

        # Check formula
        phi = (1 + np.sqrt(5)) / 2
        expected = 27**phi
        assert abs(ratio - expected) < 1e-10

        # Check experimental agreement
        experimental = 206.7682827
        deviation = abs(ratio - experimental) / experimental * 100
        assert deviation < 0.01  # Exceptional precision!

    def test_m_tau_m_mu(self, gift):
        """Test tau/muon ratio = (7+77)/5."""
        ratio = gift.lepton.m_tau_m_mu()

        # Check exact formula
        expected = (7 + 77) / 5
        assert ratio == expected

        # Check experimental agreement
        experimental = 16.8167
        deviation = abs(ratio - experimental) / experimental * 100
        assert deviation < 1.0

    def test_m_tau_m_e_exact(self, gift):
        """Test tau/electron ratio = 3477 (EXACT!)."""
        ratio = gift.lepton.m_tau_m_e()

        # Check EXACT formula
        expected = 77 + 10 * 248 + 10 * 99
        assert ratio == expected
        assert ratio == 3477

        # Check experimental agreement (spectacular!)
        experimental = 3477.23
        deviation = abs(ratio - experimental) / experimental * 100
        assert deviation < 0.01  # Essentially exact!

    def test_Q_Koide_exact(self, gift):
        """Test Koide parameter Q = 2/3 (EXACT!)."""
        Q = gift.lepton.Q_Koide()

        # Check EXACT formula
        assert abs(Q - 2 / 3) < 1e-15

        # Check topological origin: 14/21
        assert abs(Q - 14 / 21) < 1e-15

        # Check experimental agreement (amazing!)
        experimental = 0.666661
        deviation = abs(Q - experimental) / experimental * 100
        assert deviation < 0.001  # < 0.001%!

    def test_koide_formula_verification(self, gift):
        """Test Koide formula with actual masses."""
        result = gift.lepton.verify_koide_formula()

        assert "Q_empirical" in result
        assert "Q_GIFT" in result
        assert "deviation_%" in result

        # Empirical Q should match GIFT prediction very closely
        assert result["deviation_%"] < 0.01


class TestNeutrinoSector:
    """Test neutrino sector."""

    @pytest.fixture
    def gift(self):
        return GIFT()

    def test_theta_12(self, gift):
        """Test solar angle θ₁₂ = π/9."""
        theta = gift.neutrino.theta_12()

        # Check formula
        expected = np.pi / 9
        assert abs(theta - expected) < 1e-12

        # Check in degrees
        theta_deg = gift.neutrino.theta_12(degrees=True)
        assert abs(theta_deg - 20.0) < 1e-10

    def test_delta_CP(self, gift):
        """Test CP phase δ_CP = ζ(3) + √5."""
        delta = gift.neutrino.delta_CP()

        # Check formula
        expected = gift.constants.zeta3 + gift.constants.sqrt5
        assert abs(delta - expected) < 1e-12

        # Check in degrees (should be ~ 197°)
        delta_deg = gift.neutrino.delta_CP(degrees=True)
        assert 196 < delta_deg < 198


class TestQuarkSector:
    """Test quark sector."""

    @pytest.fixture
    def gift(self):
        return GIFT()

    def test_m_s_m_d_exact(self, gift):
        """Test strange/down ratio = 20 (EXACT!)."""
        ratio = gift.quark.m_s_m_d()

        # Check exact formula: 2² × 5 = 20
        assert ratio == 20

        # Check experimental agreement
        experimental = 20.0
        assert abs(ratio - experimental) < 1.0

    def test_V_us(self, gift):
        """Test CKM element V_us = 1/√5."""
        V_us = gift.quark.V_us()

        # Check formula
        expected = 1 / np.sqrt(5)
        assert abs(V_us - expected) < 1e-12


class TestCosmologySector:
    """Test cosmology sector."""

    @pytest.fixture
    def gift(self):
        return GIFT()

    def test_Omega_DE(self, gift):
        """Test dark energy Ω_DE = ln(2)."""
        Omega = gift.cosmology.Omega_DE()

        # Check formula
        expected = np.log(2)
        assert abs(Omega - expected) < 1e-12

    def test_n_s(self, gift):
        """Test spectral index n_s = ξ²."""
        n_s = gift.cosmology.n_s()

        # Check formula
        xi = gift.constants.xi
        expected = xi**2
        assert abs(n_s - expected) < 1e-12


def test_all_deviations_under_1_percent():
    """Test that ALL observables are within 1% of experiment."""
    gift = GIFT()
    results = gift.compute_all()

    # Filter observables with experimental data
    with_exp = results[results["experimental"].notna()]

    # Check all deviations < 1%
    max_deviation = with_exp["deviation_%"].max()
    assert max_deviation < 1.0, f"Max deviation {max_deviation:.3f}% exceeds 1%!"

    # Print summary for visibility
    mean_dev = with_exp["deviation_%"].mean()
    print(f"\nMean deviation across {len(with_exp)} observables: {mean_dev:.4f}%")
    print(f"Max deviation: {max_deviation:.4f}%")
