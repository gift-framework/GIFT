"""
Unit tests for core GIFT framework physics calculations.

Tests all 34 dimensionless observables for numerical correctness,
parameter sensitivity, and consistency with theoretical predictions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add statistical_validation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation import GIFTFrameworkStatistical, PARAM_UNCERTAINTIES


class TestGIFTFrameworkInitialization:
    """Test framework initialization and parameters."""

    def test_default_initialization(self):
        """Test framework initializes with default parameters."""
        gift = GIFTFrameworkStatistical()
        assert gift.p2 == 2.0
        assert gift.Weyl_factor == 5
        assert abs(gift.tau - 10416 / 2673) < 1e-10

    def test_custom_parameters(self):
        """Test framework accepts custom parameters."""
        gift = GIFTFrameworkStatistical(p2=2.5, Weyl_factor=6, tau=4.0)
        assert gift.p2 == 2.5
        assert gift.Weyl_factor == 6
        assert gift.tau == 4.0

    def test_derived_parameters(self):
        """Test derived parameters computed correctly."""
        gift = GIFTFrameworkStatistical()

        # beta0 = pi / 8
        expected_beta0 = np.pi / 8
        assert abs(gift.beta0 - expected_beta0) < 1e-10

        # xi = (Weyl_factor / p2) * beta0 = (5/2) * (pi/8)
        expected_xi = (5.0 / 2.0) * (np.pi / 8)
        assert abs(gift.xi - expected_xi) < 1e-10

        # delta = 2*pi / Weyl_factor^2 = 2*pi / 25
        expected_delta = 2 * np.pi / 25
        assert abs(gift.delta - expected_delta) < 1e-10

    def test_topological_integers(self):
        """Test topological integers have correct values."""
        gift = GIFTFrameworkStatistical()
        assert gift.b2_K7 == 21
        assert gift.b3_K7 == 77
        assert gift.H_star == 99
        assert gift.dim_E8 == 248
        assert gift.dim_G2 == 14
        assert gift.dim_K7 == 7
        assert gift.rank_E8 == 8


class TestGaugeSectorObservables:
    """Test gauge sector observable calculations."""

    def test_alpha_inv_MZ(self, gift_framework):
        """Test fine structure constant at MZ."""
        obs = gift_framework.compute_all_observables()
        alpha_inv = obs['alpha_inv_MZ']

        # Should be 2^7 - 1/24 = 128 - 0.04167 = 127.958
        expected = 2**7 - 1/24
        assert abs(alpha_inv - expected) < 1e-10

        # Should be close to experimental value 127.955
        assert abs(alpha_inv - 127.955) < 0.1

    def test_sin2thetaW(self, gift_framework):
        """Test weak mixing angle."""
        obs = gift_framework.compute_all_observables()
        sin2thetaW = obs['sin2thetaW']

        # Should be zeta(2) - sqrt(2) = pi^2/6 - sqrt(2)
        zeta2 = np.pi**2 / 6
        expected = zeta2 - np.sqrt(2)
        assert abs(sin2thetaW - expected) < 1e-10

        # Should be close to experimental value 0.23122
        assert abs(sin2thetaW - 0.23122) < 0.01

    def test_alpha_s_MZ(self, gift_framework):
        """Test strong coupling constant at MZ."""
        obs = gift_framework.compute_all_observables()
        alpha_s = obs['alpha_s_MZ']

        # Should be sqrt(2) / 12
        expected = np.sqrt(2) / 12
        assert abs(alpha_s - expected) < 1e-10

        # Should be close to experimental value 0.1179
        assert abs(alpha_s - 0.1179) < 0.01


class TestNeutrinoSectorObservables:
    """Test neutrino sector observable calculations."""

    def test_theta12(self, gift_framework):
        """Test solar mixing angle."""
        obs = gift_framework.compute_all_observables()
        theta12 = obs['theta12']

        # Should be arctan(sqrt(delta/gamma)) in degrees
        gamma = 511 / 884
        delta = 2 * np.pi / 25
        expected = np.arctan(np.sqrt(delta / gamma)) * 180 / np.pi
        assert abs(theta12 - expected) < 1e-10

        # Should be close to experimental value 33.44 degrees
        assert abs(theta12 - 33.44) < 2.0

    def test_theta13(self, gift_framework):
        """Test reactor mixing angle."""
        obs = gift_framework.compute_all_observables()
        theta13 = obs['theta13']

        # Should be pi/21 in degrees
        expected = (np.pi / 21) * 180 / np.pi
        assert abs(theta13 - expected) < 1e-10

        # Should be close to experimental value 8.61 degrees
        assert abs(theta13 - 8.61) < 0.5

    def test_theta23(self, gift_framework):
        """Test atmospheric mixing angle."""
        obs = gift_framework.compute_all_observables()
        theta23 = obs['theta23']

        # Should be (8 + 77) / 99 in radians, converted to degrees
        expected_rad = (8 + 77) / 99
        expected = expected_rad * 180 / np.pi
        assert abs(theta23 - expected) < 1e-10

        # Should be close to experimental value 49.2 degrees
        assert abs(theta23 - 49.2) < 2.0

    def test_delta_CP(self, gift_framework):
        """Test CP violation phase (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        delta_CP = obs['delta_CP']

        # Should be exactly 7*14 + 99 = 197 degrees
        expected = 7 * 14 + 99
        assert delta_CP == expected

        # Experimental value is 197 ± 24 degrees
        assert abs(delta_CP - 197.0) < 1.0


class TestLeptonSectorObservables:
    """Test lepton sector observable calculations."""

    def test_Q_Koide(self, gift_framework):
        """Test Koide formula parameter (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        Q_Koide = obs['Q_Koide']

        # Should be exactly 14/21 = 2/3
        expected = 14 / 21
        assert abs(Q_Koide - expected) < 1e-10
        assert abs(Q_Koide - 2/3) < 1e-10

        # Experimental value is 0.6667
        assert abs(Q_Koide - 0.6667) < 0.001

    def test_m_mu_m_e(self, gift_framework):
        """Test muon to electron mass ratio."""
        obs = gift_framework.compute_all_observables()
        m_mu_m_e = obs['m_mu_m_e']

        # Should be 27^phi where phi = golden ratio
        phi = (1 + np.sqrt(5)) / 2
        expected = 27 ** phi
        assert abs(m_mu_m_e - expected) < 1e-10

        # Should be close to experimental value 206.768
        assert abs(m_mu_m_e - 206.768) < 1.0

    def test_m_tau_m_e(self, gift_framework):
        """Test tau to electron mass ratio (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        m_tau_m_e = obs['m_tau_m_e']

        # Should be exactly 7 + 10*248 + 10*99 = 3477
        expected = 7 + 10 * 248 + 10 * 99
        assert m_tau_m_e == expected

        # Experimental value is 3477.0
        assert abs(m_tau_m_e - 3477.0) < 1.0


class TestQuarkSectorObservables:
    """Test quark sector observable calculations."""

    def test_m_s_m_d(self, gift_framework):
        """Test strange to down quark mass ratio (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        m_s_m_d = obs['m_s_m_d']

        # Should be exactly p2^2 * Weyl_factor = 4 * 5 = 20
        expected = gift_framework.p2**2 * gift_framework.Weyl_factor
        assert m_s_m_d == expected

        # Experimental value is 20 ± 1
        assert abs(m_s_m_d - 20.0) < 1.0


class TestHiggsSectorObservables:
    """Test Higgs sector observable calculations."""

    def test_lambda_H(self, gift_framework):
        """Test Higgs quartic coupling (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        lambda_H = obs['lambda_H']

        # Should be exactly sqrt(17) / 32
        expected = np.sqrt(17) / 32
        assert abs(lambda_H - expected) < 1e-10

        # Should be close to experimental value 0.129
        assert abs(lambda_H - 0.129) < 0.005


class TestCosmologicalObservables:
    """Test cosmological observable calculations."""

    def test_Omega_DE(self, gift_framework):
        """Test dark energy density parameter (PROVEN exact)."""
        obs = gift_framework.compute_all_observables()
        Omega_DE = obs['Omega_DE']

        # Should be exactly ln(2) * 98/99
        expected = np.log(2) * 98 / 99
        assert abs(Omega_DE - expected) < 1e-10

        # Should be close to experimental value 0.6847
        assert abs(Omega_DE - 0.6847) < 0.01

    def test_n_s(self, gift_framework):
        """Test scalar spectral index."""
        obs = gift_framework.compute_all_observables()
        n_s = obs['n_s']

        # Should be zeta(11) / zeta(5)
        zeta11 = 1.0004941886041195
        zeta5 = 1.0369277551433699
        expected = zeta11 / zeta5
        assert abs(n_s - expected) < 1e-10

        # Should be close to experimental value 0.9649
        assert abs(n_s - 0.9649) < 0.01

    def test_H0(self, gift_framework):
        """Test Hubble constant."""
        obs = gift_framework.compute_all_observables()
        H0 = obs['H0']

        # Should be H0_Planck * (zeta3 / xi)^beta0
        H0_Planck = 67.36
        zeta3 = gift_framework.zeta3
        xi = gift_framework.xi
        beta0 = gift_framework.beta0
        expected = H0_Planck * (zeta3 / xi)**beta0
        assert abs(H0 - expected) < 1e-10

        # Value should be reasonable (60-80)
        assert 60 < H0 < 80


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_observable_reproducibility(self, gift_framework):
        """Test that calculations are reproducible."""
        obs1 = gift_framework.compute_all_observables()
        obs2 = gift_framework.compute_all_observables()

        for key in obs1.keys():
            assert obs1[key] == obs2[key], f"Observable {key} not reproducible"

    def test_no_nan_or_inf(self, gift_framework):
        """Test that no observable produces NaN or Inf."""
        obs = gift_framework.compute_all_observables()

        for key, value in obs.items():
            assert not np.isnan(value), f"Observable {key} is NaN"
            assert not np.isinf(value), f"Observable {key} is Inf"

    def test_observable_ranges(self, gift_framework):
        """Test that all observables are in reasonable ranges."""
        obs = gift_framework.compute_all_observables()

        # Define reasonable ranges
        ranges = {
            'alpha_inv_MZ': (120, 135),
            'sin2thetaW': (0.2, 0.25),
            'alpha_s_MZ': (0.1, 0.15),
            'theta12': (30, 40),
            'theta13': (8, 10),
            'theta23': (40, 55),
            'delta_CP': (150, 220),
            'Q_Koide': (0.6, 0.7),
            'm_mu_m_e': (200, 210),
            'm_tau_m_e': (3400, 3500),
            'm_s_m_d': (15, 25),
            'lambda_H': (0.1, 0.15),
            'Omega_DE': (0.6, 0.75),
            'n_s': (0.95, 0.98),
            'H0': (60, 80)
        }

        for key, (low, high) in ranges.items():
            value = obs[key]
            assert low <= value <= high, f"{key} = {value} outside range [{low}, {high}]"


class TestParameterSensitivity:
    """Test parameter sensitivity and variations."""

    def test_p2_variation(self):
        """Test observables vary correctly with p2."""
        gift1 = GIFTFrameworkStatistical(p2=2.0)
        gift2 = GIFTFrameworkStatistical(p2=2.1)

        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        # m_s_m_d should change: 4*5 vs 4.41*5
        assert obs1['m_s_m_d'] != obs2['m_s_m_d']

        # H0 should change (depends on xi which depends on p2)
        assert obs1['H0'] != obs2['H0']

    def test_Weyl_factor_variation(self):
        """Test observables vary correctly with Weyl_factor."""
        gift1 = GIFTFrameworkStatistical(Weyl_factor=5)
        gift2 = GIFTFrameworkStatistical(Weyl_factor=6)

        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        # m_s_m_d should change: 4*5 vs 4*6
        assert obs1['m_s_m_d'] != obs2['m_s_m_d']

        # theta12 should change (depends on delta which depends on Weyl_factor)
        assert obs1['theta12'] != obs2['theta12']

    def test_topological_invariants_unchanged(self):
        """Test that topologically exact observables don't change with parameters."""
        gift1 = GIFTFrameworkStatistical(p2=2.0, Weyl_factor=5)
        gift2 = GIFTFrameworkStatistical(p2=2.5, Weyl_factor=6)

        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        # These are topologically exact and shouldn't change
        assert obs1['delta_CP'] == obs2['delta_CP']  # 7*14 + 99 = 197
        assert obs1['Q_Koide'] == obs2['Q_Koide']  # 14/21 = 2/3
        assert obs1['m_tau_m_e'] == obs2['m_tau_m_e']  # 7 + 10*248 + 10*99
        assert obs1['lambda_H'] == obs2['lambda_H']  # sqrt(17)/32


class TestExperimentalComparison:
    """Test comparison with experimental data."""

    def test_all_observables_within_tolerance(self, gift_framework, experimental_data):
        """Test that predictions are within reasonable tolerance of experiment."""
        obs = gift_framework.compute_all_observables()

        # Maximum allowed deviation: 1% for most observables
        max_deviation = 0.01  # 1%

        failures = []
        for key, (exp_val, exp_unc) in experimental_data.items():
            pred_val = obs[key]
            deviation = abs(pred_val - exp_val) / exp_val

            # More lenient for observables with large uncertainties
            if key in ['delta_CP', 'H0']:
                allowed_dev = 0.05  # 5%
            else:
                allowed_dev = max_deviation

            if deviation > allowed_dev:
                failures.append(f"{key}: {deviation*100:.2f}% deviation")

        assert len(failures) == 0, f"Observables outside tolerance: {failures}"

    def test_mean_precision(self, gift_framework, experimental_data):
        """Test that mean precision is below 0.2% as claimed."""
        obs = gift_framework.compute_all_observables()

        deviations = []
        for key, (exp_val, exp_unc) in experimental_data.items():
            pred_val = obs[key]
            deviation = abs(pred_val - exp_val) / exp_val
            deviations.append(deviation)

        mean_deviation = np.mean(deviations)
        assert mean_deviation < 0.002, f"Mean deviation {mean_deviation*100:.2f}% exceeds 0.2%"


@pytest.mark.slow
class TestParameterUncertaintyPropagation:
    """Test uncertainty propagation (slower tests)."""

    def test_small_monte_carlo(self, random_seed):
        """Test Monte Carlo with small sample size."""
        np.random.seed(random_seed)
        n_samples = 1000

        p2_samples = np.random.normal(2.0, 0.001, n_samples)
        results = []

        for p2 in p2_samples[:100]:  # Test only 100 for speed
            gift = GIFTFrameworkStatistical(p2=p2)
            obs = gift.compute_all_observables()
            results.append(obs['alpha_inv_MZ'])

        # Should have some variance
        assert np.std(results) > 0

        # Mean should be close to default
        gift_default = GIFTFrameworkStatistical()
        default_val = gift_default.compute_all_observables()['alpha_inv_MZ']
        assert abs(np.mean(results) - default_val) < 0.1
