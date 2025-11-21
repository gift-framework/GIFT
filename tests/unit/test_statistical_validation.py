"""
Unit tests for statistical validation functions.

Tests Monte Carlo, bootstrap, and Sobol analysis components in isolation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add statistical_validation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation import (
    GIFTFrameworkStatistical,
    PARAM_UNCERTAINTIES,
    monte_carlo_uncertainty_propagation,
    bootstrap_experimental_validation,
)


class TestGIFTFrameworkStatistical:
    """Test the GIFTFrameworkStatistical class."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        gift = GIFTFrameworkStatistical()
        assert gift.p2 == 2.0
        assert gift.Weyl_factor == 5
        assert np.isclose(gift.tau, 10416 / 2673)

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        gift = GIFTFrameworkStatistical(p2=2.5, Weyl_factor=6, tau=4.0)
        assert gift.p2 == 2.5
        assert gift.Weyl_factor == 6
        assert gift.tau == 4.0

    def test_derived_parameters(self):
        """Test derived parameters are computed correctly."""
        gift = GIFTFrameworkStatistical()
        assert np.isclose(gift.beta0, np.pi / 8)
        assert np.isclose(gift.xi, (5 / 2) * (np.pi / 8))
        assert np.isclose(gift.delta, 2 * np.pi / 25)

    def test_compute_all_observables_returns_dict(self):
        """Test compute_all_observables returns a dictionary."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        assert isinstance(obs, dict)
        assert len(obs) == 15  # 15 observables

    def test_observable_names(self):
        """Test all expected observables are present."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        expected_names = [
            "alpha_inv_MZ", "sin2thetaW", "alpha_s_MZ",
            "theta12", "theta13", "theta23", "delta_CP",
            "Q_Koide", "m_mu_m_e", "m_tau_m_e", "m_s_m_d",
            "lambda_H", "Omega_DE", "n_s", "H0"
        ]
        for name in expected_names:
            assert name in obs, f"Missing observable: {name}"


class TestParameterUncertainties:
    """Test parameter uncertainty definitions."""

    def test_uncertainties_defined(self):
        """Test all parameter uncertainties are defined."""
        assert "p2" in PARAM_UNCERTAINTIES
        assert "Weyl_factor" in PARAM_UNCERTAINTIES
        assert "tau" in PARAM_UNCERTAINTIES

    def test_uncertainty_structure(self):
        """Test uncertainty dictionaries have correct structure."""
        for param, unc in PARAM_UNCERTAINTIES.items():
            assert "central" in unc, f"{param} missing 'central'"
            assert "uncertainty" in unc, f"{param} missing 'uncertainty'"
            assert unc["uncertainty"] > 0, f"{param} uncertainty must be positive"

    def test_central_values(self):
        """Test central values match defaults."""
        assert PARAM_UNCERTAINTIES["p2"]["central"] == 2.0
        assert PARAM_UNCERTAINTIES["Weyl_factor"]["central"] == 5
        assert np.isclose(PARAM_UNCERTAINTIES["tau"]["central"], 10416 / 2673)


class TestMonteCarloFunctions:
    """Test Monte Carlo uncertainty propagation (with small samples)."""

    @pytest.mark.slow
    def test_monte_carlo_returns_distributions(self):
        """Test MC returns distributions and statistics."""
        distributions, stats = monte_carlo_uncertainty_propagation(
            n_samples=1000, seed=42
        )
        assert isinstance(distributions, dict)
        assert isinstance(stats, dict)
        assert len(distributions) == 15
        assert len(stats) == 15

    @pytest.mark.slow
    def test_monte_carlo_distribution_shape(self):
        """Test MC distributions have correct shape."""
        n_samples = 1000
        distributions, _ = monte_carlo_uncertainty_propagation(
            n_samples=n_samples, seed=42
        )
        for name, dist in distributions.items():
            assert len(dist) == n_samples, f"{name} has wrong length"

    @pytest.mark.slow
    def test_monte_carlo_statistics_structure(self):
        """Test MC statistics have all required fields."""
        _, stats = monte_carlo_uncertainty_propagation(n_samples=1000, seed=42)
        required_fields = ["mean", "std", "median", "q16", "q84", "q025", "q975", "min", "max"]
        for obs_name, obs_stats in stats.items():
            for field in required_fields:
                assert field in obs_stats, f"{obs_name} missing {field}"

    @pytest.mark.slow
    def test_monte_carlo_reproducibility(self):
        """Test MC with same seed gives same results."""
        _, stats1 = monte_carlo_uncertainty_propagation(n_samples=100, seed=42)
        _, stats2 = monte_carlo_uncertainty_propagation(n_samples=100, seed=42)
        for obs_name in stats1:
            assert stats1[obs_name]["mean"] == stats2[obs_name]["mean"]

    @pytest.mark.slow
    def test_monte_carlo_different_seeds(self):
        """Test MC with different seeds gives different results."""
        # Use 10000 samples to ensure batches run (batch_size=10000 in implementation)
        _, stats1 = monte_carlo_uncertainty_propagation(n_samples=10000, seed=42)
        _, stats2 = monte_carlo_uncertainty_propagation(n_samples=10000, seed=123)
        # At least one observable should differ
        differences = sum(
            1 for obs in stats1
            if stats1[obs]["mean"] != stats2[obs]["mean"]
        )
        assert differences > 0


class TestBootstrapFunctions:
    """Test bootstrap validation functions."""

    @pytest.mark.slow
    def test_bootstrap_returns_stats(self):
        """Test bootstrap returns statistics dict."""
        stats = bootstrap_experimental_validation(n_bootstrap=100, seed=42)
        assert isinstance(stats, dict)

    @pytest.mark.slow
    def test_bootstrap_statistics_structure(self):
        """Test bootstrap statistics have required fields."""
        stats = bootstrap_experimental_validation(n_bootstrap=100, seed=42)
        required_fields = ["mean", "median", "std", "q025", "q975"]
        for obs_name, obs_stats in stats.items():
            for field in required_fields:
                assert field in obs_stats, f"{obs_name} missing {field}"

    @pytest.mark.slow
    def test_bootstrap_reproducibility(self):
        """Test bootstrap with same seed is reproducible."""
        stats1 = bootstrap_experimental_validation(n_bootstrap=100, seed=42)
        stats2 = bootstrap_experimental_validation(n_bootstrap=100, seed=42)
        for obs_name in stats1:
            assert stats1[obs_name]["mean"] == stats2[obs_name]["mean"]

    @pytest.mark.slow
    def test_bootstrap_deviations_positive(self):
        """Test all bootstrap deviations are positive percentages."""
        stats = bootstrap_experimental_validation(n_bootstrap=100, seed=42)
        for obs_name, obs_stats in stats.items():
            assert obs_stats["mean"] >= 0, f"{obs_name} has negative deviation"


class TestParameterSensitivity:
    """Test sensitivity of observables to parameter variations."""

    def test_p2_sensitivity(self):
        """Test which observables depend on p2."""
        gift1 = GIFTFrameworkStatistical(p2=2.0)
        gift2 = GIFTFrameworkStatistical(p2=2.1)
        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        # m_s_m_d should change (depends on p2^2)
        assert obs1["m_s_m_d"] != obs2["m_s_m_d"]

        # Topological observables should NOT change
        assert obs1["delta_CP"] == obs2["delta_CP"]
        assert obs1["Q_Koide"] == obs2["Q_Koide"]
        assert obs1["m_tau_m_e"] == obs2["m_tau_m_e"]

    def test_Weyl_factor_sensitivity(self):
        """Test which observables depend on Weyl_factor."""
        gift1 = GIFTFrameworkStatistical(Weyl_factor=5)
        gift2 = GIFTFrameworkStatistical(Weyl_factor=5.1)
        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        # m_s_m_d should change
        assert obs1["m_s_m_d"] != obs2["m_s_m_d"]

        # Pure topological should NOT change
        assert obs1["delta_CP"] == obs2["delta_CP"]
        assert obs1["Q_Koide"] == obs2["Q_Koide"]

    def test_topological_invariance(self):
        """Test topological observables are parameter-independent."""
        params_sets = [
            {"p2": 2.0, "Weyl_factor": 5, "tau": 3.9},
            {"p2": 2.5, "Weyl_factor": 6, "tau": 4.0},
            {"p2": 1.5, "Weyl_factor": 4, "tau": 3.5},
        ]

        topological_obs = ["delta_CP", "Q_Koide", "lambda_H", "Omega_DE", "alpha_inv_MZ"]
        first_values = None

        for params in params_sets:
            gift = GIFTFrameworkStatistical(**params)
            obs = gift.compute_all_observables()

            if first_values is None:
                first_values = {name: obs[name] for name in topological_obs}
            else:
                for name in topological_obs:
                    assert obs[name] == first_values[name], (
                        f"{name} changed with parameters: {obs[name]} vs {first_values[name]}"
                    )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_p2(self):
        """Test with small p2 value."""
        gift = GIFTFrameworkStatistical(p2=0.1)
        obs = gift.compute_all_observables()
        # Should still compute without NaN/Inf
        for name, value in obs.items():
            assert np.isfinite(value), f"{name} not finite with small p2"

    def test_large_p2(self):
        """Test with large p2 value."""
        gift = GIFTFrameworkStatistical(p2=100.0)
        obs = gift.compute_all_observables()
        for name, value in obs.items():
            assert np.isfinite(value), f"{name} not finite with large p2"

    def test_small_Weyl_factor(self):
        """Test with small Weyl factor."""
        gift = GIFTFrameworkStatistical(Weyl_factor=0.1)
        obs = gift.compute_all_observables()
        for name, value in obs.items():
            assert np.isfinite(value), f"{name} not finite with small Weyl"

    def test_negative_parameters_handled(self):
        """Test behavior with negative parameters."""
        # The framework should handle these gracefully
        gift = GIFTFrameworkStatistical(p2=-2.0)
        obs = gift.compute_all_observables()
        # m_s_m_d = p2^2 * Weyl, so should still be positive
        assert obs["m_s_m_d"] > 0
