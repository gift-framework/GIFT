"""
Integration tests for full GIFT framework pipeline.

Tests end-to-end workflows including:
- Observable calculation
- Statistical validation
- Experimental comparison
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation import (
    GIFTFrameworkStatistical,
    monte_carlo_uncertainty_propagation,
    bootstrap_experimental_validation
)


@pytest.mark.integration
class TestFullObservablePipeline:
    """Test complete observable calculation pipeline."""

    def test_end_to_end_observable_calculation(self):
        """Test full pipeline from parameters to experimental comparison."""
        # 1. Initialize framework
        gift = GIFTFrameworkStatistical()

        # 2. Compute all observables
        observables = gift.compute_all_observables()

        # 3. Check all observables are computed
        expected_observables = [
            'alpha_inv_MZ', 'sin2thetaW', 'alpha_s_MZ',
            'theta12', 'theta13', 'theta23', 'delta_CP',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e', 'm_s_m_d',
            'lambda_H', 'Omega_DE', 'n_s', 'H0'
        ]

        for obs in expected_observables:
            assert obs in observables
            assert not np.isnan(observables[obs])
            assert not np.isinf(observables[obs])

        # 4. Compare with experimental data
        exp_data = gift.experimental_data

        for obs_name in expected_observables:
            if obs_name in exp_data:
                pred = observables[obs_name]
                exp_val, exp_unc = exp_data[obs_name]

                # Should be within reasonable range
                deviation = abs(pred - exp_val) / exp_val
                assert deviation < 0.05, f"{obs_name}: {deviation*100:.2f}% deviation"

    def test_parameter_variation_consistency(self):
        """Test that parameter variations produce consistent results."""
        params = [
            (2.0, 5, 10416/2673),
            (2.001, 5, 10416/2673),
            (2.0, 5.01, 10416/2673)
        ]

        results = []
        for p2, weyl, tau in params:
            gift = GIFTFrameworkStatistical(p2=p2, Weyl_factor=weyl, tau=tau)
            obs = gift.compute_all_observables()
            results.append(obs)

        # Results should be similar but not identical
        for key in results[0].keys():
            vals = [r[key] for r in results]

            # Topologically exact observables that don't depend on p2/Weyl_factor
            # These are derived from mathematical/topological constants only
            topological_constants = [
                'delta_CP',      # 7*14 + 99 = 197
                'Q_Koide',       # 14/21 = 2/3
                'm_tau_m_e',     # 7 + 10*248 + 10*99 = 3477
                'lambda_H',      # sqrt(17)/32
                'alpha_inv_MZ',  # 2^7 - 1/24
                'Omega_DE',      # ln(2) * 98/99
                'alpha_s_MZ',    # sqrt(2)/12
                'sin2thetaW',    # zeta(2) - sqrt(2)
                'theta13',       # pi/21
                'm_mu_m_e',      # 27^phi (golden ratio)
                'n_s'            # zeta(11)/zeta(5)
            ]

            if key not in topological_constants:
                # These should vary with parameters
                std = np.std(vals)
                assert std > 0, f"{key} should vary with parameters"


@pytest.mark.integration
@pytest.mark.slow
class TestStatisticalValidationPipeline:
    """Test statistical validation pipeline."""

    def test_small_monte_carlo_pipeline(self):
        """Test Monte Carlo pipeline with small sample."""
        n_samples = 1000  # Small for testing

        distributions, statistics = monte_carlo_uncertainty_propagation(
            n_samples=n_samples,
            seed=42
        )

        # Check outputs
        assert len(distributions) > 0
        assert len(statistics) > 0

        # Check statistics structure
        for obs_name, stats in statistics.items():
            assert 'mean' in stats
            assert 'std' in stats
            assert 'median' in stats
            assert stats['std'] >= 0

    def test_small_bootstrap_pipeline(self):
        """Test bootstrap pipeline with small sample."""
        n_bootstrap = 100  # Small for testing

        bootstrap_stats = bootstrap_experimental_validation(
            n_bootstrap=n_bootstrap,
            seed=42
        )

        # Check outputs
        assert len(bootstrap_stats) > 0

        for obs_name, stats in bootstrap_stats.items():
            assert 'mean' in stats
            assert 'median' in stats
            assert stats['mean'] >= 0  # Deviation percentage


@pytest.mark.integration
class TestExperimentalComparisonPipeline:
    """Test experimental comparison workflow."""

    def test_precision_calculation(self):
        """Test overall precision calculation."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        exp_data = gift.experimental_data

        deviations = []
        for obs_name, (exp_val, exp_unc) in exp_data.items():
            pred_val = obs[obs_name]
            dev = abs(pred_val - exp_val) / exp_val
            deviations.append(dev * 100)  # Convert to percentage

        mean_deviation = np.mean(deviations)

        # Should be under 0.5% as claimed (0.13% nominal)
        assert mean_deviation < 0.5, f"Mean deviation {mean_deviation:.3f}% too high"

    def test_all_observables_predicted(self):
        """Test that all experimental observables have predictions."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        exp_data = gift.experimental_data

        for obs_name in exp_data.keys():
            assert obs_name in obs, f"Missing prediction for {obs_name}"

    def test_no_outliers(self):
        """Test that no predictions are wildly wrong."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        exp_data = gift.experimental_data

        for obs_name, (exp_val, exp_unc) in exp_data.items():
            pred_val = obs[obs_name]
            dev = abs(pred_val - exp_val) / exp_val

            # No prediction should be off by more than 10%
            assert dev < 0.10, f"{obs_name} deviation {dev*100:.2f}% is too large"


@pytest.mark.integration
class TestMultiVersionCompatibility:
    """Test compatibility across framework versions."""

    def test_default_parameters_consistent(self):
        """Test that default parameters produce consistent results."""
        gift1 = GIFTFrameworkStatistical()
        gift2 = GIFTFrameworkStatistical()

        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        for key in obs1.keys():
            assert obs1[key] == obs2[key], f"{key} not reproducible"

    def test_seeded_calculations_reproducible(self):
        """Test that seeded calculations are reproducible."""
        np.random.seed(42)
        gift1 = GIFTFrameworkStatistical()
        obs1 = gift1.compute_all_observables()

        np.random.seed(42)
        gift2 = GIFTFrameworkStatistical()
        obs2 = gift2.compute_all_observables()

        for key in obs1.keys():
            assert obs1[key] == obs2[key]
