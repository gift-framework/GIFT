"""
Regression tests for observable values.

Ensures that observable calculations remain consistent across code changes.
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from run_validation import GIFTFrameworkStatistical


@pytest.fixture
def reference_data():
    """Load reference observable data."""
    reference_file = Path(__file__).parent.parent / "fixtures" / "reference_observables.json"
    with open(reference_file, 'r') as f:
        return json.load(f)


@pytest.mark.regression
class TestObservableRegression:
    """Test that observables match reference values."""

    def test_all_observables_match_reference(self, reference_data):
        """Test that all observables match reference within tolerance."""
        # Create framework with same parameters as reference
        params = reference_data['parameters']
        gift = GIFTFrameworkStatistical(
            p2=params['p2'],
            Weyl_factor=params['Weyl_factor'],
            tau=params['tau']
        )

        obs = gift.compute_all_observables()
        ref_obs = reference_data['observables']

        failures = []
        for key, ref_value in ref_obs.items():
            calc_value = obs[key]
            relative_diff = abs(calc_value - ref_value) / abs(ref_value) if ref_value != 0 else abs(calc_value)

            # Reasonable tolerance for regression (2%)
            # Allows for minor numerical variations across platforms/versions
            tolerance = 0.02  # 2%
            if relative_diff > tolerance:
                failures.append(f"{key}: calculated={calc_value}, reference={ref_value}, diff={relative_diff*100:.2f}%")

        assert len(failures) == 0, f"Regression failures (>{tolerance*100}%):\n" + "\n".join(failures)

    def test_exact_observables_unchanged(self, reference_data):
        """Test that PROVEN exact observables haven't changed."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()

        exact_observables = {
            'delta_CP': 197.0,           # 7*14 + 99
            'Q_Koide': 2/3,              # 14/21
            'm_tau_m_e': 3477.0,         # 7 + 10*248 + 10*99
            'm_s_m_d': 20.0,             # 4 * 5
            'lambda_H': np.sqrt(17)/32,  # sqrt(17)/32
        }

        for key, expected in exact_observables.items():
            assert abs(obs[key] - expected) < 1e-10, f"{key} exact value changed!"

    def test_topological_invariants_stable(self):
        """Test that topological invariants are stable across parameter variations."""
        # These should not change with parameters
        invariant_observables = ['delta_CP', 'Q_Koide', 'm_tau_m_e', 'lambda_H']

        # Test with different parameters
        gift1 = GIFTFrameworkStatistical(p2=2.0, Weyl_factor=5)
        gift2 = GIFTFrameworkStatistical(p2=2.5, Weyl_factor=6)

        obs1 = gift1.compute_all_observables()
        obs2 = gift2.compute_all_observables()

        for key in invariant_observables:
            assert obs1[key] == obs2[key], f"Topological invariant {key} changed with parameters!"

    def test_precision_not_degraded(self, reference_data):
        """Test that precision hasn't degraded from reference."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()

        ref_deviations = reference_data['deviations_percent']
        exp_values = reference_data['experimental_values']

        for key, ref_dev in ref_deviations.items():
            calc_value = obs[key]
            exp_value = exp_values[key]

            current_dev = abs(calc_value - exp_value) / exp_value * 100

            # Current deviation should not be significantly worse than reference
            # Allow up to 2x degradation OR absolute deviation < 1%
            # (accounts for numerical variations and platform differences)
            acceptable = max(ref_dev * 2.0, 1.0)  # 2x reference or 1% max
            assert current_dev <= acceptable, \
                f"{key}: precision degraded from {ref_dev}% to {current_dev}% (max: {acceptable}%)"

    def test_mean_precision_maintained(self, reference_data):
        """Test that mean precision is maintained."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()
        exp_data = gift.experimental_data

        current_deviations = []
        for key, (exp_val, _) in exp_data.items():
            pred_val = obs[key]
            dev = abs(pred_val - exp_val) / exp_val * 100
            current_deviations.append(dev)

        current_mean = np.mean(current_deviations)

        # Reference mean precision should be around 0.13%
        # Allow up to 0.2%
        assert current_mean < 0.2, f"Mean precision degraded to {current_mean}%"


@pytest.mark.regression
class TestNumericalStability:
    """Test numerical stability over multiple runs."""

    def test_reproducibility_over_runs(self):
        """Test that multiple runs produce identical results."""
        results = []

        for _ in range(10):
            gift = GIFTFrameworkStatistical()
            obs = gift.compute_all_observables()
            results.append(obs)

        # All runs should be identical
        for i in range(1, len(results)):
            for key in results[0].keys():
                assert results[0][key] == results[i][key], \
                    f"{key} not reproducible: run 0 = {results[0][key]}, run {i} = {results[i][key]}"

    def test_numerical_precision_maintained(self):
        """Test that numerical precision is maintained."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()

        # Check that values are well-defined and representable
        for key, value in obs.items():
            # No NaN or Inf
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"

            # Values should be in reasonable range for float64
            assert abs(value) < 1e100, f"{key} exceeds reasonable float range"
            assert abs(value) > 1e-100 or value == 0, f"{key} underflows float range"


@pytest.mark.regression
class TestBackwardCompatibility:
    """Test backward compatibility with previous versions."""

    def test_default_parameters_unchanged(self):
        """Test that default parameters haven't changed."""
        gift = GIFTFrameworkStatistical()

        assert gift.p2 == 2.0
        assert gift.Weyl_factor == 5
        assert abs(gift.tau - 10416/2673) < 1e-10
        assert gift.b2_K7 == 21
        assert gift.b3_K7 == 77
        assert gift.H_star == 99

    def test_experimental_data_present(self):
        """Test that all expected experimental data is present."""
        gift = GIFTFrameworkStatistical()

        expected_keys = [
            'alpha_inv_MZ', 'sin2thetaW', 'alpha_s_MZ',
            'theta12', 'theta13', 'theta23', 'delta_CP',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e', 'm_s_m_d',
            'lambda_H', 'Omega_DE', 'n_s', 'H0'
        ]

        for key in expected_keys:
            assert key in gift.experimental_data, f"Missing experimental data for {key}"

    def test_observable_names_unchanged(self):
        """Test that observable names haven't changed."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()

        expected_observables = [
            'alpha_inv_MZ', 'sin2thetaW', 'alpha_s_MZ',
            'theta12', 'theta13', 'theta23', 'delta_CP',
            'Q_Koide', 'm_mu_m_e', 'm_tau_m_e', 'm_s_m_d',
            'lambda_H', 'Omega_DE', 'n_s', 'H0'
        ]

        for key in expected_observables:
            assert key in obs, f"Observable {key} missing from output"
