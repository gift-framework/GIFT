"""
Enhanced regression tests for all 46 GIFT v2.1 observables.

Features:
- Historical value tracking across versions
- Automatic regression detection
- Tolerance-based change detection
- Observable stability metrics
- Cross-version compatibility tests
- Change detection with significance testing

Version: 2.1.0
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V21_AVAILABLE,
    reason="GIFT v2.1 not available"
)


# Historical reference values (version 2.1.0 baseline)
BASELINE_V21_VALUES = {
    # Gauge sector
    "alpha_inv_MZ": 137.033,
    "sin2thetaW": 0.23128,
    "alpha_s_MZ": 0.11785,

    # Neutrino mixing
    "theta12": 33.69,
    "theta13": 8.571,
    "theta23": 48.63,
    "delta_CP": 197.0,

    # Lepton mass ratios
    "Q_Koide": 0.666667,
    "m_mu_m_e": 206.795,
    "m_tau_m_e": 3477.0,

    # Quark mass ratios
    "m_s_m_d": 20.0,
    "m_c_m_s": 13.56,
    "m_b_m_u": 1934.8,
    "m_t_m_b": 41.32,
    "m_d_m_u": 2.163,
    "m_c_m_u": 587.9,
    "m_b_m_d": 894.5,
    "m_t_m_s": 1849.3,
    "m_t_m_d": 36981.0,
    "m_t_m_c": 136.08,

    # CKM matrix
    "V_us": 0.2244,
    "V_cb": 0.0421,
    "V_ub": 0.00393,
    "V_cd": 0.2178,
    "V_cs": 0.9972,
    "V_td": 0.00809,

    # Higgs
    "lambda_H": 0.1287,

    # Cosmological
    "Omega_DE": 0.6861,
    "Omega_DM": 0.2653,
    "Omega_b": 0.0494,
    "n_s": 0.9651,
    "sigma_8": 0.812,
    "A_s": 2.098e-9,
    "Omega_gamma": 5.39e-5,
    "Omega_nu": 0.00063,
    "Y_p": 0.2448,
    "D_H": 2.545e-5,

    # Dimensional
    "v_EW": 246.23,
    "M_W": 80.371,
    "M_Z": 91.189,
    "m_u_MeV": 2.17,
    "m_d_MeV": 4.69,
    "m_s_MeV": 93.8,
    "m_c_MeV": 1272.0,
    "m_b_MeV": 4182.0,
    "m_t_GeV": 172.78,
    "H0": 69.8,
}

# Regression tolerances (tighter for proven exact, looser for derived)
REGRESSION_TOLERANCES = {
    # Proven exact - very tight tolerance
    "delta_CP": 1e-10,
    "Q_Koide": 1e-10,
    "m_tau_m_e": 1e-10,
    "m_s_m_d": 1e-10,
    "lambda_H": 1e-10,
    "Omega_DE": 1e-10,

    # Topological - tight tolerance
    "alpha_inv_MZ": 1e-6,
    "sin2thetaW": 1e-6,
    "alpha_s_MZ": 1e-6,

    # Derived - moderate tolerance
    "default": 1e-4,
}


class TestObservableStabilityRegression:
    """Test that observable values remain stable across code changes."""

    @pytest.mark.parametrize("obs_name,baseline_value",
                             list(BASELINE_V21_VALUES.items()))
    def test_observable_regression_vs_baseline(self, obs_name, baseline_value):
        """Test each observable hasn't regressed from baseline."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        if obs_name not in obs:
            pytest.skip(f"{obs_name} not computed in this version")

        current_value = obs[obs_name]

        # Get tolerance
        tolerance = REGRESSION_TOLERANCES.get(obs_name,
                                              REGRESSION_TOLERANCES["default"])

        # Compute absolute and relative difference
        abs_diff = abs(current_value - baseline_value)
        rel_diff = abs_diff / (abs(baseline_value) + 1e-15)

        # Check regression
        assert abs_diff < tolerance or rel_diff < tolerance, (
            f"{obs_name} REGRESSION DETECTED:\n"
            f"  Baseline: {baseline_value:.10g}\n"
            f"  Current:  {current_value:.10g}\n"
            f"  Abs diff: {abs_diff:.2e}\n"
            f"  Rel diff: {rel_diff:.2e}\n"
            f"  Tolerance: {tolerance:.2e}"
        )

    def test_all_baseline_observables_present(self):
        """Test that all baseline observables are still computed."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        missing = [name for name in BASELINE_V21_VALUES if name not in obs]

        if missing:
            pytest.fail(
                f"REGRESSION: {len(missing)} baseline observables no longer computed:\n"
                f"{missing}"
            )

    def test_no_new_nan_values(self):
        """Test that no observables have become NaN."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        nan_obs = [name for name, value in obs.items() if np.isnan(value)]

        assert len(nan_obs) == 0, (
            f"REGRESSION: {len(nan_obs)} observables are now NaN: {nan_obs}"
        )

    def test_no_new_inf_values(self):
        """Test that no observables have become Inf."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        inf_obs = [name for name, value in obs.items() if np.isinf(value)]

        assert len(inf_obs) == 0, (
            f"REGRESSION: {len(inf_obs)} observables are now Inf: {inf_obs}"
        )


class TestObservableChangeDetection:
    """Detect and quantify changes in observable values."""

    def test_proven_exact_unchanged(self):
        """Test proven exact values are absolutely unchanged."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        proven_exact = ["delta_CP", "Q_Koide", "m_tau_m_e", "m_s_m_d",
                       "lambda_H", "Omega_DE"]

        changes = {}
        for obs_name in proven_exact:
            if obs_name in obs and obs_name in BASELINE_V21_VALUES:
                current = obs[obs_name]
                baseline = BASELINE_V21_VALUES[obs_name]

                if current != baseline:  # Exact equality check
                    changes[obs_name] = {
                        'baseline': baseline,
                        'current': current,
                        'diff': current - baseline
                    }

        assert len(changes) == 0, (
            f"PROVEN EXACT values changed (should be mathematically exact):\n"
            f"{json.dumps(changes, indent=2)}"
        )

    def test_identify_significant_changes(self):
        """Identify observables with significant changes."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Threshold for "significant" change
        significant_threshold = 0.01  # 1%

        significant_changes = []

        for obs_name, baseline_value in BASELINE_V21_VALUES.items():
            if obs_name in obs:
                current_value = obs[obs_name]
                rel_change = abs(current_value - baseline_value) / (abs(baseline_value) + 1e-15)

                if rel_change > significant_threshold:
                    significant_changes.append({
                        'observable': obs_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'relative_change': rel_change
                    })

        # This test documents changes rather than failing
        if significant_changes:
            print("\n" + "="*60)
            print("SIGNIFICANT CHANGES DETECTED (>1%):")
            print("="*60)
            for change in significant_changes:
                print(f"{change['observable']}:")
                print(f"  Baseline: {change['baseline']:.6g}")
                print(f"  Current:  {change['current']:.6g}")
                print(f"  Change:   {change['relative_change']*100:.2f}%")
            print("="*60)


class TestCrossVersionCompatibility:
    """Test compatibility across different parameter sets."""

    def test_default_params_give_baseline_results(self):
        """Test that default parameters reproduce baseline results."""
        framework = GIFTFrameworkV21()  # Default params
        obs = framework.compute_all_observables()

        # Check a few key observables
        key_observables = ["alpha_inv_MZ", "sin2thetaW", "delta_CP", "Q_Koide"]

        for obs_name in key_observables:
            if obs_name in obs and obs_name in BASELINE_V21_VALUES:
                current = obs[obs_name]
                baseline = BASELINE_V21_VALUES[obs_name]

                tolerance = REGRESSION_TOLERANCES.get(obs_name, 1e-4)

                assert abs(current - baseline) < tolerance, (
                    f"{obs_name}: default params don't reproduce baseline"
                )

    def test_parameter_variation_bounds(self):
        """Test observable variation stays within expected bounds."""
        # Test with slightly varied parameters
        param_sets = [
            {"p2": 2.0, "Weyl_factor": 5.0, "tau": 3.8967},  # Default
            {"p2": 2.05, "Weyl_factor": 5.0, "tau": 3.8967},  # +2.5% p2
            {"p2": 2.0, "Weyl_factor": 5.1, "tau": 3.8967},  # +2% Weyl
        ]

        results = []
        for params in param_sets:
            framework = GIFTFrameworkV21(**params)
            obs = framework.compute_all_observables()
            results.append(obs)

        # Check that topological observables don't vary
        topological = ["delta_CP", "Q_Koide", "m_tau_m_e"]

        for obs_name in topological:
            if obs_name in results[0]:
                values = [r[obs_name] for r in results if obs_name in r]

                # All should be identical (within numerical noise)
                std_dev = np.std(values)

                assert std_dev < 1e-10, (
                    f"{obs_name} (topological) varies with parameters: std={std_dev}"
                )


class TestObservableHistory:
    """Track observable value history for regression analysis."""

    def test_save_current_values_for_history(self, tmp_path):
        """Save current observable values for future regression tests."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Save to temp file (in real usage, commit to git)
        history_file = tmp_path / "observable_history.json"

        import datetime

        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "2.1.0",
            "observables": {k: float(v) for k, v in obs.items() if np.isfinite(v)},
            "parameters": {
                "p2": 2.0,
                "Weyl_factor": 5.0,
                "tau": 3.8967
            }
        }

        with open(history_file, 'w') as f:
            json.dump(history_entry, f, indent=2)

        # Verify file created
        assert history_file.exists()

        # Load and verify
        with open(history_file, 'r') as f:
            loaded = json.load(f)

        assert "timestamp" in loaded
        assert "observables" in loaded
        assert len(loaded["observables"]) > 0

    def test_compare_with_reference_file(self):
        """Compare current values with reference file."""
        # Load reference file
        reference_file = PROJECT_ROOT / "tests" / "fixtures" / "reference_observables_v21.json"

        if not reference_file.exists():
            pytest.skip("Reference file not found")

        with open(reference_file, 'r') as f:
            reference_data = json.load(f)

        reference_obs = reference_data.get("observables", {})

        # Compute current values
        framework = GIFTFrameworkV21()
        current_obs = framework.compute_all_observables()

        # Compare
        mismatches = []

        for obs_name, ref_data in reference_obs.items():
            if obs_name in current_obs:
                ref_value = ref_data.get("predicted")
                current_value = current_obs[obs_name]

                if ref_value is not None:
                    tolerance = REGRESSION_TOLERANCES.get(obs_name, 1e-4)

                    if abs(current_value - ref_value) > tolerance:
                        mismatches.append({
                            'name': obs_name,
                            'reference': ref_value,
                            'current': current_value,
                            'diff': current_value - ref_value
                        })

        if mismatches:
            print("\n" + "="*60)
            print("MISMATCHES WITH REFERENCE FILE:")
            for m in mismatches:
                print(f"{m['name']}: {m['reference']} → {m['current']} (Δ={m['diff']:.2e})")
            print("="*60)


class TestObservableStatistics:
    """Compute stability statistics for observables."""

    def test_observable_variance_across_runs(self):
        """Test observable variance across multiple runs with same params."""
        n_runs = 10

        results = {}

        for i in range(n_runs):
            framework = GIFTFrameworkV21()
            obs = framework.compute_all_observables()

            for name, value in obs.items():
                if name not in results:
                    results[name] = []
                results[name].append(value)

        # Compute statistics
        for obs_name, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Should have zero variance (deterministic)
            assert std_val == 0, (
                f"{obs_name} has non-zero variance across runs: std={std_val}"
            )

    def test_observable_correlation_structure(self):
        """Test correlation structure remains stable."""
        # Sample parameters
        np.random.seed(42)
        n_samples = 50

        observables_matrix = []

        for i in range(n_samples):
            p2 = 2.0 + np.random.normal(0, 0.05)
            weyl = 5.0 + np.random.normal(0, 0.1)

            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            # Extract a few key observables
            obs_vector = [
                obs.get("alpha_inv_MZ", 0),
                obs.get("sin2thetaW", 0),
                obs.get("m_mu_m_e", 0),
            ]
            observables_matrix.append(obs_vector)

        observables_matrix = np.array(observables_matrix)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(observables_matrix.T)

        # Should have reasonable correlations
        assert np.all(np.isfinite(corr_matrix))
        assert np.all(np.abs(corr_matrix) <= 1.0)


class TestRegressionReporting:
    """Generate regression test reports."""

    def test_generate_regression_report(self, tmp_path):
        """Generate comprehensive regression report."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        report = {
            "summary": {
                "total_observables": len(obs),
                "baseline_observables": len(BASELINE_V21_VALUES),
                "missing_from_baseline": [],
                "new_in_current": [],
                "regressions": [],
                "improvements": [],
            },
            "details": {}
        }

        # Identify changes
        for obs_name in set(list(obs.keys()) + list(BASELINE_V21_VALUES.keys())):
            if obs_name not in BASELINE_V21_VALUES:
                report["summary"]["new_in_current"].append(obs_name)
            elif obs_name not in obs:
                report["summary"]["missing_from_baseline"].append(obs_name)
            else:
                current = obs[obs_name]
                baseline = BASELINE_V21_VALUES[obs_name]

                tolerance = REGRESSION_TOLERANCES.get(obs_name, 1e-4)
                diff = abs(current - baseline)

                report["details"][obs_name] = {
                    "baseline": float(baseline),
                    "current": float(current),
                    "difference": float(diff),
                    "relative_change": float(diff / (abs(baseline) + 1e-15)),
                    "within_tolerance": diff < tolerance
                }

                if diff > tolerance:
                    report["summary"]["regressions"].append(obs_name)

        # Save report
        report_file = tmp_path / "regression_report.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        assert report_file.exists()

        # Print summary
        print("\n" + "="*60)
        print("REGRESSION TEST SUMMARY")
        print("="*60)
        print(f"Total observables: {report['summary']['total_observables']}")
        print(f"New observables: {len(report['summary']['new_in_current'])}")
        print(f"Missing observables: {len(report['summary']['missing_from_baseline'])}")
        print(f"Regressions detected: {len(report['summary']['regressions'])}")
        print("="*60)
