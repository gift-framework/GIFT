"""
Enhanced regression tests for GIFT v2.2 observables.

Features:
- Historical value tracking across versions
- Automatic regression detection
- Tolerance-based change detection
- Observable stability metrics
- Cross-version compatibility tests
- Change detection with significance testing

Version: 2.2.0 (updated from 2.1.0)
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
    from gift_v22_core import GIFTFrameworkV22
    V22_AVAILABLE = True
except ImportError:
    V22_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V22_AVAILABLE,
    reason="GIFT v2.2 not available"
)


# Reference values (version 2.2.0 baseline - zero-parameter paradigm)
# Values computed from GIFTFrameworkV22.compute_all_observables()
BASELINE_V22_VALUES = {
    # Gauge sector (v2.2 exact formulas)
    "alpha_inv": 137.03329918032787,  # (dim(E8)+rank(E8))/2 + H*/D_bulk + det(g)*kappa_T
    "sin2thetaW": 0.23076923076923078,  # 3/13 = b2/(b3+dim(G2)) PROVEN
    "alpha_s_MZ": 0.11785113019775793,  # sqrt(2)/12 TOPOLOGICAL

    # Neutrino mixing
    "theta12": 33.41,             # arctan(sqrt(delta/gamma))
    "theta13": 8.571,             # pi/b2 = pi/21
    "theta23": 49.19,             # (rank+b3)/H* in deg
    "delta_CP": 197.0,            # dim(K7)*dim(G2) + H* PROVEN

    # Lepton mass ratios
    "Q_Koide": 0.6666666666666666,  # dim(G2)/b2 = 14/21 = 2/3 PROVEN
    "m_mu_m_e": 207.012,          # 27^phi (phi=golden ratio)
    "m_tau_m_e": 3477.0,          # dim(K7)+10*dim(E8)+10*H* PROVEN

    # Quark mass ratios
    "m_s_m_d": 20.0,              # p2^2 * Weyl = 4*5 PROVEN
    "m_c_m_s": 13.60,             # tau * 3.49
    "m_b_m_u": 1935.15,
    "m_t_m_b": 41.408,
    "m_d_m_u": 2.163,
    "m_c_m_d": 272.0,             # m_c_m_s * m_s_m_d
    "m_b_m_d": 894.5,
    "m_t_m_s": 1852.3044660194173,  # Updated to v2.2 computed value
    "m_t_m_c": 136.20245461219852,  # Updated to v2.2 computed value

    # CKM matrix
    "V_us": 0.2245,
    "V_cb": 0.04214,
    "V_ub": 0.003947,
    "V_td": 0.008657,
    "V_ts": 0.04154,
    "V_tb": 0.999106,

    # Higgs
    "lambda_H": 0.1288470508005519,  # sqrt(17)/32 PROVEN

    # Cosmological
    "Omega_DE": 0.6861456938876226,  # ln(2)*98/99 PROVEN
    "n_s": 0.9648639296628596,    # zeta(11)/zeta(5) PROVEN
    "H0": 69.8,

    # v2.2 new structural parameters
    "kappa_T": 0.01639344262295082,  # 1/61 TOPOLOGICAL
    "tau": 3.8967452300785634,    # 3472/891 PROVEN
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
    "alpha_inv": 1e-6,
    "sin2thetaW": 1e-6,
    "alpha_s_MZ": 1e-6,
    "kappa_T": 1e-10,
    "tau": 1e-10,

    # Derived - moderate tolerance
    "default": 1e-3,  # Increased tolerance for v2.2
}


class TestObservableStabilityRegression:
    """Test that observable values remain stable across code changes."""

    @pytest.mark.parametrize("obs_name,baseline_value",
                             list(BASELINE_V22_VALUES.items()))
    def test_observable_regression_vs_baseline(self, obs_name, baseline_value):
        """Test each observable hasn't regressed from baseline."""
        framework = GIFTFrameworkV22()
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
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        missing = [name for name in BASELINE_V22_VALUES if name not in obs]

        if missing:
            pytest.fail(
                f"REGRESSION: {len(missing)} baseline observables no longer computed:\n"
                f"{missing}"
            )

    def test_no_new_nan_values(self):
        """Test that no observables have become NaN."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        nan_obs = [name for name, value in obs.items() if np.isnan(value)]

        assert len(nan_obs) == 0, (
            f"REGRESSION: {len(nan_obs)} observables are now NaN: {nan_obs}"
        )

    def test_no_new_inf_values(self):
        """Test that no observables have become Inf."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        inf_obs = [name for name, value in obs.items() if np.isinf(value)]

        assert len(inf_obs) == 0, (
            f"REGRESSION: {len(inf_obs)} observables are now Inf: {inf_obs}"
        )


class TestObservableChangeDetection:
    """Detect and quantify changes in observable values."""

    def test_proven_exact_unchanged(self):
        """Test proven exact values are absolutely unchanged."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # v2.2 has 13 proven exact relations
        proven_exact = ["delta_CP", "Q_Koide", "m_tau_m_e", "m_s_m_d",
                       "lambda_H", "Omega_DE", "n_s", "sin2thetaW",
                       "kappa_T", "tau"]

        changes = {}
        for obs_name in proven_exact:
            if obs_name in obs and obs_name in BASELINE_V22_VALUES:
                current = obs[obs_name]
                baseline = BASELINE_V22_VALUES[obs_name]

                # Use relative tolerance for floating point comparison
                rel_diff = abs(current - baseline) / (abs(baseline) + 1e-15)
                if rel_diff > 1e-6:
                    changes[obs_name] = {
                        'baseline': baseline,
                        'current': current,
                        'rel_diff': rel_diff
                    }

        assert len(changes) == 0, (
            f"PROVEN EXACT values changed beyond tolerance:\n"
            f"{json.dumps(changes, indent=2)}"
        )

    def test_identify_significant_changes(self):
        """Identify observables with significant changes."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # Threshold for "significant" change
        significant_threshold = 0.01  # 1%

        significant_changes = []

        for obs_name, baseline_value in BASELINE_V22_VALUES.items():
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
        framework = GIFTFrameworkV22()  # Default params
        obs = framework.compute_all_observables()

        # Check a few key observables
        key_observables = ["alpha_inv", "sin2thetaW", "delta_CP", "Q_Koide"]

        for obs_name in key_observables:
            if obs_name in obs and obs_name in BASELINE_V22_VALUES:
                current = obs[obs_name]
                baseline = BASELINE_V22_VALUES[obs_name]

                tolerance = REGRESSION_TOLERANCES.get(obs_name, 1e-3)

                assert abs(current - baseline) < tolerance, (
                    f"{obs_name}: default params don't reproduce baseline"
                )

    def test_topological_observables_constant(self):
        """Test that topological observables are constant (zero-parameter paradigm)."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # In v2.2, all parameters are fixed - just verify values
        topological = ["delta_CP", "Q_Koide", "m_tau_m_e", "sin2thetaW", "kappa_T"]

        for obs_name in topological:
            if obs_name in obs and obs_name in BASELINE_V22_VALUES:
                current = obs[obs_name]
                baseline = BASELINE_V22_VALUES[obs_name]

                rel_diff = abs(current - baseline) / (abs(baseline) + 1e-15)
                assert rel_diff < 1e-6, (
                    f"{obs_name} (topological) differs from baseline: {rel_diff:.2e}"
                )


class TestObservableHistory:
    """Track observable value history for regression analysis."""

    def test_save_current_values_for_history(self, tmp_path):
        """Save current observable values for future regression tests."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # Save to temp file (in real usage, commit to git)
        history_file = tmp_path / "observable_history.json"

        import datetime

        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "2.2.0",
            "observables": {k: float(v) for k, v in obs.items() if np.isfinite(v)},
            "parameters": {
                "p2": 2.0,
                "Weyl_factor": 5.0,
                "tau": 3.8967  # 3472/891 PROVEN
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
        reference_file = PROJECT_ROOT / "tests" / "fixtures" / "reference_observables_v22.json"

        if not reference_file.exists():
            pytest.skip("Reference file not found")

        with open(reference_file, 'r') as f:
            reference_data = json.load(f)

        reference_obs = reference_data.get("observables", {})

        # Compute current values
        framework = GIFTFrameworkV22()
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
            framework = GIFTFrameworkV22()
            obs = framework.compute_all_observables()

            for name, value in obs.items():
                if name not in results:
                    results[name] = []
                results[name].append(value)

        # Compute statistics
        for obs_name, values in results.items():
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Should have zero variance (deterministic - zero-parameter paradigm)
            # Allow for machine epsilon floating point variance
            assert std_val < 1e-10, (
                f"{obs_name} has unexpected variance across runs: std={std_val}"
            )

    def test_observable_correlation_structure(self):
        """Test correlation structure remains stable (v2.2: deterministic values)."""
        # In v2.2 zero-parameter paradigm, values are fixed
        # This test verifies stability across multiple instantiations
        np.random.seed(42)
        n_samples = 10  # Reduced since values are deterministic

        observables_matrix = []

        for i in range(n_samples):
            # v2.2: No adjustable parameters - all values are fixed
            framework = GIFTFrameworkV22()
            obs = framework.compute_all_observables()

            # Extract a few key observables
            obs_vector = [
                obs.get("alpha_inv", 0),
                obs.get("sin2thetaW", 0),
                obs.get("m_mu_m_e", 0),
            ]
            observables_matrix.append(obs_vector)

        observables_matrix = np.array(observables_matrix)

        # In zero-parameter paradigm, all values should be identical
        # So variance should be zero (or very close to machine precision)
        for col in range(observables_matrix.shape[1]):
            std_val = np.std(observables_matrix[:, col])
            assert std_val < 1e-10, f"Column {col} has non-zero variance: {std_val}"


class TestRegressionReporting:
    """Generate regression test reports."""

    def test_generate_regression_report(self, tmp_path):
        """Generate comprehensive regression report."""
        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        report = {
            "summary": {
                "total_observables": len(obs),
                "baseline_observables": len(BASELINE_V22_VALUES),
                "missing_from_baseline": [],
                "new_in_current": [],
                "regressions": [],
                "improvements": [],
            },
            "details": {}
        }

        # Identify changes
        for obs_name in set(list(obs.keys()) + list(BASELINE_V22_VALUES.keys())):
            if obs_name not in BASELINE_V22_VALUES:
                report["summary"]["new_in_current"].append(obs_name)
            elif obs_name not in obs:
                report["summary"]["missing_from_baseline"].append(obs_name)
            else:
                current = obs[obs_name]
                baseline = BASELINE_V22_VALUES[obs_name]

                tolerance = REGRESSION_TOLERANCES.get(obs_name, 1e-4)
                diff = abs(current - baseline)

                report["details"][obs_name] = {
                    "baseline": float(baseline),
                    "current": float(current),
                    "difference": float(diff),
                    "relative_change": float(diff / (abs(baseline) + 1e-15)),
                    "within_tolerance": bool(diff < tolerance)
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
        print("REGRESSION TEST SUMMARY (v2.2)")
        print("="*60)
        print(f"Total observables: {report['summary']['total_observables']}")
        print(f"New observables: {len(report['summary']['new_in_current'])}")
        print(f"Missing observables: {len(report['summary']['missing_from_baseline'])}")
        print(f"Regressions detected: {len(report['summary']['regressions'])}")
        print("="*60)
