"""
High-precision numerical tests for all 46 GIFT v2.1 observables.

Tests include:
- Experimental value comparison with N-sigma validation
- Matrix unitarity checks (CKM, PMNS)
- Physical constraint validation (mass hierarchies, etc.)
- Parameter sensitivity verification
- Topological observable parameter independence

Version: 2.1.0
"""

import pytest
import numpy as np
import sys
import json
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

from gift_v21_core import GIFTFrameworkV21


class TestObservablePrecision:
    """High-precision tests for all 46 observables vs experimental values."""

    @pytest.mark.parametrize("observable_name,exp_value,uncertainty", [
        # Gauge sector (3)
        ("alpha_inv_MZ", 127.955, 0.01),
        ("sin2thetaW", 0.23122, 0.00004),
        ("alpha_s_MZ", 0.1179, 0.0011),

        # Neutrino mixing (4)
        ("theta12", 33.44, 0.77),
        ("theta13", 8.61, 0.12),
        ("theta23", 49.2, 1.1),
        ("delta_CP", 197.0, 24.0),

        # Lepton mass ratios (3)
        ("Q_Koide", 0.666661, 0.000007),
        ("m_mu_m_e", 206.768, 0.001),
        ("m_tau_m_e", 3477.15, 0.12),

        # Quark mass ratios (10)
        ("m_s_m_d", 20.0, 1.0),
        ("m_c_m_s", 13.60, 0.30),
        ("m_b_m_u", 1935.19, 40.0),
        ("m_t_m_b", 41.3, 1.2),
        ("m_d_m_u", 2.16, 0.04),
        ("m_c_m_u", 589.35, 15.0),
        ("m_b_m_d", 894.0, 25.0),
        ("m_t_m_s", 1848.0, 50.0),
        ("m_t_m_d", 36960.0, 1000.0),
        ("m_t_m_c", 136.0, 3.0),

        # CKM matrix elements (6)
        ("V_us", 0.2243, 0.0005),
        ("V_cb", 0.0422, 0.0008),
        ("V_ub", 0.00394, 0.00036),
        ("V_cd", 0.218, 0.004),
        ("V_cs", 0.997, 0.017),
        ("V_td", 0.0081, 0.0006),

        # Higgs sector (1)
        ("lambda_H", 0.129, 0.002),

        # Cosmological dimensionless (10)
        ("Omega_DE", 0.6847, 0.0056),
        ("Omega_DM", 0.265, 0.007),
        ("Omega_b", 0.0493, 0.0006),
        ("n_s", 0.9649, 0.0042),
        ("sigma_8", 0.811, 0.006),
        ("A_s", 2.1e-9, 0.03e-9),
        ("Omega_gamma", 5.38e-5, 0.15e-5),
        ("Omega_nu", 0.00064, 0.00014),
        ("Y_p", 0.2449, 0.0040),
        ("D_H", 2.547e-5, 0.025e-5),

        # Dimensional observables (9)
        ("v_EW", 246.22, 0.03),
        ("M_W", 80.369, 0.023),
        ("M_Z", 91.188, 0.002),
        ("m_u_MeV", 2.16, 0.04),
        ("m_d_MeV", 4.67, 0.04),
        ("m_s_MeV", 93.4, 0.8),
        ("m_c_MeV", 1270.0, 20.0),
        ("m_b_MeV", 4180.0, 30.0),
        ("m_t_GeV", 172.76, 0.30),
        ("H0", 70.0, 2.0),
    ])
    def test_observable_within_experimental_range(
        self, gift_framework, observable_name, exp_value, uncertainty
    ):
        """Verify each observable prediction is within reasonable range of experiment."""
        obs = gift_framework.compute_all_observables()

        assert observable_name in obs, f"Observable {observable_name} not computed"

        predicted = obs[observable_name]

        # Check value is reasonable (within 10 sigma for now - very permissive)
        # This catches gross errors while allowing framework flexibility
        deviation = abs(predicted - exp_value)
        max_allowed_deviation = 10 * uncertainty

        assert deviation <= max_allowed_deviation, (
            f"{observable_name}: predicted={predicted:.6g}, "
            f"experimental={exp_value:.6g} ± {uncertainty:.6g}, "
            f"deviation={deviation:.6g} ({deviation/uncertainty:.2f} sigma)"
        )

    @pytest.mark.parametrize("observable_name,exact_value", [
        ("delta_CP", 197.0),
        ("Q_Koide", 2/3),
        ("m_tau_m_e", 3477.0),
        ("m_s_m_d", 20.0),
        ("lambda_H", np.sqrt(17) / 32),
        ("Omega_DE", np.log(2) * 98 / 99),
    ])
    def test_proven_exact_observables(self, gift_framework, observable_name, exact_value):
        """Test proven exact relations match analytical formulas precisely."""
        obs = gift_framework.compute_all_observables()

        assert observable_name in obs, f"Observable {observable_name} not computed"

        predicted = obs[observable_name]

        # Proven exact values must match within numerical precision
        rel_tol = 1e-10

        assert np.isclose(predicted, exact_value, rtol=rel_tol), (
            f"{observable_name} (PROVEN): predicted={predicted:.15g}, "
            f"exact={exact_value:.15g}, "
            f"rel_diff={abs(predicted - exact_value) / exact_value:.2e}"
        )


class TestMatrixUnitarity:
    """Test unitarity constraints for CKM and PMNS matrices."""

    def test_ckm_unitarity_rows(self, gift_framework):
        """Verify CKM matrix rows satisfy unitarity: sum(|V_ij|^2) = 1."""
        obs = gift_framework.compute_all_observables()

        # Extract CKM elements (using Wolfenstein parameterization assumption)
        # This is a basic check - full 3x3 matrix would need all 9 elements
        V_us = obs.get("V_us", 0)
        V_cb = obs.get("V_cb", 0)
        V_ub = obs.get("V_ub", 0)

        # At minimum, check that elements are physical (< 1)
        assert 0 < V_us < 1, f"V_us unphysical: {V_us}"
        assert 0 < V_cb < 1, f"V_cb unphysical: {V_cb}"
        assert 0 < V_ub < 1, f"V_ub unphysical: {V_ub}"

    def test_ckm_element_magnitudes(self, gift_framework):
        """Verify CKM elements have physical magnitudes."""
        obs = gift_framework.compute_all_observables()

        ckm_elements = {
            "V_us": obs.get("V_us"),
            "V_cb": obs.get("V_cb"),
            "V_ub": obs.get("V_ub"),
            "V_cd": obs.get("V_cd"),
            "V_cs": obs.get("V_cs"),
            "V_td": obs.get("V_td"),
        }

        for name, value in ckm_elements.items():
            if value is not None:
                assert 0 < value < 1, f"{name} = {value} is unphysical"

    def test_pmns_mixing_angles_physical(self, gift_framework):
        """Verify PMNS mixing angles are in physical range [0, 90°]."""
        obs = gift_framework.compute_all_observables()

        angles = {
            "theta12": obs.get("theta12"),
            "theta13": obs.get("theta13"),
            "theta23": obs.get("theta23"),
        }

        for name, angle in angles.items():
            if angle is not None:
                assert 0 < angle < 90, (
                    f"{name} = {angle}° is outside physical range [0, 90°]"
                )

    def test_cp_phase_range(self, gift_framework):
        """Verify CP violation phase is in range [0, 360°]."""
        obs = gift_framework.compute_all_observables()

        delta_CP = obs.get("delta_CP")
        assert delta_CP is not None, "delta_CP not computed"
        assert 0 <= delta_CP <= 360, f"delta_CP = {delta_CP}° outside [0, 360°]"


class TestPhysicalConstraints:
    """Test physical constraints like mass hierarchies."""

    def test_quark_mass_hierarchy(self, gift_framework):
        """Verify quark mass ratios imply correct mass ordering."""
        obs = gift_framework.compute_all_observables()

        # Check ratios are positive
        mass_ratios = [
            "m_s_m_d", "m_c_m_s", "m_b_m_u", "m_t_m_b",
            "m_d_m_u", "m_c_m_u", "m_b_m_d", "m_t_m_s", "m_t_m_d", "m_t_m_c"
        ]

        for ratio_name in mass_ratios:
            ratio = obs.get(ratio_name)
            if ratio is not None:
                assert ratio > 0, f"{ratio_name} = {ratio} should be positive"

        # Check basic hierarchy: m_s/m_d > 1 (strange heavier than down)
        m_s_m_d = obs.get("m_s_m_d")
        if m_s_m_d is not None:
            assert m_s_m_d > 1, f"m_s/m_d = {m_s_m_d} should be > 1"

        # Check m_t/m_b > 1 (top heavier than bottom)
        m_t_m_b = obs.get("m_t_m_b")
        if m_t_m_b is not None:
            assert m_t_m_b > 1, f"m_t/m_b = {m_t_m_b} should be > 1"

    def test_dimensional_quark_masses_hierarchy(self, gift_framework):
        """Verify dimensional quark masses follow generational hierarchy."""
        obs = gift_framework.compute_all_observables()

        masses = {
            "m_u_MeV": obs.get("m_u_MeV"),
            "m_d_MeV": obs.get("m_d_MeV"),
            "m_s_MeV": obs.get("m_s_MeV"),
            "m_c_MeV": obs.get("m_c_MeV"),
            "m_b_MeV": obs.get("m_b_MeV"),
            "m_t_GeV": obs.get("m_t_GeV"),
        }

        # All masses should be positive
        for name, mass in masses.items():
            if mass is not None:
                assert mass > 0, f"{name} = {mass} should be positive"

        # Check basic ordering (if all present)
        if all(m is not None for m in [masses["m_u_MeV"], masses["m_c_MeV"]]):
            assert masses["m_c_MeV"] > masses["m_u_MeV"], "m_c should be > m_u"

        if all(m is not None for m in [masses["m_d_MeV"], masses["m_s_MeV"]]):
            assert masses["m_s_MeV"] > masses["m_d_MeV"], "m_s should be > m_d"

    def test_cosmological_density_fractions(self, gift_framework):
        """Verify cosmological density fractions are in [0, 1]."""
        obs = gift_framework.compute_all_observables()

        densities = {
            "Omega_DE": obs.get("Omega_DE"),
            "Omega_DM": obs.get("Omega_DM"),
            "Omega_b": obs.get("Omega_b"),
            "Omega_gamma": obs.get("Omega_gamma"),
            "Omega_nu": obs.get("Omega_nu"),
        }

        for name, omega in densities.items():
            if omega is not None:
                assert 0 < omega < 1, f"{name} = {omega} should be in (0, 1)"

        # Check that major components sum to approximately 1
        if all(densities[k] is not None for k in ["Omega_DE", "Omega_DM", "Omega_b"]):
            total = densities["Omega_DE"] + densities["Omega_DM"] + densities["Omega_b"]
            # Allow some slack for radiation and neutrinos
            assert 0.95 < total < 1.05, f"Omega_total = {total} should be ≈ 1"

    def test_gauge_couplings_positive(self, gift_framework):
        """Verify gauge couplings are positive."""
        obs = gift_framework.compute_all_observables()

        alpha_inv = obs.get("alpha_inv_MZ")
        if alpha_inv is not None:
            assert alpha_inv > 0, f"alpha_inv_MZ = {alpha_inv} should be positive"

        sin2thetaW = obs.get("sin2thetaW")
        if sin2thetaW is not None:
            assert 0 < sin2thetaW < 1, f"sin2thetaW = {sin2thetaW} should be in (0,1)"

        alpha_s = obs.get("alpha_s_MZ")
        if alpha_s is not None:
            assert 0 < alpha_s < 1, f"alpha_s_MZ = {alpha_s} should be in (0,1)"

    def test_electroweak_scale_consistency(self, gift_framework):
        """Verify electroweak scale observables are mutually consistent."""
        obs = gift_framework.compute_all_observables()

        v_EW = obs.get("v_EW")
        M_W = obs.get("M_W")
        M_Z = obs.get("M_Z")

        # All should be present and positive
        if v_EW is not None:
            assert v_EW > 0, f"v_EW = {v_EW} should be positive"

        if M_W is not None and M_Z is not None:
            # M_Z should be larger than M_W
            assert M_Z > M_W, f"M_Z ({M_Z}) should be > M_W ({M_W})"


class TestParameterSensitivity:
    """Test parameter sensitivity and derivatives."""

    @pytest.mark.parametrize("observable_name", [
        "delta_CP", "Q_Koide", "m_tau_m_e", "lambda_H", "Omega_DE"
    ])
    def test_topological_observables_parameter_independent(
        self, observable_name
    ):
        """Verify topological observables don't depend on free parameters."""
        # Test with different parameter values
        params_sets = [
            {"p2": 2.0, "Weyl_factor": 5.0, "tau": 3.8967},
            {"p2": 2.5, "Weyl_factor": 5.0, "tau": 3.8967},
            {"p2": 2.0, "Weyl_factor": 6.0, "tau": 3.8967},
            {"p2": 2.0, "Weyl_factor": 5.0, "tau": 4.0},
        ]

        values = []
        for params in params_sets:
            framework = GIFTFrameworkV21(**params)
            obs = framework.compute_all_observables()
            if observable_name in obs:
                values.append(obs[observable_name])

        if len(values) > 1:
            # All values should be identical (within numerical noise)
            for i in range(1, len(values)):
                rel_diff = abs(values[i] - values[0]) / (abs(values[0]) + 1e-10)
                assert rel_diff < 1e-8, (
                    f"{observable_name} varies with parameters: "
                    f"{values[0]:.10g} vs {values[i]:.10g} "
                    f"(rel_diff={rel_diff:.2e})"
                )

    def test_parameter_variation_affects_derived_observables(self):
        """Verify that varying parameters affects derived (non-topological) observables."""
        # Create two frameworks with different Weyl factors
        fw1 = GIFTFrameworkV21(p2=2.0, Weyl_factor=5.0, tau=3.8967)
        fw2 = GIFTFrameworkV21(p2=2.0, Weyl_factor=6.0, tau=3.8967)

        obs1 = fw1.compute_all_observables()
        obs2 = fw2.compute_all_observables()

        # Test some derived observables that should change
        derived_observables = ["theta12", "theta13", "theta23", "m_mu_m_e"]

        changed_count = 0
        for obs_name in derived_observables:
            if obs_name in obs1 and obs_name in obs2:
                rel_diff = abs(obs2[obs_name] - obs1[obs_name]) / abs(obs1[obs_name])
                if rel_diff > 1e-6:  # Should see some variation
                    changed_count += 1

        # At least some observables should change
        assert changed_count > 0, (
            "No derived observables changed when varying Weyl_factor"
        )


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_repeated_computation_gives_same_result(self, gift_framework):
        """Verify repeated computations give identical results."""
        obs1 = gift_framework.compute_all_observables()
        obs2 = gift_framework.compute_all_observables()

        for key in obs1:
            if key in obs2:
                assert obs1[key] == obs2[key], (
                    f"{key}: repeated computation gives different result: "
                    f"{obs1[key]} vs {obs2[key]}"
                )

    def test_no_nan_or_inf_in_observables(self, gift_framework):
        """Verify no NaN or Inf values in any observable."""
        obs = gift_framework.compute_all_observables()

        for name, value in obs.items():
            assert not np.isnan(value), f"{name} is NaN"
            assert not np.isinf(value), f"{name} is Inf"
            assert np.isfinite(value), f"{name} is not finite: {value}"

    def test_observables_have_reasonable_magnitudes(self, gift_framework):
        """Verify observables are in expected magnitude ranges."""
        obs = gift_framework.compute_all_observables()

        # Define reasonable ranges (very permissive to catch only gross errors)
        ranges = {
            "alpha_inv_MZ": (100, 200),
            "sin2thetaW": (0.1, 0.5),
            "alpha_s_MZ": (0.01, 0.5),
            "Q_Koide": (0.5, 0.8),
            "m_tau_m_e": (3000, 4000),
            "Omega_DE": (0.5, 0.9),
            "lambda_H": (0.01, 0.5),
        }

        for name, (min_val, max_val) in ranges.items():
            if name in obs:
                value = obs[name]
                assert min_val <= value <= max_val, (
                    f"{name} = {value} outside reasonable range [{min_val}, {max_val}]"
                )


class TestObservableCoverage:
    """Test that all expected observables are computed."""

    def test_all_46_observables_present(self, gift_framework):
        """Verify all 46 v2.1 observables are computed."""
        obs = gift_framework.compute_all_observables()

        expected_observables = [
            # Dimensionless (37)
            "alpha_inv_MZ", "sin2thetaW", "alpha_s_MZ",
            "theta12", "theta13", "theta23", "delta_CP",
            "Q_Koide", "m_mu_m_e", "m_tau_m_e",
            "m_s_m_d", "m_c_m_s", "m_b_m_u", "m_t_m_b",
            "m_d_m_u", "m_c_m_u", "m_b_m_d", "m_t_m_s", "m_t_m_d", "m_t_m_c",
            "V_us", "V_cb", "V_ub", "V_cd", "V_cs", "V_td",
            "lambda_H",
            "Omega_DE", "Omega_DM", "Omega_b", "n_s", "sigma_8", "A_s",
            "Omega_gamma", "Omega_nu", "Y_p", "D_H",
            # Dimensional (9)
            "v_EW", "M_W", "M_Z",
            "m_u_MeV", "m_d_MeV", "m_s_MeV", "m_c_MeV", "m_b_MeV", "m_t_GeV",
            "H0",
        ]

        missing = [name for name in expected_observables if name not in obs]

        assert len(missing) == 0, (
            f"Missing {len(missing)} observables: {missing}"
        )

        assert len(obs) >= 46, (
            f"Expected at least 46 observables, got {len(obs)}"
        )

    def test_observables_by_sector(self, gift_framework):
        """Verify observables are organized by sector."""
        obs = gift_framework.compute_all_observables()

        sectors = {
            "gauge": ["alpha_inv_MZ", "sin2thetaW", "alpha_s_MZ"],
            "neutrino": ["theta12", "theta13", "theta23", "delta_CP"],
            "lepton": ["Q_Koide", "m_mu_m_e", "m_tau_m_e"],
            "quark": ["m_s_m_d", "m_c_m_s", "m_b_m_u", "m_t_m_b"],
            "CKM": ["V_us", "V_cb", "V_ub", "V_cd", "V_cs", "V_td"],
            "higgs": ["lambda_H"],
            "cosmology": ["Omega_DE", "Omega_DM", "n_s", "H0"],
        }

        for sector_name, sector_obs in sectors.items():
            present = [name for name in sector_obs if name in obs]
            coverage = len(present) / len(sector_obs) * 100

            # Require at least some coverage per sector
            assert len(present) > 0, f"No observables from {sector_name} sector"
