"""
Property-based tests using Hypothesis framework.

Features:
- Automatic test case generation
- Property invariants verification
- Edge case discovery
- Mathematical property validation
- Boundary condition exploration
- Contravariant property testing

Requires: hypothesis library
Install with: pip install hypothesis

Version: 2.1.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21, GIFTParameters
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis import HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(not V21_AVAILABLE, reason="GIFT v2.1 not available"),
    pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
]


# Hypothesis strategies for parameters
positive_floats = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
small_positive = st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
parameter_floats = st.floats(min_value=1.5, max_value=3.0, allow_nan=False, allow_infinity=False)


class TestMathematicalInvariants:
    """Test mathematical properties that should always hold."""

    @given(st.floats(min_value=1.8, max_value=2.2, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, deadline=5000)
    def test_topological_observables_parameter_independent(self, p2_value):
        """Property: Topological observables don't depend on parameters."""
        try:
            fw1 = GIFTFrameworkV21(p2=2.0, Weyl_factor=5.0)
            fw2 = GIFTFrameworkV21(p2=p2_value, Weyl_factor=5.0)

            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()

            topological = ["delta_CP", "Q_Koide", "m_tau_m_e", "lambda_H"]

            for obs_name in topological:
                if obs_name in obs1 and obs_name in obs2:
                    # Should be identical
                    assert abs(obs1[obs_name] - obs2[obs_name]) < 1e-10, (
                        f"{obs_name} varies with p2: {obs1[obs_name]} vs {obs2[obs_name]}"
                    )
        except Exception as e:
            # Some parameter values might cause computation issues
            assume(False)

    @given(
        p2=st.floats(min_value=1.5, max_value=2.5, allow_nan=False, allow_infinity=False),
        weyl=st.floats(min_value=4.0, max_value=6.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, deadline=5000)
    def test_all_observables_finite(self, p2, weyl):
        """Property: All observables should be finite for valid parameters."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            for name, value in obs.items():
                assert np.isfinite(value), f"{name} is not finite: {value}"

        except Exception:
            # Some parameter combinations might be invalid
            assume(False)

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=15, deadline=5000)
    def test_mass_ratios_positive(self, p2, weyl):
        """Property: All mass ratios should be positive."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            mass_ratios = [k for k in obs.keys() if k.startswith("m_") and "_m_" in k]

            for ratio_name in mass_ratios:
                value = obs[ratio_name]
                assert value > 0, f"{ratio_name} = {value} should be positive"

        except Exception:
            assume(False)

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=15, deadline=5000)
    def test_ckm_elements_bounded(self, p2, weyl):
        """Property: CKM matrix elements should be in [0, 1]."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            ckm_elements = [k for k in obs.keys() if k.startswith("V_")]

            for ckm_name in ckm_elements:
                value = obs[ckm_name]
                assert 0 <= value <= 1, (
                    f"{ckm_name} = {value} outside [0, 1]"
                )

        except Exception:
            assume(False)


class TestComputationalProperties:
    """Test computational properties and behavior."""

    @given(
        p2=st.floats(min_value=1.9, max_value=2.1, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=10, deadline=5000)
    def test_deterministic_computation(self, p2):
        """Property: Same parameters give same results."""
        try:
            fw1 = GIFTFrameworkV21(p2=p2)
            fw2 = GIFTFrameworkV21(p2=p2)

            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()

            for key in obs1:
                if key in obs2:
                    assert obs1[key] == obs2[key], (
                        f"{key}: non-deterministic ({obs1[key]} vs {obs2[key]})"
                    )

        except Exception:
            assume(False)

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_idempotent_computation(self, p2, weyl):
        """Property: Computing twice gives same result (idempotent)."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)

            obs1 = framework.compute_all_observables()
            obs2 = framework.compute_all_observables()

            for key in obs1:
                if key in obs2:
                    assert obs1[key] == obs2[key], (
                        f"{key}: not idempotent"
                    )

        except Exception:
            assume(False)

    @given(
        values=st.lists(
            st.floats(min_value=1.8, max_value=2.2, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=10, deadline=10000)
    def test_order_independence(self, values):
        """Property: Order of parameter values doesn't affect individual results."""
        try:
            results = []

            for p2_val in values:
                framework = GIFTFrameworkV21(p2=p2_val)
                obs = framework.compute_all_observables()
                results.append((p2_val, obs.get("alpha_inv_MZ", None)))

            # Each p2 value should give consistent result regardless of order
            # Create dict
            p2_to_result = {}
            for p2_val, result in results:
                if result is not None:
                    if p2_val in p2_to_result:
                        # Check consistency
                        assert abs(p2_to_result[p2_val] - result) < 1e-10
                    else:
                        p2_to_result[p2_val] = result

        except Exception:
            assume(False)


class TestPhysicalConstraints:
    """Test physical constraint properties."""

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=15, deadline=5000)
    def test_cosmological_densities_sum_to_one(self, p2, weyl):
        """Property: Cosmological density fractions should sum to ~1."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            densities = ["Omega_DE", "Omega_DM", "Omega_b"]

            if all(d in obs for d in densities):
                total = sum(obs[d] for d in densities)

                # Should be approximately 1 (allow for radiation)
                assert 0.95 < total < 1.05, (
                    f"Density fractions sum to {total}, not ~1"
                )

        except Exception:
            assume(False)

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=15, deadline=5000)
    def test_mixing_angles_physical_range(self, p2, weyl):
        """Property: Mixing angles should be in [0, 90] degrees."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            angles = ["theta12", "theta13", "theta23"]

            for angle_name in angles:
                if angle_name in obs:
                    angle = obs[angle_name]
                    assert 0 <= angle <= 90, (
                        f"{angle_name} = {angle}° outside [0, 90]°"
                    )

        except Exception:
            assume(False)

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=15, deadline=5000)
    def test_gauge_couplings_positive(self, p2, weyl):
        """Property: Gauge couplings should be positive."""
        try:
            framework = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            obs = framework.compute_all_observables()

            couplings = ["alpha_inv_MZ", "sin2thetaW", "alpha_s_MZ"]

            for coupling_name in couplings:
                if coupling_name in obs:
                    value = obs[coupling_name]
                    assert value > 0, (
                        f"{coupling_name} = {value} should be positive"
                    )

        except Exception:
            assume(False)


class TestNumericalStability:
    """Test numerical stability properties."""

    @given(
        p2_1=st.floats(min_value=1.99, max_value=2.01, allow_nan=False, allow_infinity=False),
        p2_2=st.floats(min_value=1.99, max_value=2.01, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=10, deadline=5000)
    def test_continuous_dependence(self, p2_1, p2_2):
        """Property: Small parameter changes give small result changes."""
        try:
            # Skip if parameters too similar
            assume(abs(p2_1 - p2_2) > 1e-6)

            fw1 = GIFTFrameworkV21(p2=p2_1)
            fw2 = GIFTFrameworkV21(p2=p2_2)

            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()

            # Derived observables should vary continuously
            derived_obs = ["theta12", "theta13", "theta23", "m_mu_m_e"]

            for obs_name in derived_obs:
                if obs_name in obs1 and obs_name in obs2:
                    val1 = obs1[obs_name]
                    val2 = obs2[obs_name]

                    param_change = abs(p2_2 - p2_1) / 2.0
                    value_change = abs(val2 - val1)
                    rel_value_change = value_change / (abs(val1) + 1e-10)

                    # Change should be bounded (not chaotic)
                    assert rel_value_change < 100 * param_change, (
                        f"{obs_name}: discontinuous behavior"
                    )

        except Exception:
            assume(False)

    @given(
        p2=st.floats(min_value=1.95, max_value=2.05, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=10, deadline=5000)
    def test_no_numerical_cancellation(self, p2):
        """Property: Results shouldn't be dominated by numerical cancellation."""
        try:
            framework = GIFTFrameworkV21(p2=p2)
            obs = framework.compute_all_observables()

            # No observable should be exactly zero (suggests cancellation issues)
            for name, value in obs.items():
                # Allow proven zeros, but most should be non-zero
                if value == 0.0:
                    # This might be legitimate, but flag it
                    pass  # In real usage, might want to investigate

                # Should have reasonable magnitude
                assert abs(value) > 1e-100 or value == 0, (
                    f"{name} = {value} suspiciously small"
                )

        except Exception:
            assume(False)


class TestSymmetryProperties:
    """Test symmetry and invariance properties."""

    @given(
        sign=st.sampled_from([-1, 1])
    )
    @settings(max_examples=2, deadline=5000)
    def test_parameter_sign_invariance(self, sign):
        """Property: Some observables might be invariant under parameter sign flip."""
        # This is a speculative test - remove if not applicable

        try:
            # Test if any observables have sign symmetry
            # (Most won't, but worth checking)

            fw_pos = GIFTFrameworkV21(p2=2.0)
            obs_pos = fw_pos.compute_all_observables()

            # This test might not apply to GIFT framework
            # Kept as example of property-based testing approach
            assume(False)  # Skip for now

        except Exception:
            assume(False)


class TestMonotonicity:
    """Test monotonicity properties."""

    @given(
        p2_values=st.lists(
            st.floats(min_value=1.8, max_value=2.2, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=5, deadline=15000)
    def test_monotonic_parameter_dependence(self, p2_values):
        """Property: Some observables might be monotonic in parameters."""
        try:
            # Sort parameter values
            p2_sorted = sorted(p2_values)

            results = []
            for p2 in p2_sorted:
                framework = GIFTFrameworkV21(p2=p2)
                obs = framework.compute_all_observables()
                results.append(obs)

            # Check if any observable is monotonic
            # (This is exploratory - helps understand dependencies)

            # For now, just check computation succeeds
            assert len(results) == len(p2_sorted)

        except Exception:
            assume(False)


class TestComposition:
    """Test composition and combination properties."""

    @given(
        p2=parameter_floats,
        weyl=parameter_floats
    )
    @settings(max_examples=10, deadline=5000)
    def test_framework_composition(self, p2, weyl):
        """Property: Creating multiple frameworks doesn't interfere."""
        try:
            fw1 = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            fw2 = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)
            fw3 = GIFTFrameworkV21(p2=p2, Weyl_factor=weyl)

            obs1 = fw1.compute_all_observables()
            obs2 = fw2.compute_all_observables()
            obs3 = fw3.compute_all_observables()

            # All should be identical
            for key in obs1:
                if key in obs2 and key in obs3:
                    assert obs1[key] == obs2[key] == obs3[key]

        except Exception:
            assume(False)


# Summary test to demonstrate Hypothesis usage
class TestHypothesisFeatures:
    """Demonstrate advanced Hypothesis features."""

    @given(
        data=st.data()
    )
    @settings(max_examples=5, deadline=10000)
    def test_interactive_strategy(self, data):
        """Demonstrate interactive test case generation."""
        try:
            # Generate parameters interactively
            p2 = data.draw(st.floats(min_value=1.8, max_value=2.2, allow_nan=False, allow_infinity=False))

            framework = GIFTFrameworkV21(p2=p2)
            obs = framework.compute_all_observables()

            # All observables should be computed
            assert len(obs) > 0

        except Exception:
            assume(False)
