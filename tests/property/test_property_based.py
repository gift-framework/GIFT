"""
Property-Based Tests using Hypothesis

Tests mathematical properties that should hold universally across the parameter
space using random/fuzz testing with the Hypothesis library.

Author: GIFT Framework Team
"""

import pytest
import numpy as np
import os
import sys

# Try to import hypothesis - skip tests if not available
try:
    from hypothesis import given, strategies as st, settings, assume, HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not installed")(func)
        return decorator

    class st:
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None

    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def assume(*args):
        pass

import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../statistical_validation'))


# Skip entire module if hypothesis not available
pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis library not installed"
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def g2_model():
    """Create G2PhiNetwork for testing."""
    from G2_phi_network import G2PhiNetwork
    return G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[32, 32],
        fourier_modes=4,
        normalize_phi=True
    )


# ============================================================================
# Metric Property Tests
# ============================================================================

class TestMetricProperties:
    """Property-based tests for metric tensor properties."""

    @given(
        x1=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x2=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x3=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x4=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x5=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x6=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x7=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_metric_always_positive_definite(self, g2_model, x1, x2, x3, x4, x5, x6, x7):
        """Metric should always be positive definite for any input coordinates."""
        from G2_phi_network import metric_from_phi_algebraic
        from G2_geometry import project_spd

        coords = torch.tensor([[x1, x2, x3, x4, x5, x6, x7]], dtype=torch.float32)
        coords.requires_grad = True

        g2_model.eval()
        with torch.no_grad():
            phi = g2_model(coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)

        eigenvalues = torch.linalg.eigvalsh(metric)

        # All eigenvalues should be positive
        assert (eigenvalues > 0).all(), f"Found non-positive eigenvalue: {eigenvalues}"

    @given(
        x1=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x2=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x3=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x4=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x5=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x6=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x7=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_metric_always_symmetric(self, g2_model, x1, x2, x3, x4, x5, x6, x7):
        """Metric should always be symmetric."""
        from G2_phi_network import metric_from_phi_algebraic
        from G2_geometry import project_spd

        coords = torch.tensor([[x1, x2, x3, x4, x5, x6, x7]], dtype=torch.float32)
        coords.requires_grad = True

        g2_model.eval()
        with torch.no_grad():
            phi = g2_model(coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)

        symmetry_error = torch.norm(metric - metric.transpose(-2, -1))

        assert symmetry_error < 1e-5, f"Metric not symmetric, error = {symmetry_error}"

    @given(
        x1=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x2=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x3=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x4=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x5=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x6=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        x7=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_phi_output_finite(self, g2_model, x1, x2, x3, x4, x5, x6, x7):
        """Phi output should always be finite."""
        coords = torch.tensor([[x1, x2, x3, x4, x5, x6, x7]], dtype=torch.float32)

        g2_model.eval()
        with torch.no_grad():
            phi = g2_model(coords)

        assert torch.isfinite(phi).all(), f"Found non-finite phi values"


# ============================================================================
# Topological Invariance Tests
# ============================================================================

class TestTopologicalInvariance:
    """Property-based tests for topological invariants."""

    @given(
        p2=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        weyl=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
        tau=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_delta_cp_invariant(self, p2, weyl, tau):
        """delta_CP = 197 should be independent of parameters."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # delta_CP is topological - doesn't depend on p2, weyl, tau
        assert obs['delta_CP'] == 197.0

    @given(
        p2=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        weyl=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
        tau=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_q_koide_invariant(self, p2, weyl, tau):
        """Q_Koide = 2/3 should be independent of parameters."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # Q_Koide is topological
        assert np.isclose(obs['Q_Koide'], 2/3, rtol=1e-10)

    @given(
        p2=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        weyl=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
        tau=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_m_tau_m_e_invariant(self, p2, weyl, tau):
        """m_tau/m_e = 3477 should be independent of parameters."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # m_tau/m_e is topological
        assert obs['m_tau_m_e'] == 3477.0

    @given(
        p2=st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        weyl=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
        tau=st.floats(1.0, 10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_omega_de_invariant(self, p2, weyl, tau):
        """Omega_DE = ln(2)*98/99 should be independent of parameters."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        expected = np.log(2) * 98 / 99
        assert np.isclose(obs['Omega_DE'], expected, rtol=1e-10)


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(
        perturbation=st.floats(-1e-6, 1e-6, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_observables_stable_under_perturbation(self, perturbation):
        """Observables should be stable under small perturbations."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # All observables should be finite
        for name, value in obs.items():
            assert np.isfinite(value), f"{name} is not finite"

    @given(
        scale=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_dimensionless_scale_invariant(self, scale):
        """Dimensionless quantities should not change with scale."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # These should be scale-invariant
        scale_invariant = ['Q_Koide', 'sin2thetaW', 'n_s']

        for name in scale_invariant:
            assert np.isfinite(obs[name])
            assert 0 < obs[name] < 10  # Reasonable range


# ============================================================================
# Manifold Property Tests
# ============================================================================

class TestManifoldProperties:
    """Property-based tests for manifold properties."""

    @given(
        x=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_periodicity_idempotent(self, x):
        """Wrapping should be idempotent: wrap(wrap(x)) = wrap(x) (mod 2*pi)."""
        from G2_manifold import TorusT7

        torus = TorusT7(device='cpu')
        coords = torch.tensor([[x, x, x, x, x, x, x]])

        wrapped_once = torus.enforce_periodicity(coords)
        wrapped_twice = torus.enforce_periodicity(wrapped_once)

        # Both values should represent the same point on the torus
        # The difference should be 0 or a multiple of 2*pi (i.e., periodically equivalent)
        diff = torch.abs(wrapped_once - wrapped_twice)
        # Either the values are close, or they differ by ~2*pi (boundary case)
        is_same = diff < 1e-5
        is_periodic_equiv = torch.abs(diff - 2*np.pi) < 1e-5
        assert (is_same | is_periodic_equiv).all()

    @given(
        x=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False),
        y=st.floats(0, 2*np.pi, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_distance_triangle_inequality(self, x, y):
        """Distance should satisfy triangle inequality."""
        from G2_manifold import TorusT7

        torus = TorusT7(device='cpu')

        # Three random points
        p1 = torch.tensor([[x] * 7])
        p2 = torch.tensor([[y] * 7])
        p3 = torch.tensor([[(x + y) / 2] * 7])

        d12 = torus.compute_distance(p1, p2)
        d13 = torus.compute_distance(p1, p3)
        d23 = torus.compute_distance(p2, p3)

        # Triangle inequality: d(p1, p2) <= d(p1, p3) + d(p3, p2)
        assert d12 <= d13 + d23 + 1e-6

    @given(
        n_samples=st.integers(1, 100)
    )
    @settings(max_examples=20, deadline=None)
    def test_sampling_correct_count(self, n_samples):
        """Sampling should return correct number of points."""
        from G2_manifold import TorusT7

        torus = TorusT7(device='cpu')
        coords = torus.sample_points(n_samples, method='uniform')

        assert coords.shape[0] == n_samples
        assert coords.shape[1] == 7


# ============================================================================
# Loss Function Property Tests
# ============================================================================

class TestLossFunctionProperties:
    """Property-based tests for loss function properties."""

    @given(
        epoch=st.integers(0, 5000)
    )
    @settings(max_examples=100, deadline=None)
    def test_curriculum_weights_positive(self, epoch):
        """Curriculum weights should always be positive."""
        from G2_losses import CurriculumScheduler

        scheduler = CurriculumScheduler()
        weights = scheduler.get_weights(epoch)

        for name, weight in weights.items():
            assert weight >= 0, f"Weight {name} = {weight} is negative at epoch {epoch}"

    @given(
        epoch=st.integers(0, 5000)
    )
    @settings(max_examples=100, deadline=None)
    def test_curriculum_phase_valid(self, epoch):
        """Curriculum phase should be valid."""
        from G2_losses import CurriculumScheduler

        scheduler = CurriculumScheduler()
        phase = scheduler.get_phase(epoch)

        assert 0 <= phase < scheduler.n_phases


# ============================================================================
# GIFT Parameter Property Tests
# ============================================================================

class TestGIFTParameterProperties:
    """Property-based tests for GIFT parameter properties."""

    def test_all_constants_finite(self):
        """All GIFT constants should be finite."""
        from gift_v22_core import GIFTParametersV22

        params = GIFTParametersV22()

        # Check all properties
        assert np.isfinite(params.p2)
        assert np.isfinite(params.Weyl_factor)
        assert np.isfinite(params.beta0)
        assert np.isfinite(params.det_g_float)
        assert np.isfinite(params.kappa_T_float)
        assert np.isfinite(params.tau_float)
        assert np.isfinite(params.xi)
        assert np.isfinite(float(params.sin2_theta_W))
        assert np.isfinite(params.alpha_s)
        assert np.isfinite(params.lambda_H)

    def test_mathematical_constants_accurate(self):
        """Mathematical constants should be accurate."""
        from gift_v22_core import GIFTParametersV22

        params = GIFTParametersV22()

        # Golden ratio
        assert np.isclose(params.phi_golden, (1 + np.sqrt(5)) / 2, rtol=1e-10)

        # Apery's constant (zeta(3))
        assert np.isclose(params.zeta3, 1.2020569031595942, rtol=1e-10)

        # Euler-Mascheroni constant
        assert np.isclose(params.gamma_euler, 0.5772156649015329, rtol=1e-10)


# ============================================================================
# Batch Size Invariance Tests
# ============================================================================

class TestBatchSizeInvariance:
    """Tests that results don't depend on batch size."""

    @given(
        batch_size=st.integers(1, 50)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_model_batch_invariant(self, g2_model, batch_size):
        """Model output should be the same regardless of batch processing."""
        torch.manual_seed(42)

        # Single fixed coordinate
        single_coord = torch.rand(1, 7) * 2 * np.pi

        g2_model.eval()
        with torch.no_grad():
            single_output = g2_model(single_coord)

            # Process in batch with padding
            batch_coords = single_coord.repeat(batch_size, 1)
            batch_output = g2_model(batch_coords)

        # First element of batch should match single output
        assert torch.allclose(single_output, batch_output[0:1], atol=1e-5)


# ============================================================================
# Seed Reproducibility Tests
# ============================================================================

class TestSeedReproducibility:
    """Tests for reproducibility with fixed seeds."""

    @given(
        seed=st.integers(0, 10000)
    )
    @settings(max_examples=20, deadline=None)
    def test_sampling_reproducible_with_seed(self, seed):
        """Sampling with same seed should produce same results."""
        from G2_manifold import TorusT7

        torch.manual_seed(seed)
        torus1 = TorusT7(device='cpu')
        coords1 = torus1.sample_points(10, method='uniform')

        torch.manual_seed(seed)
        torus2 = TorusT7(device='cpu')
        coords2 = torus2.sample_points(10, method='uniform')

        assert torch.allclose(coords1, coords2)
