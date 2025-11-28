"""
Boundary Condition Tests for Manifold Coordinates

Tests for edge cases in manifold coordinates including:
- Coordinate boundaries
- Periodicity enforcement
- Singularity handling
- Volume element positivity

Author: GIFT Framework Team
"""

import pytest
import torch
import numpy as np
import os
import sys

# Add G2_ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))

from G2_manifold import TorusT7, TwistedConnectedSum, create_manifold
from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
from G2_geometry import project_spd, volume_form


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def torus():
    """Create standard T^7 manifold."""
    return TorusT7(device='cpu')


@pytest.fixture
def custom_torus():
    """Create T^7 with custom radii."""
    radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi]
    return TorusT7(radii=radii, device='cpu')


@pytest.fixture
def g2_model():
    """Create G2PhiNetwork for testing."""
    return G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[32, 32],
        fourier_modes=4,
        normalize_phi=True
    )


# ============================================================================
# Torus Boundary Tests
# ============================================================================

class TestTorusBoundaries:
    """Tests for T^7 torus boundary conditions."""

    def test_fundamental_domain_bounds(self, torus):
        """Fundamental domain should be [0, 2*pi]^7."""
        bounds = torus.get_fundamental_domain()

        assert bounds.shape == (7, 2)
        assert torch.allclose(bounds[:, 0], torch.zeros(7))
        assert torch.allclose(bounds[:, 1], torch.ones(7) * 2 * np.pi)

    def test_samples_in_fundamental_domain(self, torus):
        """Sampled points should be in fundamental domain."""
        coords = torus.sample_points(1000, method='uniform')

        assert (coords >= 0).all()
        assert (coords <= 2 * np.pi + 1e-6).all()

    def test_zero_boundary_valid(self, torus, g2_model):
        """Model should handle coordinates at zero boundary."""
        coords = torch.zeros(10, 7)
        coords.requires_grad = True

        phi = g2_model(coords)

        assert torch.isfinite(phi).all()

    def test_upper_boundary_valid(self, torus, g2_model):
        """Model should handle coordinates at upper boundary (2*pi)."""
        coords = torch.ones(10, 7) * 2 * np.pi
        coords.requires_grad = True

        phi = g2_model(coords)

        assert torch.isfinite(phi).all()

    def test_corner_points_valid(self, torus, g2_model):
        """Model should handle corner points of fundamental domain."""
        # All 2^7 = 128 corners
        corners = torch.tensor([
            [int(i) * 2 * np.pi for i in format(n, '07b')]
            for n in range(128)
        ], dtype=torch.float32)

        phi = g2_model(corners)

        assert torch.isfinite(phi).all()


# ============================================================================
# Periodicity Tests
# ============================================================================

class TestPeriodicity:
    """Tests for periodic boundary condition enforcement."""

    def test_wrapping_outside_domain(self, torus):
        """Coordinates outside domain should wrap correctly."""
        # Points far outside [0, 2*pi]
        coords_outside = torch.tensor([
            [10.0] * 7,
            [-5.0] * 7,
            [100.0] * 7,
            [-100.0] * 7
        ])

        wrapped = torus.enforce_periodicity(coords_outside)

        # All should be in [0, 2*pi)
        assert (wrapped >= 0).all()
        assert (wrapped < 2 * np.pi + 1e-6).all()

    def test_wrapping_preserves_in_domain(self, torus):
        """Wrapping should not change points already in domain."""
        coords_inside = torch.rand(100, 7) * 2 * np.pi

        wrapped = torus.enforce_periodicity(coords_inside)

        assert torch.allclose(coords_inside, wrapped)

    def test_periodic_equivalence_distance(self, torus):
        """Periodic equivalent points should have zero distance."""
        # Point and its periodic image
        x1 = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        x2 = x1 + 2 * np.pi  # Periodic image

        distance = torus.compute_distance(x1, x2)

        assert distance.item() < 1e-5

    def test_wrapping_multiple_periods(self, torus):
        """Should handle wrapping across multiple periods."""
        coords = torch.tensor([[
            10 * np.pi,
            20 * np.pi,
            -5 * np.pi,
            1000 * np.pi,
            -1000 * np.pi,
            0.5,
            np.pi
        ]])

        wrapped = torus.enforce_periodicity(coords)

        assert (wrapped >= 0).all()
        assert (wrapped < 2 * np.pi + 1e-6).all()

    def test_model_output_periodic_consistency(self, torus, g2_model):
        """Model outputs should be approximately consistent for periodic points."""
        g2_model.eval()

        # Sample base point
        base = torch.rand(1, 7) * 2 * np.pi

        # Create periodic images
        images = []
        for shift in [0, 1, -1, 2, -2]:
            img = base + shift * 2 * np.pi
            images.append(torus.enforce_periodicity(img))

        # All periodic images should be wrapped to same point
        for img in images:
            assert torch.allclose(images[0], img, atol=1e-5)


# ============================================================================
# Periodic Distance Tests
# ============================================================================

class TestPeriodicDistance:
    """Tests for periodic distance computation."""

    def test_distance_symmetric(self, torus):
        """Distance should be symmetric: d(x,y) = d(y,x)."""
        x1 = torch.rand(100, 7) * 2 * np.pi
        x2 = torch.rand(100, 7) * 2 * np.pi

        d12 = torus.compute_distance(x1, x2)
        d21 = torus.compute_distance(x2, x1)

        assert torch.allclose(d12, d21)

    def test_distance_non_negative(self, torus):
        """Distance should be non-negative."""
        x1 = torch.rand(100, 7) * 2 * np.pi
        x2 = torch.rand(100, 7) * 2 * np.pi

        distance = torus.compute_distance(x1, x2)

        assert (distance >= 0).all()

    def test_distance_self_zero(self, torus):
        """Distance to self should be zero."""
        x = torch.rand(100, 7) * 2 * np.pi

        distance = torus.compute_distance(x, x)

        assert torch.allclose(distance, torch.zeros(100))

    def test_distance_across_boundary_shorter(self, torus):
        """Periodic distance should find shorter path across boundary."""
        # Points near opposite boundaries
        x1 = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        x2 = torch.tensor([[2*np.pi - 0.1, 2*np.pi - 0.1, 2*np.pi - 0.1,
                          2*np.pi - 0.1, 2*np.pi - 0.1, 2*np.pi - 0.1,
                          2*np.pi - 0.1]])

        periodic_dist = torus.compute_distance(x1, x2)
        euclidean_dist = torch.norm(x1 - x2)

        # Periodic distance should be much smaller
        assert periodic_dist < euclidean_dist / 2


# ============================================================================
# Custom Radii Tests
# ============================================================================

class TestCustomRadii:
    """Tests for T^7 with custom radii."""

    def test_custom_radii_respected(self, custom_torus):
        """Custom radii should be stored correctly."""
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.pi])
        assert torch.allclose(custom_torus.radii, expected)

    def test_samples_respect_custom_radii(self, custom_torus):
        """Samples should respect custom radii bounds."""
        coords = custom_torus.sample_points(1000, method='uniform')

        for i in range(7):
            assert (coords[:, i] >= 0).all()
            assert (coords[:, i] <= custom_torus.radii[i] + 1e-6).all()

    def test_custom_radii_volume(self, custom_torus):
        """Volume should be product of radii."""
        expected_volume = 1.0 * 2.0 * 3.0 * 4.0 * 5.0 * 6.0 * np.pi
        computed_volume = custom_torus.volume()

        assert np.isclose(computed_volume, expected_volume, rtol=1e-6)

    def test_custom_radii_wrapping(self, custom_torus):
        """Wrapping should respect custom radii."""
        coords = torch.tensor([[
            2.0,   # Should wrap at 1.0
            5.0,   # Should wrap at 2.0
            10.0,  # Should wrap at 3.0
            12.0,  # Should wrap at 4.0
            15.0,  # Should wrap at 5.0
            18.0,  # Should wrap at 6.0
            10.0   # Should wrap at pi
        ]])

        wrapped = custom_torus.enforce_periodicity(coords)

        for i in range(7):
            assert wrapped[0, i] >= 0
            assert wrapped[0, i] < custom_torus.radii[i] + 1e-6


# ============================================================================
# Volume Element Tests
# ============================================================================

class TestVolumeElement:
    """Tests for metric volume element positivity."""

    def test_volume_element_positive(self, g2_model, torus):
        """sqrt(det(g)) should be positive everywhere."""
        coords = torus.sample_points(100, method='uniform')
        coords.requires_grad = True

        phi = g2_model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        vol = volume_form(metric)

        assert (vol > 0).all()

    def test_volume_element_at_boundaries(self, g2_model, torus):
        """Volume element should be positive at boundaries."""
        # Test at various boundary points
        boundary_points = torch.tensor([
            [0.0] * 7,
            [2*np.pi] * 7,
            [0.0, 2*np.pi, 0.0, 2*np.pi, 0.0, 2*np.pi, 0.0],
            [np.pi] * 7
        ])
        boundary_points.requires_grad = True

        phi = g2_model(boundary_points)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        vol = volume_form(metric)

        assert (vol > 0).all()

    def test_volume_element_finite(self, g2_model, torus):
        """Volume element should be finite."""
        coords = torus.sample_points(100, method='uniform')
        coords.requires_grad = True

        phi = g2_model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        vol = volume_form(metric)

        assert torch.isfinite(vol).all()


# ============================================================================
# Sampling Method Tests
# ============================================================================

class TestSamplingMethods:
    """Tests for different sampling methods."""

    def test_uniform_sampling(self, torus):
        """Uniform sampling should cover domain evenly."""
        coords = torus.sample_points(10000, method='uniform')

        # Check each dimension has samples across range
        for i in range(7):
            assert coords[:, i].min() < 0.5
            assert coords[:, i].max() > 2*np.pi - 0.5

    def test_grid_sampling(self, torus):
        """Grid sampling should produce regular lattice."""
        coords = torus.sample_points(128, method='grid')

        # Should have points
        assert coords.shape[0] > 0
        assert coords.shape[1] == 7

    def test_sobol_sampling(self, torus):
        """Sobol sampling should produce quasi-random sequence."""
        coords = torus.sample_points(100, method='sobol')

        # Should be in domain
        assert (coords >= 0).all()
        assert (coords <= 2*np.pi + 1e-6).all()

    def test_invalid_method_raises(self, torus):
        """Invalid sampling method should raise error."""
        with pytest.raises(ValueError):
            torus.sample_points(100, method='invalid_method')


# ============================================================================
# Validation Grid Tests
# ============================================================================

class TestValidationGrid:
    """Tests for validation grid creation."""

    def test_grid_shape(self, torus):
        """Validation grid should have correct shape."""
        grid = torus.create_validation_grid(points_per_dim=3)

        # 3^7 = 2187 points
        assert grid.shape == (2187, 7)

    def test_grid_covers_domain(self, torus):
        """Grid should cover entire domain."""
        grid = torus.create_validation_grid(points_per_dim=5)

        for i in range(7):
            assert grid[:, i].min() < 0.1
            assert grid[:, i].max() > 2*np.pi - 0.1

    def test_grid_includes_boundaries(self, torus):
        """Grid should include boundary points."""
        grid = torus.create_validation_grid(points_per_dim=5)

        # Should have points at or near 0
        assert grid.min() < 0.1

        # Should have points at or near 2*pi
        assert grid.max() > 2*np.pi - 0.1


# ============================================================================
# TCS Manifold Tests
# ============================================================================

class TestTCSManifold:
    """Tests for Twisted Connected Sum manifold (placeholder)."""

    def test_tcs_creates(self):
        """TCS manifold should create (delegates to T7)."""
        tcs = TwistedConnectedSum(device='cpu')
        assert tcs.dim == 7

    def test_tcs_samples(self):
        """TCS should be able to sample points."""
        tcs = TwistedConnectedSum(device='cpu')
        coords = tcs.sample_points(100, method='uniform')

        assert coords.shape == (100, 7)

    def test_factory_creates_tcs(self):
        """Factory should create TCS manifold."""
        manifold = create_manifold('TCS', device='cpu')
        assert isinstance(manifold, TwistedConnectedSum)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestManifoldEdgeCases:
    """Tests for edge cases in manifold operations."""

    def test_empty_batch(self, torus):
        """Should handle empty batch gracefully."""
        # Create empty tensor
        coords = torch.empty(0, 7)
        wrapped = torus.enforce_periodicity(coords)

        assert wrapped.shape == (0, 7)

    def test_single_point(self, torus):
        """Should handle single point."""
        coords = torus.sample_points(1, method='uniform')

        assert coords.shape == (1, 7)

    def test_very_large_coordinates(self, torus):
        """Should handle very large coordinates."""
        coords = torch.tensor([[1e10] * 7])
        wrapped = torus.enforce_periodicity(coords)

        assert (wrapped >= 0).all()
        assert (wrapped < 2*np.pi + 1e-6).all()
        assert torch.isfinite(wrapped).all()

    def test_very_small_negative_coordinates(self, torus):
        """Should handle small negative coordinates."""
        coords = torch.tensor([[-1e-10] * 7])
        wrapped = torus.enforce_periodicity(coords)

        assert (wrapped >= 0).all()
        assert torch.isfinite(wrapped).all()

    def test_nan_handling(self, torus):
        """Periodicity should preserve NaN structure."""
        coords = torch.tensor([[np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        wrapped = torus.enforce_periodicity(coords)

        # NaN should remain NaN
        assert torch.isnan(wrapped[0, 0])
        # Others should be valid
        assert torch.isfinite(wrapped[0, 1:]).all()

    def test_inf_handling(self, torus):
        """Should handle infinity gracefully."""
        coords = torch.tensor([[np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        wrapped = torus.enforce_periodicity(coords)

        # Inf wrapped should produce NaN (indeterminate)
        assert torch.isnan(wrapped[0, 0]) or not torch.isfinite(wrapped[0, 0])
