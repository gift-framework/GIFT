"""
Unit tests for G2 manifold module.

Tests manifold construction, point sampling, and coordinate operations.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add G2_ML/archived/early_development/0.2 to path (legacy version for these tests)
sys.path.insert(0, str(Path(__file__).parent.parent / "archived" / "early_development" / "0.2"))

from G2_manifold import create_manifold


class TestManifoldCreation:
    """Test manifold creation and initialization."""

    def test_create_T7_manifold(self):
        """Test creating T^7 torus manifold."""
        manifold = create_manifold('T7', device='cpu')
        assert manifold is not None
        assert hasattr(manifold, 'sample_points')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_manifold_gpu(self):
        """Test creating manifold on GPU."""
        manifold = create_manifold('T7', device='cuda')
        points = manifold.sample_points(10)
        assert points.device.type == 'cuda'

    def test_invalid_manifold_type(self):
        """Test that invalid manifold type raises error."""
        with pytest.raises((ValueError, KeyError, AttributeError)):
            create_manifold('invalid_type', device='cpu')


class TestPointSampling:
    """Test point sampling on manifolds."""

    def test_sample_points_shape(self):
        """Test that sampled points have correct shape."""
        manifold = create_manifold('T7', device='cpu')
        n_points = 100

        points = manifold.sample_points(n_points)

        assert points.shape == (n_points, 7)

    def test_sample_points_different_sizes(self):
        """Test sampling different numbers of points."""
        manifold = create_manifold('T7', device='cpu')

        for n in [1, 10, 100, 1000]:
            points = manifold.sample_points(n)
            assert points.shape[0] == n
            assert points.shape[1] == 7

    def test_sample_points_reproducible(self):
        """Test that sampling is reproducible with same seed."""
        manifold1 = create_manifold('T7', device='cpu')
        manifold2 = create_manifold('T7', device='cpu')

        torch.manual_seed(42)
        points1 = manifold1.sample_points(50)

        torch.manual_seed(42)
        points2 = manifold2.sample_points(50)

        assert torch.allclose(points1, points2)

    def test_sample_points_in_range(self):
        """Test that sampled points are in valid range."""
        manifold = create_manifold('T7', device='cpu')
        points = manifold.sample_points(100)

        # For T^7, points should be in [0, 2*pi]^7
        assert torch.all(points >= 0)
        assert torch.all(points <= 2 * torch.pi + 1e-6)

    def test_sample_points_no_nan(self):
        """Test that sampled points contain no NaN."""
        manifold = create_manifold('T7', device='cpu')
        points = manifold.sample_points(100)

        assert not torch.any(torch.isnan(points))
        assert not torch.any(torch.isinf(points))


class TestManifoldProperties:
    """Test manifold geometric properties."""

    def test_manifold_dimension(self):
        """Test that manifold has dimension 7."""
        manifold = create_manifold('T7', device='cpu')
        points = manifold.sample_points(1)

        assert points.shape[1] == 7

    def test_manifold_coverage(self):
        """Test that many samples cover the manifold."""
        manifold = create_manifold('T7', device='cpu')
        points = manifold.sample_points(10000)

        # Check that we have reasonable coverage in each dimension
        for dim in range(7):
            coords = points[:, dim]
            assert coords.min() < torch.pi  # Some points in lower half
            assert coords.max() > torch.pi  # Some points in upper half

    def test_batch_sampling(self):
        """Test sampling in batches."""
        manifold = create_manifold('T7', device='cpu')

        batch1 = manifold.sample_points(50)
        batch2 = manifold.sample_points(50)

        # Should be different samples (with high probability)
        assert not torch.allclose(batch1, batch2)


@pytest.mark.slow
class TestLargeScaleManifold:
    """Test manifold operations at scale."""

    def test_large_batch_sampling(self):
        """Test sampling large batches."""
        manifold = create_manifold('T7', device='cpu')
        large_batch = manifold.sample_points(100000)

        assert large_batch.shape == (100000, 7)
        assert not torch.any(torch.isnan(large_batch))

    def test_memory_efficiency(self):
        """Test that sampling doesn't leak memory."""
        manifold = create_manifold('T7', device='cpu')

        # Sample multiple times, old tensors should be freed
        for _ in range(10):
            points = manifold.sample_points(10000)
            del points
