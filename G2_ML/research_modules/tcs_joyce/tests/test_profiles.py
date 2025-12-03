"""
Unit tests for TCS profile functions.

Tests:
- smooth_step sigmoid behavior
- left_plateau / right_plateau boundary conditions
- neck_bump localization
- TCSProfiles class
- Derivative computations
"""

import pytest
import torch
import sys
from pathlib import Path

# Add tcs_joyce to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from profiles import (
        smooth_step,
        left_plateau,
        right_plateau,
        neck_bump,
        neck_bump_normalized,
        derivative_smooth_step,
        TCSProfiles,
    )
    PROFILES_AVAILABLE = True
except ImportError as e:
    PROFILES_AVAILABLE = False
    PROFILES_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not PROFILES_AVAILABLE,
    reason=f"profiles module not available: {PROFILES_IMPORT_ERROR if not PROFILES_AVAILABLE else ''}"
)


# =============================================================================
# smooth_step Tests
# =============================================================================

class TestSmoothStep:
    """Test smooth_step sigmoid function."""

    def test_output_range(self):
        """Output should be in [0, 1]."""
        x = torch.linspace(-10, 10, 1000)
        y = smooth_step(x, x0=0.0, width=1.0)
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_center_value(self):
        """At x = x0, output should be 0.5."""
        x = torch.tensor([0.0])
        y = smooth_step(x, x0=0.0, width=1.0)
        assert torch.isclose(y, torch.tensor([0.5]), atol=1e-6)

    def test_asymptotic_left(self):
        """Far left should be ~0."""
        x = torch.tensor([-100.0])
        y = smooth_step(x, x0=0.0, width=1.0)
        assert y.item() < 0.01

    def test_asymptotic_right(self):
        """Far right should be ~1."""
        x = torch.tensor([100.0])
        y = smooth_step(x, x0=0.0, width=1.0)
        assert y.item() > 0.99

    def test_monotonicity(self):
        """Function should be monotonically increasing."""
        x = torch.linspace(-5, 5, 100)
        y = smooth_step(x, x0=0.0, width=1.0)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all()

    def test_width_effect(self):
        """Larger width = smoother transition."""
        x = torch.tensor([0.5])
        y_narrow = smooth_step(x, x0=0.0, width=0.1)
        y_wide = smooth_step(x, x0=0.0, width=1.0)
        # Narrow should be closer to 1 at x=0.5
        assert y_narrow > y_wide

    def test_steepness_effect(self):
        """Larger steepness = sharper transition."""
        x = torch.tensor([0.2])
        y_normal = smooth_step(x, x0=0.0, width=1.0, steepness=1.0)
        y_steep = smooth_step(x, x0=0.0, width=1.0, steepness=5.0)
        assert y_steep > y_normal

    def test_batched_input(self):
        """Should work with batched input."""
        x = torch.randn(100, 10)
        y = smooth_step(x, x0=0.0, width=1.0)
        assert y.shape == x.shape


# =============================================================================
# left_plateau Tests
# =============================================================================

class TestLeftPlateau:
    """Test left_plateau profile function."""

    def test_left_boundary(self):
        """Should be ~1 on the left side."""
        x = torch.tensor([-10.0])
        y = left_plateau(x, lambda_neck=0.0, sigma=0.1)
        assert y.item() > 0.99

    def test_right_boundary(self):
        """Should be ~0 on the right side."""
        x = torch.tensor([10.0])
        y = left_plateau(x, lambda_neck=0.0, sigma=0.1)
        assert y.item() < 0.01

    def test_monotonicity(self):
        """Should be monotonically decreasing."""
        x = torch.linspace(-5, 5, 100)
        y = left_plateau(x, lambda_neck=0.0, sigma=0.5)
        diffs = y[1:] - y[:-1]
        assert (diffs <= 1e-6).all()


# =============================================================================
# right_plateau Tests
# =============================================================================

class TestRightPlateau:
    """Test right_plateau profile function."""

    def test_left_boundary(self):
        """Should be ~0 on the left side."""
        x = torch.tensor([-10.0])
        y = right_plateau(x, lambda_neck=0.0, sigma=0.1)
        assert y.item() < 0.01

    def test_right_boundary(self):
        """Should be ~1 on the right side."""
        x = torch.tensor([10.0])
        y = right_plateau(x, lambda_neck=0.0, sigma=0.1)
        assert y.item() > 0.99

    def test_monotonicity(self):
        """Should be monotonically increasing."""
        x = torch.linspace(-5, 5, 100)
        y = right_plateau(x, lambda_neck=0.0, sigma=0.5)
        diffs = y[1:] - y[:-1]
        assert (diffs >= -1e-6).all()


# =============================================================================
# neck_bump Tests
# =============================================================================

class TestNeckBump:
    """Test neck_bump profile function."""

    def test_peak_at_center(self):
        """Maximum should be at lambda_neck."""
        x = torch.linspace(-2, 2, 1001)
        y = neck_bump(x, lambda_neck=0.0, sigma=0.5)
        max_idx = y.argmax()
        assert abs(x[max_idx].item()) < 0.01

    def test_peak_height(self):
        """Peak height should match parameter."""
        x = torch.tensor([0.0])
        y = neck_bump(x, lambda_neck=0.0, sigma=0.5, peak_height=2.5)
        assert torch.isclose(y, torch.tensor([2.5]), atol=1e-6)

    def test_decay_at_edges(self):
        """Should decay to ~0 far from center."""
        x = torch.tensor([10.0])
        y = neck_bump(x, lambda_neck=0.0, sigma=0.5)
        assert y.item() < 0.01

    def test_symmetry(self):
        """Should be symmetric around lambda_neck."""
        x_pos = torch.tensor([1.0])
        x_neg = torch.tensor([-1.0])
        y_pos = neck_bump(x_pos, lambda_neck=0.0, sigma=0.5)
        y_neg = neck_bump(x_neg, lambda_neck=0.0, sigma=0.5)
        assert torch.isclose(y_pos, y_neg, atol=1e-6)


# =============================================================================
# neck_bump_normalized Tests
# =============================================================================

class TestNeckBumpNormalized:
    """Test normalized neck bump."""

    def test_integral_approximately_one(self):
        """Integral over domain should be ~1."""
        x = torch.linspace(-1, 1, 10001)
        dx = (x[1] - x[0]).item()
        y = neck_bump_normalized(x, lambda_neck=0.0, sigma=0.2, domain=(-1, 1))
        integral = (y * dx).sum().item()
        assert abs(integral - 1.0) < 0.1  # Allow some numerical error


# =============================================================================
# derivative_smooth_step Tests
# =============================================================================

class TestDerivativeSmoothStep:
    """Test derivative of smooth_step."""

    def test_peak_at_center(self):
        """Derivative should peak at x0."""
        x = torch.linspace(-2, 2, 1001)
        dy = derivative_smooth_step(x, x0=0.0, width=0.5)
        max_idx = dy.argmax()
        assert abs(x[max_idx].item()) < 0.02

    def test_non_negative(self):
        """Derivative should be non-negative."""
        x = torch.linspace(-10, 10, 1000)
        dy = derivative_smooth_step(x, x0=0.0, width=1.0)
        assert (dy >= -1e-6).all()

    def test_numerical_derivative(self):
        """Should match numerical derivative."""
        x = torch.tensor([0.5], requires_grad=True)
        y = smooth_step(x, x0=0.0, width=1.0)
        y.backward()
        analytical = derivative_smooth_step(torch.tensor([0.5]), x0=0.0, width=1.0)
        assert torch.isclose(x.grad, analytical, atol=1e-4)


# =============================================================================
# TCSProfiles Class Tests
# =============================================================================

class TestTCSProfiles:
    """Test TCSProfiles container class."""

    def test_default_creation(self):
        """Should create with default parameters."""
        profiles = TCSProfiles()
        assert profiles.lambda_L == 0.0
        assert profiles.lambda_R == 1.0
        assert profiles.lambda_neck == 0.5

    def test_custom_creation(self):
        """Should create with custom parameters."""
        profiles = TCSProfiles(
            lambda_L=-1.0,
            lambda_R=1.0,
            lambda_neck=0.0,
            sigma_L=0.2,
            sigma_R=0.2,
            sigma_neck=0.15,
        )
        assert profiles.lambda_L == -1.0
        assert profiles.sigma_neck == 0.15

    def test_left_method(self):
        """left() should return left plateau profile."""
        profiles = TCSProfiles()
        x = torch.tensor([-1.0])
        y = profiles.left(x)
        assert y.item() > 0.9

    def test_right_method(self):
        """right() should return right plateau profile."""
        profiles = TCSProfiles()
        x = torch.tensor([2.0])
        y = profiles.right(x)
        assert y.item() > 0.9

    def test_neck_method(self):
        """neck() should return neck bump profile."""
        profiles = TCSProfiles()
        x = torch.tensor([0.5])  # At lambda_neck
        y = profiles.neck(x)
        assert y.item() > 0.9

    def test_all_profiles(self):
        """all_profiles() should return tuple of 3 profiles."""
        profiles = TCSProfiles()
        x = torch.linspace(0, 1, 100)
        left, right, neck = profiles.all_profiles(x)
        assert left.shape == x.shape
        assert right.shape == x.shape
        assert neck.shape == x.shape

    def test_profile_derivatives(self):
        """profile_derivatives() should return tuple of 3 derivatives."""
        profiles = TCSProfiles()
        x = torch.linspace(0, 1, 100)
        d_left, d_right, d_neck = profiles.profile_derivatives(x)
        assert d_left.shape == x.shape
        assert d_right.shape == x.shape
        assert d_neck.shape == x.shape

    def test_from_domain(self):
        """from_domain() should create profiles from domain spec."""
        profiles = TCSProfiles.from_domain(
            domain=(-1, 1),
            neck_fraction=0.5,
            transition_width=0.1,
        )
        assert profiles.lambda_L == -1.0
        assert profiles.lambda_R == 1.0
        assert profiles.lambda_neck == 0.0  # center of domain

    def test_partition_of_unity_approximate(self):
        """Left + right should approximately partition unity away from neck."""
        profiles = TCSProfiles.from_domain(domain=(0, 1), transition_width=0.05)
        x_left = torch.tensor([0.1])
        x_right = torch.tensor([0.9])

        # Far from neck, left + right ~ 1
        sum_left = profiles.left(x_left) + profiles.right(x_left)
        sum_right = profiles.left(x_right) + profiles.right(x_right)

        assert abs(sum_left.item() - 1.0) < 0.2
        assert abs(sum_right.item() - 1.0) < 0.2
