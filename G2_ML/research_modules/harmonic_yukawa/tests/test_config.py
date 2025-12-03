"""
Unit tests for harmonic_yukawa configuration.

Tests:
- HarmonicConfig dataclass
- GIFT topological constants
- Default configuration values
"""

import pytest
import sys
from pathlib import Path

# Add harmonic_yukawa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import HarmonicConfig, default_harmonic_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    CONFIG_AVAILABLE = False
    CONFIG_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason=f"config module not available: {CONFIG_IMPORT_ERROR if not CONFIG_AVAILABLE else ''}"
)


# =============================================================================
# HarmonicConfig Tests
# =============================================================================

class TestHarmonicConfig:
    """Test HarmonicConfig dataclass."""

    def test_default_config_exists(self):
        """default_harmonic_config should exist."""
        assert default_harmonic_config is not None
        assert isinstance(default_harmonic_config, HarmonicConfig)

    def test_custom_config_creation(self):
        """Should be able to create custom config."""
        config = HarmonicConfig(
            n_sample_points=5000,
            harmonic_epochs=1000,
        )
        assert config.n_sample_points == 5000
        assert config.harmonic_epochs == 1000


# =============================================================================
# Topological Constants (GIFT v2.2)
# =============================================================================

class TestTopologicalConstants:
    """Test GIFT topological constants are correct."""

    def test_b2_K7(self):
        """b2(K7) = 21 for Joyce manifold."""
        config = default_harmonic_config
        assert config.b2 == 21

    def test_b3_K7(self):
        """b3(K7) = 77 for Joyce manifold."""
        config = default_harmonic_config
        assert config.b3 == 77

    def test_b3_decomposition(self):
        """b3 = b3_local + b3_global = 35 + 42."""
        config = default_harmonic_config
        assert config.b3_local == 35
        assert config.b3_global == 42
        assert config.b3 == config.b3_local + config.b3_global

    def test_det_g_target(self):
        """det(g) = 65/32 = 2.03125 (PROVEN)."""
        config = default_harmonic_config
        assert config.det_g_target == 65/32
        assert abs(config.det_g_target - 2.03125) < 1e-10

    def test_kappa_T(self):
        """kappa_T = 1/61 (torsion magnitude)."""
        config = default_harmonic_config
        assert config.kappa_T == 1/61

    def test_dim_K7(self):
        """K7 is 7-dimensional."""
        config = default_harmonic_config
        assert config.dim_K7 == 7


# =============================================================================
# Component Dimensions
# =============================================================================

class TestComponentDimensions:
    """Test differential form component counts."""

    def test_2form_components(self):
        """2-form has C(7,2) = 21 components."""
        config = default_harmonic_config
        assert config.dim_2form == 21

    def test_3form_components(self):
        """3-form has C(7,3) = 35 components."""
        config = default_harmonic_config
        assert config.dim_3form == 35

    def test_binomial_coefficients(self):
        """Verify binomial coefficients."""
        from math import comb
        config = default_harmonic_config

        assert config.dim_2form == comb(7, 2)
        assert config.dim_3form == comb(7, 3)


# =============================================================================
# Derived Properties
# =============================================================================

class TestDerivedProperties:
    """Test computed properties."""

    def test_total_harmonic_dim(self):
        """total_harmonic_dim = b2 + b3 = 98."""
        config = default_harmonic_config
        assert config.total_harmonic_dim == 21 + 77
        assert config.total_harmonic_dim == 98

    def test_h_star(self):
        """h* = b2 + b3 + 1 = 99."""
        config = default_harmonic_config
        assert config.h_star == 99


# =============================================================================
# Training Parameters
# =============================================================================

class TestTrainingParameters:
    """Test training-related parameters."""

    def test_sample_points_positive(self):
        """n_sample_points should be positive."""
        config = default_harmonic_config
        assert config.n_sample_points > 0

    def test_epochs_positive(self):
        """harmonic_epochs should be positive."""
        config = default_harmonic_config
        assert config.harmonic_epochs > 0

    def test_learning_rate_positive(self):
        """harmonic_lr should be positive."""
        config = default_harmonic_config
        assert config.harmonic_lr > 0

    def test_loss_weights_positive(self):
        """Loss weights should be positive."""
        config = default_harmonic_config
        assert config.orthonormality_weight > 0
        assert config.closedness_weight > 0
        assert config.coclosedness_weight > 0

    def test_eps_small_positive(self):
        """eps should be small positive number."""
        config = default_harmonic_config
        assert config.eps > 0
        assert config.eps < 1e-6


# =============================================================================
# Yukawa Parameters
# =============================================================================

class TestYukawaParameters:
    """Test Yukawa computation parameters."""

    def test_yukawa_samples_positive(self):
        """n_yukawa_samples should be positive."""
        config = default_harmonic_config
        assert config.n_yukawa_samples > 0

    def test_yukawa_batch_size_positive(self):
        """yukawa_batch_size should be positive."""
        config = default_harmonic_config
        assert config.yukawa_batch_size > 0

    def test_yukawa_batch_size_reasonable(self):
        """yukawa_batch_size should be <= n_yukawa_samples."""
        config = default_harmonic_config
        assert config.yukawa_batch_size <= config.n_yukawa_samples


# =============================================================================
# Edge Cases
# =============================================================================

class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_minimal_config(self):
        """Config with minimal values should work."""
        config = HarmonicConfig(
            n_sample_points=10,
            n_yukawa_samples=10,
            yukawa_batch_size=5,
            harmonic_epochs=1,
        )
        assert config.n_sample_points == 10

    def test_config_immutable_defaults(self):
        """Modifying one config shouldn't affect default."""
        config = HarmonicConfig(n_sample_points=999)

        # Default should be unchanged
        assert default_harmonic_config.n_sample_points != 999

    def test_all_defaults_set(self):
        """All fields should have default values."""
        config = HarmonicConfig()

        # Should not raise AttributeError
        assert config.b2 is not None
        assert config.b3 is not None
        assert config.det_g_target is not None
        assert config.n_sample_points is not None
