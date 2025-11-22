"""Tests for topological constants."""
import pytest
from giftpy.core.constants import CONSTANTS, TopologicalConstants


class TestTopologicalConstants:
    """Test topological constants and their properties."""

    def test_primary_parameters(self):
        """Test primary topological parameters."""
        assert CONSTANTS.p2 == 2
        assert CONSTANTS.rank_E8 == 8
        assert CONSTANTS.Weyl_factor == 5

    def test_dimensions(self):
        """Test derived dimensions."""
        assert CONSTANTS.dim_E8 == 248
        assert CONSTANTS.dim_E8xE8 == 496
        assert CONSTANTS.dim_K7 == 7
        assert CONSTANTS.dim_G2 == 14
        assert CONSTANTS.dim_J3 == 27

    def test_betti_numbers(self):
        """Test Betti numbers of K₇."""
        assert CONSTANTS.b0 == 1
        assert CONSTANTS.b1 == 0
        assert CONSTANTS.b2 == 21
        assert CONSTANTS.b3 == 77
        assert CONSTANTS.b4 == 77
        assert CONSTANTS.b5 == 21
        assert CONSTANTS.b6 == 0
        assert CONSTANTS.b7 == 1

    def test_betti_constraint(self):
        """Test topological constraint: b₂ + b₃ = 2×7²."""
        assert CONSTANTS.b2 + CONSTANTS.b3 == 2 * CONSTANTS.dim_K7**2

    def test_total_cohomology(self):
        """Test H*(K₇) = 99."""
        assert CONSTANTS.H_star == 99
        # H* counts unique cohomology classes (Poincaré duality means we don't double-count)
        # H* = 1 + 0 + 21 + 77 = 99 (only up to middle dimension)
        total_unique = CONSTANTS.b0 + CONSTANTS.b1 + CONSTANTS.b2 + CONSTANTS.b3
        assert total_unique == 99

        # Full sum with Poincaré duality gives 198
        total_all = sum([CONSTANTS.b0, CONSTANTS.b1, CONSTANTS.b2, CONSTANTS.b3,
                         CONSTANTS.b4, CONSTANTS.b5, CONSTANTS.b6, CONSTANTS.b7])
        assert total_all == 198  # 2 × 99 due to duality

    def test_euler_characteristic(self):
        """Test Euler characteristic χ(K₇) = 0."""
        assert CONSTANTS.chi_K7 == 0

    def test_poincare_duality(self):
        """Test Poincaré duality for K₇."""
        assert CONSTANTS.b4 == CONSTANTS.b3  # b₄ = b₃
        assert CONSTANTS.b5 == CONSTANTS.b2  # b₅ = b₂

    def test_mathematical_constants(self):
        """Test mathematical constants precision."""
        # Golden ratio
        assert abs(CONSTANTS.phi - 1.618033988749895) < 1e-10

        # Square roots
        import numpy as np

        assert abs(CONSTANTS.sqrt2 - np.sqrt(2)) < 1e-14
        assert abs(CONSTANTS.sqrt5 - np.sqrt(5)) < 1e-14
        assert abs(CONSTANTS.sqrt17 - np.sqrt(17)) < 1e-14

        # ln(2)
        assert abs(CONSTANTS.ln2 - np.log(2)) < 1e-14

        # ζ(3) - Apéry's constant
        assert abs(CONSTANTS.zeta3 - 1.2020569) < 1e-6

    def test_gift_parameters(self):
        """Test GIFT-specific parameters."""
        # β₀ = b₂/b₃
        expected_beta0 = CONSTANTS.b2 / CONSTANTS.b3
        assert abs(CONSTANTS.beta0 - expected_beta0) < 1e-10

        # ξ = (5/2)β₀
        expected_xi = (5 / 2) * CONSTANTS.beta0
        assert abs(CONSTANTS.xi - expected_xi) < 1e-10

        # ε₀ = 1/8
        assert CONSTANTS.epsilon0 == 1 / 8

        # N_gen = 3
        assert CONSTANTS.N_gen == 3

    def test_tau_parameter(self):
        """Test temporal parameter τ."""
        assert abs(CONSTANTS.tau - 10416 / 2673) < 1e-10

    def test_delta_parameter(self):
        """Test δ = √5 - ζ(3)."""
        import numpy as np

        expected = np.sqrt(5) - CONSTANTS.zeta3
        assert abs(CONSTANTS.delta - expected) < 1e-10

    def test_verify_constraints(self):
        """Test topological constraint verification."""
        # Should not raise
        assert CONSTANTS.verify_topological_constraints() == True

    def test_custom_constants(self):
        """Test creating custom constants (for research)."""
        custom = TopologicalConstants(p2=2, rank_E8=8, Weyl_factor=5)
        assert custom.b2 == 21
        assert custom.b3 == 77
        assert custom.verify_topological_constraints() == True

    def test_immutability(self):
        """Test that constants are immutable (frozen dataclass)."""
        with pytest.raises(Exception):  # FrozenInstanceError
            CONSTANTS.b2 = 999  # Should fail

    def test_summary(self):
        """Test summary generation."""
        summary = CONSTANTS.summary()
        assert isinstance(summary, str)
        assert "GIFT Topological Constants" in summary
        assert "b₂ = 21" in summary
        assert "b₃ = 77" in summary
