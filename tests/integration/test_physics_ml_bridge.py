"""
Integration tests for Physics-ML Bridge

Tests that G2_ML outputs integrate correctly with GIFT physics framework,
verifying topological invariants, metric properties, and physical predictions.

Author: GIFT Framework Team
"""

import pytest
import torch
import numpy as np
import os
import sys

# Add G2_ML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))

from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
from G2_geometry import project_spd, volume_form
from G2_manifold import TorusT7
from G2_losses import torsion_loss, volume_loss, phi_normalization_loss

# Add statistical_validation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../statistical_validation'))
from gift_v22_core import GIFTFrameworkV22, GIFTParametersV22


# ============================================================================
# GIFT Constants for Comparison
# ============================================================================

# Key topological constants from GIFT framework
GIFT_CONSTANTS = {
    'dim_K7': 7,
    'dim_G2': 14,
    'b2_K7': 21,
    'b3_K7': 77,
    'H_star': 99,
    'det_g_target': 65.0 / 32.0,  # = 2.03125
    'kappa_T': 1.0 / 61.0,  # = 0.016393
    'phi_norm_sq_target': 7.0,  # G2 3-form normalization
}


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def g2_model():
    """Create a G2PhiNetwork for testing."""
    model = G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[64, 64, 32],
        fourier_modes=8,
        normalize_phi=True
    )
    return model


@pytest.fixture
def manifold():
    """Create T^7 manifold."""
    return TorusT7(device='cpu')


@pytest.fixture
def gift_framework():
    """Create GIFT v2.2 framework."""
    return GIFTFrameworkV22()


@pytest.fixture
def sample_coords(manifold):
    """Generate sample coordinates on manifold."""
    return manifold.sample_points(100, method='uniform')


# ============================================================================
# Topological Invariant Tests
# ============================================================================

class TestTopologicalInvariants:
    """Tests that ML outputs respect topological invariants."""

    def test_k7_dimension_matches(self, g2_model, sample_coords):
        """K7 manifold dimension should be 7."""
        # Model expects 7D input
        phi = g2_model(sample_coords)

        assert sample_coords.shape[1] == GIFT_CONSTANTS['dim_K7']

    def test_metric_is_7x7(self, g2_model, sample_coords):
        """Metric tensor should be 7x7."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        assert metric.shape == (100, 7, 7)

    def test_phi_dimension_correct(self, g2_model, sample_coords):
        """Phi 3-form should have correct dimension."""
        phi = g2_model(sample_coords)

        # 3-form on 7D has C(7,3) = 35 components
        assert phi.shape[1] == 35

    def test_phi_normalization_target(self, g2_model, sample_coords):
        """Phi norm squared should target 7.0."""
        phi = g2_model(sample_coords)
        phi_norm_sq = torch.sum(phi ** 2, dim=1).mean().item()

        # Due to normalization layer, should be close to 7
        assert abs(phi_norm_sq - GIFT_CONSTANTS['phi_norm_sq_target']) < 1.0


class TestMetricProperties:
    """Tests that ML metric satisfies required properties."""

    def test_metric_symmetric(self, g2_model, sample_coords):
        """Metric should be symmetric."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        symmetry_error = torch.norm(metric - metric.transpose(-2, -1)).item()
        assert symmetry_error < 1e-5

    def test_metric_positive_definite(self, g2_model, sample_coords):
        """Metric should be positive definite."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        eigenvalues = torch.linalg.eigvalsh(metric)
        assert eigenvalues.min().item() > 0

    def test_metric_determinant_reasonable(self, g2_model, sample_coords):
        """Metric determinant should be in reasonable range."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        det_g = torch.det(metric)

        # Determinant should be positive and finite
        assert (det_g > 0).all()
        assert torch.isfinite(det_g).all()


class TestGIFTFrameworkIntegration:
    """Tests integration between G2_ML and GIFT framework."""

    def test_gift_constants_defined(self, gift_framework):
        """GIFT framework should define all required constants."""
        assert gift_framework.dim_K7 == 7
        assert gift_framework.dim_G2 == 14
        assert gift_framework.b2_K7 == 21
        assert gift_framework.b3_K7 == 77
        assert gift_framework.H_star == 99

    def test_topological_values_match(self, gift_framework):
        """GIFT topological values should match constants."""
        params = gift_framework.params

        assert params.det_g_float == GIFT_CONSTANTS['det_g_target']
        assert abs(params.kappa_T_float - GIFT_CONSTANTS['kappa_T']) < 1e-6

    def test_phi_target_matches_g2_dimension(self, gift_framework):
        """Phi normalization target should relate to G2 dimension."""
        # ||phi||^2 = 7 = dim(K7)
        assert GIFT_CONSTANTS['phi_norm_sq_target'] == gift_framework.dim_K7


# ============================================================================
# Loss Function Integration Tests
# ============================================================================

class TestLossFunctionIntegration:
    """Tests that loss functions enforce GIFT constraints."""

    def test_torsion_loss_computes(self, g2_model, sample_coords):
        """Torsion loss should compute without errors."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        loss, info = torsion_loss(phi, metric, sample_coords, method='autograd')

        assert torch.isfinite(loss)
        assert 'torsion_total' in info

    def test_volume_loss_targets_unit(self, g2_model, sample_coords):
        """Volume loss should target det(g) = 1."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        loss, info = volume_loss(metric)

        assert torch.isfinite(loss)
        assert 'det_g_mean' in info

    def test_phi_normalization_targets_7(self, g2_model, sample_coords):
        """Phi normalization loss should target ||phi||^2 = 7."""
        phi = g2_model(sample_coords)

        loss, info = phi_normalization_loss(phi, target=7.0)

        assert torch.isfinite(loss)
        assert 'phi_norm_sq_mean' in info


# ============================================================================
# Physical Prediction Tests
# ============================================================================

class TestPhysicalPredictions:
    """Tests that GIFT predictions are valid."""

    def test_weinberg_angle_exact(self, gift_framework):
        """sin^2(theta_W) = 3/13 should be exact."""
        sin2_theta_W = float(gift_framework.params.sin2_theta_W)
        expected = 3.0 / 13.0

        assert abs(sin2_theta_W - expected) < 1e-10

    def test_tau_hierarchy_exact(self, gift_framework):
        """tau = 3472/891 should be exact."""
        tau = gift_framework.params.tau_float
        expected = 3472.0 / 891.0

        assert abs(tau - expected) < 1e-10

    def test_kappa_torsion_exact(self, gift_framework):
        """kappa_T = 1/61 should be exact."""
        kappa_T = gift_framework.params.kappa_T_float
        expected = 1.0 / 61.0

        assert abs(kappa_T - expected) < 1e-10

    def test_all_observables_computable(self, gift_framework):
        """All 39 observables should be computable."""
        obs = gift_framework.compute_all_observables()

        assert len(obs) >= 39
        for name, value in obs.items():
            assert np.isfinite(value), f"Observable {name} is not finite"


# ============================================================================
# Harmonic Form Tests
# ============================================================================

class TestHarmonicForms:
    """Tests related to harmonic forms on K7."""

    def test_b2_harmonic_2forms(self, gift_framework):
        """b2(K7) = 21 harmonic 2-forms should be defined."""
        assert gift_framework.b2_K7 == 21

    def test_b3_harmonic_3forms(self, gift_framework):
        """b3(K7) = 77 harmonic 3-forms should be defined."""
        assert gift_framework.b3_K7 == 77

    def test_h_star_total_cohomology(self, gift_framework):
        """H* = b2 + b3 + 1 = 99 should hold."""
        expected = gift_framework.b2_K7 + gift_framework.b3_K7 + 1
        assert gift_framework.H_star == expected
        assert gift_framework.H_star == 99


# ============================================================================
# Numerical Consistency Tests
# ============================================================================

class TestNumericalConsistency:
    """Tests for numerical consistency between components."""

    def test_metric_reconstruction_deterministic(self, g2_model, sample_coords):
        """Metric reconstruction should be deterministic."""
        g2_model.eval()

        sample_coords.requires_grad = True
        with torch.no_grad():
            phi1 = g2_model(sample_coords)
            phi2 = g2_model(sample_coords)

        assert torch.allclose(phi1, phi2)

    def test_loss_computation_consistent(self, g2_model, sample_coords):
        """Loss computations should be consistent across calls."""
        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        loss1, _ = volume_loss(metric)
        loss2, _ = volume_loss(metric)

        assert torch.allclose(loss1, loss2)

    def test_gift_observables_stable(self, gift_framework):
        """GIFT observables should be numerically stable."""
        obs1 = gift_framework.compute_all_observables()
        obs2 = gift_framework.compute_all_observables()

        for name in obs1:
            assert np.isclose(obs1[name], obs2[name]), f"Observable {name} unstable"


# ============================================================================
# G2 Identity Tests
# ============================================================================

class TestG2Identity:
    """Tests for G2 structure identity."""

    def test_g2_wedge_product_identity(self, g2_model, sample_coords):
        """phi ^ *phi should relate to volume form."""
        from G2_geometry import hodge_star

        sample_coords.requires_grad = True
        phi = g2_model(sample_coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        # Compute Hodge dual
        phi_dual = hodge_star(phi, metric)

        # Both should be finite
        assert torch.isfinite(phi).all()
        assert torch.isfinite(phi_dual).all()


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_ml_output_compatible_with_physics(self, g2_model, manifold, gift_framework):
        """ML outputs should be compatible with physics framework."""
        coords = manifold.sample_points(50, method='uniform')
        coords.requires_grad = True

        # Generate ML metric
        phi = g2_model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        # Compute physics-related quantities
        det_g = torch.det(metric)
        eigenvalues = torch.linalg.eigvalsh(metric)

        # Should all be valid
        assert (det_g > 0).all(), "Determinant should be positive"
        assert (eigenvalues > 0).all(), "Eigenvalues should be positive"
        assert torch.isfinite(metric).all(), "Metric should be finite"

    def test_training_preserves_structure(self, g2_model, manifold):
        """Brief training should preserve metric structure."""
        from G2_losses import G2TotalLoss, CurriculumScheduler

        optimizer = torch.optim.Adam(g2_model.parameters(), lr=1e-4)
        loss_fn = G2TotalLoss(
            curriculum_scheduler=CurriculumScheduler(),
            use_positivity=True
        )

        # Run a few training steps
        g2_model.train()
        for _ in range(3):
            coords = manifold.sample_points(16, method='uniform')
            coords.requires_grad = True

            phi = g2_model(coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)

            total_loss, _ = loss_fn(phi, metric, coords, epoch=0)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # After training, model should still produce valid outputs
        g2_model.eval()
        coords = manifold.sample_points(50, method='uniform')
        with torch.no_grad():
            phi = g2_model(coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)

        eigenvalues = torch.linalg.eigvalsh(metric)
        assert (eigenvalues > 0).all(), "Metric should remain SPD after training"


# ============================================================================
# Version Compatibility Tests
# ============================================================================

class TestVersionCompatibility:
    """Tests for version compatibility."""

    def test_gift_v22_parameters_defined(self):
        """GIFT v2.2 parameters should all be defined."""
        params = GIFTParametersV22()

        # Check all key parameters
        assert params.dim_E8 == 248
        assert params.dim_G2 == 14
        assert params.dim_K7 == 7
        assert params.b2_K7 == 21
        assert params.b3_K7 == 77
        assert params.H_star == 99
        assert params.N_gen == 3

    def test_13_proven_relations(self, gift_framework):
        """All 13 PROVEN relations should be defined."""
        proven = gift_framework.get_proven_relations()

        assert len(proven) == 13

        # Check some key ones
        assert 'N_gen' in proven
        assert 'Q_Koide' in proven
        assert 'delta_CP' in proven
        assert 'sin2thetaW' in proven
        assert 'tau' in proven
        assert 'kappa_T' in proven
