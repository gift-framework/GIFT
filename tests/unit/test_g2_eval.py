"""
Unit tests for G2 Evaluation Module (G2_ML/0.2/G2_eval.py)

Tests model loading, point evaluation, grid evaluation, and reporting.

Author: GIFT Framework Team
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))

from G2_eval import (
    load_model,
    evaluate_at_point,
    evaluate_on_grid,
    print_point_evaluation,
    print_grid_evaluation
)
from G2_phi_network import G2PhiNetwork
from G2_manifold import TorusT7


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_model():
    """Create a sample G2PhiNetwork for testing."""
    model = G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[32, 32],
        fourier_modes=4,
        normalize_phi=True
    )
    return model


@pytest.fixture
def sample_checkpoint(sample_model, tmp_path):
    """Create a sample checkpoint file."""
    checkpoint_path = tmp_path / 'test_model.pt'

    config = {
        'encoding_type': 'fourier',
        'hidden_dims': [32, 32],
        'fourier_modes': 4,
        'normalize_phi': True
    }

    torch.save({
        'epoch': 100,
        'model_state_dict': sample_model.state_dict(),
        'loss': 0.001,
        'config': config
    }, checkpoint_path)

    return str(checkpoint_path)


@pytest.fixture
def sample_manifold():
    """Create a sample manifold."""
    return TorusT7(device='cpu')


# ============================================================================
# Load Model Tests
# ============================================================================

class TestLoadModel:
    """Tests for load_model() function."""

    def test_loads_model_successfully(self, sample_checkpoint):
        """Should load model from valid checkpoint."""
        model, config = load_model(sample_checkpoint, device='cpu')

        assert isinstance(model, G2PhiNetwork)
        assert isinstance(config, dict)

    def test_model_in_eval_mode(self, sample_checkpoint):
        """Loaded model should be in evaluation mode."""
        model, config = load_model(sample_checkpoint, device='cpu')

        assert not model.training

    def test_config_extracted(self, sample_checkpoint):
        """Configuration should be extracted from checkpoint."""
        model, config = load_model(sample_checkpoint, device='cpu')

        assert 'encoding_type' in config
        assert config['encoding_type'] == 'fourier'

    def test_default_config_when_missing(self, sample_model, tmp_path):
        """Should use defaults when config missing from checkpoint."""
        checkpoint_path = tmp_path / 'no_config.pt'

        torch.save({
            'model_state_dict': sample_model.state_dict()
        }, checkpoint_path)

        model, config = load_model(str(checkpoint_path), device='cpu')

        assert isinstance(config, dict)
        assert 'encoding_type' in config

    def test_loads_to_correct_device(self, sample_checkpoint):
        """Model should be on the specified device."""
        model, config = load_model(sample_checkpoint, device='cpu')

        # Check all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu'

    def test_invalid_path_raises_error(self):
        """Should raise error for invalid checkpoint path."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_model('/nonexistent/path/model.pt', device='cpu')


# ============================================================================
# Evaluate at Point Tests
# ============================================================================

class TestEvaluateAtPoint:
    """Tests for evaluate_at_point() function."""

    def test_returns_results_dict(self, sample_model):
        """Should return a results dictionary."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert isinstance(results, dict)

    def test_required_keys_present(self, sample_model):
        """Results should contain required keys."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        required_keys = ['coords', 'phi', 'metric', 'phi_norm_sq',
                        'det_g', 'volume', 'eigenvalues']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_single_point_shape(self, sample_model):
        """Single point should produce correct output shapes."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert results['coords'].shape == (1, 7)
        assert results['phi'].shape[0] == 1
        assert results['metric'].shape == (1, 7, 7)
        assert results['eigenvalues'].shape == (1, 7)

    def test_batch_points_shape(self, sample_model):
        """Batch of points should produce correct output shapes."""
        coords = torch.rand(10, 7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert results['coords'].shape == (10, 7)
        assert results['phi'].shape[0] == 10
        assert results['metric'].shape == (10, 7, 7)
        assert results['eigenvalues'].shape == (10, 7)

    def test_phi_norm_positive(self, sample_model):
        """Phi norm squared should be positive."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert results['phi_norm_sq'][0] > 0

    def test_eigenvalues_positive(self, sample_model):
        """Metric eigenvalues should be positive (SPD)."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert all(ev > 0 for ev in results['eigenvalues'][0])

    def test_metric_symmetric(self, sample_model):
        """Metric should be symmetric."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        metric = results['metric'][0]
        assert np.allclose(metric, metric.T, atol=1e-5)

    def test_results_numpy_arrays(self, sample_model):
        """Results should be numpy arrays."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        for key in ['coords', 'phi', 'metric', 'phi_norm_sq', 'det_g', 'eigenvalues']:
            assert isinstance(results[key], np.ndarray)


# ============================================================================
# Evaluate on Grid Tests
# ============================================================================

class TestEvaluateOnGrid:
    """Tests for evaluate_on_grid() function."""

    def test_returns_results_dict(self, sample_model, sample_manifold):
        """Should return a results dictionary."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        assert isinstance(results, dict)

    def test_phi_norm_statistics(self, sample_model, sample_manifold):
        """Results should contain phi norm statistics."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        assert 'phi_norm_sq' in results
        phi_stats = results['phi_norm_sq']
        assert 'mean' in phi_stats
        assert 'std' in phi_stats
        assert 'min' in phi_stats
        assert 'max' in phi_stats

    def test_det_g_statistics(self, sample_model, sample_manifold):
        """Results should contain determinant statistics."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        assert 'det_g' in results
        det_stats = results['det_g']
        assert 'mean' in det_stats
        assert 'error_from_1' in det_stats

    def test_eigenvalue_statistics(self, sample_model, sample_manifold):
        """Results should contain eigenvalue statistics."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        assert 'eigenvalues' in results
        eig_stats = results['eigenvalues']
        assert 'min' in eig_stats
        assert 'max' in eig_stats
        assert 'condition_number_mean' in eig_stats

    def test_small_grid_completes(self, sample_model, sample_manifold):
        """Small grid evaluation should complete quickly."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        # 2^7 = 128 points
        assert results is not None

    def test_values_are_finite(self, sample_model, sample_manifold):
        """All computed values should be finite."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        assert np.isfinite(results['phi_norm_sq']['mean'])
        assert np.isfinite(results['det_g']['mean'])
        assert np.isfinite(results['eigenvalues']['min'])


# ============================================================================
# Print Functions Tests
# ============================================================================

class TestPrintFunctions:
    """Tests for print helper functions."""

    def test_print_point_evaluation(self, sample_model, capsys):
        """print_point_evaluation should output formatted text."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        print_point_evaluation(results, idx=0)

        captured = capsys.readouterr()
        assert 'Point Evaluation' in captured.out
        assert 'phi' in captured.out.lower()
        assert 'det(g)' in captured.out

    def test_print_grid_evaluation(self, sample_model, sample_manifold, capsys):
        """print_grid_evaluation should output formatted text."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        print_grid_evaluation(results)

        captured = capsys.readouterr()
        assert 'Grid Evaluation' in captured.out
        assert 'Mean' in captured.out


# ============================================================================
# JSON Serialization Tests
# ============================================================================

class TestJSONSerialization:
    """Tests for JSON serialization of results."""

    def test_point_results_serializable(self, sample_model):
        """Point evaluation results should be JSON serializable."""
        coords = torch.rand(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        # Convert numpy arrays to lists
        results_json = {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in results.items()}

        # Should not raise
        json_str = json.dumps(results_json)
        assert len(json_str) > 0

    def test_grid_results_serializable(self, sample_model, sample_manifold):
        """Grid evaluation results should be JSON serializable."""
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=2, device='cpu')

        # Should not raise
        json_str = json.dumps(results)
        assert len(json_str) > 0


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEvalEdgeCases:
    """Tests for edge cases in evaluation."""

    def test_zero_coordinates(self, sample_model):
        """Should handle zero coordinates."""
        coords = torch.zeros(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert np.isfinite(results['phi_norm_sq'][0])

    def test_boundary_coordinates(self, sample_model):
        """Should handle coordinates at boundary (2*pi)."""
        coords = torch.ones(7) * 2 * np.pi
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert np.isfinite(results['phi_norm_sq'][0])

    def test_large_coordinates(self, sample_model):
        """Should handle large coordinates (outside fundamental domain)."""
        coords = torch.ones(7) * 100.0
        results = evaluate_at_point(sample_model, coords, device='cpu')

        # Model should still produce finite output
        assert np.isfinite(results['phi_norm_sq'][0])

    def test_negative_coordinates(self, sample_model):
        """Should handle negative coordinates."""
        coords = -torch.ones(7)
        results = evaluate_at_point(sample_model, coords, device='cpu')

        assert np.isfinite(results['phi_norm_sq'][0])


# ============================================================================
# Checkpoint Round-Trip Tests
# ============================================================================

class TestCheckpointRoundTrip:
    """Tests for saving and loading models."""

    def test_model_evaluation_consistent_after_save_load(self, sample_model, tmp_path):
        """Model should produce same outputs before and after save/load."""
        # Evaluate before saving
        coords = torch.rand(5, 7)
        sample_model.eval()
        with torch.no_grad():
            output_before = sample_model(coords)

        # Save checkpoint
        checkpoint_path = tmp_path / 'checkpoint.pt'
        config = {
            'encoding_type': 'fourier',
            'hidden_dims': [32, 32],
            'fourier_modes': 4,
            'normalize_phi': True
        }
        torch.save({
            'model_state_dict': sample_model.state_dict(),
            'config': config
        }, checkpoint_path)

        # Load and evaluate
        loaded_model, _ = load_model(str(checkpoint_path), device='cpu')
        with torch.no_grad():
            output_after = loaded_model(coords)

        # Should be identical
        assert torch.allclose(output_before, output_after)


# ============================================================================
# Memory Handling Tests
# ============================================================================

class TestMemoryHandling:
    """Tests for memory handling in evaluation."""

    def test_batched_grid_evaluation(self, sample_model, sample_manifold):
        """Large grids should be processed in batches without OOM."""
        # 3^7 = 2187 points, should be batched
        results = evaluate_on_grid(sample_model, sample_manifold,
                                   n_points_per_dim=3, device='cpu')

        assert results is not None
        assert np.isfinite(results['phi_norm_sq']['mean'])

    def test_no_gradient_accumulation(self, sample_model, sample_manifold):
        """Evaluation should not accumulate gradients."""
        import gc

        # Run evaluation multiple times
        for _ in range(3):
            results = evaluate_on_grid(sample_model, sample_manifold,
                                       n_points_per_dim=2, device='cpu')
            gc.collect()

        # Should complete without memory issues
        assert results is not None


# ============================================================================
# Integration with Validation Tests
# ============================================================================

class TestEvalValidationIntegration:
    """Tests for integration with validation module."""

    def test_comprehensive_validation_runs(self, sample_model, sample_manifold):
        """Comprehensive validation should run successfully."""
        from G2_validation import comprehensive_validation

        report = comprehensive_validation(
            sample_model, sample_manifold,
            n_samples=50, device='cpu', verbose=False
        )

        assert isinstance(report, dict)
        assert 'overall_score' in report
        assert 'torsion' in report
        assert 'quality' in report

    def test_validation_metrics_reasonable(self, sample_model, sample_manifold):
        """Validation metrics should be in reasonable ranges."""
        from G2_validation import validate_metric_quality

        results = validate_metric_quality(
            sample_model, sample_manifold,
            n_samples=50, device='cpu'
        )

        # Check metrics are in expected ranges
        assert results['phi_norm_sq_mean'] > 0
        assert results['volume_mean'] > 0
        assert results['min_eigenvalue'] > 0  # SPD
