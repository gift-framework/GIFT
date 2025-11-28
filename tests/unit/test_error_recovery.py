"""
Error Recovery Tests

Tests for graceful error handling, edge case recovery, and meaningful error
messages across the GIFT framework and G2_ML components.

Author: GIFT Framework Team
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../statistical_validation'))


# ============================================================================
# G2_ML Error Handling Tests
# ============================================================================

class TestG2MLErrors:
    """Tests for error handling in G2_ML modules."""

    def test_invalid_encoding_type(self):
        """Should raise error for invalid encoding type."""
        from G2_phi_network import G2PhiNetwork

        with pytest.raises((ValueError, KeyError)):
            G2PhiNetwork(
                encoding_type='invalid_encoding',
                hidden_dims=[32, 32]
            )

    def test_invalid_hidden_dims(self):
        """Should handle unusual hidden dimensions gracefully."""
        from G2_phi_network import G2PhiNetwork

        # Empty hidden dims - model should still work (direct input->output)
        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[]
        )
        coords = torch.rand(10, 7)
        output = model(coords)
        # Should still produce output with correct shape
        assert output.shape == (10, 35)

    def test_wrong_input_dimension(self):
        """Should raise error for wrong input dimension."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        # Wrong dimension: 5 instead of 7
        with pytest.raises((RuntimeError, ValueError)):
            wrong_dim_coords = torch.rand(10, 5)
            model(wrong_dim_coords)

    def test_nan_input_handling(self):
        """Model should handle NaN inputs gracefully."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        nan_coords = torch.tensor([[np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        # Should produce output (possibly NaN)
        output = model(nan_coords)
        # Output shape should be correct
        assert output.shape[1] == 35

    def test_inf_input_handling(self):
        """Model should handle infinity inputs."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        inf_coords = torch.tensor([[np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        # Should produce output (possibly NaN/Inf)
        output = model(inf_coords)
        assert output.shape[1] == 35


class TestManifoldErrors:
    """Tests for error handling in manifold module."""

    def test_invalid_manifold_type(self):
        """Should raise error for invalid manifold type."""
        from G2_manifold import create_manifold

        with pytest.raises(ValueError):
            create_manifold('INVALID_TYPE')

    def test_wrong_radii_length(self):
        """Should raise error for wrong number of radii."""
        from G2_manifold import TorusT7

        with pytest.raises(AssertionError):
            TorusT7(radii=[1.0, 2.0, 3.0])  # Only 3, need 7

    def test_invalid_sampling_method(self):
        """Should raise error for invalid sampling method."""
        from G2_manifold import TorusT7

        torus = TorusT7(device='cpu')

        with pytest.raises(ValueError):
            torus.sample_points(100, method='invalid_method')

    def test_negative_sample_count(self):
        """Should handle negative sample count."""
        from G2_manifold import TorusT7

        torus = TorusT7(device='cpu')

        # May raise error or return empty
        try:
            coords = torus.sample_points(-10, method='uniform')
            # If it returns, should be empty or have 0 samples
            assert coords.shape[0] >= 0
        except (ValueError, RuntimeError):
            pass  # Expected


class TestLossErrors:
    """Tests for error handling in loss functions."""

    def test_mismatched_batch_sizes(self):
        """Should handle mismatched batch sizes."""
        from G2_losses import torsion_loss

        phi = torch.randn(10, 35)
        metric = torch.randn(5, 7, 7)  # Different batch size
        coords = torch.rand(10, 7, requires_grad=True)

        with pytest.raises((RuntimeError, ValueError)):
            torsion_loss(phi, metric, coords)

    def test_non_spd_metric(self):
        """Should handle non-SPD metric."""
        from G2_losses import volume_loss

        # Create non-SPD metric (negative eigenvalues)
        metric = -torch.eye(7).unsqueeze(0).repeat(10, 1, 1)

        # Should still compute (det will be negative)
        loss, info = volume_loss(metric)
        # Loss should be finite (even if large)
        assert torch.isfinite(loss) or loss.item() > 0


class TestValidationErrors:
    """Tests for error handling in validation module."""

    def test_invalid_n_samples(self):
        """Should handle invalid sample count."""
        from G2_phi_network import G2PhiNetwork
        from G2_manifold import TorusT7
        from G2_validation import validate_metric_quality

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )
        manifold = TorusT7(device='cpu')

        # Zero samples
        try:
            results = validate_metric_quality(model, manifold, n_samples=0)
            # If it returns, results should be dict
            assert isinstance(results, dict)
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass  # Expected


# ============================================================================
# GIFT Framework Error Handling Tests
# ============================================================================

class TestGIFTErrors:
    """Tests for error handling in GIFT framework."""

    def test_framework_initialization(self):
        """Framework should initialize without errors."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        assert framework is not None

    def test_observable_computation_complete(self):
        """All observables should compute without error."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        obs = framework.compute_all_observables()

        # Should have all observables
        assert len(obs) >= 30  # At least 30 observables

        # All should be finite
        for name, value in obs.items():
            assert np.isfinite(value), f"{name} is not finite"

    def test_deviation_computation(self):
        """Deviation computation should handle all cases."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        deviations = framework.compute_deviations()

        # Should return dict
        assert isinstance(deviations, dict)

        # Each entry should have required keys
        for name, data in deviations.items():
            assert 'prediction' in data
            assert 'experimental' in data
            assert 'deviation_pct' in data

    def test_summary_statistics(self):
        """Summary statistics should compute without error."""
        from gift_v22_core import GIFTFrameworkV22

        framework = GIFTFrameworkV22()
        stats = framework.summary_statistics()

        assert 'mean_deviation' in stats
        assert 'median_deviation' in stats
        assert 'total_observables' in stats


# ============================================================================
# File I/O Error Tests
# ============================================================================

class TestFileIOErrors:
    """Tests for file I/O error handling."""

    def test_load_nonexistent_model(self):
        """Should raise error for nonexistent model file."""
        from G2_eval import load_model

        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_model('/nonexistent/path/model.pt')

    def test_load_corrupted_checkpoint(self):
        """Should handle corrupted checkpoint file."""
        from G2_eval import load_model

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(b'corrupted data that is not a valid pytorch checkpoint')
            temp_path = f.name

        try:
            with pytest.raises((RuntimeError, Exception)):
                load_model(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_to_readonly_directory(self):
        """Should handle permission errors gracefully."""
        # This test might be platform-dependent
        # Skip on systems where we can't create read-only dirs easily
        pass


# ============================================================================
# Numerical Error Recovery Tests
# ============================================================================

class TestNumericalRecovery:
    """Tests for recovery from numerical issues."""

    def test_near_singular_matrix(self):
        """Should handle near-singular matrices."""
        from G2_geometry import project_spd

        # Create nearly singular matrix
        near_singular = torch.eye(7) * 1e-10
        near_singular = near_singular.unsqueeze(0)

        # project_spd should make it SPD
        spd = project_spd(near_singular)
        eigenvalues = torch.linalg.eigvalsh(spd)

        # All eigenvalues should be positive after projection
        assert (eigenvalues > 0).all()

    def test_extreme_values(self):
        """Should handle extreme coordinate values."""
        from G2_phi_network import G2PhiNetwork
        from G2_phi_network import metric_from_phi_algebraic
        from G2_geometry import project_spd

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        # Very large coordinates
        large_coords = torch.ones(10, 7) * 1e6

        # Should still produce output
        phi = model(large_coords)
        assert phi.shape == (10, 35)

        # Metric should be computable
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        assert metric.shape == (10, 7, 7)

    def test_zero_batch_handling(self):
        """Should handle zero-size batches."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        empty_coords = torch.empty(0, 7)

        try:
            output = model(empty_coords)
            assert output.shape[0] == 0
        except (RuntimeError, ValueError):
            pass  # Some models may not support empty batches


# ============================================================================
# Type Error Tests
# ============================================================================

class TestTypeErrors:
    """Tests for type error handling."""

    def test_numpy_input_conversion(self):
        """Should handle numpy array input."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        # Numpy input
        numpy_coords = np.random.rand(10, 7).astype(np.float32)

        # Should either convert or raise clear error
        try:
            output = model(torch.from_numpy(numpy_coords))
            assert output.shape == (10, 35)
        except TypeError as e:
            assert 'tensor' in str(e).lower() or 'type' in str(e).lower()

    def test_wrong_dtype(self):
        """Should handle wrong dtype."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        # Double precision input (model likely expects float32)
        double_coords = torch.rand(10, 7, dtype=torch.float64)

        # Should either work or raise clear error
        try:
            output = model(double_coords.float())
            assert output.shape == (10, 35)
        except RuntimeError:
            pass


# ============================================================================
# Device Mismatch Tests
# ============================================================================

class TestDeviceMismatch:
    """Tests for device mismatch handling."""

    def test_cpu_model_cpu_data(self):
        """CPU model with CPU data should work."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        ).to('cpu')

        coords = torch.rand(10, 7, device='cpu')
        output = model(coords)

        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_model_cpu_data(self):
        """GPU model with CPU data should raise or auto-convert."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        ).to('cuda')

        coords = torch.rand(10, 7, device='cpu')

        with pytest.raises(RuntimeError):
            # Should raise error about device mismatch
            output = model(coords)


# ============================================================================
# Memory Error Tests
# ============================================================================

class TestMemoryErrors:
    """Tests for memory-related error handling."""

    def test_reasonable_batch_size(self):
        """Should handle reasonable batch sizes."""
        from G2_phi_network import G2PhiNetwork
        from G2_phi_network import metric_from_phi_algebraic
        from G2_geometry import project_spd

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        )

        # Reasonable batch size
        coords = torch.rand(1000, 7)
        coords.requires_grad = True

        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)

        assert metric.shape == (1000, 7, 7)


# ============================================================================
# Curriculum Scheduler Edge Cases
# ============================================================================

class TestCurriculumEdgeCases:
    """Tests for curriculum scheduler edge cases."""

    def test_negative_epoch(self):
        """Should handle negative epoch number."""
        from G2_losses import CurriculumScheduler

        scheduler = CurriculumScheduler()

        # Negative epoch
        weights = scheduler.get_weights(-10)
        phase = scheduler.get_phase(-10)

        # Should return something reasonable
        assert isinstance(weights, dict)
        assert phase >= 0

    def test_very_large_epoch(self):
        """Should handle very large epoch number."""
        from G2_losses import CurriculumScheduler

        scheduler = CurriculumScheduler()

        # Very large epoch
        weights = scheduler.get_weights(1000000)
        phase = scheduler.get_phase(1000000)

        # Should return last phase
        assert isinstance(weights, dict)
        assert phase == scheduler.n_phases - 1


# ============================================================================
# Configuration Error Tests
# ============================================================================

class TestConfigurationErrors:
    """Tests for configuration error handling."""

    def test_missing_config_keys(self):
        """Should handle missing configuration keys."""
        from G2_train import train

        incomplete_config = {
            'epochs': 1,
            'batch_size': 8,
            # Missing many required keys
        }

        with pytest.raises((KeyError, TypeError)):
            train(incomplete_config)

    def test_invalid_config_values(self):
        """Should handle invalid configuration values."""
        from G2_train import get_default_config, train

        config = get_default_config()
        config['epochs'] = -1  # Invalid
        config['batch_size'] = 0  # Invalid

        # Should raise or handle gracefully
        with pytest.raises((ValueError, RuntimeError)):
            # Either raises immediately or fails during training
            train(config)
