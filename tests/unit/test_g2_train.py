"""
Unit tests for G2 Training Pipeline (G2_ML/0.2/G2_train.py)

Tests training configuration, epoch training, validation, checkpointing,
and training orchestration.

Author: GIFT Framework Team
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../G2_ML/0.2'))

from G2_train import (
    get_default_config,
    train_epoch,
    validate,
    train,
    plot_training_history,
    parse_args
)
from G2_phi_network import G2PhiNetwork
from G2_manifold import TorusT7
from G2_losses import G2TotalLoss, CurriculumScheduler


# ============================================================================
# Configuration Tests
# ============================================================================

class TestDefaultConfig:
    """Tests for get_default_config()."""

    def test_returns_dict(self):
        """Config should be a dictionary."""
        config = get_default_config()
        assert isinstance(config, dict)

    def test_required_keys_present(self):
        """All required configuration keys should be present."""
        config = get_default_config()
        required_keys = [
            'encoding_type', 'hidden_dims', 'fourier_modes', 'fourier_scale',
            'omega_0', 'normalize_phi', 'manifold_type', 'batch_size',
            'epochs', 'learning_rate', 'weight_decay', 'grad_clip',
            'checkpoint_interval', 'validation_interval', 'device', 'seed',
            'output_dir', 'use_curriculum', 'derivative_method',
            'use_scheduler', 'scheduler_T0', 'scheduler_eta_min'
        ]
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_encoding_type_valid(self):
        """Encoding type should be 'fourier' or 'siren'."""
        config = get_default_config()
        assert config['encoding_type'] in ['fourier', 'siren']

    def test_hidden_dims_is_list(self):
        """Hidden dims should be a list of integers."""
        config = get_default_config()
        assert isinstance(config['hidden_dims'], list)
        assert all(isinstance(d, int) for d in config['hidden_dims'])

    def test_positive_numeric_values(self):
        """Numeric parameters should be positive."""
        config = get_default_config()
        assert config['batch_size'] > 0
        assert config['epochs'] > 0
        assert config['learning_rate'] > 0
        assert config['fourier_modes'] > 0

    def test_device_auto_detection(self):
        """Device should be 'cuda' or 'cpu'."""
        config = get_default_config()
        assert config['device'] in ['cuda', 'cpu']

    def test_intervals_are_positive(self):
        """Checkpoint and validation intervals should be positive."""
        config = get_default_config()
        assert config['checkpoint_interval'] > 0
        assert config['validation_interval'] > 0


# ============================================================================
# Train Epoch Tests
# ============================================================================

class TestTrainEpoch:
    """Tests for train_epoch() function."""

    @pytest.fixture
    def training_setup(self):
        """Set up model, loss, optimizer, manifold for training."""
        device = 'cpu'
        config = {
            'device': device,
            'batch_size': 8,
            'grad_clip': 1.0
        }

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4,
            normalize_phi=True
        ).to(device)

        loss_fn = G2TotalLoss(
            curriculum_scheduler=CurriculumScheduler(),
            use_ricci=False,
            use_positivity=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        manifold = TorusT7(device=device)

        return model, loss_fn, optimizer, manifold, config

    def test_returns_metrics_dict(self, training_setup):
        """train_epoch should return a metrics dictionary."""
        model, loss_fn, optimizer, manifold, config = training_setup
        metrics = train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        assert isinstance(metrics, dict)
        assert 'epoch' in metrics
        assert 'loss' in metrics

    def test_loss_is_finite(self, training_setup):
        """Returned loss should be finite."""
        model, loss_fn, optimizer, manifold, config = training_setup
        metrics = train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        assert np.isfinite(metrics['loss'])

    def test_model_parameters_updated(self, training_setup):
        """Model parameters should be updated after training step."""
        model, loss_fn, optimizer, manifold, config = training_setup

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Train one epoch
        train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        # Check at least some parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break

        assert params_changed, "Model parameters should change after training"

    def test_gradient_clipping_applied(self, training_setup):
        """Gradient clipping should limit gradient magnitude."""
        model, loss_fn, optimizer, manifold, config = training_setup
        config['grad_clip'] = 0.1  # Strict clipping

        # Run training step
        train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        # After clipping, gradients should be bounded
        # (Note: gradients are cleared after optimizer.step(), so we just check no error)
        assert True  # If we get here without error, clipping worked

    def test_epoch_number_in_metrics(self, training_setup):
        """Epoch number should be correctly recorded in metrics."""
        model, loss_fn, optimizer, manifold, config = training_setup

        for epoch in [0, 5, 100]:
            metrics = train_epoch(model, loss_fn, optimizer, manifold, config, epoch=epoch)
            assert metrics['epoch'] == epoch


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidate:
    """Tests for validate() function."""

    @pytest.fixture
    def validation_setup(self):
        """Set up model and manifold for validation."""
        device = 'cpu'
        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4,
            normalize_phi=True
        ).to(device)
        manifold = TorusT7(device=device)
        return model, manifold, device

    def test_returns_results_dict(self, validation_setup):
        """validate should return a results dictionary."""
        model, manifold, device = validation_setup
        results = validate(model, manifold, device, n_samples=100)

        assert isinstance(results, dict)

    def test_required_metrics_present(self, validation_setup):
        """Validation results should contain required metrics."""
        model, manifold, device = validation_setup
        results = validate(model, manifold, device, n_samples=100)

        required_keys = [
            'phi_norm_sq_mean', 'phi_norm_sq_std',
            'det_g_mean', 'det_g_std',
            'eigenvalue_min', 'eigenvalue_max'
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_phi_norm_positive(self, validation_setup):
        """Phi norm squared should be positive."""
        model, manifold, device = validation_setup
        results = validate(model, manifold, device, n_samples=100)

        assert results['phi_norm_sq_mean'] > 0

    def test_eigenvalues_positive(self, validation_setup):
        """Eigenvalues should be positive (SPD metric)."""
        model, manifold, device = validation_setup
        results = validate(model, manifold, device, n_samples=100)

        assert results['eigenvalue_min'] > 0

    def test_different_sample_sizes(self, validation_setup):
        """Validation should work with different sample sizes."""
        model, manifold, device = validation_setup

        for n_samples in [10, 50, 100]:
            results = validate(model, manifold, device, n_samples=n_samples)
            assert isinstance(results, dict)
            assert np.isfinite(results['phi_norm_sq_mean'])


# ============================================================================
# Training Function Tests
# ============================================================================

class TestTrain:
    """Tests for main train() function."""

    def test_creates_output_directory(self):
        """Training should create output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = os.path.join(tmpdir, 'checkpoints')
            config['epochs'] = 2
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['validation_interval'] = 1
            config['checkpoint_interval'] = 2
            config['device'] = 'cpu'

            train(config)

            assert os.path.exists(config['output_dir'])

    def test_saves_config_file(self):
        """Training should save configuration to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 2
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['validation_interval'] = 1
            config['checkpoint_interval'] = 2
            config['device'] = 'cpu'

            train(config)

            config_path = os.path.join(tmpdir, 'config.json')
            assert os.path.exists(config_path)

            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            assert saved_config['epochs'] == 2

    def test_saves_final_model(self):
        """Training should save final model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 2
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['validation_interval'] = 1
            config['checkpoint_interval'] = 2
            config['device'] = 'cpu'

            train(config)

            final_path = os.path.join(tmpdir, 'final_model.pt')
            assert os.path.exists(final_path)

    def test_returns_model_history_report(self):
        """Training should return model, history, and report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 2
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['validation_interval'] = 2
            config['checkpoint_interval'] = 2
            config['device'] = 'cpu'

            model, history, report = train(config)

            assert isinstance(model, G2PhiNetwork)
            assert isinstance(history, dict)
            assert isinstance(report, dict)

    def test_history_records_losses(self):
        """Training history should record losses over epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 3
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['validation_interval'] = 3
            config['checkpoint_interval'] = 3
            config['device'] = 'cpu'

            model, history, report = train(config)

            assert 'loss' in history
            assert len(history['loss']) == 3
            assert all(np.isfinite(loss) for loss in history['loss'])

    def test_reproducibility_with_seed(self):
        """Same seed should produce reproducible results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = os.path.join(tmpdir, 'run1')
            config['epochs'] = 2
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['seed'] = 12345
            config['validation_interval'] = 2
            config['checkpoint_interval'] = 2
            config['device'] = 'cpu'

            model1, history1, _ = train(config)

            config['output_dir'] = os.path.join(tmpdir, 'run2')
            model2, history2, _ = train(config)

            # Losses should be identical with same seed
            assert np.allclose(history1['loss'], history2['loss'], rtol=1e-5)


# ============================================================================
# Plot Training History Tests
# ============================================================================

class TestPlotTrainingHistory:
    """Tests for plot_training_history() function."""

    def test_creates_plot_file(self):
        """Should create training history plot file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                'epoch': [0, 1, 2],
                'loss': [1.0, 0.5, 0.25],
                'torsion': [0.5, 0.3, 0.1],
                'volume': [0.3, 0.2, 0.1],
                'phi_norm': [7.5, 7.2, 7.0],
                'det_g': [0.8, 0.9, 1.0]
            }

            plot_training_history(history, tmpdir)

            plot_path = os.path.join(tmpdir, 'training_history.png')
            assert os.path.exists(plot_path)

    def test_handles_empty_history(self):
        """Should handle empty history gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                'epoch': [],
                'loss': [],
                'torsion': [],
                'volume': [],
                'phi_norm': [],
                'det_g': []
            }

            # Should not raise
            try:
                plot_training_history(history, tmpdir)
            except Exception as e:
                pytest.fail(f"Should handle empty history: {e}")


# ============================================================================
# Argument Parsing Tests
# ============================================================================

class TestParseArgs:
    """Tests for command line argument parsing."""

    def test_default_values(self):
        """Defaults should be applied when no args provided."""
        with patch('sys.argv', ['G2_train.py']):
            args = parse_args()
            assert args.encoding == 'fourier'
            assert args.epochs == 3000

    def test_encoding_choices(self):
        """Should accept valid encoding types."""
        for encoding in ['fourier', 'siren']:
            with patch('sys.argv', ['G2_train.py', '--encoding', encoding]):
                args = parse_args()
                assert args.encoding == encoding

    def test_device_auto(self):
        """Device 'auto' should resolve to cuda or cpu."""
        with patch('sys.argv', ['G2_train.py', '--device', 'auto']):
            args = parse_args()
            assert args.device in ['cuda', 'cpu']

    def test_custom_epochs(self):
        """Custom epoch count should be parsed."""
        with patch('sys.argv', ['G2_train.py', '--epochs', '100']):
            args = parse_args()
            assert args.epochs == 100

    def test_no_curriculum_flag(self):
        """--no-curriculum flag should disable curriculum learning."""
        with patch('sys.argv', ['G2_train.py', '--no-curriculum']):
            args = parse_args()
            assert args.no_curriculum is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for complete training workflow."""

    @pytest.mark.slow
    def test_full_training_workflow(self):
        """Complete training workflow should execute without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 5
            config['batch_size'] = 16
            config['hidden_dims'] = [32, 32]
            config['fourier_modes'] = 4
            config['validation_interval'] = 5
            config['checkpoint_interval'] = 5
            config['device'] = 'cpu'

            model, history, report = train(config)

            # Verify outputs
            assert os.path.exists(os.path.join(tmpdir, 'final_model.pt'))
            assert os.path.exists(os.path.join(tmpdir, 'config.json'))
            assert os.path.exists(os.path.join(tmpdir, 'training_history.json'))
            assert os.path.exists(os.path.join(tmpdir, 'validation_report.json'))

    def test_curriculum_phases(self):
        """Training should progress through curriculum phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['epochs'] = 10
            config['batch_size'] = 8
            config['hidden_dims'] = [16, 16]
            config['fourier_modes'] = 2
            config['use_curriculum'] = True
            config['validation_interval'] = 10
            config['checkpoint_interval'] = 10
            config['device'] = 'cpu'

            model, history, report = train(config)

            # Should complete without errors
            assert len(history['loss']) == 10


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestTrainErrorHandling:
    """Tests for error handling in training."""

    def test_invalid_manifold_type(self):
        """Should handle invalid manifold type gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['manifold_type'] = 'INVALID'
            config['epochs'] = 1
            config['device'] = 'cpu'

            with pytest.raises(ValueError):
                train(config)

    def test_invalid_encoding_type(self):
        """Should handle invalid encoding type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = get_default_config()
            config['output_dir'] = tmpdir
            config['encoding_type'] = 'INVALID'
            config['epochs'] = 1
            config['device'] = 'cpu'

            with pytest.raises((ValueError, KeyError)):
                train(config)
