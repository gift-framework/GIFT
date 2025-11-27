"""
Tests for G2 ML Training Pipeline (v0.2)

These tests cover:
- G2_train.py: Training configuration, training loop, checkpointing
- G2_eval.py: Model loading, point evaluation, grid evaluation
- G2_export.py: ONNX export, checkpoint utilities

Note: These tests are designed to run without GPU and with minimal resources.
      Full training tests are marked as slow.

Author: GIFT Framework
Date: 2025-11-27
"""

import pytest
import numpy as np
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Skip if torch not available
torch = pytest.importorskip("torch")

# Add G2_ML to path
g2_ml_path = Path(__file__).parent.parent.parent / "G2_ML" / "0.2"
sys.path.insert(0, str(g2_ml_path))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def device():
    """Get device for testing (CPU for CI)."""
    return "cpu"


@pytest.fixture
def sample_coords(device):
    """Create sample coordinates for testing."""
    coords = torch.randn(10, 7, device=device)
    return coords


@pytest.fixture
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint file."""
    config = {
        'encoding_type': 'fourier',
        'hidden_dims': [64, 64],
        'fourier_modes': 8,
        'fourier_scale': 1.0,
        'omega_0': 30.0,
        'normalize_phi': True
    }

    # Create a minimal model state dict
    from G2_phi_network import G2PhiNetwork
    model = G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[64, 64],
        fourier_modes=8
    )

    checkpoint = {
        'epoch': 100,
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss': 0.001
    }

    checkpoint_path = temp_dir / "test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


# =============================================================================
# Test: Training Configuration (G2_train.py)
# =============================================================================

class TestTrainingConfiguration:
    """Tests for training configuration."""

    def test_get_default_config(self):
        """Test default configuration is returned."""
        from G2_train import get_default_config

        config = get_default_config()

        assert isinstance(config, dict)
        assert 'encoding_type' in config
        assert 'hidden_dims' in config
        assert 'epochs' in config
        assert 'batch_size' in config
        assert 'learning_rate' in config

    def test_default_config_values(self):
        """Test default configuration has expected values."""
        from G2_train import get_default_config

        config = get_default_config()

        assert config['encoding_type'] == 'fourier'
        assert config['hidden_dims'] == [256, 256, 128]
        assert config['fourier_modes'] == 16
        assert config['batch_size'] == 512
        assert config['epochs'] == 3000
        assert config['learning_rate'] == 1e-4

    def test_default_config_has_curriculum(self):
        """Test default config enables curriculum learning."""
        from G2_train import get_default_config

        config = get_default_config()

        assert config['use_curriculum'] is True

    def test_default_config_device_detection(self):
        """Test device detection in default config."""
        from G2_train import get_default_config

        config = get_default_config()

        # Should be 'cuda' or 'cpu' based on availability
        assert config['device'] in ['cuda', 'cpu']

    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        from G2_train import parse_args

        with patch('sys.argv', ['G2_train.py']):
            args = parse_args()

        assert args.encoding == 'fourier'
        assert args.epochs == 3000
        assert args.batch_size == 512


# =============================================================================
# Test: Training Epoch (G2_train.py)
# =============================================================================

class TestTrainingEpoch:
    """Tests for single training epoch."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping GPU test")
    def test_train_epoch_returns_metrics(self, device, temp_dir):
        """Test train_epoch returns metrics dict."""
        from G2_train import train_epoch, get_default_config
        from G2_phi_network import G2PhiNetwork
        from G2_manifold import create_manifold
        from G2_losses import G2TotalLoss

        config = get_default_config()
        config['device'] = device
        config['batch_size'] = 8  # Small batch for testing

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        ).to(device)

        manifold = create_manifold('T7', device=device)
        loss_fn = G2TotalLoss()
        optimizer = torch.optim.Adam(model.parameters())

        metrics = train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        assert isinstance(metrics, dict)
        assert 'epoch' in metrics
        assert 'loss' in metrics

    def test_train_epoch_updates_model(self, device):
        """Test train_epoch updates model parameters."""
        from G2_train import train_epoch, get_default_config
        from G2_phi_network import G2PhiNetwork
        from G2_manifold import create_manifold
        from G2_losses import G2TotalLoss

        config = get_default_config()
        config['device'] = device
        config['batch_size'] = 4

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        manifold = create_manifold('T7', device=device)
        loss_fn = G2TotalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        train_epoch(model, loss_fn, optimizer, manifold, config, epoch=0)

        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break

        assert params_changed, "Model parameters should change after training step"


# =============================================================================
# Test: Validation (G2_train.py)
# =============================================================================

class TestValidation:
    """Tests for validation function."""

    def test_validate_returns_results(self, device):
        """Test validate returns results dict."""
        from G2_train import validate
        from G2_phi_network import G2PhiNetwork
        from G2_manifold import create_manifold

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        manifold = create_manifold('T7', device=device)

        results = validate(model, manifold, device, n_samples=10)

        assert isinstance(results, dict)
        assert 'phi_norm_sq_mean' in results
        assert 'det_g_mean' in results
        assert 'eigenvalue_min' in results
        assert 'eigenvalue_max' in results

    def test_validate_positive_eigenvalues(self, device):
        """Test validation reports positive eigenvalues (SPD metric)."""
        from G2_train import validate
        from G2_phi_network import G2PhiNetwork
        from G2_manifold import create_manifold

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        manifold = create_manifold('T7', device=device)

        results = validate(model, manifold, device, n_samples=10)

        # Minimum eigenvalue should be positive (SPD projection)
        assert results['eigenvalue_min'] > 0


# =============================================================================
# Test: Model Loading (G2_eval.py)
# =============================================================================

class TestModelLoading:
    """Tests for model loading from checkpoints."""

    def test_load_model_from_checkpoint(self, mock_checkpoint, device):
        """Test loading model from checkpoint file."""
        from G2_eval import load_model

        model, config = load_model(str(mock_checkpoint), device=device)

        assert model is not None
        assert config is not None
        assert config['encoding_type'] == 'fourier'

    def test_load_model_eval_mode(self, mock_checkpoint, device):
        """Test loaded model is in eval mode."""
        from G2_eval import load_model

        model, _ = load_model(str(mock_checkpoint), device=device)

        assert not model.training

    def test_load_model_config_extraction(self, mock_checkpoint, device):
        """Test config is extracted from checkpoint."""
        from G2_eval import load_model

        _, config = load_model(str(mock_checkpoint), device=device)

        assert 'hidden_dims' in config
        assert 'encoding_type' in config


# =============================================================================
# Test: Point Evaluation (G2_eval.py)
# =============================================================================

class TestPointEvaluation:
    """Tests for point evaluation functions."""

    def test_evaluate_at_point(self, mock_checkpoint, device):
        """Test evaluation at a single point."""
        from G2_eval import load_model, evaluate_at_point

        model, _ = load_model(str(mock_checkpoint), device=device)

        coords = torch.zeros(7, device=device)
        results = evaluate_at_point(model, coords, device=device)

        assert 'phi' in results
        assert 'metric' in results
        assert 'det_g' in results
        assert 'eigenvalues' in results

    def test_evaluate_at_point_batch(self, mock_checkpoint, device):
        """Test evaluation at multiple points."""
        from G2_eval import load_model, evaluate_at_point

        model, _ = load_model(str(mock_checkpoint), device=device)

        coords = torch.randn(5, 7, device=device)
        results = evaluate_at_point(model, coords, device=device)

        assert results['phi'].shape[0] == 5
        assert results['metric'].shape[0] == 5

    def test_evaluate_at_point_phi_shape(self, mock_checkpoint, device):
        """Test phi output has correct shape."""
        from G2_eval import load_model, evaluate_at_point

        model, _ = load_model(str(mock_checkpoint), device=device)

        coords = torch.zeros(7, device=device)
        results = evaluate_at_point(model, coords, device=device)

        # phi should have shape (1, num_components)
        assert len(results['phi'].shape) == 2
        assert results['phi'].shape[0] == 1

    def test_evaluate_at_point_metric_shape(self, mock_checkpoint, device):
        """Test metric output has correct shape (7x7)."""
        from G2_eval import load_model, evaluate_at_point

        model, _ = load_model(str(mock_checkpoint), device=device)

        coords = torch.zeros(7, device=device)
        results = evaluate_at_point(model, coords, device=device)

        # metric should have shape (1, 7, 7)
        assert results['metric'].shape == (1, 7, 7)


# =============================================================================
# Test: Grid Evaluation (G2_eval.py)
# =============================================================================

class TestGridEvaluation:
    """Tests for grid evaluation functions."""

    def test_evaluate_on_grid_small(self, mock_checkpoint, device):
        """Test grid evaluation with small grid."""
        from G2_eval import load_model, evaluate_on_grid
        from G2_manifold import create_manifold

        model, config = load_model(str(mock_checkpoint), device=device)
        manifold = create_manifold('T7', device=device)

        results = evaluate_on_grid(model, manifold, n_points_per_dim=2, device=device)

        assert 'phi_norm_sq' in results
        assert 'det_g' in results
        assert 'eigenvalues' in results

    def test_evaluate_on_grid_statistics(self, mock_checkpoint, device):
        """Test grid evaluation returns statistics."""
        from G2_eval import load_model, evaluate_on_grid
        from G2_manifold import create_manifold

        model, config = load_model(str(mock_checkpoint), device=device)
        manifold = create_manifold('T7', device=device)

        results = evaluate_on_grid(model, manifold, n_points_per_dim=2, device=device)

        # Check statistics structure
        assert 'mean' in results['phi_norm_sq']
        assert 'std' in results['phi_norm_sq']
        assert 'min' in results['phi_norm_sq']
        assert 'max' in results['phi_norm_sq']


# =============================================================================
# Test: ONNX Export (G2_export.py)
# =============================================================================

class TestONNXExport:
    """Tests for ONNX export functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("onnx", reason="ONNX not installed"),
        reason="ONNX not available"
    )
    def test_export_to_onnx(self, mock_checkpoint, temp_dir, device):
        """Test ONNX export creates valid file."""
        from G2_export import load_model_from_checkpoint, export_to_onnx

        model, config = load_model_from_checkpoint(str(mock_checkpoint), device=device)

        onnx_path = temp_dir / "model.onnx"
        success = export_to_onnx(model, str(onnx_path), device=device)

        assert success
        assert onnx_path.exists()

    def test_load_model_from_checkpoint(self, mock_checkpoint, device):
        """Test checkpoint loading utility."""
        from G2_export import load_model_from_checkpoint

        model, config = load_model_from_checkpoint(str(mock_checkpoint), device=device)

        assert model is not None
        assert config is not None

    def test_load_model_from_checkpoint_missing_config(self, temp_dir, device):
        """Test checkpoint loading with missing config uses defaults."""
        from G2_phi_network import G2PhiNetwork

        # Create checkpoint without config
        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[64, 64],
            fourier_modes=8
        )

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }

        checkpoint_path = temp_dir / "no_config_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        from G2_export import load_model_from_checkpoint

        loaded_model, config = load_model_from_checkpoint(str(checkpoint_path), device=device)

        assert loaded_model is not None
        # Should use default config
        assert config['encoding_type'] == 'fourier'


# =============================================================================
# Test: Checkpointing
# =============================================================================

class TestCheckpointing:
    """Tests for model checkpointing."""

    def test_checkpoint_structure(self, mock_checkpoint):
        """Test checkpoint has required structure."""
        checkpoint = torch.load(mock_checkpoint)

        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint

    def test_checkpoint_config_preserved(self, mock_checkpoint):
        """Test config is preserved in checkpoint."""
        checkpoint = torch.load(mock_checkpoint)
        config = checkpoint['config']

        assert 'encoding_type' in config
        assert 'hidden_dims' in config

    def test_save_and_load_checkpoint(self, temp_dir, device):
        """Test saving and loading checkpoint preserves model."""
        from G2_phi_network import G2PhiNetwork

        # Create model
        model1 = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        ).to(device)

        # Get output for test input
        test_input = torch.randn(1, 7, device=device)
        with torch.no_grad():
            output1 = model1(test_input)

        # Save checkpoint
        checkpoint_path = temp_dir / "save_load_test.pt"
        torch.save({
            'model_state_dict': model1.state_dict(),
            'config': {
                'encoding_type': 'fourier',
                'hidden_dims': [32, 32],
                'fourier_modes': 4,
                'fourier_scale': 1.0,
                'omega_0': 30.0,
                'normalize_phi': True
            }
        }, checkpoint_path)

        # Load into new model
        model2 = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[32, 32],
            fourier_modes=4
        ).to(device)
        checkpoint = torch.load(checkpoint_path)
        model2.load_state_dict(checkpoint['model_state_dict'])
        model2.eval()

        with torch.no_grad():
            output2 = model2(test_input)

        # Outputs should match
        assert torch.allclose(output1, output2)


# =============================================================================
# Test: Training History
# =============================================================================

class TestTrainingHistory:
    """Tests for training history tracking."""

    def test_history_structure(self):
        """Test training history has correct structure."""
        history = {
            'epoch': [],
            'loss': [],
            'torsion': [],
            'volume': [],
            'phi_norm': [],
            'det_g': []
        }

        # Simulate some epochs
        for i in range(5):
            history['epoch'].append(i)
            history['loss'].append(0.1 / (i + 1))
            history['torsion'].append(0.01 / (i + 1))
            history['volume'].append(0.05)
            history['phi_norm'].append(7.0 + 0.1 * np.random.randn())
            history['det_g'].append(1.0 + 0.01 * np.random.randn())

        assert len(history['epoch']) == 5
        assert len(history['loss']) == 5

    def test_history_json_serializable(self):
        """Test history can be serialized to JSON."""
        history = {
            'epoch': [0, 1, 2],
            'loss': [0.1, 0.05, 0.01],
            'torsion': [0.01, 0.005, 0.001],
        }

        json_str = json.dumps(history)
        parsed = json.loads(json_str)

        assert parsed['epoch'] == [0, 1, 2]


# =============================================================================
# Test: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of training components."""

    def test_metric_positive_definite(self, device):
        """Test metric projection ensures positive definiteness."""
        from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
        from G2_geometry import project_spd

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        coords = torch.randn(10, 7, device=device)

        with torch.no_grad():
            phi = model(coords)
            metric = metric_from_phi_algebraic(phi, use_approximation=True)
            metric = project_spd(metric)

            # Check eigenvalues are positive
            eigenvalues = torch.linalg.eigvalsh(metric)
            assert (eigenvalues > 0).all()

    def test_no_nan_in_forward(self, device):
        """Test forward pass produces no NaN values."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        coords = torch.randn(10, 7, device=device)

        with torch.no_grad():
            phi = model(coords)

        assert not torch.isnan(phi).any()

    def test_no_inf_in_forward(self, device):
        """Test forward pass produces no Inf values."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        coords = torch.randn(10, 7, device=device)

        with torch.no_grad():
            phi = model(coords)

        assert not torch.isinf(phi).any()

    def test_gradient_not_exploding(self, device):
        """Test gradients don't explode during backward pass."""
        from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
        from G2_geometry import project_spd

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        coords = torch.randn(4, 7, device=device, requires_grad=True)

        phi = model(coords)
        loss = phi.sum()
        loss.backward()

        # Check gradients are finite
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()
                assert not torch.isinf(p.grad).any()


# =============================================================================
# Test: Device Handling
# =============================================================================

class TestDeviceHandling:
    """Tests for proper device handling."""

    def test_model_to_device(self, device):
        """Test model can be moved to device."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        # Check model is on correct device
        for p in model.parameters():
            assert str(p.device).startswith(device.split(':')[0])

    def test_forward_respects_device(self, device):
        """Test forward pass respects input device."""
        from G2_phi_network import G2PhiNetwork

        model = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[16, 16],
            fourier_modes=2
        ).to(device)

        coords = torch.randn(5, 7, device=device)

        with torch.no_grad():
            phi = model(coords)

        assert str(phi.device).startswith(device.split(':')[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
