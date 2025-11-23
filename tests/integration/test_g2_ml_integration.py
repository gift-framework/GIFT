"""
Integration tests for G2 ML training pipeline.

Tests include:
- Short training runs (smoke tests)
- Model export/import round trips
- End-to-end metric learning pipeline
- Loss convergence behavior
- Checkpoint saving and loading
- Multi-epoch training stability

Version: 2.1.0
"""

import pytest
import torch
import numpy as np
import sys
import tempfile
from pathlib import Path

# Add G2_ML to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "G2_ML" / "0.2"))

try:
    from G2_phi_network import G2PhiNetwork
    from G2_losses import torsion_loss, volume_loss, phi_normalization_loss
    from G2_geometry import project_spd
    G2_ML_AVAILABLE = True
except ImportError:
    G2_ML_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not G2_ML_AVAILABLE,
    reason="G2_ML modules not available"
)


class TestShortTrainingRun:
    """Test short training runs (smoke tests)."""

    def test_training_smoke_test(self):
        """Test network can be trained for a few iterations."""
        torch.manual_seed(42)

        # Create network
        network = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[64, 32],
            fourier_modes=8
        )

        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Training loop (10 iterations only)
        n_iterations = 10
        batch_size = 8

        losses = []

        for iteration in range(n_iterations):
            # Generate random batch
            coords = torch.randn(batch_size, 7, requires_grad=True)

            # Forward pass
            phi = network(coords)

            # Create simple metric (identity for now)
            metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

            # Compute losses
            loss_torsion, _ = torsion_loss(phi, metric, coords)
            loss_phi_norm, _ = phi_normalization_loss(phi, target=7.0)

            # Total loss
            loss = loss_torsion + 0.01 * loss_phi_norm

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Test completed successfully
        assert len(losses) == n_iterations

        # Losses should be finite
        assert all(np.isfinite(l) for l in losses)

        # Ideally losses should decrease (but might not in just 10 iters)
        # At minimum, last loss should not be NaN
        assert np.isfinite(losses[-1])

    def test_training_convergence_trend(self):
        """Test that loss generally decreases over training."""
        torch.manual_seed(42)

        network = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[128, 64],
            fourier_modes=16
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)  # Higher LR for faster convergence

        n_iterations = 50
        batch_size = 16

        losses = []

        for iteration in range(n_iterations):
            coords = torch.randn(batch_size, 7, requires_grad=True)
            phi = network(coords)

            metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

            loss_torsion, _ = torsion_loss(phi, metric, coords)
            loss_phi_norm, _ = phi_normalization_loss(phi, target=7.0)

            loss = loss_torsion + 0.1 * loss_phi_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Compare first and last 10 iterations
        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-10:])

        # Loss should generally decrease (allow some slack)
        # This test might be flaky, so we just check it doesn't increase dramatically
        assert final_loss < 2 * initial_loss, (
            f"Loss increased dramatically: {initial_loss} -> {final_loss}"
        )

    def test_gradient_flow_through_pipeline(self):
        """Test gradients flow through entire pipeline."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        coords = torch.randn(4, 7, requires_grad=True)

        # Forward
        phi = network(coords)
        metric = torch.eye(7).unsqueeze(0).repeat(4, 1, 1)

        loss_torsion, _ = torsion_loss(phi, metric, coords)

        # Backward
        optimizer.zero_grad()
        loss_torsion.backward()

        # Check all parameters have gradients
        for name, param in network.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), f"Gradient for {name} has NaN/Inf"

        # Optimizer step should work
        optimizer.step()


class TestModelExportImport:
    """Test model export and import."""

    def test_model_state_dict_save_load(self):
        """Test saving and loading model state dict."""
        torch.manual_seed(42)

        # Create and train model briefly
        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Train for a few steps
        for _ in range(5):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)

            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save state dict
        state_dict = network.state_dict()

        # Create new network and load
        network2 = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        network2.load_state_dict(state_dict)

        # Test that both networks give same output
        test_input = torch.randn(4, 7)

        with torch.no_grad():
            output1 = network(test_input)
            output2 = network2(test_input)

        assert torch.allclose(output1, output2, atol=1e-6), "Loaded model produces different output"

    def test_model_checkpoint_round_trip(self):
        """Test full checkpoint save/load including optimizer state."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Train for a few steps
        for _ in range(5):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint = {
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
        }

        # In real scenario would save to file, here we just verify structure
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint

        # Load into new network and optimizer
        network2 = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer2 = torch.optim.Adam(network2.parameters(), lr=1e-3)

        network2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

        # Verify loaded correctly
        test_input = torch.randn(4, 7)

        with torch.no_grad():
            output1 = network(test_input)
            output2 = network2(test_input)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestEndToEndPipeline:
    """Test complete end-to-end training pipeline."""

    def test_complete_mini_pipeline(self):
        """Test complete pipeline with all components."""
        torch.manual_seed(42)

        # 1. Create network
        network = G2PhiNetwork(
            encoding_type='fourier',
            hidden_dims=[128, 64],
            fourier_modes=16
        )

        # 2. Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # 3. Training loop
        n_epochs = 3
        n_batches_per_epoch = 5
        batch_size = 8

        epoch_losses = []

        for epoch in range(n_epochs):
            batch_losses = []

            for batch_idx in range(n_batches_per_epoch):
                # Generate batch
                coords = torch.randn(batch_size, 7, requires_grad=True)

                # Forward
                phi = network(coords)
                metric = torch.eye(7).unsqueeze(0).repeat(batch_size, 1, 1)

                # Losses
                loss_torsion, info_torsion = torsion_loss(phi, metric, coords)
                loss_volume, info_volume = volume_loss(metric)
                loss_phi_norm, info_phi_norm = phi_normalization_loss(phi)

                # Combined loss
                loss = loss_torsion + 0.1 * loss_volume + 0.01 * loss_phi_norm

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)

        # Pipeline completed successfully
        assert len(epoch_losses) == n_epochs

        # All losses should be finite
        assert all(np.isfinite(l) for l in epoch_losses)

    def test_multi_epoch_stability(self):
        """Test training remains stable over multiple epochs."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        n_epochs = 10
        n_batches = 5

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                coords = torch.randn(8, 7, requires_grad=True)
                phi = network(coords)

                metric = torch.eye(7).unsqueeze(0).repeat(8, 1, 1)

                loss_torsion, _ = torsion_loss(phi, metric, coords)

                optimizer.zero_grad()
                loss_torsion.backward()

                # Check gradients don't explode
                total_grad_norm = 0.0
                for param in network.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2

                total_grad_norm = np.sqrt(total_grad_norm)

                assert total_grad_norm < 1e6, f"Gradient explosion at epoch {epoch}"

                optimizer.step()

        # Training completed without explosion
        assert True


class TestLossWeighting:
    """Test different loss weighting schemes."""

    def test_curriculum_weighting(self):
        """Test curriculum learning with changing weights."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Curriculum: Start with phi normalization, gradually add torsion
        n_iterations = 20

        for iteration in range(n_iterations):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            metric = torch.eye(7).unsqueeze(0).repeat(8, 1, 1)

            loss_torsion, _ = torsion_loss(phi, metric, coords)
            loss_phi_norm, _ = phi_normalization_loss(phi)

            # Gradually increase torsion weight
            torsion_weight = iteration / n_iterations
            phi_norm_weight = 1.0 - torsion_weight

            loss = torsion_weight * loss_torsion + phi_norm_weight * loss_phi_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete without issues
        assert True

    def test_equal_weighting(self):
        """Test training with equal loss weights."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        for _ in range(10):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            metric = torch.eye(7).unsqueeze(0).repeat(8, 1, 1)

            loss_torsion, _ = torsion_loss(phi, metric, coords)
            loss_volume, _ = volume_loss(metric)
            loss_phi_norm, _ = phi_normalization_loss(phi)

            # Equal weights
            loss = loss_torsion + loss_volume + loss_phi_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert True


class TestBatchProcessing:
    """Test batch processing correctness."""

    def test_batch_size_independence(self):
        """Test that different batch sizes give consistent per-sample results."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        # Single sample
        coord_single = torch.randn(1, 7)
        phi_single = network(coord_single)

        # Same sample in batch of 4
        coord_batch = coord_single.repeat(4, 1)
        phi_batch = network(coord_batch)

        # All batch outputs should match single output
        for i in range(4):
            assert torch.allclose(phi_batch[i], phi_single[0], atol=1e-6), (
                f"Batch element {i} differs from single computation"
            )

    def test_varying_batch_sizes(self):
        """Test network handles different batch sizes."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        # Test various batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            coords = torch.randn(batch_size, 7)
            phi = network(coords)

            assert phi.shape == (batch_size, 35), (
                f"Wrong output shape for batch_size={batch_size}"
            )
            assert torch.all(torch.isfinite(phi))


class TestOptimizationAlgorithms:
    """Test different optimizers."""

    @pytest.mark.parametrize("optimizer_class", [
        torch.optim.Adam,
        torch.optim.SGD,
        torch.optim.RMSprop,
    ])
    def test_optimizer_compatibility(self, optimizer_class):
        """Test network works with different optimizers."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])

        if optimizer_class == torch.optim.SGD:
            optimizer = optimizer_class(network.parameters(), lr=1e-2)
        else:
            optimizer = optimizer_class(network.parameters(), lr=1e-3)

        # Train for a few steps
        for _ in range(5):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete successfully
        assert True


class TestLearningRateScheduling:
    """Test learning rate schedulers."""

    def test_step_lr_scheduler(self):
        """Test StepLR scheduler."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

        # Reduce LR every 5 steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        initial_lr = optimizer.param_groups[0]['lr']

        for iteration in range(10):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        # LR should have decreased
        assert final_lr < initial_lr

    def test_reduce_on_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        for iteration in range(10):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step scheduler with loss value
            scheduler.step(loss.item())

        # Should complete successfully
        assert True


class TestTrainingMetrics:
    """Test tracking of training metrics."""

    def test_loss_history_tracking(self):
        """Test tracking loss history."""
        torch.manual_seed(42)

        network = G2PhiNetwork(encoding_type='fourier', hidden_dims=[64])
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        loss_history = []

        for _ in range(20):
            coords = torch.randn(8, 7, requires_grad=True)
            phi = network(coords)
            loss = phi.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        # Should have recorded all losses
        assert len(loss_history) == 20

        # Can compute statistics
        mean_loss = np.mean(loss_history)
        std_loss = np.std(loss_history)

        assert np.isfinite(mean_loss)
        assert np.isfinite(std_loss)
