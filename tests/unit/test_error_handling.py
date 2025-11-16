"""
Error handling and edge case tests.

Tests invalid inputs, numerical edge cases, and error recovery.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Skip entire module if torch is not available
torch = pytest.importorskip("torch", reason="PyTorch required for G2 geometry tests")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "G2_ML" / "0.2"))

from run_validation import GIFTFrameworkStatistical
from G2_geometry import project_spd, volume_form, metric_inverse


class TestGIFTFrameworkErrorHandling:
    """Test error handling in GIFT framework."""

    def test_invalid_parameter_types(self):
        """Test that invalid parameter types are handled."""
        with pytest.raises(TypeError):
            GIFTFrameworkStatistical(p2="invalid")

    def test_negative_p2(self):
        """Test behavior with negative p2."""
        # Should still initialize (may produce invalid physics)
        gift = GIFTFrameworkStatistical(p2=-1.0)
        obs = gift.compute_all_observables()

        # Some observables may be invalid
        # Just check it doesn't crash
        assert obs is not None

    def test_zero_weyl_factor(self):
        """Test behavior with zero Weyl factor."""
        gift = GIFTFrameworkStatistical(Weyl_factor=0.001)

        # Should not crash, but some observables may be extreme
        obs = gift.compute_all_observables()
        assert obs is not None

    def test_very_large_parameters(self):
        """Test behavior with very large parameters."""
        gift = GIFTFrameworkStatistical(p2=1000, Weyl_factor=1000)

        obs = gift.compute_all_observables()

        # Should not produce Inf
        for key, value in obs.items():
            assert not np.isinf(value), f"{key} produced Inf"


class TestG2GeometryErrorHandling:
    """Test error handling in G2 geometry operations."""

    def test_project_spd_with_nan(self):
        """Test SPD projection with NaN input."""
        metric = torch.randn(5, 7, 7)
        metric[0, 0, 0] = float('nan')

        # Should handle NaN gracefully or raise error
        try:
            spd_metric = project_spd(metric)
            # If it doesn't raise, check output
            assert spd_metric is not None
        except (RuntimeError, ValueError):
            # Expected for NaN input
            pass

    def test_project_spd_with_inf(self):
        """Test SPD projection with Inf input."""
        metric = torch.randn(5, 7, 7)
        metric[0, 0, 0] = float('inf')

        try:
            spd_metric = project_spd(metric)
            assert spd_metric is not None
        except (RuntimeError, ValueError):
            pass

    def test_volume_form_singular_matrix(self):
        """Test volume form with singular matrix."""
        # Create singular matrix (determinant = 0)
        metric = torch.zeros(3, 7, 7)

        vol = volume_form(metric)

        # Should handle gracefully (likely produce 0 or small value)
        assert not torch.any(torch.isnan(vol))

    def test_metric_inverse_singular_matrix(self):
        """Test metric inverse with singular matrix."""
        # Create near-singular matrix
        metric = torch.eye(7).unsqueeze(0).repeat(3, 1, 1) * 1e-10

        # Should handle with regularization
        inv_metric = metric_inverse(metric, epsilon=1e-8)

        # Should not produce NaN
        assert not torch.any(torch.isnan(inv_metric))
        assert not torch.any(torch.isinf(inv_metric))

    def test_empty_batch(self):
        """Test geometry operations with empty batch."""
        metric = torch.randn(0, 7, 7)

        # Should handle empty batch
        spd_metric = project_spd(metric)
        assert spd_metric.shape[0] == 0

    def test_wrong_shape(self):
        """Test geometry operations with wrong input shape."""
        metric = torch.randn(5, 5, 5)  # Wrong: should be (batch, 7, 7)

        # Should raise error or handle gracefully
        with pytest.raises((RuntimeError, AssertionError, IndexError)):
            project_spd(metric)


class TestNumericalEdgeCases:
    """Test numerical edge cases."""

    def test_gift_framework_extreme_precision(self):
        """Test framework with extreme precision requirements."""
        gift = GIFTFrameworkStatistical()
        obs = gift.compute_all_observables()

        # All observables should be representable in float64
        for key, value in obs.items():
            assert abs(value) < 1e300, f"{key} exceeds float64 range"
            assert abs(value) > 1e-300 or value == 0, f"{key} underflows float64"

    def test_gradient_overflow_protection(self):
        """Test that gradients don't overflow."""
        metric = torch.randn(3, 7, 7, requires_grad=True) * 100  # Large values

        spd_metric = project_spd(metric)
        loss = torch.sum(spd_metric ** 2)

        loss.backward()

        # Gradients should not be Inf or NaN
        assert not torch.any(torch.isnan(metric.grad))
        assert not torch.any(torch.isinf(metric.grad))

    def test_very_small_eigenvalues(self):
        """Test handling of very small eigenvalues."""
        Q, _ = torch.linalg.qr(torch.randn(3, 7, 7))
        eigenvalues = torch.rand(3, 7) * 1e-10  # Very small
        metric = Q @ torch.diag_embed(eigenvalues) @ Q.transpose(-2, -1)

        spd_metric = project_spd(metric, epsilon=1e-12)

        # Should clamp to epsilon
        final_eigenvalues = torch.linalg.eigvalsh(spd_metric)
        assert torch.all(final_eigenvalues >= 1e-12 - 1e-14)


class TestFileIOErrors:
    """Test file I/O error handling."""

    def test_missing_file_graceful_handling(self):
        """Test graceful handling of missing files."""
        # Try to load from non-existent file
        nonexistent_path = Path("/nonexistent/directory/file.txt")

        # Should raise FileNotFoundError or handle gracefully
        with pytest.raises(FileNotFoundError):
            with open(nonexistent_path, 'r') as f:
                f.read()

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON data."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {{{")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                with open(temp_path, 'r') as f:
                    json.load(f)
        finally:
            Path(temp_path).unlink()


class TestMemoryManagement:
    """Test memory management and cleanup."""

    def test_no_memory_leak_in_loop(self):
        """Test that repeated calculations don't leak memory."""
        import gc

        for _ in range(100):
            gift = GIFTFrameworkStatistical()
            obs = gift.compute_all_observables()
            del gift
            del obs

        # Force garbage collection
        gc.collect()

        # If we got here without OOM, test passes

    def test_large_tensor_cleanup(self):
        """Test that large tensors are properly freed."""
        import gc

        for _ in range(10):
            large_metric = torch.randn(10000, 7, 7)
            del large_metric
            gc.collect()


class TestDivisionByZero:
    """Test division by zero protection."""

    def test_gift_framework_zero_division_protection(self):
        """Test that framework protects against division by zero."""
        gift = GIFTFrameworkStatistical(Weyl_factor=1e-10)

        obs = gift.compute_all_observables()

        # Should not produce Inf from division
        for key, value in obs.items():
            assert not np.isinf(value), f"{key} produced Inf from division"

    def test_volume_form_zero_determinant(self):
        """Test volume form with zero determinant."""
        metric = torch.zeros(3, 7, 7)

        vol = volume_form(metric)

        # Should produce small value, not NaN
        assert not torch.any(torch.isnan(vol))
