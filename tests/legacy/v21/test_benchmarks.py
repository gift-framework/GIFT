"""
Performance benchmarks and profiling tests for GIFT framework.

Features:
- Observable computation timing
- Memory usage profiling
- Scalability tests
- Performance regression detection
- Optimization validation
- Bottleneck identification

Version: 2.1.0
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path
import gc

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V21_AVAILABLE,
    reason="GIFT v2.1 not available"
)


# Performance baselines (in seconds, on reference hardware)
PERFORMANCE_BASELINES = {
    "framework_initialization": 0.1,
    "single_observable_computation": 1.0,
    "all_observables_computation": 2.0,
    "monte_carlo_100_samples": 10.0,
}


class TestComputationTiming:
    """Test computation timing and performance."""

    def test_framework_initialization_time(self):
        """Test framework initialization is fast."""
        start = time.time()

        framework = GIFTFrameworkV21()

        elapsed = time.time() - start

        # Should be very fast (< 100ms)
        assert elapsed < PERFORMANCE_BASELINES["framework_initialization"], (
            f"Initialization took {elapsed:.3f}s (baseline: {PERFORMANCE_BASELINES['framework_initialization']}s)"
        )

    def test_observable_computation_time(self):
        """Test all observables computation time."""
        framework = GIFTFrameworkV21()

        start = time.time()

        obs = framework.compute_all_observables()

        elapsed = time.time() - start

        print(f"\n⏱️  All observables computed in {elapsed:.3f}s")

        # Should complete in reasonable time
        assert elapsed < PERFORMANCE_BASELINES["all_observables_computation"], (
            f"Computation took {elapsed:.3f}s (baseline: {PERFORMANCE_BASELINES['all_observables_computation']}s)"
        )

    def test_repeated_computation_overhead(self):
        """Test overhead of repeated computations."""
        framework = GIFTFrameworkV21()

        # First computation (may include lazy initialization)
        start = time.time()
        obs1 = framework.compute_all_observables()
        first_time = time.time() - start

        # Second computation (should be fast if cached)
        start = time.time()
        obs2 = framework.compute_all_observables()
        second_time = time.time() - start

        print(f"\n⏱️  First: {first_time:.3f}s, Second: {second_time:.3f}s")

        # Second should not be significantly slower
        assert second_time < 2 * first_time, (
            "Repeated computation much slower than first"
        )

    def test_single_framework_instance_overhead(self):
        """Test overhead of creating single framework instance."""
        timings = []

        for i in range(10):
            start = time.time()
            framework = GIFTFrameworkV21()
            obs = framework.compute_all_observables()
            elapsed = time.time() - start

            timings.append(elapsed)

        mean_time = np.mean(timings)
        std_time = np.std(timings)

        print(f"\n⏱️  Mean: {mean_time:.3f}s ± {std_time:.3f}s")

        # Standard deviation should be reasonable
        assert std_time < 0.5 * mean_time, "High variability in computation time"


class TestMemoryUsage:
    """Test memory usage and efficiency."""

    def test_framework_memory_footprint(self):
        """Test framework instance doesn't use excessive memory."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Measure before
            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Create framework
            framework = GIFTFrameworkV21()
            obs = framework.compute_all_observables()

            # Measure after
            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            mem_increase = mem_after - mem_before

            print(f"\n💾 Memory increase: {mem_increase:.1f} MB")

            # Should not use excessive memory (< 100 MB for basic computation)
            assert mem_increase < 100, (
                f"Excessive memory usage: {mem_increase:.1f} MB"
            )

        except ImportError:
            pytest.skip("psutil not available for memory profiling")

    def test_memory_cleanup_after_computation(self):
        """Test memory is properly cleaned up."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            gc.collect()
            mem_baseline = process.memory_info().rss / 1024 / 1024

            # Create and destroy many instances
            for i in range(10):
                framework = GIFTFrameworkV21()
                obs = framework.compute_all_observables()
                del framework
                del obs

            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024

            mem_increase = mem_after - mem_baseline

            print(f"\n💾 Memory after 10 iterations: +{mem_increase:.1f} MB")

            # Should not have significant memory leak
            assert mem_increase < 50, (
                f"Possible memory leak: {mem_increase:.1f} MB after 10 iterations"
            )

        except ImportError:
            pytest.skip("psutil not available")

    def test_large_monte_carlo_memory_efficiency(self):
        """Test memory efficiency with large Monte Carlo simulation."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024

            # Run Monte Carlo
            n_samples = 100
            results = []

            for i in range(n_samples):
                framework = GIFTFrameworkV21()
                obs = framework.compute_all_observables()

                # Only store one value to test memory efficiency
                if 'alpha_inv_MZ' in obs:
                    results.append(obs['alpha_inv_MZ'])

                # Cleanup
                del framework
                del obs

                if i % 10 == 0:
                    gc.collect()

            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024

            mem_increase = mem_after - mem_before

            print(f"\n💾 Memory for 100 samples: +{mem_increase:.1f} MB")

            # Should scale linearly, not quadratically
            assert mem_increase < 200, (
                f"Excessive memory usage for 100 samples: {mem_increase:.1f} MB"
            )

        except ImportError:
            pytest.skip("psutil not available")


class TestScalability:
    """Test computational scalability."""

    def test_parameter_sweep_scalability(self):
        """Test performance scales linearly with number of evaluations."""
        sample_sizes = [10, 20, 50]
        timings = []

        for n_samples in sample_sizes:
            start = time.time()

            for i in range(n_samples):
                framework = GIFTFrameworkV21()
                obs = framework.compute_all_observables()

            elapsed = time.time() - start
            time_per_sample = elapsed / n_samples

            timings.append((n_samples, elapsed, time_per_sample))

            print(f"\n⏱️  {n_samples} samples: {elapsed:.2f}s ({time_per_sample:.3f}s/sample)")

        # Time per sample should be roughly constant
        times_per_sample = [t[2] for t in timings]
        variation = np.std(times_per_sample) / np.mean(times_per_sample)

        assert variation < 0.5, (
            f"High variation in time per sample: {variation*100:.1f}%"
        )

    def test_parallel_potential(self):
        """Test that computations are independent (parallelizable)."""
        # Test that results are identical regardless of order
        frameworks = [GIFTFrameworkV21() for _ in range(5)]

        results = [fw.compute_all_observables() for fw in frameworks]

        # All should give same results (order-independent)
        for i in range(1, len(results)):
            for key in results[0]:
                if key in results[i]:
                    assert results[0][key] == results[i][key], (
                        f"Results not order-independent for {key}"
                    )


class TestPerformanceRegression:
    """Detect performance regressions."""

    def test_computation_time_regression(self):
        """Test computation time hasn't regressed."""
        framework = GIFTFrameworkV21()

        # Warm up
        obs = framework.compute_all_observables()

        # Measure performance
        start = time.time()

        for i in range(5):
            obs = framework.compute_all_observables()

        elapsed = time.time() - start
        avg_time = elapsed / 5

        print(f"\n⏱️  Average computation time: {avg_time:.3f}s")

        # Compare with baseline
        baseline = PERFORMANCE_BASELINES["all_observables_computation"]

        # Should not be significantly slower than baseline
        assert avg_time < 1.5 * baseline, (
            f"PERFORMANCE REGRESSION: {avg_time:.3f}s vs baseline {baseline:.3f}s"
        )

    def test_no_quadratic_slowdown(self):
        """Test computation time doesn't grow quadratically."""
        # Test with increasing parameter variations
        n_values = [5, 10, 20]
        timings = []

        for n in n_values:
            start = time.time()

            for i in range(n):
                p2 = 2.0 + 0.1 * (i / n)
                framework = GIFTFrameworkV21(p2=p2)
                obs = framework.compute_all_observables()

            elapsed = time.time() - start
            timings.append((n, elapsed))

        # Fit linear model: time = a * n + b
        # If growth is linear, R^2 should be high
        n_array = np.array([t[0] for t in timings])
        time_array = np.array([t[1] for t in timings])

        # Simple linear fit
        slope = np.cov(n_array, time_array)[0, 1] / np.var(n_array)
        intercept = np.mean(time_array) - slope * np.mean(n_array)

        predicted = slope * n_array + intercept
        residuals = time_array - predicted

        r_squared = 1 - (np.sum(residuals**2) / np.sum((time_array - np.mean(time_array))**2))

        print(f"\n📊 Linear fit R²: {r_squared:.3f}")

        # Should be approximately linear
        assert r_squared > 0.8, (
            f"Non-linear scaling detected: R²={r_squared:.3f}"
        )


class TestBottleneckIdentification:
    """Identify performance bottlenecks."""

    def test_identify_slowest_observables(self):
        """Identify which observables are slowest to compute."""
        framework = GIFTFrameworkV21()

        # This is a simplified test - real profiling would use cProfile
        # For now, just test that we can compute all observables

        start = time.time()
        obs = framework.compute_all_observables()
        total_time = time.time() - start

        print(f"\n⏱️  Total time for {len(obs)} observables: {total_time:.3f}s")
        print(f"⏱️  Average time per observable: {total_time/len(obs):.4f}s")

        # Should be efficient
        assert total_time / len(obs) < 0.1, "Observables taking too long on average"

    def test_initialization_overhead(self):
        """Test initialization overhead vs computation time."""
        # Measure initialization
        start = time.time()
        framework = GIFTFrameworkV21()
        init_time = time.time() - start

        # Measure computation
        start = time.time()
        obs = framework.compute_all_observables()
        compute_time = time.time() - start

        print(f"\n⏱️  Init: {init_time:.4f}s, Compute: {compute_time:.3f}s")

        # Initialization should be small fraction of total
        assert init_time < 0.5 * compute_time, (
            "Initialization overhead too high"
        )


class TestCachingEffectiveness:
    """Test effectiveness of any caching mechanisms."""

    def test_repeated_access_performance(self):
        """Test that repeated access to same observable is fast."""
        framework = GIFTFrameworkV21()

        # First access (compute)
        start = time.time()
        obs1 = framework.compute_all_observables()
        first_time = time.time() - start

        # Second access (may be cached)
        start = time.time()
        obs2 = framework.compute_all_observables()
        second_time = time.time() - start

        print(f"\n⏱️  First access: {first_time:.3f}s")
        print(f"⏱️  Second access: {second_time:.3f}s")

        # If there's caching, second should be faster or same
        # If no caching, should still be comparable
        assert second_time <= 2 * first_time


class TestNumericalPrecisionVsSpeed:
    """Test trade-offs between precision and speed."""

    def test_precision_maintained_despite_optimizations(self):
        """Test that any speed optimizations don't compromise precision."""
        framework = GIFTFrameworkV21()

        # Compute observables
        obs = framework.compute_all_observables()

        # Known exact values
        exact_values = {
            "delta_CP": 197.0,
            "Q_Koide": 2/3,
            "m_tau_m_e": 3477.0,
            "m_s_m_d": 20.0,
        }

        for obs_name, exact_val in exact_values.items():
            if obs_name in obs:
                computed_val = obs[obs_name]

                # Should maintain high precision
                rel_error = abs(computed_val - exact_val) / exact_val

                assert rel_error < 1e-10, (
                    f"{obs_name}: precision compromised ({rel_error:.2e} relative error)"
                )


class TestConcurrentPerformance:
    """Test performance with concurrent operations."""

    def test_no_interference_between_instances(self):
        """Test multiple instances don't interfere."""
        # Create multiple instances
        frameworks = [GIFTFrameworkV21() for _ in range(3)]

        # Time sequential execution
        start = time.time()
        results_seq = []
        for fw in frameworks:
            results_seq.append(fw.compute_all_observables())
        sequential_time = time.time() - start

        # Create fresh instances
        frameworks2 = [GIFTFrameworkV21() for _ in range(3)]

        # Time again (should be similar)
        start = time.time()
        results_seq2 = []
        for fw in frameworks2:
            results_seq2.append(fw.compute_all_observables())
        sequential_time2 = time.time() - start

        print(f"\n⏱️  Sequential run 1: {sequential_time:.3f}s")
        print(f"⏱️  Sequential run 2: {sequential_time2:.3f}s")

        # Should have similar performance
        assert abs(sequential_time - sequential_time2) < 0.5 * sequential_time


class TestBenchmarkReporting:
    """Generate benchmark reports."""

    def test_generate_performance_report(self, tmp_path):
        """Generate comprehensive performance benchmark report."""
        import json
        import datetime

        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "2.1.0",
            "benchmarks": {}
        }

        # Framework initialization
        start = time.time()
        framework = GIFTFrameworkV21()
        init_time = time.time() - start

        report["benchmarks"]["initialization"] = {
            "time_seconds": init_time,
            "baseline_seconds": PERFORMANCE_BASELINES["framework_initialization"],
            "within_baseline": init_time < PERFORMANCE_BASELINES["framework_initialization"]
        }

        # Observable computation
        start = time.time()
        obs = framework.compute_all_observables()
        compute_time = time.time() - start

        report["benchmarks"]["all_observables"] = {
            "time_seconds": compute_time,
            "n_observables": len(obs),
            "time_per_observable": compute_time / len(obs),
            "baseline_seconds": PERFORMANCE_BASELINES["all_observables_computation"],
            "within_baseline": compute_time < PERFORMANCE_BASELINES["all_observables_computation"]
        }

        # Save report
        report_file = tmp_path / "performance_benchmark.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*60)
        print(f"Initialization: {init_time*1000:.1f}ms")
        print(f"All observables: {compute_time:.3f}s ({len(obs)} observables)")
        print(f"Per observable: {compute_time/len(obs)*1000:.1f}ms")
        print("="*60)

        assert report_file.exists()
