"""Contract tests for performance benchmark API - 这些测试必须在实现前编写且必须失败."""

import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.contract
@pytest.mark.performance
class TestBenchmarkContract:
    """Contract tests for drawing performance benchmark functionality."""

    def test_benchmark_function_exists(self):
        """Contract: benchmark_drawing_performance function must exist."""
        try:
            from utils.drawing import benchmark_drawing_performance
            assert callable(benchmark_drawing_performance), "Function must be callable"
        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_function_signature(self):
        """Contract: benchmark_drawing_performance must have correct signature."""
        try:
            import inspect
            from utils.drawing import benchmark_drawing_performance

            sig = inspect.signature(benchmark_drawing_performance)
            params = list(sig.parameters.keys())

            expected_params = ['image', 'detections_data', 'iterations']
            assert params == expected_params, f"Expected {expected_params}, got {params}"

            # Check default value for iterations
            assert sig.parameters['iterations'].default == 100, "iterations default must be 100"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_return_format(self, sample_image, sample_detections):
        """Contract: benchmark function must return performance metrics in correct format."""
        try:
            from utils.drawing import benchmark_drawing_performance

            result = benchmark_drawing_performance(sample_image, sample_detections, iterations=10)

            # Check return type
            assert isinstance(result, dict), "Must return dictionary"

            # Check required keys
            required_keys = ['pil_avg_time', 'supervision_avg_time', 'improvement_ratio']
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"

            # Check value types
            assert isinstance(result['pil_avg_time'], (int, float)), "pil_avg_time must be numeric"
            assert isinstance(result['supervision_avg_time'], (int, float)), "supervision_avg_time must be numeric"
            assert isinstance(result['improvement_ratio'], (int, float)), "improvement_ratio must be numeric"

            # Check value ranges
            assert result['pil_avg_time'] > 0, "pil_avg_time must be positive"
            assert result['supervision_avg_time'] > 0, "supervision_avg_time must be positive"
            assert result['improvement_ratio'] > 0, "improvement_ratio must be positive"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_with_varying_iterations(self, sample_image, sample_detections):
        """Contract: benchmark must work with different iteration counts."""
        try:
            from utils.drawing import benchmark_drawing_performance

            # Test with different iteration counts
            for iterations in [1, 5, 10, 50]:
                result = benchmark_drawing_performance(sample_image, sample_detections, iterations=iterations)
                assert isinstance(result, dict), f"Failed with {iterations} iterations"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_performance_target(self, sample_image, benchmark_config):
        """Contract: supervision implementation must meet performance target."""
        try:
            from utils.drawing import benchmark_drawing_performance

            # Create test data with target number of objects
            large_detections = []
            detections_list = []
            for i in range(benchmark_config["max_objects"]):
                x1, y1 = 50 + i * 25, 50 + i * 15
                x2, y2 = x1 + 50, y1 + 30
                detections_list.append([x1, y1, x2, y2, 0.9, i % 2])
            large_detections.append(detections_list)

            result = benchmark_drawing_performance(
                sample_image, large_detections,
                iterations=benchmark_config["iterations"]
            )

            # Check performance target
            target_time = benchmark_config["target_time_ms"]
            actual_time = result["supervision_avg_time"]

            assert actual_time < target_time, \
                f"Performance target not met: {actual_time:.2f}ms > {target_time}ms"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_improvement_validation(self, sample_image, sample_detections):
        """Contract: supervision implementation should show improvement over PIL."""
        try:
            from utils.drawing import benchmark_drawing_performance

            result = benchmark_drawing_performance(sample_image, sample_detections, iterations=50)

            # Check improvement ratio makes sense
            improvement_ratio = result["improvement_ratio"]
            assert improvement_ratio >= 1.0, \
                f"Supervision should be faster than PIL: ratio={improvement_ratio:.2f}"

            # Expected improvement should be significant (at least 1.5x)
            assert improvement_ratio >= 1.5, \
                f"Performance improvement insufficient: {improvement_ratio:.2f}x (target: ≥1.5x)"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_consistency(self, sample_image, sample_detections):
        """Contract: benchmark results should be consistent across runs."""
        try:
            from utils.drawing import benchmark_drawing_performance

            # Run benchmark multiple times
            results = []
            for _ in range(3):
                result = benchmark_drawing_performance(sample_image, sample_detections, iterations=20)
                results.append(result)

            # Check consistency (within 50% variance)
            supervision_times = [r["supervision_avg_time"] for r in results]
            pil_times = [r["pil_avg_time"] for r in results]

            supervision_std = np.std(supervision_times)
            supervision_mean = np.mean(supervision_times)
            supervision_cv = supervision_std / supervision_mean if supervision_mean > 0 else 0

            pil_std = np.std(pil_times)
            pil_mean = np.mean(pil_times)
            pil_cv = pil_std / pil_mean if pil_mean > 0 else 0

            assert supervision_cv < 0.5, f"Supervision timing too inconsistent: CV={supervision_cv:.2f}"
            assert pil_cv < 0.5, f"PIL timing too inconsistent: CV={pil_cv:.2f}"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_with_empty_detections(self, sample_image):
        """Contract: benchmark must handle empty detections."""
        try:
            from utils.drawing import benchmark_drawing_performance

            empty_detections = [[]]
            result = benchmark_drawing_performance(sample_image, empty_detections, iterations=10)

            assert isinstance(result, dict), "Must handle empty detections"
            assert all(result[key] >= 0 for key in result.keys()), "Times must be non-negative"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_with_large_image(self, sample_detections):
        """Contract: benchmark must work with different image sizes."""
        try:
            from utils.drawing import benchmark_drawing_performance

            # Create larger image
            large_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            large_image[:, :] = [100, 100, 100]

            result = benchmark_drawing_performance(large_image, sample_detections, iterations=5)

            assert isinstance(result, dict), "Must handle large images"
            assert result["supervision_avg_time"] > 0, "Must process large images"

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_error_handling(self, sample_image):
        """Contract: benchmark must handle invalid inputs gracefully."""
        try:
            from utils.drawing import benchmark_drawing_performance

            # Test with invalid detections format
            with pytest.raises((ValueError, TypeError, IndexError)):
                benchmark_drawing_performance(sample_image, "invalid", iterations=1)

            # Test with zero iterations
            with pytest.raises((ValueError, TypeError)):
                benchmark_drawing_performance(sample_image, [[]], iterations=0)

        except ImportError:
            pytest.fail("benchmark_drawing_performance function must be implemented")

    def test_benchmark_memory_efficiency(self, sample_image, sample_detections):
        """Contract: benchmark should not cause memory leaks during repeated runs."""
        try:
            import psutil
            import os
            from utils.drawing import benchmark_drawing_performance

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Run multiple benchmarks
            for _ in range(10):
                benchmark_drawing_performance(sample_image, sample_detections, iterations=5)

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 50MB)
            max_increase = 50 * 1024 * 1024  # 50MB
            assert memory_increase < max_increase, \
                f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase"

        except ImportError:
            pytest.skip("psutil not available for memory testing")