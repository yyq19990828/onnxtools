"""Integration tests for pipeline compatibility - 这些测试必须在实现前编写且必须失败."""

import pytest
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline compatibility with supervision changes."""

    def test_pipeline_drawing_integration(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Pipeline should work seamlessly with new drawing functions."""
        try:
            from onnxtools.pipeline import process_frame
            from onnxtools.utils.drawing import draw_detections

            # Mock the pipeline call to drawing function
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image.shape

        except ImportError:
            # If pipeline is not available, test direct integration
            from onnxtools.utils.drawing import draw_detections
            result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)
            assert isinstance(result, np.ndarray)

    def test_pipeline_data_format_compatibility(self, sample_image):
        """Integration: Pipeline detection format should work with new conversion functions."""
        # Simulate pipeline detection output format
        pipeline_detections = [
            [
                [100.0, 100.0, 200.0, 150.0, 0.95, 0],  # vehicle
                [150.0, 120.0, 180.0, 135.0, 0.88, 1],  # plate
            ]
        ]

        pipeline_class_names = {0: "vehicle", 1: "plate"}
        pipeline_colors = [(255, 0, 0), (0, 255, 0)]

        # Test conversion compatibility
        try:
            from onnxtools.utils.supervision_converter import convert_to_supervision_detections
            import supervision as sv

            sv_detections = convert_to_supervision_detections(pipeline_detections, pipeline_class_names)
            assert isinstance(sv_detections, sv.Detections)

        except ImportError:
            pytest.fail("Pipeline data format conversion must be supported")

    def test_pipeline_ocr_integration_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Pipeline OCR results should integrate correctly."""
        # Simulate pipeline OCR output
        pipeline_ocr_results = [
            None,  # vehicle (no OCR)
            {
                "plate_text": "京A12345",
                "color": "蓝色",
                "layer": "单层",
                "should_display_ocr": True
            }
        ]

        from onnxtools.utils.drawing import draw_detections

        result = draw_detections(
            sample_image, sample_detections, sample_class_names,
            sample_colors, plate_results=pipeline_ocr_results
        )

        assert isinstance(result, np.ndarray)
        assert not np.array_equal(result, sample_image)

    def test_main_script_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Main script should work with supervision integration."""
        # Test main.py compatibility by calling draw_detections as main.py would
        from onnxtools.utils.drawing import draw_detections

        # Simulate main.py call pattern
        result = draw_detections(
            image=sample_image,
            detections=sample_detections,
            class_names=sample_class_names,
            colors=sample_colors,
            plate_results=None,
            font_path="SourceHanSans-VF.ttf"
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_batch_processing_compatibility(self, sample_class_names, sample_colors):
        """Integration: Batch processing should work efficiently."""
        # Simulate batch processing scenario
        batch_images = []
        batch_detections = []

        for i in range(5):  # 5 images in batch
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[:, :] = [100 + i * 20, 100 + i * 20, 100 + i * 20]
            batch_images.append(image)

            detections = [[[100.0 + i * 50, 100.0, 200.0 + i * 50, 150.0, 0.9, 0]]]
            batch_detections.append(detections)

        from onnxtools.utils.drawing import draw_detections
        import time

        # Process batch
        start_time = time.time()
        results = []
        for image, detections in zip(batch_images, batch_detections):
            result = draw_detections(image, detections, sample_class_names, sample_colors)
            results.append(result)
        processing_time = time.time() - start_time

        # Validate results
        assert len(results) == 5
        for result in results:
            assert isinstance(result, np.ndarray)

        # Performance check
        avg_time_per_frame = processing_time / 5 * 1000  # ms
        assert avg_time_per_frame < 50.0, f"Batch processing too slow: {avg_time_per_frame:.2f}ms per frame"

    def test_video_processing_compatibility(self, sample_detections, sample_class_names, sample_colors):
        """Integration: Video processing pipeline should work correctly."""
        # Simulate video frame processing
        video_frames = []
        for i in range(10):  # 10 frames
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            video_frames.append(frame)

        from onnxtools.utils.drawing import draw_detections

        processed_frames = []
        for frame in video_frames:
            result = draw_detections(frame, sample_detections, sample_class_names, sample_colors)
            processed_frames.append(result)

        assert len(processed_frames) == 10
        for frame in processed_frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)

    def test_real_time_processing_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Real-time processing requirements should be met."""
        from onnxtools.utils.drawing import draw_detections
        import time

        # Simulate real-time processing (30 FPS requirement = ~33ms per frame)
        frame_times = []
        for _ in range(10):
            start_time = time.time()
            result = draw_detections(sample_image.copy(), sample_detections, sample_class_names, sample_colors)
            frame_time = (time.time() - start_time) * 1000  # ms
            frame_times.append(frame_time)

            assert isinstance(result, np.ndarray)

        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)

        # Real-time requirements
        assert avg_frame_time < 33.0, f"Average frame time too slow: {avg_frame_time:.2f}ms (target: <33ms)"
        assert max_frame_time < 50.0, f"Maximum frame time too slow: {max_frame_time:.2f}ms (target: <50ms)"

    def test_cli_argument_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: CLI arguments should work with new implementation."""
        from onnxtools.utils.drawing import draw_detections

        # Test different CLI-like configurations
        cli_configs = [
            {"font_path": "SourceHanSans-VF.ttf"},
            {"font_path": None},
            {},  # Default configuration
        ]

        for config in cli_configs:
            try:
                result = draw_detections(
                    sample_image, sample_detections, sample_class_names, sample_colors,
                    **config
                )
                assert isinstance(result, np.ndarray)
            except TypeError:
                # Some configurations might not be supported yet
                pass

    def test_output_format_pipeline_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Output formats should remain compatible with existing pipeline."""
        from onnxtools.utils.drawing import draw_detections
        import cv2

        result = draw_detections(sample_image, sample_detections, sample_class_names, sample_colors)

        # Test cv2 compatibility
        assert result.dtype == np.uint8, "Output must be uint8 for cv2 compatibility"
        assert len(result.shape) == 3, "Output must be 3D for cv2"
        assert result.shape[2] == 3, "Output must be BGR format"

        # Test that cv2 operations work
        try:
            # Simulate cv2.imwrite operation
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            assert gray.shape == result.shape[:2]

            # Simulate cv2.imshow operation
            assert result.min() >= 0 and result.max() <= 255, "Values must be in valid range"

        except Exception as e:
            pytest.fail(f"cv2 compatibility failed: {e}")

    def test_error_handling_pipeline_compatibility(self, sample_image, sample_class_names, sample_colors):
        """Integration: Error handling should maintain pipeline stability."""
        from onnxtools.utils.drawing import draw_detections

        # Test error scenarios that pipeline might encounter
        error_scenarios = [
            [[]],  # Empty detections
            [[[float('inf'), 100.0, 200.0, 150.0, 0.9, 0]]],  # Invalid coordinates
            [[[100.0, 100.0, 200.0, 150.0, float('nan'), 0]]],  # Invalid confidence
            [[[-100.0, -100.0, 50.0, 50.0, 0.9, 0]]],  # Negative coordinates
        ]

        for scenario in error_scenarios:
            try:
                result = draw_detections(sample_image, scenario, sample_class_names, sample_colors)
                # If no error, result should still be valid
                assert isinstance(result, np.ndarray)
            except (ValueError, TypeError, OverflowError):
                # These errors are acceptable for invalid data
                pass

    def test_memory_usage_pipeline_compatibility(self, sample_class_names, sample_colors):
        """Integration: Memory usage should be efficient for pipeline processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        from onnxtools.utils.drawing import draw_detections

        # Process multiple frames to check for memory leaks
        for i in range(20):
            image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            detections = [[[100.0, 100.0, 200.0, 150.0, 0.9, 0]]]

            result = draw_detections(image, detections, sample_class_names, sample_colors)
            assert isinstance(result, np.ndarray)

            # Clean up references
            del result, image, detections

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 20MB for 20 frames)
        max_increase = 20 * 1024 * 1024  # 20MB
        assert memory_increase < max_increase, \
            f"Memory usage increased too much: {memory_increase / 1024 / 1024:.1f}MB"

    def test_configuration_pipeline_compatibility(self, sample_image, sample_detections):
        """Integration: Configuration loading should work with pipeline."""
        # Test with different configuration formats
        config_variants = [
            # Standard configuration
            {
                "class_names": {0: "vehicle", 1: "plate"},
                "colors": [(255, 0, 0), (0, 255, 0)]
            },
            # List-based class names
            {
                "class_names": ["vehicle", "plate"],
                "colors": [(255, 0, 0), (0, 255, 0)]
            },
            # Extended colors
            {
                "class_names": {0: "vehicle", 1: "plate"},
                "colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            }
        ]

        from onnxtools.utils.drawing import draw_detections

        for config in config_variants:
            try:
                result = draw_detections(
                    sample_image, sample_detections,
                    config["class_names"], config["colors"]
                )
                assert isinstance(result, np.ndarray)
            except (ValueError, TypeError, KeyError):
                # Some configurations might not be supported
                pass

    def test_concurrent_pipeline_compatibility(self, sample_image, sample_detections, sample_class_names, sample_colors):
        """Integration: Concurrent pipeline processing should work correctly."""
        import threading
        import queue

        result_queue = queue.Queue()
        error_queue = queue.Queue()

        def pipeline_worker(worker_id):
            try:
                from onnxtools.utils.drawing import draw_detections
                for i in range(5):  # Process 5 frames per worker
                    image = sample_image.copy()
                    result = draw_detections(image, sample_detections, sample_class_names, sample_colors)
                    result_queue.put((worker_id, i, result))
            except Exception as e:
                error_queue.put((worker_id, e))

        # Create multiple workers
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=pipeline_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert error_queue.empty(), f"Concurrent processing errors: {list(error_queue.queue)}"

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == 15, "All workers should complete"  # 3 workers * 5 frames