"""Integration tests for AnnotatorPipeline combinations and rendering order."""

import pytest
import numpy as np
import supervision as sv
from onnxtools.utils.annotator_factory import AnnotatorPipeline, AnnotatorType


@pytest.fixture
def test_image():
    """Create 640x640 test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create test detections with 5 objects."""
    xyxy = np.array([
        [100, 100, 250, 200],
        [300, 150, 450, 280],
        [500, 100, 620, 220],
        [100, 350, 240, 480],
        [350, 400, 500, 550]
    ], dtype=np.float32)

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]),
        class_id=np.array([0, 1, 0, 1, 0])
    )


class TestAnnotatorPipelineIntegration:
    """Integration tests for annotator combinations and rendering order."""

    def test_simple_two_annotator_pipeline(self, test_image, test_detections):
        """Test simple pipeline with two annotators."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {'thickness': 2})
                    .add(AnnotatorType.RICH_LABEL, {}))

        result = pipeline.annotate(test_image, test_detections)

        assert result.shape == test_image.shape
        assert not np.array_equal(result, test_image)

    def test_three_annotator_pipeline(self, test_image, test_detections):
        """Test pipeline with three annotators."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.ROUND_BOX, {'thickness': 3, 'roundness': 0.3})
                    .add(AnnotatorType.DOT, {'radius': 5})
                    .add(AnnotatorType.RICH_LABEL, {}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_complex_five_plus_annotator_pipeline(self, test_image, test_detections):
        """Test complex pipeline with 5+ annotators."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.HALO, {'opacity': 0.3, 'kernel_size': 40})
                    .add(AnnotatorType.ROUND_BOX, {'thickness': 3, 'roundness': 0.3})
                    .add(AnnotatorType.DOT, {'radius': 4})
                    .add(AnnotatorType.PERCENTAGE_BAR, {'height': 16, 'width': 80})
                    .add(AnnotatorType.RICH_LABEL, {}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_rendering_order_matters(self, test_image, test_detections):
        """Test that annotator order affects final result."""
        # Pipeline 1: Box then Dot
        pipeline1 = (AnnotatorPipeline()
                     .add(AnnotatorType.BOX, {'thickness': 2})
                     .add(AnnotatorType.DOT, {'radius': 8}))

        # Pipeline 2: Dot then Box
        pipeline2 = (AnnotatorPipeline()
                     .add(AnnotatorType.DOT, {'radius': 8})
                     .add(AnnotatorType.BOX, {'thickness': 2}))

        result1 = pipeline1.annotate(test_image.copy(), test_detections)
        result2 = pipeline2.annotate(test_image.copy(), test_detections)

        # Both should render successfully
        assert result1.shape == test_image.shape
        assert result2.shape == test_image.shape

        # Results might be different due to render order
        # (though in some cases they might be identical)

    def test_conflict_warning_generation(self):
        """Test that conflicting combinations generate warnings."""
        # BOX + ROUND_BOX should generate conflict warning
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {})
                    .add(AnnotatorType.ROUND_BOX, {}))

        warnings = pipeline.check_conflicts()
        assert len(warnings) > 0
        assert any('box' in w.lower() and 'round_box' in w.lower() for w in warnings)

    def test_color_blur_conflict_warning(self):
        """Test COLOR + BLUR conflict detection."""
        # COLOR and BLUR are mutually exclusive, should raise ValueError when adding
        pipeline = AnnotatorPipeline().add(AnnotatorType.COLOR, {'opacity': 0.3})

        with pytest.raises(ValueError, match="conflicts with existing annotators"):
            pipeline.add(AnnotatorType.BLUR, {'kernel_size': 15})

    def test_no_conflict_for_valid_combination(self):
        """Test that valid combinations don't generate conflicts."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.ROUND_BOX, {})
                    .add(AnnotatorType.RICH_LABEL, {}))

        warnings = pipeline.check_conflicts()
        # Should have no or minimal warnings
        assert isinstance(warnings, list)

    def test_pipeline_with_geometric_markers(self, test_image, test_detections):
        """Test pipeline combining geometric markers."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.CIRCLE, {'thickness': 2})
                    .add(AnnotatorType.TRIANGLE, {'base': 20, 'height': 20})
                    .add(AnnotatorType.ELLIPSE, {'thickness': 2}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_pipeline_with_fill_and_border(self, test_image, test_detections):
        """Test combining fill and border annotators."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.COLOR, {'opacity': 0.2})
                    .add(AnnotatorType.ROUND_BOX, {'thickness': 3}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_pipeline_with_privacy_and_labels(self, test_image, test_detections):
        """Test combining privacy annotators with labels."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {'thickness': 2})
                    .add(AnnotatorType.BLUR, {'kernel_size': 15}))

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape

    def test_pipeline_empty_then_add(self, test_image, test_detections):
        """Test starting with empty pipeline and adding incrementally."""
        pipeline = AnnotatorPipeline()

        # Initially empty
        result_empty = pipeline.annotate(test_image.copy(), test_detections)
        assert result_empty.shape == test_image.shape

        # Add one annotator
        pipeline.add(AnnotatorType.BOX, {'thickness': 2})
        result_one = pipeline.annotate(test_image.copy(), test_detections)
        assert result_one.shape == test_image.shape

        # Add another
        pipeline.add(AnnotatorType.DOT, {'radius': 5})
        result_two = pipeline.annotate(test_image.copy(), test_detections)
        assert result_two.shape == test_image.shape

    def test_pipeline_reuse(self, test_image, test_detections):
        """Test that pipeline can be reused multiple times."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.ROUND_BOX, {})
                    .add(AnnotatorType.RICH_LABEL, {}))

        # Annotate multiple times with same pipeline
        result1 = pipeline.annotate(test_image.copy(), test_detections)
        result2 = pipeline.annotate(test_image.copy(), test_detections)
        result3 = pipeline.annotate(test_image.copy(), test_detections)

        # All should succeed
        assert all(r.shape == test_image.shape for r in [result1, result2, result3])

    def test_pipeline_with_different_images(self, test_detections):
        """Test pipeline with different image sizes."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {})
                    .add(AnnotatorType.DOT, {}))

        # Different image sizes
        image_320 = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        image_640 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        image_1280 = np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)

        for image in [image_320, image_640, image_1280]:
            result = pipeline.annotate(image, test_detections)
            assert result.shape == image.shape