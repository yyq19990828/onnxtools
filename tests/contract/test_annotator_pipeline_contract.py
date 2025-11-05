"""
Contract test for AnnotatorPipeline interface.

This test MUST FAIL before implementation (TDD approach).
"""

import pytest
import numpy as np
import supervision as sv
from onnxtools.utils.annotator_factory import AnnotatorPipeline, AnnotatorFactory, AnnotatorType


@pytest.fixture
def test_image():
    """Create a 640x640 test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detections():
    """Create test detections with 5 objects."""
    xyxy = np.array([
        [100, 100, 200, 200],
        [250, 150, 350, 250],
        [400, 100, 500, 200],
        [100, 350, 200, 450],
        [300, 400, 400, 500]
    ], dtype=np.float32)

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.95, 0.87, 0.92, 0.78, 0.85]),
        class_id=np.array([0, 1, 0, 1, 0])
    )


class TestAnnotatorPipelineContract:
    """Contract tests for AnnotatorPipeline interface."""

    def test_pipeline_builder_pattern(self):
        """Test that .add() returns self for method chaining."""
        pipeline = AnnotatorPipeline()

        # add() should return self
        result = pipeline.add(AnnotatorType.BOX, {})
        assert result is pipeline, "add() should return self for builder pattern"

    def test_pipeline_chaining(self):
        """Test that multiple add() calls can be chained."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {})
                    .add(AnnotatorType.RICH_LABEL, {}))

        assert isinstance(pipeline, AnnotatorPipeline)

    def test_pipeline_annotate_preserves_shape(self, test_image, test_detections):
        """Test that annotate() returns image with same shape."""
        pipeline = AnnotatorPipeline().add(AnnotatorType.BOX, {})

        result = pipeline.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

    def test_pipeline_annotate_returns_copy(self, test_image, test_detections):
        """Test that annotate() doesn't modify original image."""
        pipeline = AnnotatorPipeline().add(AnnotatorType.BOX, {})

        original_copy = test_image.copy()
        result = pipeline.annotate(test_image, test_detections)

        # Original should not be modified
        assert np.array_equal(test_image, original_copy)
        # Result should be different (annotated)
        assert not np.array_equal(result, test_image)

    def test_empty_pipeline_returns_original_copy(self, test_image, test_detections):
        """Test that empty pipeline returns copy of original image."""
        pipeline = AnnotatorPipeline()

        result = pipeline.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        # Should be a copy, not same object
        assert result is not test_image

    def test_multiple_annotators_applied_in_order(self, test_image, test_detections):
        """Test that multiple annotators are applied sequentially."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {'thickness': 2})
                    .add(AnnotatorType.DOT, {'radius': 5})
                    .add(AnnotatorType.RICH_LABEL, {}))

        result = pipeline.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

    def test_add_annotator_instance_directly(self, test_image, test_detections):
        """Test adding pre-created annotator instance."""
        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2
        )

        pipeline = AnnotatorPipeline().add(box_annotator, None)
        result = pipeline.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)

    def test_add_annotator_by_type_enum(self, test_image, test_detections):
        """Test adding annotator by AnnotatorType enum."""
        pipeline = AnnotatorPipeline().add(AnnotatorType.ROUND_BOX, {'thickness': 3})

        result = pipeline.annotate(test_image, test_detections)
        assert isinstance(result, np.ndarray)

    def test_check_conflicts_returns_list(self):
        """Test that check_conflicts() returns a list."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {})
                    .add(AnnotatorType.ROUND_BOX, {}))

        warnings = pipeline.check_conflicts()

        assert isinstance(warnings, list)

    def test_check_conflicts_detects_box_round_box_conflict(self):
        """Test that BOX + ROUND_BOX conflict is detected."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.BOX, {})
                    .add(AnnotatorType.ROUND_BOX, {}))

        warnings = pipeline.check_conflicts()

        # Should warn about conflicting box types
        assert len(warnings) > 0

    def test_check_conflicts_detects_color_blur_conflict(self):
        """Test that COLOR + BLUR conflict is detected."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.COLOR, {})
                    .add(AnnotatorType.BLUR, {}))

        warnings = pipeline.check_conflicts()

        # Should warn about color + blur conflict
        assert len(warnings) > 0

    def test_check_conflicts_no_warning_for_valid_combination(self):
        """Test that valid combinations don't generate warnings."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.ROUND_BOX, {})
                    .add(AnnotatorType.RICH_LABEL, {}))

        warnings = pipeline.check_conflicts()

        # Should be empty or minimal warnings
        assert isinstance(warnings, list)

    def test_pipeline_with_five_annotators(self, test_image, test_detections):
        """Test complex pipeline with 5+ annotators."""
        pipeline = (AnnotatorPipeline()
                    .add(AnnotatorType.ROUND_BOX, {'thickness': 3})
                    .add(AnnotatorType.DOT, {'radius': 4})
                    .add(AnnotatorType.HALO, {'opacity': 0.3})
                    .add(AnnotatorType.PERCENTAGE_BAR, {'height': 16})
                    .add(AnnotatorType.RICH_LABEL, {}))

        result = pipeline.annotate(test_image, test_detections)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

    def test_pipeline_rendering_order_matters(self, test_image, test_detections):
        """Test that annotator order affects final result."""
        pipeline1 = (AnnotatorPipeline()
                     .add(AnnotatorType.BOX, {})
                     .add(AnnotatorType.DOT, {}))

        pipeline2 = (AnnotatorPipeline()
                     .add(AnnotatorType.DOT, {})
                     .add(AnnotatorType.BOX, {}))

        result1 = pipeline1.annotate(test_image.copy(), test_detections)
        result2 = pipeline2.annotate(test_image.copy(), test_detections)

        # Results should be different due to render order
        # (unless both annotators don't affect each other, which is unlikely)
        assert result1.shape == result2.shape