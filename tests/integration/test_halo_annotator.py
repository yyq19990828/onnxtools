"""Integration test for HaloAnnotator."""


from onnxtools.utils.supervision_annotator import AnnotatorFactory, AnnotatorType


class TestHaloAnnotatorIntegration:
    """Integration tests for HaloAnnotator."""

    def test_halo_basic_rendering(self, test_image, test_detections):
        """Test basic halo effect rendering."""
        annotator = AnnotatorFactory.create(AnnotatorType.HALO, {"opacity": 0.3, "kernel_size": 40})
        result = annotator.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
        # Image rendered successfully

    def test_halo_different_kernel_sizes(self, test_image, test_detections):
        """Test different kernel sizes for halo effect."""
        for kernel_size in [20, 40, 60, 80]:
            annotator = AnnotatorFactory.create(AnnotatorType.HALO, {"opacity": 0.3, "kernel_size": kernel_size})
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_halo_different_opacity(self, test_image, test_detections):
        """Test different opacity values."""
        for opacity in [0.1, 0.3, 0.5, 0.7]:
            annotator = AnnotatorFactory.create(AnnotatorType.HALO, {"opacity": opacity, "kernel_size": 40})
            result = annotator.annotate(test_image.copy(), test_detections)
            assert result.shape == test_image.shape

    def test_halo_with_other_annotators(self, test_image, test_detections):
        """Test halo combined with box annotator."""
        from onnxtools.utils.supervision_annotator import AnnotatorPipeline

        pipeline = (
            AnnotatorPipeline()
            .add(AnnotatorType.HALO, {"opacity": 0.3, "kernel_size": 40})
            .add(AnnotatorType.BOX, {"thickness": 2})
        )

        result = pipeline.annotate(test_image, test_detections)
        assert result.shape == test_image.shape
