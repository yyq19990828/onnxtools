"""Result class for wrapping BaseORT inference outputs.

This module provides a Result class that wraps detection results from BaseORT
subclasses, offering object-oriented access to bounding boxes, scores, class IDs,
and providing convenient methods for visualization, filtering, and data conversion.

Author: ONNX Vehicle Plate Recognition Team
Date: 2025-11-05
Version: 1.0.0
"""

from typing import Optional, Union, List, Dict, Any
import warnings
import numpy as np
import numpy.typing as npt

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Result:
    """Detection result wrapper class for BaseORT inference outputs.

    This class provides a convenient object-oriented interface for accessing
    and manipulating detection results, including:
    - Read-only property access to boxes, scores, class_ids, etc.
    - Indexing and slicing operations
    - Visualization methods (plot, show, save)
    - Filtering and data conversion utilities

    The class implements shallow immutability: attributes are read-only,
    but internal numpy array elements can be modified.

    Attributes:
        boxes: Bounding boxes in xyxy format [N, 4]
        scores: Confidence scores [N]
        class_ids: Class ID integers [N]
        orig_img: Original input image (BGR format)
        orig_shape: Original image shape (height, width)
        names: Mapping from class ID to class name
        path: Image file path (optional)

    Example:
        >>> result = Result(
        ...     boxes=np.array([[10, 20, 100, 150]]),
        ...     scores=np.array([0.95]),
        ...     class_ids=np.array([0]),
        ...     orig_shape=(640, 640),
        ...     names={0: 'vehicle'}
        ... )
        >>> print(len(result))  # 1
        >>> print(result.boxes)  # [[10 20 100 150]]
        >>> first_det = result[0]  # Index access
    """

    def __init__(
        self,
        boxes: Optional[npt.NDArray[np.float32]] = None,
        scores: Optional[npt.NDArray[np.float32]] = None,
        class_ids: Optional[npt.NDArray[np.int32]] = None,
        orig_img: Optional[npt.NDArray[np.uint8]] = None,
        orig_shape: Optional[tuple[int, int]] = None,
        names: Optional[dict[int, str]] = None,
        path: Optional[str] = None
    ) -> None:
        """Initialize Result object with detection data.

        Args:
            boxes: Bounding boxes in xyxy format [N, 4]. Can be None for empty results.
            scores: Confidence scores [N]. Can be None for empty results.
            class_ids: Class ID integers [N]. Can be None for empty results.
            orig_img: Original input image (BGR format). Can be None if visualization not needed.
            orig_shape: Original image shape (height, width). Required, cannot be None.
            names: Mapping from class ID to class name. Defaults to empty dict.
            path: Image file path. Optional.

        Raises:
            TypeError: If orig_shape is None.
            ValueError: If orig_shape is not a tuple of length 2.
            ValueError: If boxes, scores, or class_ids have inconsistent shapes.
        """
        # V1: orig_shape validation - must not be None
        if orig_shape is None:
            raise TypeError("orig_shape is required and cannot be None")

        # V2: orig_shape validation - must be a tuple of length 2
        if not (isinstance(orig_shape, tuple) and len(orig_shape) == 2):
            raise ValueError("orig_shape must be a tuple of (height, width)")

        # V3: boxes shape validation (only if not None)
        if boxes is not None:
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}")

        # V4: scores shape validation (only if not None)
        if scores is not None:
            if scores.ndim != 1:
                raise ValueError(f"scores must have shape (N,), got {scores.shape}")

        # V5: class_ids shape validation (only if not None)
        if class_ids is not None:
            if class_ids.ndim != 1:
                raise ValueError(f"class_ids must have shape (N,), got {class_ids.shape}")

        # V6: Length consistency validation
        lengths = []
        if boxes is not None:
            lengths.append(('boxes', len(boxes)))
        if scores is not None:
            lengths.append(('scores', len(scores)))
        if class_ids is not None:
            lengths.append(('class_ids', len(class_ids)))

        if lengths:
            first_len = lengths[0][1]
            for name, length in lengths[1:]:
                if length != first_len:
                    raise ValueError("boxes, scores, and class_ids must have the same length")

        # Store private attributes
        self._boxes = boxes
        self._scores = scores
        self._class_ids = class_ids
        self._orig_img = orig_img
        self._orig_shape = orig_shape
        self._names = names if names is not None else {}
        self._path = path

    # ============================================================================
    # Read-only properties (T006)
    # ============================================================================

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        """Bounding boxes in xyxy format [N, 4].

        Returns:
            np.ndarray: Bounding boxes array. If None during init, returns empty array with shape (0, 4).
        """
        if self._boxes is None:
            return np.empty((0, 4), dtype=np.float32)
        return self._boxes

    @property
    def scores(self) -> npt.NDArray[np.float32]:
        """Confidence scores [N].

        Returns:
            np.ndarray: Confidence scores array. If None during init, returns empty array with shape (0,).
        """
        if self._scores is None:
            return np.empty((0,), dtype=np.float32)
        return self._scores

    @property
    def class_ids(self) -> npt.NDArray[np.int32]:
        """Class ID integers [N].

        Returns:
            np.ndarray: Class IDs array. If None during init, returns empty array with shape (0,).
        """
        if self._class_ids is None:
            return np.empty((0,), dtype=np.int32)
        return self._class_ids

    @property
    def orig_img(self) -> Optional[npt.NDArray[np.uint8]]:
        """Original input image (BGR format).

        Returns:
            np.ndarray or None: Original image array, or None if not provided.
        """
        return self._orig_img

    @property
    def orig_shape(self) -> tuple[int, int]:
        """Original image shape (height, width).

        Returns:
            tuple[int, int]: Original image dimensions.
        """
        return self._orig_shape

    @property
    def names(self) -> dict[int, str]:
        """Mapping from class ID to class name.

        Returns:
            dict[int, str]: Class names dictionary.
        """
        return self._names

    @property
    def path(self) -> Optional[str]:
        """Image file path.

        Returns:
            str or None: File path if provided, otherwise None.
        """
        return self._path

    # ============================================================================
    # Magic methods (T007, T008)
    # ============================================================================

    def __len__(self) -> int:
        """Return the number of detections.

        Returns:
            int: Number of detected objects (N).

        Example:
            >>> result = Result(boxes=np.array([[10, 20, 30, 40]]), orig_shape=(640, 640))
            >>> len(result)  # 1
        """
        return len(self.boxes)

    def __repr__(self) -> str:
        """Return developer-friendly string representation.

        Returns:
            str: Detailed representation of Result object.
        """
        return (
            f"Result(n={len(self)}, "
            f"orig_shape={self.orig_shape}, "
            f"has_img={self.orig_img is not None})"
        )

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns:
            str: Summary of detection results.
        """
        num_dets = len(self)
        if num_dets == 0:
            return "Result: No detections"

        class_counts = {}
        for class_id in self.class_ids:
            class_name = self.names.get(int(class_id), f"class_{class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        counts_str = ", ".join([f"{name}: {count}" for name, count in class_counts.items()])
        return f"Result: {num_dets} detection(s) ({counts_str})"

    def __getitem__(self, index: Union[int, slice]) -> "Result":
        """Get detection(s) by index or slice (T011, T012).

        This method returns a new Result object containing a subset of detections.
        Uses numpy views to avoid unnecessary array copies for memory efficiency.

        Args:
            index: Integer index (single detection) or slice object (range of detections).

        Returns:
            Result: New Result object with selected detection(s).

        Raises:
            IndexError: If index is out of bounds.

        Example:
            >>> result[0]  # First detection
            >>> result[1:3]  # Detections 1 and 2
            >>> result[-1]  # Last detection
        """
        # Handle integer index: need to preserve 2D shape for boxes
        if isinstance(index, int):
            # Convert negative index to positive for slice construction
            if index < 0:
                # For negative index, convert to positive: -1 -> len-1
                actual_index = len(self) + index
                if actual_index < 0:
                    raise IndexError(f"index {index} is out of bounds")
                # Use internal arrays to preserve None when appropriate
                boxes_indexed = self._boxes[actual_index:actual_index+1] if self._boxes is not None else None
                scores_indexed = self._scores[actual_index:actual_index+1] if self._scores is not None else None
                class_ids_indexed = self._class_ids[actual_index:actual_index+1] if self._class_ids is not None else None
            else:
                # Positive index: check bounds first
                if index >= len(self):
                    raise IndexError(f"index {index} is out of bounds for axis 0 with size {len(self)}")
                # Use internal arrays to preserve None when appropriate
                boxes_indexed = self._boxes[index:index+1] if self._boxes is not None else None
                scores_indexed = self._scores[index:index+1] if self._scores is not None else None
                class_ids_indexed = self._class_ids[index:index+1] if self._class_ids is not None else None
        else:
            # Slice: numpy automatically preserves dimensions
            boxes_indexed = self._boxes[index] if self._boxes is not None else None
            scores_indexed = self._scores[index] if self._scores is not None else None
            class_ids_indexed = self._class_ids[index] if self._class_ids is not None else None

        return Result(
            boxes=boxes_indexed,
            scores=scores_indexed,
            class_ids=class_ids_indexed,
            orig_img=self.orig_img,  # Shared original image
            orig_shape=self.orig_shape,  # Shared shape
            names=self.names,  # Shared names dict
            path=self.path  # Shared path
        )

    # ============================================================================
    # Data conversion methods (T013, T015)
    # ============================================================================

    def numpy(self) -> "Result":
        """Ensure all internal data are numpy.ndarray format (T013).

        This is an idempotent operation. Since Result class already stores
        data as numpy arrays, this method simply returns self.

        Returns:
            Result: Self reference (all data already numpy arrays).

        Example:
            >>> result = result.numpy()  # Idempotent
        """
        return self

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert Result object to dictionary format (T015).

        .. deprecated:: 0.2.0
            `to_dict()` will be removed in version 0.3.0.
            Use Result object's property access instead (e.g., `result.boxes`).

        Returns:
            dict: Dictionary with 'boxes', 'scores', 'class_ids' keys.

        Example:
            >>> result_dict = result.to_dict()  # Deprecated, use result.boxes instead
        """
        warnings.warn(
            "to_dict()方法已废弃，将在第2个迭代（v0.3.0）移除。"
            "请使用Result对象的属性访问（如result.boxes）代替。",
            DeprecationWarning,
            stacklevel=2  # Show caller location, not to_dict() itself
        )
        return {
            'boxes': self.boxes,
            'scores': self.scores,
            'class_ids': self.class_ids
        }

    def to_supervision(self) -> "sv.Detections":
        """Convert Result object to supervision.Detections format (T025).

        This method creates a supervision.Detections object compatible with
        the Supervision library's annotator pipeline for advanced visualization.

        Returns:
            supervision.Detections: Supervision detections object with xyxy,
                confidence, class_id, and class_name data.

        Raises:
            ImportError: If supervision library is not installed.

        Example:
            >>> from onnxtools import create_detector
            >>> detector = create_detector('yolo', 'models/yolo11n.onnx')
            >>> result = detector(image)
            >>> sv_detections = result.to_supervision()
            >>> # Use with Supervision annotators
            >>> import supervision as sv
            >>> box_annotator = sv.BoxAnnotator()
            >>> annotated = box_annotator.annotate(image.copy(), sv_detections)
        """
        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library is required for to_supervision() method. "
                "Install it with: pip install supervision==0.26.1"
            )

        # Handle empty results
        if len(self) == 0:
            return sv.Detections.empty()

        # Build class_name list from names dict
        class_names_list = []
        for cls_id in self.class_ids:
            cls_id_int = int(cls_id)
            if cls_id_int in self.names:
                class_names_list.append(self.names[cls_id_int])
            else:
                class_names_list.append(f"class_{cls_id_int}")

        # Create supervision.Detections object
        return sv.Detections(
            xyxy=self.boxes,
            confidence=self.scores,
            class_id=self.class_ids,
            data={'class_name': class_names_list}
        )

    # ============================================================================
    # Visualization methods (T026, T027, T028, T029)
    # ============================================================================

    def plot(self, annotator_preset: str = 'standard') -> npt.NDArray[np.uint8]:
        """Plot detections on the original image using Supervision pipeline (T026).

        This method draws bounding boxes and labels on the original image using
        the Supervision library's annotator pipeline. Supports multiple visualization
        presets for different use cases.

        Args:
            annotator_preset: Visualization preset name. Options:
                - 'standard': Default boxes + labels
                - 'debug': Detailed info with confidence bars
                - 'lightweight': Minimal annotations
                - 'privacy': Blur plate regions
                - 'high_contrast': Enhanced visibility

        Returns:
            np.ndarray: Annotated image in BGR format (uint8).

        Raises:
            ValueError: If orig_img is None (T029).
            ImportError: If supervision or cv2 is not installed.

        Example:
            >>> result = detector(image)
            >>> annotated = result.plot(annotator_preset='debug')
            >>> cv2.imwrite('output.jpg', annotated)
        """
        # T029: Validate orig_img is not None
        if self.orig_img is None:
            raise ValueError(
                "Cannot plot detections: orig_img is None. "
                "Ensure the image was provided when creating the Result object."
            )

        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library is required for plot() method. "
                "Install it with: pip install supervision==0.26.1"
            )

        # Import visualization utilities
        from onnxtools.utils.visualization_preset import create_preset_pipeline

        # Handle empty results - return original image copy
        if len(self) == 0:
            return self.orig_img.copy()

        # Convert to supervision format
        sv_detections = self.to_supervision()

        # Create annotator pipeline from preset
        pipeline = create_preset_pipeline(annotator_preset)

        # Annotate image
        annotated_image = pipeline.annotate(self.orig_img.copy(), sv_detections)

        return annotated_image

    def show(self, window_name: str = 'Result', annotator_preset: str = 'standard') -> None:
        """Display annotated image in a window using cv2.imshow() (T027).

        This is a convenience method that calls plot() and displays the result.
        Press any key to close the window.

        Args:
            window_name: Name of the display window.
            annotator_preset: Visualization preset (see plot() for options).

        Raises:
            ValueError: If orig_img is None (T029).
            ImportError: If cv2 is not installed.

        Example:
            >>> result = detector(image)
            >>> result.show(annotator_preset='debug')  # Press any key to close
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for show() method. "
                "Install it with: pip install opencv-contrib-python"
            )

        # plot() will handle orig_img validation (T029)
        annotated = self.plot(annotator_preset=annotator_preset)

        # Display image
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, output_path: str, annotator_preset: str = 'standard') -> None:
        """Save annotated image to file using cv2.imwrite() (T028).

        This is a convenience method that calls plot() and saves the result.

        Args:
            output_path: Path where the annotated image will be saved.
            annotator_preset: Visualization preset (see plot() for options).

        Raises:
            ValueError: If orig_img is None (T029).
            ImportError: If cv2 is not installed.
            RuntimeError: If image save fails.

        Example:
            >>> result = detector(image)
            >>> result.save('output/annotated.jpg', annotator_preset='debug')
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for save() method. "
                "Install it with: pip install opencv-contrib-python"
            )

        # plot() will handle orig_img validation (T029)
        annotated = self.plot(annotator_preset=annotator_preset)

        # Save image
        success = cv2.imwrite(output_path, annotated)
        if not success:
            raise RuntimeError(f"Failed to save image to {output_path}")

    # ===== Phase 5: Filtering and Transformations (US3) =====

    def filter(
        self,
        conf_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> "Result":
        """Filter detections by confidence threshold and/or class IDs.

        Creates a new Result object containing only detections that match
        the specified filter criteria. Original Result remains unchanged.

        Args:
            conf_threshold: Minimum confidence threshold (0.0 - 1.0).
                          If None, no confidence filtering is applied.
            classes: List of class IDs to keep. If None, no class filtering.

        Returns:
            Result: New Result object with filtered detections.

        Raises:
            ValueError: If conf_threshold is not in [0, 1] range or classes
                       contains non-integer values.

        Examples:
            >>> # Filter by confidence threshold
            >>> high_conf = result.filter(conf_threshold=0.8)
            >>>
            >>> # Filter by class IDs
            >>> vehicles_only = result.filter(classes=[0])
            >>>
            >>> # Combined filtering
            >>> high_conf_plates = result.filter(conf_threshold=0.7, classes=[1])
        """
        # T039: Parameter validation
        if conf_threshold is not None:
            if not isinstance(conf_threshold, (int, float)):
                raise ValueError(
                    f"conf_threshold must be a number, got {type(conf_threshold).__name__}"
                )
            if not 0.0 <= conf_threshold <= 1.0:
                raise ValueError(
                    f"conf_threshold must be in [0.0, 1.0] range, got {conf_threshold}"
                )

        if classes is not None:
            if not isinstance(classes, (list, tuple)):
                raise ValueError(
                    f"classes must be a list or tuple, got {type(classes).__name__}"
                )
            if not all(isinstance(c, (int, np.integer)) for c in classes):
                raise ValueError(
                    "classes must contain only integer values"
                )

        # Start with all True mask
        mask = np.ones(len(self), dtype=bool)

        # T036: Apply confidence threshold filter
        if conf_threshold is not None and len(self) > 0:
            mask &= (self.scores >= conf_threshold)

        # T037: Apply class filter
        if classes is not None and len(self) > 0:
            mask &= np.isin(self.class_ids, classes)

        # T038: Create new Result with filtered data
        if not mask.any():
            # Return empty Result
            return Result(
                boxes=None,
                scores=None,
                class_ids=None,
                orig_shape=self.orig_shape,
                names=self.names,
                path=self.path,
                orig_img=self.orig_img
            )

        # Return new Result with filtered detections
        return Result(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            orig_shape=self.orig_shape,
            names=self.names,
            path=self.path,
            orig_img=self.orig_img
        )

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of detection results.

        Returns a dictionary containing:
        - total_detections: Total number of detections
        - class_counts: Count of detections per class (dict)
        - avg_confidence: Average confidence score (float)
        - min_confidence: Minimum confidence score (float)
        - max_confidence: Maximum confidence score (float)

        Returns:
            Dict[str, Any]: Dictionary with summary statistics.

        Examples:
            >>> result = Result(boxes=boxes, scores=scores, class_ids=class_ids,
            ...                orig_shape=(640, 640), names={0: 'vehicle', 1: 'plate'})
            >>> stats = result.summary()
            >>> print(f"Total: {stats['total_detections']}")
            >>> print(f"Average confidence: {stats['avg_confidence']:.2f}")
            >>> print(f"Class distribution: {stats['class_counts']}")
        """
        n = len(self)

        if n == 0:
            # Return empty statistics for empty results
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }

        # Count detections per class
        class_counts = {}
        for cls_id in np.unique(self.class_ids):
            cls_id_int = int(cls_id)
            count = int(np.sum(self.class_ids == cls_id))
            class_name = self.names.get(cls_id_int, f'class_{cls_id_int}')
            class_counts[class_name] = count

        return {
            'total_detections': n,
            'class_counts': class_counts,
            'avg_confidence': float(np.mean(self.scores)),
            'min_confidence': float(np.min(self.scores)),
            'max_confidence': float(np.max(self.scores))
        }

