"""
Classification ONNX Runtime Base Class and Implementations.

This module provides:
- ClsResult: NamedTuple for classification results (supports single and multi-branch)
- BaseClsORT: Abstract base class for classification models (Template Method Pattern)
- ColorLayerORT: Vehicle plate color and layer classification (dual-branch)

Architecture Design:
- Mirrors BaseORT's Template Method Pattern for consistency
- Returns ClsResult (NamedTuple) instead of Result object
- Supports arbitrary number of classification branches
- Backward compatible with tuple unpacking

Example:
    >>> # Dual-branch classification (ColorLayerORT)
    >>> classifier = ColorLayerORT('models/color_layer.onnx')
    >>> result = classifier(plate_image)
    >>> print(f"Color: {result.labels[0]}, Layer: {result.labels[1]}")
    >>>
    >>> # Backward compatible tuple unpacking
    >>> color, layer, conf = classifier(plate_image)
"""

import cv2
import numpy as np
import logging
import onnxruntime
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from numpy.typing import NDArray


class ClsResult:
    """Classification result supporting both single and multi-branch models.

    This class provides a unified interface for classification results,
    supporting arbitrary number of classification branches (single, dual, multi).

    Attributes:
        labels: List of classification labels (e.g., ['blue'] or ['blue', 'single'])
        confidences: List of confidence scores for each label
        avg_confidence: Average confidence across all branches
        logits: Optional raw logits from model output

    Supports tuple unpacking for backward compatibility:
        >>> color, layer, conf = result  # For dual-branch (ColorLayerORT)
        >>> label, conf = result         # For single-branch

    Example:
        >>> result = ClsResult(
        ...     labels=['blue', 'single'],
        ...     confidences=[0.95, 0.88],
        ...     avg_confidence=0.915
        ... )
        >>> # Attribute access
        >>> print(result.labels[0])  # 'blue'
        >>> print(result.confidences[0])  # 0.95
        >>> # Tuple unpacking
        >>> color, layer, conf = result
    """

    __slots__ = ('labels', 'confidences', 'avg_confidence', 'logits')

    def __init__(
        self,
        labels: List[str],
        confidences: List[float],
        avg_confidence: float,
        logits: Optional[List[NDArray[np.float32]]] = None
    ):
        """Initialize ClsResult.

        Args:
            labels: List of classification labels
            confidences: List of confidence scores for each label
            avg_confidence: Average confidence across all branches
            logits: Optional raw logits from model output
        """
        self.labels = labels
        self.confidences = confidences
        self.avg_confidence = avg_confidence
        self.logits = logits

    def __iter__(self):
        """Enable tuple unpacking for backward compatibility.

        Unpacking behavior depends on number of branches:
        - Single-branch (1 label): yields (label, confidence)
        - Dual-branch (2 labels): yields (label1, label2, avg_confidence)
        - Multi-branch (3+ labels): yields (labels, confidences, avg_confidence)
        """
        n = len(self.labels)
        if n == 1:
            # Single-branch: (label, confidence)
            yield self.labels[0]
            yield self.avg_confidence
        elif n == 2:
            # Dual-branch: (label1, label2, avg_confidence)
            yield self.labels[0]
            yield self.labels[1]
            yield self.avg_confidence
        else:
            # Multi-branch: (labels, confidences, avg_confidence)
            yield self.labels
            yield self.confidences
            yield self.avg_confidence

    def __repr__(self) -> str:
        """String representation of ClsResult."""
        labels_str = ', '.join(self.labels)
        return f"ClsResult(labels=[{labels_str}], avg_confidence={self.avg_confidence:.3f})"

    def __len__(self) -> int:
        """Return number of classification branches."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[str, float]:
        """Get label and confidence for a specific branch.

        Args:
            index: Branch index

        Returns:
            Tuple of (label, confidence) for the specified branch
        """
        return (self.labels[index], self.confidences[index])


class BaseClsORT(ABC):
    """ONNX Runtime Classification Model Base Class (Template Method Pattern).

    This abstract base class provides a unified interface for classification models,
    mirroring BaseORT's design pattern for consistency.

    Supports:
    - Single-branch classification (e.g., ImageNet classifier)
    - Multi-branch classification (e.g., ColorLayerORT: color + layer)

    Subclasses must implement:
    - preprocess(): Static preprocessing method
    - postprocess(): Post-processing with label mapping

    Template Method Pattern:
    - __call__() orchestrates the inference pipeline
    - Phase 1: _prepare_inference() - preprocessing
    - Phase 2: _execute_inference() - ONNX Runtime inference
    - Phase 3: _finalize_inference() - post-processing

    Example:
        >>> class MyClassifier(BaseClsORT):
        ...     @staticmethod
        ...     def preprocess(image, input_shape, **kwargs):
        ...         # Custom preprocessing
        ...         return input_tensor, scale, original_shape
        ...
        ...     def postprocess(self, outputs, conf_thres, **kwargs):
        ...         # Custom postprocessing
        ...         return ClsResult(labels=['cat'], confidences=[0.95], avg_confidence=0.95)
    """

    def __init__(
        self,
        onnx_path: str,
        input_shape: Tuple[int, int] = (48, 168),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None
    ):
        """Initialize classification model.

        Args:
            onnx_path: ONNX model file path
            input_shape: Input image size (height, width), may be overridden by model metadata
            conf_thres: Confidence threshold for low-confidence warnings
            providers: ONNX Runtime execution providers
        """
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self._requested_input_shape = input_shape
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Create ONNX Runtime session
        self._onnx_session = onnxruntime.InferenceSession(
            self.onnx_path,
            providers=self.providers
        )
        logging.info(f"ONNX Runtime session created: {self._onnx_session.get_providers()}")

        # Get input/output names from model
        self.input_name = self._onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self._onnx_session.get_outputs()]
        logging.info(f"Model info: input={self.input_name}, outputs={self.output_names}")

        # Get actual input shape from model metadata
        self.input_shape = self._get_model_input_shape()

        logging.info(f"Classification model initialized: {self.onnx_path}")

    def _get_model_input_shape(self) -> Tuple[int, int]:
        """Get input shape from model metadata.

        Returns:
            Tuple[int, int]: Input shape (height, width)
        """
        model_input_shape = self._onnx_session.get_inputs()[0].shape

        if (len(model_input_shape) == 4 and
            model_input_shape[2] is not None and
            model_input_shape[3] is not None):
            # Shape format: [batch, channels, height, width]
            actual_height = int(model_input_shape[2])
            actual_width = int(model_input_shape[3])
            logging.info(f"Input shape from model: ({actual_height}, {actual_width})")
            return (actual_height, actual_width)
        else:
            logging.info(f"Using default input shape: {self._requested_input_shape}")
            return self._requested_input_shape

    @staticmethod
    @abstractmethod
    def preprocess(
        image: NDArray[np.uint8],
        input_shape: Tuple[int, int],
        **kwargs
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """Static preprocessing method.

        Args:
            image: Input image, BGR format [H, W, C]
            input_shape: Target input size (height, width)
            **kwargs: Additional parameters

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, C, H, W] float32
                - scale: Scale factor (usually 1.0 for classification)
                - original_shape: Original image shape (H, W)

        Raises:
            NotImplementedError: Subclass must implement this method
        """
        raise NotImplementedError(
            f"{__class__.__name__}.preprocess() must be implemented by subclass"
        )

    @abstractmethod
    def postprocess(
        self,
        outputs: List[NDArray[np.float32]],
        conf_thres: float,
        **kwargs
    ) -> ClsResult:
        """Post-processing method.

        Args:
            outputs: Model output list
            conf_thres: Confidence threshold for warnings
            **kwargs: Additional parameters

        Returns:
            ClsResult: Classification result

        Raises:
            NotImplementedError: Subclass must implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.postprocess() must be implemented by subclass"
        )

    def _prepare_inference(
        self,
        image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """Phase 1: Preprocess image.

        Args:
            image: Input image, BGR format [H, W, C]

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor
                - scale: Scale factor
                - original_shape: Original image shape
        """
        return self.preprocess(image, self.input_shape)

    def _execute_inference(
        self,
        input_tensor: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """Phase 2: Execute ONNX Runtime inference.

        Args:
            input_tensor: Preprocessed input tensor

        Returns:
            List of model output arrays
        """
        feed_dict = {self.input_name: input_tensor}
        outputs = self._onnx_session.run(self.output_names, feed_dict)
        return outputs

    def _finalize_inference(
        self,
        outputs: List[NDArray[np.float32]],
        conf_thres: Optional[float],
        **kwargs
    ) -> ClsResult:
        """Phase 3: Post-process outputs.

        Args:
            outputs: Model output arrays
            conf_thres: Confidence threshold
            **kwargs: Additional parameters for postprocess

        Returns:
            ClsResult: Classification result
        """
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        return self.postprocess(outputs, effective_conf_thres, **kwargs)

    def __call__(
        self,
        image: NDArray[np.uint8],
        conf_thres: Optional[float] = None,
        **kwargs
    ) -> ClsResult:
        """Execute classification inference.

        Template method that orchestrates the inference pipeline:
        1. _prepare_inference: Preprocessing
        2. _execute_inference: ONNX Runtime inference
        3. _finalize_inference: Post-processing

        Args:
            image: Input image, BGR format [H, W, C]
            conf_thres: Optional confidence threshold override
            **kwargs: Additional parameters for postprocess

        Returns:
            ClsResult: Classification result (supports tuple unpacking)

        Example:
            >>> result = classifier(image)
            >>> # Attribute access
            >>> print(result.labels[0])
            >>> # Tuple unpacking (backward compatible)
            >>> color, layer, conf = classifier(image)
        """
        # Phase 1: Prepare
        input_tensor, scale, original_shape = self._prepare_inference(image)

        # Phase 2: Execute
        outputs = self._execute_inference(input_tensor)

        # Phase 3: Finalize
        result = self._finalize_inference(outputs, conf_thres, **kwargs)

        return result


class ColorLayerORT(BaseClsORT):
    """Vehicle plate color and layer classification (dual-branch).

    Dual-branch classification model for vehicle plates:
    - Branch 1: Color classification (5 classes: blue, yellow, white, black, green)
    - Branch 2: Layer classification (2 classes: single, double)

    Inherits from BaseClsORT and implements:
    - preprocess(): BGR->RGB, resize, normalize to [-1,1], HWC->CHW
    - postprocess(): Softmax + argmax for each branch, map to labels

    Example:
        >>> classifier = ColorLayerORT(
        ...     'models/color_layer.onnx',
        ...     color_map={0: 'blue', 1: 'yellow', 2: 'white', 3: 'black', 4: 'green'},
        ...     layer_map={0: 'single', 1: 'double'}
        ... )
        >>> result = classifier(plate_image)
        >>> print(f"Color: {result.labels[0]}, Layer: {result.labels[1]}")
        >>>
        >>> # Backward compatible tuple unpacking
        >>> color, layer, conf = classifier(plate_image)
    """

    def __init__(
        self,
        onnx_path: str,
        color_map: Optional[Dict[int, str]] = None,
        layer_map: Optional[Dict[int, str]] = None,
        input_shape: Tuple[int, int] = (48, 168),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None,
        plate_config_path: Optional[str] = None
    ):
        """Initialize color and layer classification model.

        Args:
            onnx_path: Path to ONNX model file
            color_map: Mapping from color index to color name (optional)
            layer_map: Mapping from layer index to layer name (optional)
            input_shape: Input image size (height, width), default (48, 168)
            conf_thres: Confidence threshold, default 0.5
            providers: ONNX Runtime execution providers
            plate_config_path: Optional path to plate configuration file

        Raises:
            ValueError: If color_map or layer_map cannot be loaded
        """
        # Load from config if not provided
        if color_map is None or layer_map is None:
            color_map, layer_map = self._load_mappings(color_map, layer_map, plate_config_path)

        # Validate inputs
        if not color_map:
            raise ValueError("color_map cannot be empty")
        if not layer_map:
            raise ValueError("layer_map cannot be empty")

        # Call parent constructor
        super().__init__(onnx_path, input_shape, conf_thres, providers)

        # Store mappings
        self.color_map = color_map
        self.layer_map = layer_map

        logging.info(f"ColorLayerORT initialized: {len(color_map)} colors, {len(layer_map)} layers")

    def _load_mappings(
        self,
        color_map: Optional[Dict[int, str]],
        layer_map: Optional[Dict[int, str]],
        plate_config_path: Optional[str]
    ) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Load color and layer mappings from config.

        Priority:
        1. Provided color_map/layer_map parameters
        2. External config file (plate_config_path)
        3. Default constants from onnxtools.config

        Args:
            color_map: Provided color mapping (or None)
            layer_map: Provided layer mapping (or None)
            plate_config_path: External config file path (or None)

        Returns:
            Tuple[Dict, Dict]: (color_map, layer_map)

        Raises:
            ValueError: If mappings cannot be loaded
        """
        try:
            if plate_config_path:
                # Load from external config file
                from onnxtools.config import load_plate_config
                plate_config = load_plate_config(plate_config_path)

                if color_map is None:
                    color_map = plate_config.get('color_dict')
                    if not color_map:
                        raise ValueError("Failed to load color_map from external config")
                    logging.info(f"Loaded color_map from external config: {len(color_map)} colors")

                if layer_map is None:
                    layer_map = plate_config.get('layer_dict')
                    if not layer_map:
                        raise ValueError("Failed to load layer_map from external config")
                    logging.info(f"Loaded layer_map from external config: {len(layer_map)} layers")
            else:
                # Use default constants
                if color_map is None:
                    from onnxtools.config import COLOR_MAP
                    color_map = COLOR_MAP
                    logging.info(f"Using default color_map: {len(color_map)} colors")

                if layer_map is None:
                    from onnxtools.config import LAYER_MAP
                    layer_map = LAYER_MAP
                    logging.info(f"Using default layer_map: {len(layer_map)} layers")

        except Exception as e:
            raise ValueError(f"Failed to load plate configuration: {e}")

        return color_map, layer_map

    @staticmethod
    def preprocess(
        image: NDArray[np.uint8],
        input_shape: Tuple[int, int],
        **kwargs
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """Preprocess image for classification.

        Processing pipeline:
        1. Store original shape
        2. BGR -> RGB conversion
        3. Resize to target size
        4. Normalize to [-1, 1] range
        5. HWC -> CHW conversion
        6. Add batch dimension

        Args:
            image: Input image, BGR format [H, W, C]
            input_shape: Target size (height, width)
            **kwargs: Unused, for interface compatibility

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, 3, H, W] float32
                - scale: Fixed scale value (1.0)
                - original_shape: Original image shape (H, W)
        """
        # Store original shape
        original_shape = (image.shape[0], image.shape[1])

        # BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.asarray(img_rgb).astype(np.float32)

        # Resize to target size (note: cv2.resize expects (width, height))
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

        # Normalize to [-1, 1]
        mean_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = (img / 255.0 - mean_value) / std_value

        # HWC to CHW
        img = img.transpose([2, 0, 1])

        # Add batch dimension
        input_tensor = img[np.newaxis, :, :, :]
        input_tensor = np.array(input_tensor, dtype=np.float32)

        return input_tensor, 1.0, original_shape

    @staticmethod
    def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute softmax activation.

        Args:
            x: Input logits

        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def postprocess(
        self,
        outputs: List[NDArray[np.float32]],
        conf_thres: float,
        **kwargs
    ) -> ClsResult:
        """Post-process dual-branch outputs.

        Processing:
        1. Extract color and layer logits
        2. Apply softmax activation
        3. Get argmax predictions and confidences
        4. Map indices to label names
        5. Create ClsResult with both branches

        Args:
            outputs: Model outputs [color_logits, layer_logits]
            conf_thres: Confidence threshold for warnings
            **kwargs: Unused, for interface compatibility

        Returns:
            ClsResult with labels=[color, layer], confidences=[color_conf, layer_conf]
        """
        # Expected: outputs = [color_logits, layer_logits]
        if len(outputs) != 2:
            logging.warning(f"Expected 2 outputs, got {len(outputs)}")

        color_logits = outputs[0]
        layer_logits = outputs[1] if len(outputs) > 1 else outputs[0]

        # Apply softmax
        color_probs = self._softmax(color_logits)
        layer_probs = self._softmax(layer_logits)

        # Get predictions
        color_idx = int(np.argmax(color_probs, axis=-1)[0])
        layer_idx = int(np.argmax(layer_probs, axis=-1)[0])

        color_conf = float(color_probs[0, color_idx])
        layer_conf = float(layer_probs[0, layer_idx])

        # Map to names
        color_name = self.color_map.get(color_idx, f'unknown_{color_idx}')
        layer_name = self.layer_map.get(layer_idx, f'unknown_{layer_idx}')

        # Low confidence warnings
        if color_conf < conf_thres:
            logging.warning(f"Low color confidence: {color_conf:.3f}")
        if layer_conf < conf_thres:
            logging.warning(f"Low layer confidence: {layer_conf:.3f}")

        # Calculate average confidence
        avg_conf = (color_conf + layer_conf) / 2.0

        return ClsResult(
            labels=[color_name, layer_name],
            confidences=[color_conf, layer_conf],
            avg_confidence=avg_conf,
            logits=[color_logits, layer_logits]
        )


class VehicleAttributeORT(BaseClsORT):
    """Vehicle attribute classification (multi-label: type + color).

    Multi-label classification model with single output containing both branches:
    - First 13 values: Vehicle type (car, truck, bus, etc.)
    - Last 11 values: Vehicle color (black, white, gray, etc.)

    The model output shape is [batch, 24], which is split into:
    - type_logits: [batch, 13]
    - color_logits: [batch, 11]

    Inherits from BaseClsORT and implements:
    - preprocess(): Resize to 224x224, normalize with ImageNet mean/std
    - postprocess(): Split output, apply softmax to each branch, map to labels

    Example:
        >>> classifier = VehicleAttributeORT('models/vehicle_attribute.onnx')
        >>> result = classifier(vehicle_image)
        >>> print(f"Type: {result.labels[0]}, Color: {result.labels[1]}")
        >>>
        >>> # Tuple unpacking
        >>> vehicle_type, color, conf = classifier(vehicle_image)
    """

    # Output split indices
    NUM_VEHICLE_TYPES = 13
    NUM_COLORS = 11

    def __init__(
        self,
        onnx_path: str,
        type_map: Optional[Dict[int, str]] = None,
        color_map: Optional[Dict[int, str]] = None,
        input_shape: Tuple[int, int] = (224, 224),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None
    ):
        """Initialize vehicle attribute classification model.

        Args:
            onnx_path: Path to ONNX model file
            type_map: Mapping from type index to type name (optional)
            color_map: Mapping from color index to color name (optional)
            input_shape: Input image size (height, width), default (224, 224)
            conf_thres: Confidence threshold, default 0.5
            providers: ONNX Runtime execution providers

        Raises:
            ValueError: If type_map or color_map cannot be loaded
        """
        # Load from config if not provided
        if type_map is None:
            from onnxtools.config import VEHICLE_TYPE_MAP
            type_map = VEHICLE_TYPE_MAP
            logging.info(f"Using default vehicle type_map: {len(type_map)} types")

        if color_map is None:
            from onnxtools.config import VEHICLE_COLOR_MAP
            color_map = VEHICLE_COLOR_MAP
            logging.info(f"Using default vehicle color_map: {len(color_map)} colors")

        # Validate inputs
        if not type_map:
            raise ValueError("type_map cannot be empty")
        if not color_map:
            raise ValueError("color_map cannot be empty")

        # Call parent constructor
        super().__init__(onnx_path, input_shape, conf_thres, providers)

        # Store mappings
        self.type_map = type_map
        self.color_map = color_map

        logging.info(f"VehicleAttributeORT initialized: {len(type_map)} types, {len(color_map)} colors")

    @staticmethod
    def preprocess(
        image: NDArray[np.uint8],
        input_shape: Tuple[int, int],
        **kwargs
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """Preprocess image for vehicle attribute classification.

        Processing pipeline (based on PaddleClas config):
        1. Store original shape
        2. BGR -> RGB conversion
        3. Resize to target size (224x224)
        4. Normalize: scale=1/255, mean=[0,0,0], std=[1,1,1]
        5. HWC -> CHW conversion
        6. Add batch dimension

        Args:
            image: Input image, BGR format [H, W, C]
            input_shape: Target size (height, width)
            **kwargs: Unused, for interface compatibility

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, 3, H, W] float32
                - scale: Fixed scale value (1.0)
                - original_shape: Original image shape (H, W)
        """
        # Store original shape
        original_shape = (image.shape[0], image.shape[1])

        # BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.asarray(img_rgb).astype(np.float32)

        # Resize to target size
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

        # Normalize: only scale by 1/255, no mean/std adjustment
        img = img / 255.0

        # HWC to CHW
        img = img.transpose([2, 0, 1])

        # Add batch dimension
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor, 1.0, original_shape

    def postprocess(
        self,
        outputs: List[NDArray[np.float32]],
        conf_thres: float,
        **kwargs
    ) -> ClsResult:
        """Post-process single output with multi-label classification.

        Note: Model output already has sigmoid applied, so we directly use
        the values as probabilities without additional activation.

        Processing:
        1. Split single output into type_probs and color_probs
        2. Get argmax predictions and confidences (no softmax needed)
        3. Map indices to label names
        4. Create ClsResult with both branches

        Args:
            outputs: Model outputs [combined_probs] with shape [batch, 24]
                     Values are already sigmoid-activated probabilities
            conf_thres: Confidence threshold for warnings
            **kwargs: Unused, for interface compatibility

        Returns:
            ClsResult with labels=[vehicle_type, color], confidences=[type_conf, color_conf]
        """
        # Get combined output (already sigmoid-activated)
        combined_probs = outputs[0]  # Shape: [batch, 24]

        # Split into type and color probabilities
        type_probs = combined_probs[:, :self.NUM_VEHICLE_TYPES]  # [batch, 13]
        color_probs = combined_probs[:, self.NUM_VEHICLE_TYPES:]  # [batch, 11]

        # Get predictions (no softmax needed, model already has sigmoid)
        type_idx = int(np.argmax(type_probs, axis=-1)[0])
        color_idx = int(np.argmax(color_probs, axis=-1)[0])

        type_conf = float(type_probs[0, type_idx])
        color_conf = float(color_probs[0, color_idx])

        # Map to names
        type_name = self.type_map.get(type_idx, f'unknown_type_{type_idx}')
        color_name = self.color_map.get(color_idx, f'unknown_color_{color_idx}')

        # Low confidence warnings
        if type_conf < conf_thres:
            logging.warning(f"Low vehicle type confidence: {type_conf:.3f}")
        if color_conf < conf_thres:
            logging.warning(f"Low vehicle color confidence: {color_conf:.3f}")

        # Calculate average confidence
        avg_conf = (type_conf + color_conf) / 2.0

        return ClsResult(
            labels=[type_name, color_name],
            confidences=[type_conf, color_conf],
            avg_confidence=avg_conf,
            logits=[type_probs, color_probs]  # Store probs instead of logits
        )
