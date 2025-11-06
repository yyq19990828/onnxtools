"""
OCR and Color/Layer Classification ONNX Inference Module.

This module provides two main classes:
- ColorLayerORT: Vehicle plate color and layer classification
- OcrORT: Optical Character Recognition for plate numbers

These are independent inference classes (not inheriting BaseORT) because:
1. They perform classification/OCR tasks, not object detection
2. They return tuples (natural for classification) instead of Result objects
3. No bounding box concept - incompatible with Result's boxes/scores/class_ids model

See onnxtools/infer_onnx/CLAUDE.md for architecture design rationale.
"""

import cv2
import numpy as np
import logging
import onnxruntime
from typing import List, Tuple, Optional, Dict, TypeAlias
from numpy.typing import NDArray


# Type Aliases for OCR and Color/Layer Classification
OCRResult: TypeAlias = Tuple[str, float, List[float]]  # (text, avg_confidence, char_confidences)
ColorLogits: TypeAlias = Tuple[NDArray[np.float32], float]  # (color_logits, confidence)
LayerLogits: TypeAlias = Tuple[NDArray[np.float32], float]  # (layer_logits, confidence)
PreprocessResult: TypeAlias = Tuple[
    NDArray[np.float32],           # input_tensor
    float,                          # scale
    Tuple[int, int],               # original_shape (H, W)
    Optional[Tuple[Tuple[float, float], Tuple[float, float]]]  # ratio_pad
]
OCROutput: TypeAlias = Tuple[NDArray[np.int_], Optional[NDArray[np.float32]]]


class ColorLayerORT:
    """
    Vehicle plate color and layer classification inference engine.

    Independent inference class for classification tasks (does not inherit BaseORT).
    Supports:
    - 5 color categories: blue, yellow, white, black, green
    - 2 layer categories: single, double

    Example:
        >>> classifier = ColorLayerORT(
        ...     'models/color_layer.onnx',
        ...     color_map={0: 'blue', 1: 'yellow', 2: 'white', 3: 'black', 4: 'green'},
        ...     layer_map={0: 'single', 1: 'double'}
        ... )
        >>> color, layer, conf = classifier(plate_image)
        >>> print(f"Color: {color}, Layer: {layer}")
    """

    def __init__(
        self,
        onnx_path: str,
        color_map: Dict[int, str],
        layer_map: Dict[int, str],
        input_shape: Tuple[int, int] = (48, 168),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize color and layer classification model.

        Args:
            onnx_path: Path to ONNX model file
            color_map: Mapping from color index to color name
            layer_map: Mapping from layer index to layer name
            input_shape: Input image size (height, width), default (48, 168)
            conf_thres: Confidence threshold, default 0.5
            providers: ONNX Runtime execution providers

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If color_map or layer_map is empty
        """
        # Validate inputs
        if not color_map:
            raise ValueError("color_map cannot be empty")
        if not layer_map:
            raise ValueError("layer_map cannot be empty")

        # Initialize ONNX Runtime session
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self._onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        logging.info(f"ONNX Runtime会话已创建: {self._onnx_session.get_providers()}")

        # Get input/output names from model
        self.input_name = self._onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self._onnx_session.get_outputs()]
        logging.info(f"从ONNX模型读取信息: input={self.input_name}, outputs={self.output_names}")

        # Store configuration
        self.input_shape = input_shape
        self.conf_thres = conf_thres

        # Store mappings
        self.color_map = color_map
        self.layer_map = layer_map

        logging.info(f"ColorLayerORT initialized: {len(color_map)} colors, {len(layer_map)} layers")

    def _preprocess(self, image: NDArray[np.uint8]) -> PreprocessResult:
        """
        Preprocess image for color/layer classification (instance method).

        Args:
            image: Input plate image, BGR format [H, W, 3]

        Returns:
            PreprocessResult: (input_tensor, scale, original_shape, ratio_pad)
        """
        input_tensor, scale, original_shape = self._preprocess_static(image, self.input_shape)
        ratio_pad = None
        return input_tensor, scale, original_shape, ratio_pad

    @staticmethod
    def _preprocess_static(
        img: NDArray[np.uint8],
        image_shape: Tuple[int, int]
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """
        Preprocess image for classification (static method).

        This method:
        1. Resizes image to target size
        2. Normalizes to [-1, 1] range
        3. Converts HWC to CHW format
        4. Adds batch dimension

        Args:
            img: Input image, BGR format [H, W, 3]
            image_shape: Target size (height, width)

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, 3, H, W] float32
                - scale: Fixed scale value (1.0, no scaling)
                - original_shape: Original image shape (H, W)

        Source:
            Original utils/ocr_image_processing.py::image_pretreatment()
        """
        # Store original shape
        original_shape = (img.shape[0], img.shape[1])

        # Convert BGR to RGB (cv2 reads as BGR, but model expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.asarray(img_rgb).astype(np.float32)
        # Resize to target size (note: cv2.resize expects (width, height))
        img = cv2.resize(img, (image_shape[1], image_shape[0]))

        # Normalize to [-1, 1]
        mean_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std_value = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = (img / 255.0 - mean_value) / std_value

        # HWC to CHW
        img = img.transpose([2, 0, 1])

        # Add batch dimension
        onnx_infer_data = img[np.newaxis, :, :, :]
        onnx_infer_data = np.array(onnx_infer_data, dtype=np.float32)

        return onnx_infer_data, 1.0, original_shape

    def __call__(
        self,
        image: NDArray[np.uint8],
        conf_thres: Optional[float] = None
    ) -> Tuple[str, str, float]:
        """
        Execute color and layer classification inference.

        Args:
            image: Input plate image, BGR format [H, W, 3]
            conf_thres: Optional confidence threshold override

        Returns:
            Tuple[str, str, float]:
                - color: Color classification result
                - layer: Layer classification result
                - confidence: Average confidence of both predictions

        Example:
            >>> color, layer, conf = classifier(plate_img)
            >>> print(f"Color: {color}, Layer: {layer}, Confidence: {conf:.3f}")
        """
        # Preprocess
        input_tensor, scale, original_shape, ratio_pad = self._preprocess(image)

        # Inference using ONNX Runtime session
        feed_dict = {self.input_name: input_tensor}
        outputs = self._onnx_session.run(self.output_names, feed_dict)

        # Expected: outputs = [color_logits, layer_logits]
        if len(outputs) != 2:
            logging.warning(f"Expected 2 outputs, got {len(outputs)}")

        color_logits = outputs[0]
        layer_logits = outputs[1] if len(outputs) > 1 else outputs[0]

        # Apply softmax
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        color_probs = softmax(color_logits)
        layer_probs = softmax(layer_logits)

        # Get predictions
        color_idx = int(np.argmax(color_probs, axis=-1)[0])
        layer_idx = int(np.argmax(layer_probs, axis=-1)[0])

        color_conf = float(color_probs[0, color_idx])
        layer_conf = float(layer_probs[0, layer_idx])

        # Map to names
        color_name = self.color_map.get(color_idx, f'unknown_{color_idx}')
        layer_name = self.layer_map.get(layer_idx, f'unknown_{layer_idx}')

        # Filter by confidence threshold
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres

        if color_conf < effective_conf_thres:
            logging.warning(f"Low color confidence: {color_conf:.3f}")
        if layer_conf < effective_conf_thres:
            logging.warning(f"Low layer confidence: {layer_conf:.3f}")

        # Return simple tuple format
        average_conf = (color_conf + layer_conf) / 2.0
        return color_name, layer_name, average_conf



class OcrORT:
    """
    Optical Character Recognition inference engine for vehicle plates.

    Independent inference class for OCR tasks (does not inherit BaseORT).
    Supports:
    - Single-layer plate OCR
    - Double-layer plate processing (skew correction, split, stitch)
    - Chinese character and alphanumeric recognition

    Example:
        >>> ocr_model = OcrORT(
        ...     'models/ocr.onnx',
        ...     character=['京', 'A', 'B', '0', '1', '2', ...]
        ... )
        >>> result = ocr_model(plate_image, is_double_layer=True)
        >>> if result:
        ...     text, conf, char_confs = result
        ...     print(f"Plate: {text}, Confidence: {conf:.3f}")
    """

    def __init__(
        self,
        onnx_path: str,
        character: List[str],
        input_shape: Tuple[int, int] = (48, 168),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize OCR inference engine.

        Args:
            onnx_path: Path to ONNX model file
            character: OCR character dictionary (list of characters)
            input_shape: Input image size (height, width), default (48, 168)
            conf_thres: Confidence threshold, default 0.5
            providers: ONNX Runtime execution providers

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If character list is empty
        """
        # Validate inputs
        if not character:
            raise ValueError("character list cannot be empty")

        # Initialize ONNX Runtime session
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self._onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        logging.info(f"ONNX Runtime会话已创建: {self._onnx_session.get_providers()}")

        # Get input/output names from model
        self.input_name = self._onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self._onnx_session.get_outputs()]
        logging.info(f"从ONNX模型读取信息: input={self.input_name}, outputs={self.output_names}")

        # Store configuration
        self.input_shape = input_shape
        self.conf_thres = conf_thres

        # Store character dictionary
        self.character = character

        logging.info(f"OcrORT initialized: {len(character)} characters in dictionary")

    def _preprocess(
        self,
        image: NDArray[np.uint8],
        is_double_layer: bool = False
    ) -> PreprocessResult:
        """
        Preprocess plate image for OCR (instance method).

        This method handles both single and double-layer plates:
        - Single-layer: skew correction only
        - Double-layer: skew correction + split + stitch

        Args:
            image: Input plate image, BGR format [H, W, 3]
            is_double_layer: Whether this is a double-layer plate

        Returns:
            PreprocessResult: (input_tensor, scale, original_shape, ratio_pad)
        """
        # Step 1: Process plate (skew correction and double-layer handling)
        processed = self._process_plate_image_static(image, is_double_layer)

        if processed is None:
            # If processing failed, use original image
            logging.warning("Plate processing failed, using original image")
            processed = image

        # Step 2: Resize and normalize
        # Convert input_shape (H, W) to full shape [C, H, W]
        full_shape = [3, self.input_shape[0], self.input_shape[1]]
        input_tensor = self._resize_norm_img_static(processed, full_shape)

        original_shape = (image.shape[0], image.shape[1])
        scale = 1.0
        ratio_pad = None

        return input_tensor, scale, original_shape, ratio_pad

    @staticmethod
    def _preprocess_static(
        image: NDArray[np.uint8],
        input_shape: Tuple[int, int]
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int]]:
        """
        Preprocess plate image for OCR (static method, single-layer only).

        This is the base static preprocessing method required by BaseORT.
        For double-layer processing, use the instance method _preprocess() instead.

        Args:
            image: Input plate image, BGR format [H, W, 3]
            input_shape: Target size (height, width)

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, 3, H, W] float32
                - scale: Fixed scale value (1.0)
                - original_shape: Original image shape (H, W)

        Note:
            This method only handles single-layer plates with basic preprocessing.
            For full OCR preprocessing including double-layer support, use the
            instance method which calls more specialized static methods.
        """
        # Store original shape
        original_shape = (image.shape[0], image.shape[1])

        # Step 1: Simple skew correction (single-layer only)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        skew_angle = OcrORT._detect_skew_angle(gray_img)
        corrected_img = OcrORT._correct_skew(image, skew_angle)

        # Step 2: Resize and normalize
        full_shape = [3, input_shape[0], input_shape[1]]
        input_tensor = OcrORT._resize_norm_img_static(corrected_img, full_shape)

        return input_tensor, 1.0, original_shape

    @staticmethod
    def _detect_skew_angle(image: NDArray[np.uint8]) -> float:
        """
        Detect image skew angle using Hough line transform.

        Args:
            image: Grayscale image [H, W]

        Returns:
            Skew angle in degrees, range [-45, 45]

        Source:
            Original utils/ocr_image_processing.py::detect_skew_angle()
        """
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is None:
            return 0.0

        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        return float(np.median(angles))

    @staticmethod
    def _correct_skew(
        image: NDArray[np.uint8],
        angle: float
    ) -> NDArray[np.uint8]:
        """
        Correct image skew using affine transformation.

        Args:
            image: Input image (grayscale or color) [H, W] or [H, W, 3]
            angle: Skew angle in degrees

        Returns:
            Corrected image, same shape as input

        Source:
            Original utils/ocr_image_processing.py::correct_skew()
        """
        if abs(angle) < 0.5:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return corrected

    @staticmethod
    def _find_optimal_split_line(gray_img: NDArray[np.uint8]) -> int:
        """
        Find optimal split line for double-layer plate using horizontal projection.

        Algorithm:
        1. Compute horizontal projection histogram
        2. Smooth with Gaussian filter
        3. Find maximum point in search region (25%-65% of height)
        4. Return split point y-coordinate

        Args:
            gray_img: Grayscale image [H, W]

        Returns:
            Split line y-coordinate

        Source:
            Original utils/ocr_image_processing.py::find_optimal_split_line()
        """
        height, width = gray_img.shape
        search_start = int(height * 0.25)
        search_end = int(height * 0.65)

        # Compute horizontal projection
        horizontal_projection = np.sum(gray_img, axis=1)

        # Smooth projection
        kernel_size = max(3, height // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1

        smoothed_projection = cv2.GaussianBlur(
            horizontal_projection.astype(np.float32).reshape(-1, 1),
            (1, kernel_size),
            0
        ).flatten()

        # Search for split line
        search_region = smoothed_projection[search_start:search_end]

        if len(search_region) == 0:
            return int(height * 0.35)

        # Find maximum value region
        max_val = np.max(search_region)
        max_positions = []
        threshold = max_val * 0.9

        for i in range(len(search_region)):
            if search_region[i] >= threshold:
                max_positions.append(search_start + i)

        if max_positions:
            split_point = max_positions[len(max_positions) // 2]
            return split_point

        # Fallback: find maximum gradient change
        projection_diff = np.abs(np.diff(smoothed_projection[search_start:search_end]))
        if len(projection_diff) > 0:
            max_change_idx = np.argmax(projection_diff)
            return search_start + max_change_idx

        return int(height * 0.35)

    @staticmethod
    def _split_double_layer(
        image: NDArray[np.uint8],
        split_y: int
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Split double-layer plate into top and bottom parts.

        Args:
            image: Input image [H, W, 3]
            split_y: Split line y-coordinate

        Returns:
            Tuple[top_layer, bottom_layer]
        """
        top_part = image[0:split_y, :]
        bottom_part = image[split_y:, :]
        return top_part, bottom_part

    @staticmethod
    def _stitch_layers(
        top_layer: NDArray[np.uint8],
        bottom_layer: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Horizontally stitch top and bottom layers into single-layer plate.

        Algorithm:
        1. Resize top layer to bottom layer height
        2. Narrow top layer width to 50% of proportional width
        3. Horizontally concatenate layers

        Args:
            top_layer: Top layer image [H1, W1, 3]
            bottom_layer: Bottom layer image [H2, W2, 3]

        Returns:
            Stitched single-layer image [H2, W1'+W2, 3]
        """
        bottom_h, bottom_w = bottom_layer.shape[:2]
        top_h, top_w = top_layer.shape[:2]

        # Calculate target width for top layer (50% of proportional width)
        target_height = bottom_h
        top_aspect_ratio = top_w / top_h
        target_top_width = int(target_height * top_aspect_ratio * 0.5)

        # Resize top layer
        top_resized = cv2.resize(
            top_layer,
            (target_top_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Stitch horizontally
        stitched_plate = cv2.hconcat([top_resized, bottom_layer])

        return stitched_plate

    @staticmethod
    def _process_plate_image_static(
        img: NDArray[np.uint8],
        is_double_layer: bool = False,
        verbose: bool = False
    ) -> Optional[NDArray[np.uint8]]:
        """
        Process plate image: skew correction and double-layer handling.

        Processing pipeline:
        1. Convert to grayscale
        2. Detect and correct skew angle
        3. If double-layer:
           a. Apply CLAHE contrast enhancement
           b. Find optimal split line
           c. Split into top and bottom parts
           d. Stitch into single-layer
        4. Return processed image

        Args:
            img: Input plate image, BGR format [H, W, 3]
            is_double_layer: Whether this is a double-layer plate
            verbose: Enable verbose logging

        Returns:
            Processed single-layer plate image, or None if processing failed

        Source:
            Original utils/ocr_image_processing.py::process_plate_image()
        """
        if img is None or img.size == 0:
            if verbose:
                logging.warning("Input image is empty in process_plate_image")
            return None

        # Step 1: Grayscale conversion
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Skew correction
        skew_angle = OcrORT._detect_skew_angle(gray_img)
        corrected_img = OcrORT._correct_skew(img, skew_angle)

        if not is_double_layer:
            return corrected_img

        # Step 3: Double-layer processing
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray_img = clahe.apply(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))

        # Find split line
        split_point = OcrORT._find_optimal_split_line(enhanced_gray_img)

        # Split layers
        top_part, bottom_part = OcrORT._split_double_layer(corrected_img, split_point)

        # Validate parts
        bottom_h, bottom_w = bottom_part.shape[:2]
        top_h, top_w = top_part.shape[:2]

        if bottom_h <= 0 or bottom_w <= 0 or top_h <= 0 or top_w <= 0:
            if verbose:
                logging.warning("Part of the plate is empty after splitting.")
            return None

        # Stitch layers
        stitched_plate = OcrORT._stitch_layers(top_part, bottom_part)

        return stitched_plate

    @staticmethod
    def _resize_norm_img_static(
        img: NDArray[np.uint8],
        image_shape: List[int] = [3, 48, 168]
    ) -> NDArray[np.float32]:
        """
        Resize and normalize plate image for OCR.

        Processing:
        1. Maintain aspect ratio while resizing to target height
        2. Convert HWC to CHW format
        3. Normalize to [-1, 1] range
        4. Right-pad to target width
        5. Add batch dimension

        Args:
            img: Input plate image, BGR format [H, W, 3]
            image_shape: Target shape [C, H, W], default [3, 48, 168]

        Returns:
            Normalized tensor [1, C, H, W] float32

        Source:
            Original utils/ocr_image_processing.py::resize_norm_img()
        """
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]

        # Calculate resize width maintaining aspect ratio
        ratio = w / float(h)
        if int(np.ceil(imgH * ratio)) > imgW:
            resized_w = imgW
        else:
            resized_w = int(np.ceil(imgH * ratio))

        # Resize image
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')

        # Normalize
        if image_shape[0] == 1:
            # Grayscale
            resized_image = resized_image / 255.0
            resized_image = resized_image[np.newaxis, :]
        else:
            # Color: transpose to CHW then normalize
            resized_image = resized_image.transpose((2, 0, 1)) / 255.0

        # Normalize to [-1, 1]
        resized_image -= 0.5
        resized_image /= 0.5

        # Right-pad to target width
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        # Add batch dimension
        onnx_infer_data = padding_im[np.newaxis, :, :, :]
        onnx_infer_data = np.array(onnx_infer_data, dtype=np.float32)

        return onnx_infer_data

    @staticmethod
    def _get_ignored_tokens_static() -> List[int]:
        """
        Get list of ignored token indices for OCR decoding.

        Returns:
            List of token indices to ignore (e.g., blank token)

        Source:
            Original utils/ocr_post_processing.py::get_ignored_tokens()
        """
        return [0]

    @staticmethod
    def _decode_static(
        character: List[str],
        text_index: NDArray[np.int_],
        text_prob: Optional[NDArray[np.float32]] = None,
        is_remove_duplicate: bool = False
    ) -> List[OCRResult]:
        """
        Decode OCR output indices to text and confidence.

        Algorithm:
        1. For each batch sample:
           a. Filter ignored tokens
           b. Optionally remove duplicate characters
           c. Map indices to characters
           d. Calculate average confidence
           e. Apply post-processing rules (e.g., '苏' -> '京')
        2. Return list of (text, confidence, char_confidences)

        Args:
            character: Character dictionary list
            text_index: Character index array [B, seq_len]
            text_prob: Character probability array [B, seq_len], optional
            is_remove_duplicate: Whether to remove consecutive duplicates

        Returns:
            List of OCR results: [(text, avg_confidence, char_confidences), ...]

        Source:
            Original utils/ocr_post_processing.py::decode()
        """
        result_list = []
        ignored_tokens = OcrORT._get_ignored_tokens_static()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            # Create selection mask
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            # Remove duplicates if requested
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            # Filter ignored tokens
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            # Map indices to characters
            char_list = [
                character[int(text_id)].replace('\n', '')
                for text_id in text_index[batch_idx][selection]
            ]

            # Get confidences
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1.0] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0.0]

            # Join text
            text = ''.join(char_list)
            
            # NOTE: hack
            # Post-processing: Replace '苏' with '京'
            # if text.startswith('苏'):
            #     text = '京' + text[1:]

            # Convert confidences to float list
            float_conf_list = [float(c) for c in conf_list]
            result_list.append((text, float(np.mean(conf_list)), float_conf_list))

        return result_list

    def _postprocess(
        self,
        prediction: NDArray[np.float32],
        conf_thres: float,
        **kwargs
    ) -> List[OCRResult]:
        """
        Post-process OCR model output.

        This method:
        1. Extracts text indices and probabilities from prediction
        2. Calls _decode_static() to decode characters
        3. Filters results by confidence threshold

        Args:
            prediction: Model output [B, seq_len, num_classes]
            conf_thres: Confidence threshold
            **kwargs: Additional parameters

        Returns:
            List of OCR results
        """
        # Extract text indices and probabilities
        text_index = np.argmax(prediction, axis=-1)
        text_prob = np.max(prediction, axis=-1) if len(prediction.shape) > 2 else None

        # Decode
        results = self._decode_static(
            self.character,
            text_index,
            text_prob,
            is_remove_duplicate=True
        )

        # Filter by confidence
        filtered_results = [
            (text, conf, char_confs)
            for text, conf, char_confs in results
            if conf >= conf_thres
        ]

        return filtered_results if filtered_results else results

    def __call__(
        self,
        image: NDArray[np.uint8],
        conf_thres: Optional[float] = None,
        is_double_layer: bool = False
    ) -> Optional[OCRResult]:
        """
        Execute OCR inference on plate image.

        Args:
            image: Input plate image, BGR format [H, W, 3]
            conf_thres: Optional confidence threshold override
            is_double_layer: Whether this is a double-layer plate

        Returns:
            OCRResult or None:
                - (text, avg_conf, char_confs) if successful
                - None if OCR failed

        Example:
            >>> result = ocr_model(plate_img, is_double_layer=True)
            >>> if result:
            ...     text, conf, char_confs = result
            ...     print(f"Plate: {text}, Confidence: {conf:.3f}")
        """
        # Preprocess (handles double-layer processing internally)
        input_tensor, scale, original_shape, ratio_pad = self._preprocess(image, is_double_layer)

        # Inference using ONNX Runtime session
        feed_dict = {self.input_name: input_tensor}
        outputs = self._onnx_session.run(self.output_names, feed_dict)

        # Get prediction (first output)
        prediction = outputs[0]

        # Post-process
        effective_conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        results = self._postprocess(prediction, effective_conf_thres)

        # Return first result or None
        return results[0] if results else None
