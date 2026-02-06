from typing import Tuple

import cv2
import numpy as np


class UltralyticsLetterBox:
    """
    Optimized LetterBox implementation based on ultralytics/data/augment.py::LetterBox

    This class provides efficient image resizing and padding for YOLO model inference,
    directly migrated from ultralytics for better performance and compatibility.

    Source: ultralytics/data/augment.py::LetterBox
    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py
    """

    def __init__(
        self,
        new_shape: Tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
        half: bool = False,
    ):
        """
        Initialize LetterBox object for resizing and padding images.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (int): Value for padding the image. Default is 114.
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
            half (bool): If True, use half precision (FP16). Default is False.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center
        self.padding_value = padding_value
        self.interpolation = interpolation
        self.half = half

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """
        Apply letterboxing to an image for object detection inference.

        Based on ultralytics LetterBox.__call__ method, optimized for inference-only use.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            Tuple containing:
                - preprocessed_tensor (np.ndarray): Preprocessed image tensor [1, C, H, W]
                - scale (float): Scaling factor applied to the image
                - original_shape (Tuple[int, int]): Original image shape (height, width)
                - ratio_pad (Tuple[int, int]): Padding information (pad_w, pad_h)
        """
        # Get original shape
        original_shape = image.shape[:2]  # (height, width)
        shape = original_shape
        new_shape = self.new_shape

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        # Resize if needed
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(image, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]
        else:
            img = image

        # Add padding
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        # Apply ultralytics-style preprocessing (复刻DetectionValidator.preprocess逻辑)
        # Convert to float32 and normalize to [0, 1] (batch["img"].float() / 255)
        img = img.astype(np.float32 if not self.half else np.float16) / 255.0

        # Transpose from HWC to CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        preprocessed_tensor = np.expand_dims(img, axis=0)

        # Return information needed for post-processing
        scale = r  # Single scale factor
        ratio_pad = (left, top)  # Padding info for coordinate scaling

        return preprocessed_tensor, scale, original_shape, ratio_pad
