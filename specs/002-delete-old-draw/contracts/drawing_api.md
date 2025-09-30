# API Contract: Drawing Module

**Feature**: 002-delete-old-draw
**Date**: 2025-09-30
**Status**: Contract Definition

## Overview

This document specifies the API contract for the drawing module after removing the PIL-based implementation. The contract ensures backward compatibility while simplifying the interface by removing the `use_supervision` parameter.

## Primary Function: draw_detections

### Function Signature

```python
def draw_detections(
    image: np.ndarray,
    detections: List[List[List[float]]],
    class_names: Union[Dict[int, str], List[str]],
    colors: List[tuple],
    plate_results: Optional[List[Optional[Dict[str, Any]]]] = None,
    font_path: str = "SourceHanSans-VF.ttf"
) -> np.ndarray
```

### Parameters

#### image (required)
- **Type**: `np.ndarray`
- **Description**: Input image to annotate
- **Constraints**:
  - Shape: `(H, W, 3)` where H, W > 0
  - Dtype: `uint8`
  - Color space: BGR (OpenCV convention)
- **Example**:
  ```python
  image = cv2.imread("input.jpg")  # Shape: (1080, 1920, 3)
  ```

#### detections (required)
- **Type**: `List[List[List[float]]]`
- **Description**: Detection results from ONNX inference
- **Format**: `[batch][detection][x1, y1, x2, y2, confidence, class_id]`
- **Constraints**:
  - Batch size typically 1 for visualization
  - Each detection: 6 floats [x1, y1, x2, y2, conf, cls]
  - Valid bounding boxes: `x2 > x1, y2 > y1`
  - Confidence: `0.0 <= conf <= 1.0`
  - Class ID: `cls >= 0`
- **Example**:
  ```python
  detections = [
      [[100.0, 200.0, 300.0, 400.0, 0.95, 0],  # Vehicle
       [420.0, 529.0, 509.0, 562.0, 0.93, 1]]  # Plate
  ]
  ```

#### class_names (required)
- **Type**: `Union[Dict[int, str], List[str]]`
- **Description**: Mapping from class IDs to human-readable names
- **Constraints**:
  - Dictionary: Keys must be non-negative integers
  - List: Non-empty, index corresponds to class ID
  - Supports UTF-8 (Chinese characters)
- **Example**:
  ```python
  # Dictionary variant
  class_names = {0: "vehicle", 1: "plate"}

  # List variant
  class_names = ["vehicle", "plate"]
  ```

#### colors (required)
- **Type**: `List[tuple]`
- **Description**: RGB color tuples for bounding boxes
- **Constraints**:
  - Non-empty list
  - Each tuple: `(R, G, B)` where `0 <= R,G,B <= 255`
  - Class ID maps via modulo: `colors[class_id % len(colors)]`
- **Example**:
  ```python
  colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
  ```

#### plate_results (optional)
- **Type**: `Optional[List[Optional[Dict[str, Any]]]]`
- **Default**: `None`
- **Description**: OCR metadata for plate detections
- **Format**: List aligned with detections, `None` for non-plates
- **Dict Schema**:
  ```python
  {
      "plate_text": str,           # e.g., "苏A88888"
      "plate_conf": float,         # [0.0, 1.0]
      "color": str,                # e.g., "blue"
      "layer": str,                # e.g., "single"
      "should_display_ocr": bool   # Whether to draw OCR text
  }
  ```
- **Example**:
  ```python
  plate_results = [
      None,  # Vehicle (no plate info)
      {"plate_text": "苏A88888", "plate_conf": 0.95,
       "color": "blue", "layer": "single", "should_display_ocr": True}
  ]
  ```

#### font_path (optional)
- **Type**: `str`
- **Default**: `"SourceHanSans-VF.ttf"`
- **Description**: Path to TrueType font file for text rendering
- **Constraints**:
  - Must be valid file path
  - Font should support Chinese characters
  - If file not found, supervision uses fallback font
- **Example**:
  ```python
  font_path = "/usr/share/fonts/SourceHanSans-VF.ttf"
  ```

### Return Value

- **Type**: `np.ndarray`
- **Description**: Annotated image with bounding boxes and labels
- **Properties**:
  - Same shape as input `image`
  - Same dtype: `uint8`
  - Same color space: BGR
  - Contains rendered detection boxes, class labels, and OCR text
- **Example**:
  ```python
  result = draw_detections(image, detections, class_names, colors)
  assert result.shape == image.shape
  assert result.dtype == np.uint8
  ```

### Exceptions

#### ImportError
- **Condition**: supervision library not installed
- **Message**: `"supervision library is required for drawing functionality. Install it with: pip install supervision>=0.16.0"`
- **Handling**: Fail-fast at module import time
- **Example**:
  ```python
  try:
      from utils.drawing import draw_detections
  except ImportError as e:
      print(f"Missing dependency: {e}")
  ```

#### TypeError
- **Condition**: Invalid parameter types
- **Example triggers**:
  - `image` is not numpy array
  - `detections` is not list
  - `class_names` is neither dict nor list
- **Handling**: Raised by Python type system or validation

#### ValueError
- **Condition**: Invalid parameter values
- **Example triggers**:
  - Empty `detections` list
  - Invalid bounding box coordinates
  - Empty `class_names`
- **Handling**: Validation in supervision converter module

### Behavior Guarantees

1. **Idempotent**: Calling with same inputs produces same output
2. **Stateless**: No side effects (except logging)
3. **Non-destructive**: Input `image` is copied, not modified
4. **Performance**: <10ms for 20 detections on 1920x1080 image (GPU)
5. **Memory**: Peak usage ~2x input image size during rendering

## Breaking Changes from Previous API

### Removed Parameters

#### use_supervision (REMOVED)
- **Previous signature**: `use_supervision: bool = True`
- **Reason for removal**: Only supervision implementation remains
- **Migration**:
  - **Before**: `draw_detections(..., use_supervision=True)`
  - **After**: `draw_detections(...)` (simply omit parameter)
- **Impact**: Minimal - parameter was rarely used explicitly

### Behavioral Changes

1. **Always uses supervision**:
   - Previous: Could fall back to PIL if `use_supervision=False`
   - Current: Always uses supervision implementation
   - Impact: None (PIL fallback was never used in production)

2. **Fail-fast on missing supervision**:
   - Previous: Silent fallback to PIL with warning
   - Current: ImportError at module load time
   - Impact: Clearer error messages, faster failure detection

## Contract Tests

Contract tests verify the following properties:

### Test 1: Function Signature
```python
def test_draw_detections_signature():
    """Verify function signature matches contract"""
    import inspect
    sig = inspect.signature(draw_detections)
    assert len(sig.parameters) == 6
    assert 'use_supervision' not in sig.parameters  # REMOVED
    assert sig.return_annotation == np.ndarray
```

### Test 2: Input Validation
```python
def test_draw_detections_input_validation():
    """Verify invalid inputs raise appropriate errors"""
    with pytest.raises(TypeError):
        draw_detections(None, [], {}, [])  # Invalid image type

    with pytest.raises(ValueError):
        draw_detections(image, [], {}, [])  # Empty detections
```

### Test 3: Output Format
```python
def test_draw_detections_output_format():
    """Verify output matches expected format"""
    result = draw_detections(image, detections, class_names, colors)
    assert isinstance(result, np.ndarray)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
```

### Test 4: Supervision Integration
```python
def test_draw_detections_uses_supervision():
    """Verify supervision annotators are called"""
    with patch('utils.drawing.create_box_annotator') as mock_box:
        with patch('utils.drawing.create_rich_label_annotator') as mock_label:
            draw_detections(image, detections, class_names, colors)
            assert mock_box.called
            assert mock_label.called
```

### Test 5: Backward Compatibility
```python
def test_backward_compatible_call():
    """Verify existing code continues to work"""
    # Old code that doesn't pass use_supervision explicitly
    result = draw_detections(
        image=test_image,
        detections=test_detections,
        class_names={"0": "vehicle", "1": "plate"},
        colors=[(255, 0, 0), (0, 255, 0)],
        plate_results=test_plate_results
    )
    assert result is not None
```

## Performance Contract

### Latency Requirements
- **Target**: <10ms per frame for 20 detections (1920x1080 image)
- **Current**: ~7ms with supervision (GPU)
- **Measurement**: Averaged over 100 iterations

### Memory Requirements
- **Peak usage**: ~2x input image size during rendering
- **Example**: 1920x1080x3 image = ~6MB input, ~12MB peak
- **Cleanup**: Memory released after function return

### Benchmark Validation
```python
def test_performance_benchmark():
    """Verify performance meets target"""
    results = benchmark_drawing_performance(
        image=test_image,
        detections_data=test_detections,
        iterations=100
    )
    assert results['supervision_avg_time'] < 10.0  # milliseconds
```

## Deprecation Notice

### Deprecated (as of 002-delete-old-draw)
- `use_supervision` parameter - Remove from calling code

### Removed (as of 002-delete-old-draw)
- PIL-based implementation - No longer available as fallback
- `SUPERVISION_AVAILABLE` flag - Supervision is now required

## Migration Guide

### For Library Users

**Before** (old code with PIL fallback):
```python
from utils.drawing import draw_detections

result = draw_detections(
    image, detections, class_names, colors,
    plate_results=None,
    font_path="font.ttf",
    use_supervision=True  # ← Remove this parameter
)
```

**After** (new code, supervision only):
```python
from utils.drawing import draw_detections

result = draw_detections(
    image, detections, class_names, colors,
    plate_results=None,
    font_path="font.ttf"
    # use_supervision parameter removed
)
```

### For Library Maintainers

**Ensure supervision installed**:
```bash
pip install supervision>=0.16.0
```

**Update imports**:
- No changes needed - supervision import remains at module level

**Update tests**:
- Remove tests for `use_supervision=False` behavior
- Remove PIL fallback tests
- Keep all supervision-based tests

---
*API Contract Version: 1.0.0 - Effective: 2025-09-30*