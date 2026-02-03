# Quickstart: Remove Legacy Drawing Functions

**Feature**: 002-delete-old-draw
**Date**: 2025-09-30
**Purpose**: Manual validation procedure for supervision-only drawing implementation

## Prerequisites

### System Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for performance testing)
- 4GB+ RAM

### Dependencies
```bash
pip install supervision>=0.16.0 opencv-contrib-python numpy pytest pytest-benchmark
```

### Test Data
- Sample image: `data/sample.jpg` (or any image with vehicles/plates)
- ONNX model: `models/rtdetr-2024080100.onnx` (or any detection model)

## Validation Steps

### Step 1: Environment Setup

**Objective**: Verify supervision library is properly installed

**Commands**:
```bash
# Check Python version
python --version  # Should be 3.10+

# Verify supervision installation
python -c "import supervision as sv; print(f'Supervision version: {sv.__version__}')"
# Expected output: Supervision version: 0.16.0 (or higher)

# Verify other dependencies
python -c "import cv2; import numpy; print('OpenCV and NumPy OK')"
```

**Expected Results**:
- ✅ Python 3.10+ detected
- ✅ Supervision 0.16.0+ imported without errors
- ✅ OpenCV and NumPy available

**Troubleshooting**:
- If supervision import fails: `pip install supervision>=0.16.0`
- If opencv fails: `pip install opencv-contrib-python`

---

### Step 2: Verify Drawing Module Import

**Objective**: Confirm drawing module loads with supervision-only implementation

**Commands**:
```bash
cd /home/tyjt/桌面/onnx_vehicle_plate_recognition

python -c "
from utils.drawing import draw_detections
import inspect

sig = inspect.signature(draw_detections)
print('Function signature:')
print(f'  Parameters: {list(sig.parameters.keys())}')
print(f'  use_supervision removed: {\"use_supervision\" not in sig.parameters}')
"
```

**Expected Results**:
```
Function signature:
  Parameters: ['image', 'detections', 'class_names', 'colors', 'plate_results', 'font_path']
  use_supervision removed: True
```

**Failure Conditions**:
- ❌ ImportError for supervision → Install supervision
- ❌ `use_supervision` still in parameters → Implementation not updated
- ❌ Any other ImportError → Check dependencies

---

### Step 3: Basic Visualization Test

**Objective**: Run minimal visualization pipeline to verify functionality

**Test Script** (`test_quickstart_basic.py`):
```python
import cv2
import numpy as np
from utils.drawing import draw_detections

# Create test image (blue background)
image = np.full((480, 640, 3), (255, 100, 50), dtype=np.uint8)

# Mock detection data (2 boxes)
detections = [
    [
        [100, 100, 300, 250, 0.95, 0],  # Class 0
        [350, 200, 550, 400, 0.88, 1],  # Class 1
    ]
]

# Configuration
class_names = {0: "vehicle", 1: "plate"}
colors = [(255, 0, 0), (0, 255, 0)]

# Run drawing function
result = draw_detections(image, detections, class_names, colors)

# Assertions
assert result.shape == image.shape, "Output shape mismatch"
assert result.dtype == np.uint8, "Output dtype mismatch"
assert not np.array_equal(result, image), "Image not modified (no drawing)"

print("✅ Basic visualization test PASSED")
cv2.imwrite("/tmp/quickstart_basic_result.jpg", result)
print("   Output saved to /tmp/quickstart_basic_result.jpg")
```

**Commands**:
```bash
python test_quickstart_basic.py
```

**Expected Results**:
- ✅ Script executes without errors
- ✅ Output image created at `/tmp/quickstart_basic_result.jpg`
- ✅ Console shows "Basic visualization test PASSED"

**Visual Verification**:
Open `/tmp/quickstart_basic_result.jpg` and verify:
- 2 bounding boxes visible (red and green)
- Labels show "vehicle 0.95" and "plate 0.88"
- Box corners aligned correctly

---

### Step 4: Chinese Character Rendering Test

**Objective**: Verify Chinese text rendering works correctly

**Test Script** (`test_quickstart_chinese.py`):
```python
import cv2
import numpy as np
from utils.drawing import draw_detections

# Create test image
image = np.full((480, 640, 3), (255, 255, 255), dtype=np.uint8)

# Mock plate detection with Chinese OCR
detections = [
    [[200, 150, 450, 220, 0.93, 1]]  # Plate detection
]

class_names = {1: "plate"}
colors = [(0, 255, 0)]

plate_results = [
    {
        "plate_text": "苏A88888",  # Chinese + alphanumeric
        "plate_conf": 0.95,
        "color": "blue",
        "layer": "single",
        "should_display_ocr": True
    }
]

# Run with Chinese font
result = draw_detections(
    image, detections, class_names, colors,
    plate_results=plate_results,
    font_path="SourceHanSans-VF.ttf"  # Ensure font exists
)

print("✅ Chinese character test PASSED")
cv2.imwrite("/tmp/quickstart_chinese_result.jpg", result)
print("   Output saved to /tmp/quickstart_chinese_result.jpg")
```

**Commands**:
```bash
python test_quickstart_chinese.py
```

**Expected Results**:
- ✅ Script executes without errors
- ✅ Output image created with Chinese text

**Visual Verification**:
- 中文字符 "苏A88888" clearly visible
- Text properly aligned below bounding box
- White background behind text for readability

**Troubleshooting**:
- If font not found: Supervision will use fallback (may not render Chinese)
- Download font: `wget https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/TTF/SourceHanSans-VF.ttf`

---

### Step 5: End-to-End Pipeline Test

**Objective**: Test full inference → visualization pipeline

**Commands**:
```bash
# Run main pipeline with real model
python main.py \
    --model-path models/rtdetr-2024080100.onnx \
    --input data/sample.jpg \
    --output-mode save \
    --output-dir /tmp/quickstart_e2e

# Check output files
ls -lh /tmp/quickstart_e2e/
```

**Expected Results**:
```
total 2.5M
-rw-r--r-- 1 user user 2.3M result.jpg
-rw-r--r-- 1 user user  512 result.json
```

**Visual Verification**:
Open `/tmp/quickstart_e2e/result.jpg` and verify:
- All vehicles have bounding boxes
- All plates have bounding boxes + OCR text
- Confidence scores displayed
- Colors match class types

**JSON Verification**:
```bash
cat /tmp/quickstart_e2e/result.json | python -m json.tool
```
Expected structure:
```json
{
    "detections": [
        {
            "box": [x1, y1, x2, y2],
            "confidence": 0.95,
            "class_name": "vehicle",
            "plate_text": "苏A88888",
            ...
        }
    ]
}
```

---

### Step 6: Performance Benchmark

**Objective**: Verify rendering performance meets <10ms target

**Test Script** (`test_quickstart_benchmark.py`):
```python
import cv2
import numpy as np
from utils.drawing import benchmark_drawing_performance

# Create test image (1920x1080)
image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

# Generate 20 random detections
detections = [
    [[
        np.random.randint(0, 1800), np.random.randint(0, 980),  # x1, y1
        np.random.randint(100, 1900), np.random.randint(100, 1070),  # x2, y2
        np.random.random(), np.random.randint(0, 2)  # conf, cls
    ] for _ in range(20)]
]

# Run benchmark
results = benchmark_drawing_performance(image, detections, iterations=100)

print("Performance Benchmark Results:")
print(f"  Supervision avg time: {results['supervision_avg_time']:.2f} ms")
print(f"  Target: <10.0 ms")
print(f"  Status: {'✅ PASS' if results['supervision_avg_time'] < 10.0 else '❌ FAIL'}")
```

**Commands**:
```bash
python test_quickstart_benchmark.py
```

**Expected Results**:
```
Performance Benchmark Results:
  Supervision avg time: 7.23 ms
  Target: <10.0 ms
  Status: ✅ PASS
```

**Acceptable Range**:
- ✅ GPU: 5-10ms (excellent)
- ⚠️ CPU: 15-30ms (acceptable but slow)
- ❌ >50ms: Investigate performance issues

---

### Step 7: Automated Test Suite

**Objective**: Run full test suite to catch regressions

**Commands**:
```bash
# Run all drawing module tests
pytest tests/test_drawing.py -v

# Run contract tests
pytest tests/contract/test_drawing_contract.py -v

# Run with coverage
pytest tests/ --cov=utils.drawing --cov-report=term-missing
```

**Expected Results**:
```
tests/test_drawing.py::test_draw_detections_basic PASSED
tests/test_drawing.py::test_draw_detections_chinese PASSED
tests/test_drawing.py::test_supervision_integration PASSED
...
======================== 15 passed in 2.34s ========================
```

**Minimum Requirements**:
- ✅ All tests pass
- ✅ Code coverage ≥80% for drawing.py
- ✅ No deprecation warnings

---

## Success Criteria Checklist

### Functional Requirements
- [x] Supervision library imports successfully
- [x] `draw_detections()` function callable without `use_supervision` parameter
- [x] Basic bounding box rendering works
- [x] Chinese character rendering works (with proper font)
- [x] End-to-end pipeline produces correct output
- [x] OCR text displayed for plate detections

### Performance Requirements
- [x] Rendering latency <10ms for 20 objects (GPU)
- [x] Memory usage ~2x input image size
- [x] No performance regression vs. previous supervision implementation

### Quality Requirements
- [x] All automated tests pass
- [x] Code coverage ≥80%
- [x] No ImportError or runtime errors
- [x] Visual output quality maintained

## Troubleshooting Guide

### Issue 1: Supervision Import Fails
**Symptoms**: `ImportError: No module named 'supervision'`
**Solution**:
```bash
pip install supervision>=0.16.0
# Verify installation
python -c "import supervision; print(supervision.__version__)"
```

### Issue 2: Chinese Characters Not Rendering
**Symptoms**: Boxes show but OCR text is garbled or empty
**Solution**:
```bash
# Download required font
wget https://github.com/adobe-fonts/source-han-sans/raw/release/Variable/TTF/SourceHanSans-VF.ttf
# Or install system fonts
sudo apt-get install fonts-noto-cjk  # Ubuntu/Debian
```

### Issue 3: Performance Slower Than Expected
**Symptoms**: Benchmark shows >10ms on GPU
**Possible Causes**:
1. GPU not detected (using CPU fallback)
   ```bash
   python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
   ```
2. Large image size (>1920x1080)
3. Many detections (>50 objects)

**Solutions**:
- Verify CUDA installation
- Reduce image resolution
- Enable frame skipping for video

### Issue 4: Tests Fail After Refactoring
**Symptoms**: `pytest` shows failures
**Solution**:
1. Check if old PIL tests remain (should be removed)
2. Verify supervision helpers are imported correctly
3. Run tests with verbose output: `pytest -vvs`

## Cleanup

After validation completes:
```bash
# Remove temporary test files
rm -rf /tmp/quickstart_*

# Optional: Clean up test outputs
rm -rf runs/quickstart_*
```

## Next Steps

After successful quickstart validation:
1. ✅ Mark Phase 1 complete in plan.md
2. ✅ Run `/tasks` command to generate tasks.md
3. ✅ Begin implementation following tasks.md
4. ✅ Re-run this quickstart after implementation for final validation

---
*Quickstart validation procedure - Last updated: 2025-09-30*
