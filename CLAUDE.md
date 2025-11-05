# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Project Overview

ONNX-based vehicle and license plate recognition system supporting multiple detection architectures (YOLO, RT-DETR, RF-DETR), OCR recognition, and color/layer classification. The project uses spec-driven development with OpenSpec and emphasizes modular design with strong abstraction patterns.

## Essential Commands

### Setup and Installation
```bash
# Install core dependencies (recommended: uv)
uv sync

# Install with TensorRT support (optional, 2-5x speedup)
uv pip install pip setuptools wheel
uv pip install -e ".[trt]"

# Verify installation
python -c "import onnxtools; print('OK')"
```

### Running Inference
```bash
# Quick start with default model
./run.sh

# Image inference
python main.py \
    --model-path models/rtdetr-2024080100.onnx \
    --model-type rtdetr \
    --input data/sample.jpg \
    --output-mode show

# Video inference with frame skip
python main.py \
    --model-path models/yolo11n.onnx \
    --model-type yolo \
    --input video.mp4 \
    --source-type video \
    --output-mode save \
    --frame-skip 2

# Camera inference
python main.py \
    --model-path models/rtdetr.onnx \
    --model-type rtdetr \
    --input 0 \
    --source-type camera \
    --output-mode show
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/contract/ -v                # Contract tests
pytest tests/performance/ -v --benchmark-only  # Performance tests

# Run tests with coverage
pytest --cov=onnxtools --cov-report=html

# Run single test file
pytest tests/unit/test_ocr_metrics.py -v
```

### TensorRT Engine Building
```bash
# Build FP16 engine
python tools/build_engine.py \
    --onnx-path models/rtdetr.onnx \
    --engine-path models/rtdetr_fp16.engine \
    --fp16

# Build with accuracy comparison
python tools/build_engine.py \
    --onnx-path models/yolov8s_640.onnx \
    --compare

# Compare ONNX and TensorRT outputs
python tools/compare_onnx_engine.py \
    --onnx models/yolov8s_640.onnx \
    --engine models/yolov8s_640.engine
```

### Model Evaluation
```bash
# COCO dataset evaluation
python tools/eval.py \
    --model-type rtdetr \
    --model-path models/rtdetr.onnx \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25 \
    --iou-threshold 0.7

# OCR dataset evaluation
python -m onnxtools.infer_onnx.eval_ocr \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5
```

### Polygraphy Debugging
```bash
# Inspect model structure
polygraphy inspect model models/yolov8s_640.onnx

# Run inference and save outputs
polygraphy run models/yolov8s_640.onnx \
    --onnxrt --save-outputs results.json

# Compare ONNX and TensorRT
polygraphy run models/yolov8s_640.onnx \
    --onnxrt --trt --compare
```

### OpenSpec Workflow
```bash
# List active changes and specs
openspec list                  # Active changes
openspec list --specs          # Existing specs

# Show details
openspec show <change-id>      # Show change
openspec show <spec-id> --type spec  # Show spec

# Validate changes
openspec validate <change-id> --strict

# Archive after deployment
openspec archive <change-id> --yes
```

## Architecture

### Core Design Patterns

**Template Method Pattern (BaseORT)**
```python
# All inference classes inherit from BaseORT and implement:
class BaseORT(ABC):
    def __call__(self, img, **kwargs):
        # Template method orchestrating inference pipeline
        prepared = self._prepare_inference(img, **kwargs)
        outputs = self._execute_inference(prepared)
        return self._finalize_inference(outputs, **kwargs)

    @abstractmethod
    def _preprocess_static(img, **kwargs):
        # Must be implemented by subclasses
        pass

    @abstractmethod
    def _postprocess(outputs, **kwargs):
        # Must be implemented by subclasses
        pass
```

**Factory Pattern (Model Creation)**
```python
# Centralized model creation through factory function
from onnxtools import create_detector

detector = create_detector(
    model_type='rtdetr',  # 'yolo', 'rtdetr', 'rfdetr'
    onnx_path='models/rtdetr.onnx',
    conf_thres=0.5,
    iou_thres=0.5
)
```

**Lazy Loading (Polygraphy Integration)**
- ONNX Runtime sessions created on-demand
- TensorRT engines loaded lazily
- Reduces initialization time by 93% (800ms → 50ms)

### Module Structure

```
onnxtools/                      # Core Python package
├── infer_onnx/                 # Inference engines
│   ├── onnx_base.py            # BaseORT abstract class (KEY)
│   ├── onnx_yolo.py            # YOLO implementation
│   ├── onnx_rtdetr.py          # RT-DETR implementation
│   ├── onnx_rfdetr.py          # RF-DETR implementation
│   ├── onnx_ocr.py             # OCR + Color/Layer classifier
│   ├── eval_coco.py            # COCO dataset evaluator
│   └── eval_ocr.py             # OCR dataset evaluator
│
├── utils/                      # Utilities
│   ├── drawing.py              # Supervision-based visualization
│   ├── annotator_factory.py   # 13 annotator types (Factory)
│   ├── visualization_preset.py # 5 visualization presets
│   ├── ocr_metrics.py          # OCR evaluation metrics
│   └── supervision_*.py        # Supervision integration
│
└── pipeline.py                 # Full inference pipeline

tools/                          # Debugging and optimization
├── eval.py                     # Model evaluation
├── build_engine.py             # TensorRT engine builder
├── compare_onnx_engine.py      # ONNX vs TensorRT comparison
└── debug/                      # Debugging scripts

tests/                          # Test suite
├── unit/                       # Component-level tests
├── integration/                # End-to-end tests
├── contract/                   # API contract tests
└── performance/                # Benchmark tests

specs/                          # Feature specifications (OpenSpec)
openspec/                       # OpenSpec system
configs/                        # YAML configurations
models/                         # ONNX models and TensorRT engines
```

### Key Data Flows

**Detection Pipeline**
```
Input Image → BaseORT.__call__() → _preprocess_static()
    → ONNX/TensorRT Inference → _postprocess()
    → {boxes, scores, class_ids} → Supervision Converter
    → Annotator Pipeline → Rendered Image
```

**OCR Pipeline**
```
Detected Plate → OcrORT.__call__() → _process_plate_image_static()
    → ONNX Inference → CTC Decode → (text, confidence, char_scores)

Detected Plate → ColorLayerORT.__call__() → _preprocess_static()
    → ONNX Inference → Argmax → (color, layer, confidence)
```

## Critical Conventions

### Adding New Model Architecture

1. **Create inference class** in `onnxtools/infer_onnx/`:
```python
from onnxtools.infer_onnx.onnx_base import BaseORT

class NewModelORT(BaseORT):
    @staticmethod
    def _preprocess_static(img, input_shape, **kwargs):
        # Implement preprocessing
        return preprocessed_img

    def _postprocess(self, outputs, **kwargs):
        # Implement postprocessing
        return {'boxes': ..., 'scores': ..., 'class_ids': ...}
```

2. **Register in factory** (`onnxtools/__init__.py`):
```python
def create_detector(model_type, onnx_path, **kwargs):
    if model_type == 'newmodel':
        return NewModelORT(onnx_path, **kwargs)
```

3. **Add tests** in `tests/unit/` and `tests/integration/`

### OpenSpec Workflow for New Features

When adding features, architecture changes, or breaking changes:

1. **Search existing work**: `openspec list --specs`
2. **Create proposal**: Scaffold under `openspec/changes/<change-id>/`
   - `proposal.md` - What and why
   - `tasks.md` - Implementation checklist
   - `design.md` - Technical decisions (if needed)
   - Delta specs for affected capabilities
3. **Validate**: `openspec validate <change-id> --strict`
4. **Get approval** before implementation
5. **Implement** tasks sequentially, marking off in `tasks.md`
6. **Archive**: `openspec archive <change-id> --yes` after deployment

Skip proposals for: bug fixes, typos, dependency updates, configuration changes.

### Visualization Customization

**Using presets** (5 built-in):
```bash
# Standard: box corners + simple labels
--annotator-preset standard

# Debug: round box + confidence bar + detailed labels (OCR text visible)
--annotator-preset debug

# Lightweight: dot markers + small labels
--annotator-preset lightweight

# Privacy: boxes + blur plates
--annotator-preset privacy

# High contrast: filled regions + background dimming
--annotator-preset high_contrast
```

**Custom annotator combination**:
```python
from onnxtools.utils import AnnotatorFactory

factory = AnnotatorFactory()
annotators = [
    factory.create('round_box', roundness=0.4, thickness=3),
    factory.create('percentage_bar'),
    factory.create('rich_label')
]
```

Supported types: `box`, `round_box`, `box_corner`, `circle`, `triangle`, `ellipse`, `dot`, `color`, `background_overlay`, `halo`, `percentage_bar`, `blur`, `pixelate`

### Code Style and Testing

- **PEP 8** compliance
- **Type hints** required for all functions
- **Google-style docstrings**
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `BaseORT`)
  - Functions: `snake_case` (e.g., `create_detector`)
  - Constants: `UPPER_CASE`
- **Test requirements**:
  - Unit tests for new utilities/functions
  - Integration tests for new inference classes
  - Contract tests for API changes
  - Performance benchmarks for optimization claims

### Model Configuration

Detection models require `configs/det_config.yaml`:
```yaml
names:
  0: vehicle
  1: plate

visual_colors:
  0: [255, 0, 0]    # red for vehicle
  1: [0, 255, 0]    # green for plate
```

OCR models require `configs/plate.yaml`:
```yaml
plate_dict:
  character: "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

color_map:
  0: blue
  1: yellow
  2: white
  3: black
  4: green

layer_map:
  0: single
  1: double
```

## Important Implementation Details

### BaseORT Lifecycle
```python
# Initialization
detector = create_detector(...)  # Creates ONNX Runtime session immediately

# Inference (lazy TensorRT loading)
result = detector(image)         # Loads TensorRT if .engine file

# Cleanup (automatic)
del detector                     # ONNX Runtime session auto-cleanup
```

### Error Handling for Missing Abstractions
```python
# If subclass doesn't implement required methods:
raise NotImplementedError(
    f"{self.__class__.__name__} must implement _postprocess(). "
    f"See base class docstring for requirements."
)
```

### OCR Double-Layer Detection
```python
# Single-layer: 7-8 characters (e.g., "京A12345")
# Double-layer: 7-8 chars + separator (e.g., "京AF1234学")
result = ocr_model(plate_image, is_double_layer=True)
if result:
    text, conf, char_confs = result
```

### Polygraphy Lazy Loading
```python
# Import at class level (fast)
from polygraphy.backend.onnxrt import SessionFromOnnx

# Create session immediately in __init__ (cached for reuse)
self._onnx_session = onnxruntime.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

## Performance Targets

- **Inference latency**: < 50ms (640x640, ONNX)
- **TensorRT speedup**: 2-5x vs ONNX
- **GPU memory**: < 2GB (batch_size=1)
- **Annotator rendering**: < 30ms (20 objects)
- **Test passing rate**: > 95%

## Documentation Structure

Each module has a `CLAUDE.md` with:
- Breadcrumb navigation (`[Root] > [onnxtools] > [module]`)
- Entry points and quick start
- External API documentation
- Data models
- Test coverage
- FAQ

Key docs:
- `onnxtools/CLAUDE.md` - Package overview
- `onnxtools/infer_onnx/CLAUDE.md` - Inference engines
- `onnxtools/utils/CLAUDE.md` - Utilities
- `docs/polygraphy使用指南/` - Polygraphy debugging
- `README.md` - User-facing documentation

---

Last updated: 2025-11-05

## Active Technologies
- Python 3.10+ （项目现有版本） + numpy>=2.2.6, opencv-contrib-python>=4.12.0, supervision==0.26.1 （项目现有依赖） (001-baseort-result-third)
- N/A （内存中的数据结构，不涉及持久化） (001-baseort-result-third)

## Recent Changes
- 001-baseort-result-third: Added Python 3.10+ （项目现有版本） + numpy>=2.2.6, opencv-contrib-python>=4.12.0, supervision==0.26.1 （项目现有依赖）
