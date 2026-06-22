# CLAUDE.md

> Navigation and conventions file for AI assistants working in this repo. For user-facing install/CLI/perf, see [README.md](README.md).

ONNX-based vehicle and license plate recognition toolkit. Supports YOLO / RT-DETR / RF-DETR detection, OCR, and color/layer classification.

## Quick Navigation

| I want to... | Go here |
|--------------|---------|
| Install, run, CLI args, performance numbers | [README.md](README.md) |
| Add a new model architecture | [Core Conventions](#core-conventions) below + [onnxtools/infer_onnx/CLAUDE.md](onnxtools/infer_onnx/CLAUDE.md) |
| Inference engine internals (BaseORT, Result, ORT subclasses) | [onnxtools/infer_onnx/CLAUDE.md](onnxtools/infer_onnx/CLAUDE.md) |
| Visualization customization (13 annotators, 5 presets) | [onnxtools/utils/CLAUDE.md](onnxtools/utils/CLAUDE.md) · [docs/annotator_usage.md](docs/annotator_usage.md) |
| Model evaluation (COCO mAP, OCR metrics, MOT HOTA/MOTA/IDF1) | [onnxtools/eval/CLAUDE.md](onnxtools/eval/CLAUDE.md) · [docs/evaluation_guide.md](docs/evaluation_guide.md) |
| 2D multi-object tracking (ByteTrack, OC-SORT) | [onnxtools/tracking/CLAUDE.md](onnxtools/tracking/CLAUDE.md) · [docs/api/tracking.md](docs/api/tracking.md) |
| TensorRT build / ONNX debugging | [tools/README.md](tools/README.md) · [docs/polygraphy使用指南/](docs/polygraphy使用指南/) |
| Run/write tests | [tests/CLAUDE.md](tests/CLAUDE.md) |
| Config files (det_config / plate / visualization presets) | [configs/CLAUDE.md](configs/CLAUDE.md) |
| MCP server / LLM tool interface | [mcp_tools/CLAUDE.md](mcp_tools/CLAUDE.md) |

## Module Index

| Path | Responsibility | Docs |
|------|---------------|------|
| `onnxtools/` | Core Python package, public API entry | [CLAUDE.md](onnxtools/CLAUDE.md) |
| `onnxtools/infer_onnx/` | YOLO / RT-DETR / RF-DETR / OCR inference classes | [CLAUDE.md](onnxtools/infer_onnx/CLAUDE.md) |
| `onnxtools/utils/` | Image processing, Supervision visualization, OCR metrics | [CLAUDE.md](onnxtools/utils/CLAUDE.md) |
| `onnxtools/eval/` | COCO / OCR dataset evaluators | [CLAUDE.md](onnxtools/eval/CLAUDE.md) |
| `onnxtools/tracking/` | 2D MOT — ByteTrack (supervision / native) + OC-SORT | [CLAUDE.md](onnxtools/tracking/CLAUDE.md) |
| `tools/` | Evaluation, TensorRT build, ONNX vs TRT comparison | [README.md](tools/README.md) |
| `tests/` | unit / integration / contract / performance | [CLAUDE.md](tests/CLAUDE.md) |
| `configs/` | Detection classes, OCR dict, visualization presets (YAML) | [CLAUDE.md](configs/CLAUDE.md) |
| `models/` | ONNX models and TensorRT engines (.gitignore) | [CLAUDE.md](models/CLAUDE.md) |
| `specs/` | Completed feature specs (spec-kit) | [CLAUDE.md](specs/CLAUDE.md) |
| `openspec/` | OpenSpec change proposals and archive | [CLAUDE.md](openspec/CLAUDE.md) |
| `docs/` | User docs, Polygraphy guide, API reference | [CLAUDE.md](docs/CLAUDE.md) |
| `third_party/` | Ultralytics / Polygraphy / RF-DETR integrations | [CLAUDE.md](third_party/CLAUDE.md) |
| `mcp_tools/` | FastMCP server, LLM tool interface | [CLAUDE.md](mcp_tools/CLAUDE.md) |
| `examples/` | Demo scripts (demo_pipeline, demo_crop) | — |

## Core Architectural Contracts

Invariants AI must understand before modifying code in this repo. Implementation details in [onnxtools/infer_onnx/CLAUDE.md](onnxtools/infer_onnx/CLAUDE.md).

### BaseORT Template Method
- All inference classes inherit from [BaseORT](onnxtools/infer_onnx/onnx_base.py). `__call__` is provided by the base — **subclasses must not override it**.
- Subclasses **must** implement two abstract methods: `_preprocess_static(img, **kwargs)` and `_postprocess(outputs, **kwargs)`.
- When unimplemented, the base raises `NotImplementedError` with the class name. Do not silently fall back.

### Factory Function Is the Only Entry Point
- Detectors **must** be created via `create_detector(model_type, onnx_path, **kwargs)` (see [onnxtools/__init__.py](onnxtools/__init__.py)).
- Supported `model_type`: `yolo`, `rtdetr`, `rfdetr`, `rfdetr_unified`.
- Do not instantiate `YoloORT(...)` directly in `pipeline.py` or external code — bypasses future registration logic.

### Result Data Class
- All detectors return a unified [Result](onnxtools/infer_onnx/result.py) object with fields `boxes` / `scores` / `class_ids`.
- New fields must stay backward-compatible (default values or `Optional`).

### Polygraphy Lazy Loading
- The ONNX `InferenceSession` is created and cached in `__init__`; TensorRT engines are loaded on demand only when an `.engine` file is detected.
- This brings init from ~800ms down to ~50ms. Do not break this property when editing `__init__`.

### Data Flows
- Detection: `Image → BaseORT.__call__ → Result → Supervision Converter → Annotator Pipeline → Rendered`
- OCR: `Plate Crop → OcrORT → CTC Decode → (text, conf, char_scores)`
- Classification: `Plate Crop → ColorLayerORT → argmax → (color, layer, conf)`

## Core Conventions

### Adding a New Model Architecture

1. **Create the inference class** at `onnxtools/infer_onnx/onnx_<name>.py`, inheriting `BaseORT`. Implement `_preprocess_static` and `_postprocess` returning a `Result`.
2. **Register in the factory**: edit `create_detector` in [onnxtools/__init__.py](onnxtools/__init__.py), adding a new `model_type` branch.
3. **Add tests**: `tests/unit/test_<name>.py` + `tests/integration/test_<name>_pipeline.py`.

Full template in [onnxtools/infer_onnx/CLAUDE.md](onnxtools/infer_onnx/CLAUDE.md).

### Code Style
- PEP 8, type hints (required), Google-style docstrings on every public class/function.
- **Keep docstrings in sync with code.** When you modify a function's signature, behavior, return value, or raised exceptions, update its docstring in the same change. Stale docstrings are worse than no docstrings.
- Naming: classes `PascalCase`, functions/variables `snake_case`, constants `UPPER_CASE`.
- New utility/function → unit test; new inference class → integration test; API change → contract test; performance claim → benchmark.

### Keep Docs in Sync
- When code changes affect documented behavior, update the relevant docs **in the same change** — not in a follow-up.
- Scope check before committing: did you touch a public API, module structure, CLI flag, config schema, or workflow? If yes, update:
  - The owning module's `CLAUDE.md` (see [Module Index](#module-index))
  - `README.md` for user-facing CLI / install / behavior changes
  - `docs/` guides or API reference if mentioned there
- New module or capability → add a `CLAUDE.md` and link it from the [Module Index](#module-index).
- Removed/renamed symbols → grep all `*.md` for the old name and fix every reference. Stale doc links are bugs.

### Configuration Files
- Detection models need [configs/det_config.yaml](configs/det_config.yaml) (class names, visualization colors).
- OCR models need [configs/plate.yaml](configs/plate.yaml) (character dict, color/layer maps).
- Defaults are built into the code; external YAML takes precedence. Field reference in [configs/CLAUDE.md](configs/CLAUDE.md).

## Anti-patterns (Do Not Do)

- ❌ Override `BaseORT.__call__` — breaks the template-method contract.
- ❌ Instantiate `YoloORT` / `RtdetrORT` directly, bypassing `create_detector` — defeats future registration logic.
- ❌ Write new PIL-based drawing code — fully migrated to Supervision; see [onnxtools/utils/CLAUDE.md](onnxtools/utils/CLAUDE.md).
- ❌ Edit completed specs under `specs/` — historical archive; new requirements go through an OpenSpec proposal.
- ❌ Treat TensorRT as a required dependency — it is an optional extra `[trt]`; remote/CI environments usually lack it.
- ❌ Load a TensorRT engine eagerly in `__init__` — lazy loading must be preserved.

## Active Technologies
- Python 3.10+ · `numpy>=2.2.6` · `opencv-contrib-python>=4.12.0` · `supervision==0.26.1`
- `onnxruntime-gpu==1.22.0` · `polygraphy>=0.49.26`
- TensorRT 8.6.1 (optional, `[trt]` extra, requires NVIDIA PyPI source)

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
