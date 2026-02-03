<!--
Sync Impact Report:
- Version change: (new constitution) → 1.0.0
- Initial version ratified with 7 core principles for Python ONNX inference library
- Modified sections: N/A (initial creation)
- Added sections: All core principles, performance standards, development workflow
- Removed sections: N/A
- Templates requiring updates:
  ✅ plan-template.md: Constitution Check section already generic
  ✅ spec-template.md: No constitution-specific references found
  ✅ tasks-template.md: Compatible with principle-driven task types
- Follow-up TODOs: None - all placeholders resolved
-->

# ONNX Vehicle Plate Recognition Constitution

## Core Principles

### I. Modular Architecture
All inference engines, utilities, and tools MUST be self-contained modules with clear boundaries. Each module SHALL:
- Inherit from a common base class (e.g., `BaseOnnx`) for consistency
- Expose a well-defined public interface through `__init__.py`
- Be independently testable without requiring the full system
- Document its dependencies and API contracts

**Rationale**: Enables independent development, testing, and optimization of detection models (YOLO, RT-DETR, RF-DETR) without affecting other components.

### II. Configuration-Driven Design
Model configurations, parameters, and deployment settings MUST be externalized in YAML files. Code SHALL NOT contain hardcoded paths, thresholds, or model-specific constants.

**Rationale**: Supports rapid model switching, A/B testing, and deployment across different environments without code changes.

### III. Performance First
All inference operations MUST prioritize throughput and latency. Performance requirements:
- Batch processing over frame-by-frame where applicable
- TensorRT optimization support for production deployments
- Profiling hooks for identifying bottlenecks
- Memory-efficient tensor operations

**Rationale**: Real-time vehicle detection requires <50ms p95 latency for video streams; inefficient code blocks production deployment.

### IV. Type Safety and Contract Validation
All public APIs MUST use Python type hints (PEP 484). Input/output contracts SHALL be validated at module boundaries using:
- Type checking with mypy or similar tools
- Runtime validation for critical paths (model inputs, detection outputs)
- Explicit error messages for contract violations

**Rationale**: ONNX tensor shapes and dtypes are error-prone; explicit contracts catch issues early and improve maintainability.

### V. Test-Driven Development (TDD)
New inference engines and utilities MUST follow red-green-refactor:
1. Write failing tests for the interface contract
2. Obtain approval from project maintainers or users
3. Implement to pass tests
4. Refactor for performance and clarity

**Integration testing required for**:
- New model architecture support (e.g., adding YOLO v12)
- Changes to inference pipeline contracts
- TensorRT engine building and optimization
- Cross-model consistency (detection + OCR + classification)

**Rationale**: Model inference logic is complex and fragile; TDD ensures correctness and prevents regressions during optimization.

### VI. Observability and Debugging
All modules MUST provide structured logging and debugging capabilities:
- Use Python `logging` module with configurable levels
- Log critical operations: model loading, inference timing, error conditions
- Support for debugging tools: Polygraphy inspection, TensorRT profiling
- Clear error messages with actionable guidance

**Rationale**: Debugging ONNX/TensorRT issues requires visibility into model behavior; opaque failures waste hours of investigation.

### VII. Simplicity and Incremental Growth (YAGNI)
Start with the simplest implementation that satisfies requirements:
- No premature optimization (profile first, optimize proven bottlenecks)
- No speculative features (implement when needed, not "might need later")
- Prefer composition over inheritance beyond the base inference class
- Delete unused code and dependencies aggressively

**Rationale**: Computer vision projects accumulate technical debt quickly; strict simplicity discipline keeps the codebase maintainable.

## Performance Standards

### Inference Latency
- **Image (single frame)**: <100ms p95 on GPU, <500ms on CPU
- **Video (30fps)**: Real-time processing with frame skip support
- **Batch inference**: Linear scaling up to batch size 8

### Memory Constraints
- **Model loading**: <2GB GPU memory per model
- **Inference pipeline**: <500MB overhead beyond model weights
- **Video streaming**: Configurable buffer sizes for memory-constrained systems

### Accuracy Baselines
- **Vehicle detection**: mAP@0.5 ≥ 0.90 on validation set
- **Plate localization**: Precision ≥ 0.95, Recall ≥ 0.92
- **OCR accuracy**: Character error rate (CER) ≤ 0.02

## Development Workflow

### Code Review Requirements
All changes MUST:
- Pass automated tests (unit, integration, contract)
- Include type hints for new functions/methods
- Update relevant documentation (docstrings, CLAUDE.md)
- Demonstrate performance impact for inference path changes

### Model Integration Process
Adding new detection architectures (e.g., RT-DETR v3) requires:
1. Subclass `BaseOnnx` with standard interface
2. Create YAML configuration file in `models/`
3. Add contract tests verifying input/output shapes
4. Benchmark against existing models using `tools/eval.py`
5. Update documentation with performance characteristics

### TensorRT Optimization
Engine building MUST:
- Use `tools/build_engine.py` with documented parameters
- Validate accuracy matches ONNX model (threshold: <0.01 mAP difference)
- Profile with Polygraphy before production deployment
- Document precision mode (FP32/FP16/INT8) and calibration data

## Governance

This constitution supersedes all other development practices. Violations MUST be justified in the "Complexity Tracking" section of implementation plans with:
- Specific technical reason requiring deviation
- Simpler alternatives evaluated and rejected with rationale
- Migration path to constitutional compliance if temporary

**Amendment Procedure**:
- Propose changes via pull request to `.specify/memory/constitution.md`
- Require approval from 2+ project maintainers
- Document version bump rationale (MAJOR/MINOR/PATCH)
- Update all dependent templates and documentation

**Compliance Reviews**:
- Constitution gates in `plan.md` checked at Phase 0 and Phase 1
- Task generation in `tasks.md` reflects principle-driven categories
- Runtime guidance maintained in `CLAUDE.md` for AI-assisted development

**Version**: 1.0.0 | **Ratified**: 2025-09-30 | **Last Amended**: 2025-09-30
