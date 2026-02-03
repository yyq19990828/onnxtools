# Implementation Plan: Remove Legacy Drawing Functions

**Branch**: `002-delete-old-draw` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/tyjt/桌面/onnx_vehicle_plate_recognition/specs/002-delete-old-draw/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✅ Feature spec loaded successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✅ Project Type: single (Python library)
   → ✅ Structure Decision: utils/ module refactoring
3. Fill the Constitution Check section based on the constitution document
   → ✅ Constitution v1.0.0 principles applied
4. Evaluate Constitution Check section below
   → ✅ No violations detected
   → ✅ Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → ✅ No NEEDS CLARIFICATION - straightforward refactoring
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → ✅ Contracts defined, quickstart created
7. Re-evaluate Constitution Check section
   → ✅ No new violations introduced
   → ✅ Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
   → ✅ Task planning strategy documented
9. STOP - Ready for /tasks command
   → ✅ Plan complete, ready for /tasks
```

**IMPORTANT**: The /plan command STOPS at step 9. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

This feature removes the deprecated PIL-based drawing implementation from `utils/drawing.py`, consolidating on the modern supervision library for all visualization needs. The change eliminates 110+ lines of legacy code, removes the `use_supervision` flag, and simplifies maintenance by having a single, well-tested rendering path. The supervision library provides better performance, more professional visualizations, and easier extensibility for future enhancements.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: supervision>=0.16.0, opencv-contrib-python, numpy, Pillow
**Storage**: N/A (visualization only, no persistence)
**Testing**: pytest>=7.0.0, pytest-benchmark (for performance validation)
**Target Platform**: Linux/Windows (cross-platform Python library)
**Project Type**: single (Python library with utils module)
**Performance Goals**: Maintain or improve current drawing performance (<10ms for 20 objects)
**Constraints**: Must maintain backward API compatibility except for removed `use_supervision` parameter
**Scale/Scope**: Single module refactoring (~250 LOC → ~140 LOC), 3 supervision helper modules remain unchanged

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ I. Modular Architecture
- Drawing module remains self-contained in `utils/drawing.py`
- Public interface through `utils/__init__.py` unchanged (except removed parameter)
- Supervision helper modules (converter, labels, config) remain independent
- Clear dependency: drawing.py → supervision helpers → supervision library

### ✅ II. Configuration-Driven Design
- No configuration changes required (supervision already uses external font paths)
- Font configuration remains externalized via function parameters
- No hardcoded constants introduced or removed

### ✅ III. Performance First
- Supervision library already benchmarked and proven faster than PIL
- `benchmark_drawing_performance()` retained to validate performance claims
- No performance degradation expected (supervision is optimized C++ backend)

### ✅ IV. Type Safety and Contract Validation
- Existing type hints in `draw_detections_supervision()` preserved
- Function signatures maintain type safety
- Input/output contracts remain identical (numpy arrays in/out)

### ✅ V. Test-Driven Development (TDD)
- Existing tests for supervision implementation remain
- Integration tests validate end-to-end visualization pipeline
- Performance benchmarks verify no regression

### ✅ VI. Observability and Debugging
- Existing logging for supervision failures preserved
- Clear error messages when supervision unavailable (now hard requirement)
- Debugging capability maintained through supervision's built-in tools

### ✅ VII. Simplicity and Incremental Growth (YAGNI)
- **PERFECT ALIGNMENT**: This change embodies YAGNI by removing unused PIL fallback
- Eliminates 110+ lines of redundant code
- Single implementation path reduces cognitive load
- Deletion of speculative "fallback" code that's never used in production

**Overall Assessment**: ✅ PASS - This refactoring actively improves constitutional compliance by simplifying the codebase.

## Project Structure

### Documentation (this feature)
```
specs/002-delete-old-draw/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
└── contracts/           # Phase 1 output (/plan command)
    └── drawing_api.md   # Function contract specification
```

### Source Code (repository root)
```
utils/
├── drawing.py                    # MODIFIED: Remove PIL implementation (lines 43-152)
├── supervision_converter.py      # UNCHANGED: Keep supervision integration
├── supervision_labels.py         # UNCHANGED: Keep label creation logic
├── supervision_config.py         # UNCHANGED: Keep annotator config
└── __init__.py                   # MODIFIED: Update exports if needed

tests/
├── test_drawing.py              # MODIFIED: Remove PIL-specific tests
└── test_drawing_supervision.py  # UNCHANGED: Keep supervision tests
```

**Structure Decision**: Single project structure using `utils/` module. This is a focused refactoring task affecting only the drawing module within the existing utilities package. No new modules or restructuring required.

## Phase 0: Outline & Research

### Research Areas

Since this is a straightforward code removal task with no technical unknowns, research focuses on validation and migration safety:

1. **Supervision Library API Stability**
   - **Decision**: Use supervision>=0.16.0 (already in requirements.txt)
   - **Rationale**: Stable API, mature library with 3.5k+ GitHub stars, actively maintained
   - **Alternatives considered**: Keep PIL fallback (rejected - adds complexity, never used)

2. **Backward Compatibility Requirements**
   - **Decision**: Maintain function signature except `use_supervision` parameter
   - **Rationale**: Callers expect same input/output format (numpy arrays, detection format)
   - **Breaking change**: Removal of `use_supervision=False` option (acceptable - unused)

3. **Performance Validation Approach**
   - **Decision**: Retain `benchmark_drawing_performance()` function
   - **Rationale**: Provides empirical evidence that supervision is faster
   - **Migration strategy**: Document performance improvement in commit message

4. **Error Handling for Missing Supervision**
   - **Decision**: Raise ImportError with actionable message on missing supervision
   - **Rationale**: Clear, fail-fast behavior better than silent degradation
   - **User guidance**: Error message includes `pip install supervision>=0.16.0`

### No NEEDS CLARIFICATION

All technical context is clear:
- PIL code to remove: lines 43-152 in drawing.py
- Supervision code to keep: draw_detections_supervision() and helpers
- API contract: Maintain except removed `use_supervision` flag
- Testing: Use existing supervision tests, remove PIL-specific tests

**Output**: research.md (see separate file)

## Phase 1: Design & Contracts

### 1. Data Model

See `data-model.md` for complete specification. Key entities:

**DetectionData**:
- Format: `List[List[List[float]]]` - nested list of [x1, y1, x2, y2, confidence, class_id]
- Source: ONNX model inference output (YOLO/RT-DETR/RF-DETR)
- Constraints: Valid bounding boxes (x2>x1, y2>y1), confidence in [0,1]

**PlateResults**:
- Format: `List[Optional[Dict[str, Any]]]` - OCR metadata per detection
- Fields: plate_text, color, layer, should_display_ocr, confidence
- Optional: May be None for non-plate detections

**AnnotatedImage**:
- Format: `np.ndarray` - BGR image with drawn boxes and labels
- Shape: Same as input image (H, W, 3)
- Dtype: uint8

### 2. API Contracts

See `contracts/drawing_api.md` for complete specification. Key contract:

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

**Contract Changes**:
- ❌ REMOVED: `use_supervision: bool = True` parameter
- ✅ PRESERVED: All other parameters and return type
- ✅ BEHAVIOR: Now always uses supervision implementation

### 3. Contract Tests

Contract tests verify:
1. Function signature matches specification
2. Input validation (raises TypeError for invalid inputs)
3. Output format (returns numpy array with correct shape/dtype)
4. Supervision integration (calls supervision annotators)

Test file: `tests/contract/test_drawing_contract.py`

### 4. Integration Test Scenarios

From spec acceptance scenarios:

**Scenario 1: Visualization functionality preserved**
```python
def test_supervision_only_visualization():
    """Given PIL code removed, When draw_detections called, Then supervision renders correctly"""
    # Setup: Load test image and mock detections
    # Action: Call draw_detections() without use_supervision flag
    # Assert: Image has bounding boxes and labels drawn
```

**Scenario 2: Backward compatibility**
```python
def test_backward_compatible_api():
    """Given existing code calling draw_detections, When use_supervision removed, Then calls still work"""
    # Setup: Prepare detection data
    # Action: Call with old parameter set (minus use_supervision)
    # Assert: No errors, correct visualization
```

**Scenario 3: Missing supervision error handling**
```python
def test_missing_supervision_error():
    """Given supervision not installed, When draw_detections called, Then clear error raised"""
    # Setup: Mock ImportError for supervision
    # Action: Import drawing module
    # Assert: ImportError with installation instructions
```

### 5. Quickstart Validation

See `quickstart.md` for manual testing procedure:

1. Run existing visualization pipeline with supervision
2. Verify performance meets <10ms threshold
3. Test font loading and fallback behavior
4. Validate Chinese character rendering
5. Benchmark against baseline (PIL implementation)

### 6. Agent Context Update

Running agent context update script:

```bash
.specify/scripts/bash/update-agent-context.sh claude
```

This will update `CLAUDE.md` with:
- Recent change: Removed legacy PIL drawing implementation
- Technical note: Supervision now required dependency
- Testing guidance: Use supervision-based tests only

**Output**: data-model.md, contracts/drawing_api.md, quickstart.md, updated CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Setup tasks: Validate environment, check supervision installation
2. Test tasks (TDD): Write/update contract tests for new signature
3. Implementation tasks: Remove PIL code, update function signature, update imports
4. Integration tasks: Update callers, remove `use_supervision` flag usage
5. Validation tasks: Run tests, benchmarks, manual quickstart verification
6. Cleanup tasks: Remove unused PIL imports, update documentation

**Ordering Strategy**:
- Phase 3.1: Setup and validation (1-2 tasks)
- Phase 3.2: Tests first - update contract tests (2-3 tasks) [P]
- Phase 3.3: Implementation - remove PIL code (3-4 tasks)
- Phase 3.4: Integration - update callers (1-2 tasks)
- Phase 3.5: Validation and cleanup (3-4 tasks)

**Estimated Output**: 12-15 numbered, ordered tasks in tasks.md

Key parallelization opportunities:
- [P] Update contract tests (different test files)
- [P] Update documentation files
- [P] Cleanup unused imports across modules

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

**Success Criteria**:
- All tests pass with supervision-only implementation
- Performance benchmark shows ≥1.0x speed (no regression)
- Manual quickstart validation confirms visualization quality
- Code coverage maintained or improved
- Documentation updated to reflect changes

## Complexity Tracking

**No constitutional violations detected.**

This refactoring actively improves code quality by:
- Reducing complexity (single implementation path)
- Eliminating dead code (PIL fallback never used)
- Improving maintainability (fewer dependencies)
- Following YAGNI principle (remove unused features)

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (no violations)
- [x] Post-Design Constitution Check: PASS (improves compliance)
- [x] All NEEDS CLARIFICATION resolved (none required)
- [x] Complexity deviations documented (none - improves simplicity)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
