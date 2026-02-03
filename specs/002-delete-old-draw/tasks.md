# Tasks: Remove Legacy Drawing Functions

**Input**: Design documents from `/home/tyjt/桌面/onnx_vehicle_plate_recognition/specs/002-delete-old-draw/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/drawing_api.md, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Implementation plan loaded
   → ✅ Tech stack: Python 3.10+, supervision>=0.16.0
2. Load optional design documents:
   → ✅ research.md: Technical decisions loaded
   → ✅ data-model.md: Data entities extracted
   → ✅ contracts/drawing_api.md: API contract loaded
3. Generate tasks by category:
   → ✅ Setup: Environment validation
   → ✅ Tests: Contract tests, integration tests
   → ✅ Core: Remove PIL code, update function
   → ✅ Integration: Update callers, cleanup
   → ✅ Polish: Performance tests, documentation
4. Apply task rules:
   → ✅ Different files = mark [P] for parallel
   → ✅ Same file = sequential (no [P])
   → ✅ Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
   → ✅ 14 tasks generated
6. Generate dependency graph
   → ✅ Dependencies documented
7. Create parallel execution examples
   → ✅ Examples provided
8. Validate task completeness:
   → ✅ Contract tests cover draw_detections API
   → ✅ Integration tests validate end-to-end flow
   → ✅ Performance validation included
9. Return: SUCCESS (tasks ready for execution)
   → ✅ tasks.md ready for /implement command
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `utils/`, `tests/` at repository root
- This is a single Python library project

---

## Phase 3.1: Setup

### T001 Verify supervision library installation ✅
**Description**: Verify that supervision>=0.16.0 is installed and importable. This ensures the environment is ready for the refactoring.

**Commands**:
```bash
python -c "import supervision as sv; print(f'Supervision {sv.__version__} installed')"
pytest --version  # Verify pytest available
```

**Acceptance Criteria**:
- supervision 0.16.0+ imports without errors
- pytest 7.0.0+ available
- No ImportError exceptions

**Files Modified**: None (validation only)

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### T002 [P] Update contract test for draw_detections signature ✅
**Description**: Update the contract test in `tests/contract/test_draw_detections_contract.py` to verify the new function signature without `use_supervision` parameter.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/contract/test_draw_detections_contract.py`

**Test Cases to Add/Update**:
1. Test function signature has exactly 6 parameters (not 7)
2. Test `use_supervision` parameter is NOT in signature
3. Test all required parameters present (image, detections, class_names, colors, plate_results, font_path)
4. Test return type is np.ndarray
5. Test supervision annotators are always called (no PIL fallback)

**Expected**: Tests PASS (contract definition correct)

**Example Test**:
```python
def test_draw_detections_signature_no_use_supervision():
    """Verify use_supervision parameter removed from signature"""
    import inspect
    from utils.drawing import draw_detections

    sig = inspect.signature(draw_detections)
    params = list(sig.parameters.keys())

    assert 'use_supervision' not in params, "use_supervision should be removed"
    assert len(params) == 6, f"Expected 6 params, got {len(params)}"
    assert sig.return_annotation == np.ndarray
```

---

### T003 [P] Create integration test for supervision-only rendering ✅
**Description**: Create integration test in `tests/integration/test_supervision_only.py` to verify visualization works correctly with only supervision implementation.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/integration/test_supervision_only.py` (NEW)

**Test Scenarios**:
1. Basic detection rendering (vehicles + plates)
2. Chinese character rendering for plate OCR
3. Empty detection list handling
4. Large detection count (50+ boxes)
5. Font file missing (fallback behavior)

**Expected**: Tests will FAIL initially (file uses old PIL code), then PASS after T008

**Example Test**:
```python
def test_supervision_only_basic_rendering():
    """Test that supervision renders detections correctly"""
    image = create_test_image(640, 480)
    detections = [[[100, 100, 200, 200, 0.9, 0]]]

    result = draw_detections(image, detections, {0: "vehicle"}, [(255, 0, 0)])

    assert result.shape == image.shape
    assert not np.array_equal(result, image)  # Image modified
    # Visual verification: box should be drawn
```

---

### T004 [P] Update benchmark test to remove PIL comparison ✅
**Description**: Update `tests/contract/test_benchmark_contract.py` to remove PIL performance testing since only supervision remains.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/contract/test_benchmark_contract.py`

**Changes**:
1. Remove `pil_avg_time` from expected results
2. Update benchmark to only measure supervision performance
3. Add assertion: supervision_avg_time < 10.0ms (performance target)
4. Remove PIL-specific test cases

**Expected**: Tests PASS after updating benchmark expectations

---

## Phase 3.3: Core Implementation (ONLY after tests are failing/updated)

### T005 Remove PIL import and SUPERVISION_AVAILABLE flag
**Description**: Remove PIL imports and the `SUPERVISION_AVAILABLE` fallback flag from `utils/drawing.py` lines 5-17.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py`

**Code to Remove**:
```python
from PIL import Image, ImageDraw, ImageFont  # REMOVE
...
try:
    import supervision as sv
    ...
    SUPERVISION_AVAILABLE = True  # REMOVE
except ImportError:
    SUPERVISION_AVAILABLE = False  # REMOVE
    logging.warning("...")  # REMOVE
```

**Code to Keep**:
```python
import supervision as sv
from .supervision_converter import convert_to_supervision_detections
from .supervision_labels import create_ocr_labels
from .supervision_config import create_box_annotator, create_rich_label_annotator
```

**Changes**:
- Remove lines 5-6 (PIL imports)
- Remove lines 10-17 (try/except fallback logic)
- Keep direct `import supervision as sv` (line 10)

**Acceptance Criteria**:
- No PIL imports remain
- `SUPERVISION_AVAILABLE` flag removed
- supervision import is unconditional
- ImportError raised if supervision missing (Python default behavior)

---

### T006 Remove PIL implementation from draw_detections
**Description**: Remove the PIL-based drawing implementation (lines 43-152) from `draw_detections()` function in `utils/drawing.py`.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py`

**Lines to Delete**: 43-152 (approximately 110 lines)

**Code Block to Remove**:
- Line 43-42: PIL implementation fallback logic
- Lines 43-152: Entire PIL drawing code (Image.fromarray, ImageDraw, font loading, box drawing, text rendering)

**What Remains**:
- Lines 1-42: Imports and function signature
- Lines 36-42: Supervision implementation call (currently in try/except)
- Lines 155-209: `draw_detections_supervision()` function (KEEP)
- Lines 211-253: `benchmark_drawing_performance()` function (KEEP)

**After Deletion**:
- Function body should be ~5 lines (just call supervision implementation)
- Remove try/except wrapper around supervision call
- Direct delegation to `draw_detections_supervision()`

**Acceptance Criteria**:
- PIL code completely removed
- Function reduced from ~150 lines to ~5 lines
- No references to PIL objects (Image, ImageDraw, ImageFont)

---

### T007 Remove use_supervision parameter from draw_detections
**Description**: Remove the `use_supervision` parameter from the `draw_detections()` function signature and update the function to always use supervision.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py`

**Current Signature (line 19)**:
```python
def draw_detections(image, detections, class_names, colors, plate_results=None,
                   font_path="SourceHanSans-VF.ttf", use_supervision=True):
```

**New Signature**:
```python
def draw_detections(image, detections, class_names, colors, plate_results=None,
                   font_path="SourceHanSans-VF.ttf"):
```

**Changes**:
1. Remove `use_supervision=True` parameter
2. Remove conditional logic checking `use_supervision` value (lines 36-42)
3. Update function to directly call `draw_detections_supervision()`
4. Update docstring to reflect supervision-only behavior

**New Function Body**:
```python
def draw_detections(image, detections, class_names, colors, plate_results=None,
                   font_path="SourceHanSans-VF.ttf"):
    """
    Draws detection boxes on the image using supervision library.

    Args:
        image: Input image as numpy array (BGR)
        detections: Detection results
        class_names: Class name mapping
        colors: Colors for different classes
        plate_results: Optional OCR results for plates
        font_path: Path to font file

    Returns:
        Annotated image as numpy array (BGR)
    """
    return draw_detections_supervision(
        image, detections, class_names, colors, plate_results, font_path
    )
```

**Acceptance Criteria**:
- Parameter removed from signature
- Function body is simple delegation (4-5 lines)
- Docstring updated
- No conditional logic remains

---

### T008 Update draw_detections_supervision to handle ImportError
**Description**: Update `draw_detections_supervision()` docstring and add clear error message if supervision not available (already handled by Python import, but improve error message).

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py`

**Changes**:
1. Update function docstring to indicate it's now the primary (only) implementation
2. Remove `if not SUPERVISION_AVAILABLE` check (lines 176-177)
3. Ensure clear error propagates from module import failure

**Module-level Import Error Handling** (add at top after imports):
```python
try:
    import supervision as sv
except ImportError as e:
    raise ImportError(
        "supervision library is required for drawing functionality. "
        "Install it with: pip install supervision>=0.16.0"
    ) from e
```

**Acceptance Criteria**:
- Clear ImportError message if supervision missing
- No runtime fallback logic
- Fail-fast at import time

---

### T009 Update benchmark_drawing_performance function
**Description**: Update `benchmark_drawing_performance()` function to remove PIL comparison and only benchmark supervision implementation.

**File**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py`

**Current Function** (lines 211-253):
- Benchmarks both PIL and supervision
- Returns dict with `pil_avg_time`, `supervision_avg_time`, `improvement_ratio`

**Updated Function**:
- Only benchmark supervision (no PIL)
- Return simpler dict: `{'avg_time_ms': float, 'target_met': bool}`
- Remove `use_supervision=False` call (line 235)

**New Signature**:
```python
def benchmark_drawing_performance(
    image: np.ndarray,
    detections_data: List[List[List[float]]],
    iterations: int = 100,
    target_ms: float = 10.0
) -> Dict[str, float]:
    """
    Benchmark drawing performance with supervision implementation.

    Args:
        image: Test image
        detections_data: Detection data for testing
        iterations: Number of test iterations
        target_ms: Performance target in milliseconds

    Returns:
        {'avg_time_ms': float, 'target_met': bool}
    """
```

**Implementation**:
```python
start_time = time.time()
for _ in range(iterations):
    _ = draw_detections(image.copy(), detections_data, class_names, colors)
avg_time = (time.time() - start_time) / iterations * 1000  # ms

return {
    'avg_time_ms': avg_time,
    'target_met': avg_time < target_ms
}
```

**Acceptance Criteria**:
- No PIL benchmarking
- Simpler return format
- Performance target configurable

---

## Phase 3.4: Integration

### T010 Search and update callers of draw_detections
**Description**: Search the codebase for all calls to `draw_detections()` and remove any explicit `use_supervision=False` arguments (if any exist).

**Search Command**:
```bash
grep -r "draw_detections" --include="*.py" utils/ tests/ main.py
grep -r "use_supervision" --include="*.py" .
```

**Expected Findings**:
- `utils/pipeline.py` - Likely calls `draw_detections()`
- `main.py` - May call drawing function
- Test files - Update test calls

**Files to Check**:
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/pipeline.py`
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/main.py`
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/integration/test_*.py`

**Changes**:
- Remove `use_supervision=True` or `use_supervision=False` from function calls
- Verify all calls work with new signature

**Acceptance Criteria**:
- No `use_supervision` parameter passed anywhere
- All callers updated
- Code still runs without errors

---

### T011 Remove PIL-specific tests from test files
**Description**: Remove test functions that specifically test PIL implementation or `use_supervision=False` behavior.

**Files to Update**:
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/integration/test_fallback_mechanism.py` (likely entire file)
- Other test files with PIL-specific tests

**Test Functions to Remove** (examples):
- `test_draw_detections_pil_fallback()`
- `test_use_supervision_flag_false()`
- `test_pil_font_loading()`
- `test_supervision_unavailable_fallback()`

**Acceptance Criteria**:
- All PIL-specific tests removed
- Test suite still passes
- No references to PIL in test assertions

---

## Phase 3.5: Polish

### T012 [P] Run full test suite and verify all tests pass
**Description**: Execute the complete test suite to ensure no regressions were introduced by the refactoring.

**Commands**:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=utils.drawing --cov-report=term-missing

# Run contract tests specifically
pytest tests/contract/ -v
```

**Acceptance Criteria**:
- All tests pass (0 failures)
- Code coverage for `utils/drawing.py` ≥ 80%
- No deprecation warnings
- No ImportError or runtime errors

**Expected Results**:
- Contract tests validate new signature
- Integration tests confirm visualization works
- Performance tests meet <10ms target

---

### T013 [P] Execute quickstart validation procedure
**Description**: Run the manual validation steps from `quickstart.md` to verify the refactoring works end-to-end.

**Reference**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/specs/002-delete-old-draw/quickstart.md`

**Steps to Execute**:
1. Step 1: Environment Setup - Verify supervision installed
2. Step 2: Verify Drawing Module Import - Check function signature
3. Step 3: Basic Visualization Test - Run test script
4. Step 4: Chinese Character Rendering Test - Verify OCR display
5. Step 5: End-to-End Pipeline Test - Run main.py
6. Step 6: Performance Benchmark - Verify <10ms target
7. Step 7: Automated Test Suite - Run pytest

**Acceptance Criteria**:
- All 7 quickstart steps pass
- Visual outputs look correct
- Performance meets target
- No errors in console output

**Deliverable**: Validation report confirming success

---

### T014 [P] Update documentation and docstrings
**Description**: Update all documentation to reflect the supervision-only implementation.

**Files to Update**:
1. `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/drawing.py` - Update module docstring
2. `/home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/CLAUDE.md` - Update drawing module notes (already done by update script)
3. `/home/tyjt/桌面/onnx_vehicle_plate_recognition/README.md` - Update dependencies section if needed

**Documentation Updates**:
- Module docstring: Mention supervision as sole dependency
- Function docstrings: Remove references to PIL fallback
- README: Ensure supervision listed in requirements
- CLAUDE.md: Note recent refactoring (already updated)

**Acceptance Criteria**:
- No documentation references to PIL
- supervision library clearly documented as requirement
- Docstrings accurate and complete

---

## Dependencies

### Dependency Graph
```
T001 (Setup)
  └─> T002, T003, T004 (Tests) [P]
       └─> T005 (Remove imports)
            └─> T006 (Remove PIL code)
                 └─> T007 (Remove parameter)
                      └─> T008 (Error handling)
                           └─> T009 (Update benchmark)
                                └─> T010 (Update callers)
                                     └─> T011 (Remove PIL tests)
                                          └─> T012, T013, T014 (Validation) [P]
```

### Sequential Constraints
- **T001** must complete before T002-T004 (need environment)
- **T002-T004** can run in parallel [P] (different files)
- **T005-T009** must be sequential (same file: utils/drawing.py)
- **T010** depends on T007 (parameter removal must be complete)
- **T011** depends on T006 (PIL code must be gone)
- **T012-T014** can run in parallel [P] (different activities)

---

## Parallel Execution Examples

### Parallel Group 1: Tests (after T001)
```bash
# Launch T002, T003, T004 together:
pytest tests/contract/test_draw_detections_contract.py::test_draw_detections_signature_no_use_supervision
pytest tests/integration/test_supervision_only.py
pytest tests/contract/test_benchmark_contract.py
```

### Parallel Group 2: Validation (after T011)
```bash
# Terminal 1: Run automated tests (T012)
pytest tests/ -v --cov=utils.drawing

# Terminal 2: Run quickstart validation (T013)
bash specs/002-delete-old-draw/quickstart.md

# Terminal 3: Update docs (T014)
# Edit documentation files
```

---

## Notes

### Critical Reminders
- [P] tasks = different files, no dependencies
- Verify tests FAIL before implementing (TDD)
- Commit after each task for easy rollback
- Run tests frequently during implementation

### Common Pitfalls to Avoid
- ❌ Removing supervision helper modules (keep them!)
- ❌ Breaking API compatibility (preserve all params except `use_supervision`)
- ❌ Forgetting to update test expectations
- ❌ Leaving PIL imports in other files

### Performance Validation
- Target: <10ms for 20 detections (1920x1080 image)
- Current baseline: ~7ms with supervision
- Run benchmark after each major change

---

## Validation Checklist
*GATE: All items must be checked before marking feature complete*

### Contract Coverage
- [x] draw_detections API contract tested
- [x] Function signature validates correctly
- [x] Return type verified (np.ndarray)
- [x] Supervision integration tested

### Implementation Complete
- [ ] PIL imports removed (T005)
- [ ] PIL implementation deleted (T006)
- [ ] `use_supervision` parameter removed (T007)
- [ ] Error handling updated (T008)
- [ ] Benchmark function updated (T009)
- [ ] All callers updated (T010)
- [ ] PIL tests removed (T011)

### Quality Gates
- [ ] All automated tests pass (T012)
- [ ] Quickstart validation complete (T013)
- [ ] Documentation updated (T014)
- [ ] Performance target met (<10ms)
- [ ] Code coverage ≥80%
- [ ] No TODO or FIXME comments remain

---

*Tasks generated: 2025-09-30 | Total: 14 tasks | Estimated time: 2-3 hours*
