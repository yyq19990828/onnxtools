# Tasks: 使用Supervision库增强可视化功能

**Input**: Design documents from `/specs/001-supervision-plate-box/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.10+, supervision, opencv-python, PIL, numpy
   → Structure: Single project (utils/ module modification)
   → Target file: utils/drawing.py
2. Load design documents:
   → data-model.md: 5 entities → 5 model tasks
   → contracts/drawing_api.yaml: 3 endpoints → 3 contract tests
   → quickstart.md: 4 scenarios → 4 integration tests
3. Generate tasks by category:
   → Setup: dependencies, supervision library, testing framework
   → Tests: contract tests for 3 API endpoints, integration tests
   → Core: format conversion, annotation config, drawing functions
   → Integration: pipeline compatibility, fallback mechanism
   → Polish: performance tests, visual regression, documentation
4. Apply task rules:
   → Different files = mark [P] for parallel execution
   → Same file (utils/drawing.py) = sequential
   → Tests before implementation (TDD)
5. Total tasks: 23 numbered sequentially (T001-T023)
6. Dependencies: Setup → Tests → Implementation → Integration → Polish
7. Parallel execution: Test tasks and independent utility functions
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: Repository root structure
- **Target files**: `utils/drawing.py` (main), `tests/` (new), `requirements.txt`

---

## Phase 3.1: Setup
- [ ] T001 Install supervision library and update requirements.txt
- [ ] T002 [P] Create test directory structure tests/{contract,integration,unit}
- [ ] T003 [P] Setup pytest configuration and visual regression testing tools

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (API Validation)
- [ ] T004 [P] Contract test draw_detections API in tests/contract/test_draw_detections_contract.py
- [ ] T005 [P] Contract test convert_to_supervision_detections API in tests/contract/test_convert_detections_contract.py
- [ ] T006 [P] Contract test performance benchmark API in tests/contract/test_benchmark_contract.py

### Integration Tests (End-to-End Scenarios)
- [ ] T007 [P] Integration test basic detection drawing in tests/integration/test_basic_drawing.py
- [ ] T008 [P] Integration test OCR text overlay in tests/integration/test_ocr_integration.py
- [ ] T009 [P] Integration test fallback mechanism in tests/integration/test_fallback_mechanism.py
- [ ] T010 [P] Integration test pipeline compatibility in tests/integration/test_pipeline_integration.py

### Data Model Validation Tests
- [ ] T011 [P] Test PlateOCRResult validation in tests/unit/test_ocr_models.py
- [ ] T012 [P] Test VisualizationConfig models in tests/unit/test_config_models.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Format Conversion Functions
- [ ] T013 [P] Implement convert_to_supervision_detections() in utils/supervision_converter.py
- [ ] T014 [P] Implement create_ocr_labels() in utils/supervision_labels.py
- [ ] T015 [P] Implement font detection utilities in utils/font_utils.py

### Annotation Configuration
- [ ] T016 [P] Implement BoxAnnotatorConfig class in utils/supervision_config.py
- [ ] T017 [P] Implement RichLabelAnnotatorConfig class in utils/supervision_config.py

### Core Drawing Functions
- [ ] T018 Implement draw_detections_supervision() in utils/drawing.py
- [ ] T019 Update main draw_detections() with supervision integration in utils/drawing.py
- [ ] T020 Implement performance benchmark function in utils/drawing.py

## Phase 3.4: Integration
- [ ] T021 Add supervision fallback mechanism to utils/drawing.py
- [ ] T022 Update utils/__init__.py to export new functions
- [ ] T023 Verify pipeline.py compatibility and update if needed

## Phase 3.5: Polish
- [ ] T024 [P] Performance tests (<10ms for 20 objects) in tests/performance/test_drawing_performance.py
- [ ] T025 [P] Visual regression tests in tests/visual/test_output_quality.py
- [ ] T026 [P] Update documentation in docs/ and utils/CLAUDE.md
- [ ] T027 Code cleanup and optimization review

---

## Dependencies
**Critical Ordering:**
1. **Setup (T001-T003)** before everything
2. **Tests (T004-T012)** before implementation (T013-T023)
3. **Core functions (T013-T017)** before integration (T018-T020)
4. **Integration (T021-T023)** before polish (T024-T027)

**Specific Dependencies:**
- T013 (converter) blocks T018 (main drawing function)
- T014 (labels) blocks T018 (main drawing function)
- T015 (fonts) blocks T017 (config) and T018 (drawing)
- T016-T017 (config) blocks T018 (drawing)
- T018-T020 (core) blocks T021 (fallback)
- T021 (fallback) blocks T022 (exports)

## Parallel Execution Examples

### Phase 3.2: All Test Tasks (Run Together)
```bash
# Contract tests (different files, no dependencies)
Task: "Contract test draw_detections API in tests/contract/test_draw_detections_contract.py"
Task: "Contract test convert_to_supervision_detections API in tests/contract/test_convert_detections_contract.py"
Task: "Contract test performance benchmark API in tests/contract/test_benchmark_contract.py"

# Integration tests (different files, no dependencies)
Task: "Integration test basic detection drawing in tests/integration/test_basic_drawing.py"
Task: "Integration test OCR text overlay in tests/integration/test_ocr_integration.py"
Task: "Integration test fallback mechanism in tests/integration/test_fallback_mechanism.py"
Task: "Integration test pipeline compatibility in tests/integration/test_pipeline_integration.py"

# Unit tests (different files, no dependencies)
Task: "Test PlateOCRResult validation in tests/unit/test_ocr_models.py"
Task: "Test VisualizationConfig models in tests/unit/test_config_models.py"
```

### Phase 3.3: Core Utility Functions (Run Together)
```bash
# Utility functions (different files, no dependencies)
Task: "Implement convert_to_supervision_detections() in utils/supervision_converter.py"
Task: "Implement create_ocr_labels() in utils/supervision_labels.py"
Task: "Implement font detection utilities in utils/font_utils.py"
Task: "Implement BoxAnnotatorConfig class in utils/supervision_config.py"
Task: "Implement RichLabelAnnotatorConfig class in utils/supervision_config.py"
```

### Phase 3.5: Polish Tasks (Run Together)
```bash
# Polish tasks (different areas, no dependencies)
Task: "Performance tests (<10ms for 20 objects) in tests/performance/test_drawing_performance.py"
Task: "Visual regression tests in tests/visual/test_output_quality.py"
Task: "Update documentation in docs/ and utils/CLAUDE.md"
```

---

## Task Details

### File Modification Summary
**New Files** (can be parallel):
- `utils/supervision_converter.py` - Format conversion utilities
- `utils/supervision_labels.py` - Label generation functions
- `utils/supervision_config.py` - Configuration classes
- `utils/font_utils.py` - Font detection utilities
- `tests/contract/` - All contract test files
- `tests/integration/` - All integration test files
- `tests/unit/` - All unit test files
- `tests/performance/` - Performance test files
- `tests/visual/` - Visual regression tests

**Modified Files** (must be sequential):
- `utils/drawing.py` - Main target file (T018, T019, T020, T021)
- `utils/__init__.py` - Export updates (T022)
- `utils/pipeline.py` - Compatibility check (T023)
- `requirements.txt` - Dependency update (T001)

### Key Technical Requirements
1. **API Compatibility**: Maintain exact function signature of draw_detections()
2. **Performance Target**: <10ms drawing time for 20 detection objects
3. **Fallback Mechanism**: PIL backend if supervision fails
4. **Chinese Font Support**: SourceHanSans-VF.ttf with RichLabelAnnotator
5. **Output Format**: BGR numpy array compatible with cv2

### Test Requirements
1. **Contract Tests**: Validate API schemas match implementation
2. **Integration Tests**: End-to-end scenarios from quickstart.md
3. **Performance Tests**: Benchmark supervision vs PIL performance
4. **Visual Tests**: Compare output quality and correctness

---

## Validation Checklist
*GATE: Checked before task execution*

- [x] All contracts (3) have corresponding contract tests (T004-T006)
- [x] All entities (5) have model/validation tasks (T011-T017)
- [x] All tests (T004-T012) come before implementation (T013-T027)
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Critical ordering: Setup → Tests → Core → Integration → Polish

**Ready for execution**: ✅ All 27 tasks generated with proper dependencies and parallel execution plan.
