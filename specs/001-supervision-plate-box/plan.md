# Implementation Plan: 使用Supervision库增强可视化功能

**Branch**: `001-supervision-plate-box` | **Date**: 2025-09-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-supervision-plate-box/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → Feature spec loaded successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project Type: Single project (Python vehicle detection system)
   → Structure Decision: Option 1 (DEFAULT)
3. Evaluate Constitution Check section below
   → Initial Constitution Check: PENDING
4. Execute Phase 0 → research.md
   → Research supervision library APIs and integration patterns
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
6. Re-evaluate Constitution Check section
   → Post-Design Constitution Check: PENDING
7. Plan Phase 2 → Describe task generation approach
8. STOP - Ready for /tasks command
```

## Summary
Replace custom visualization functions in utils/drawing.py with supervision library to provide more professional and customizable detection result visualization while maintaining OCR text display compatibility.

## Technical Context
**Language/Version**: Python 3.10+
**Primary Dependencies**: supervision, opencv-python, PIL, numpy
**Storage**: N/A (visualization only, no data persistence)
**Testing**: pytest with visual regression testing
**Target Platform**: Linux (primary), Windows/macOS compatible
**Project Type**: single - Python vehicle detection system
**Performance Goals**: <10ms visualization time for 20 detection objects
**Constraints**: Maintain OCR text display functionality, preserve existing API compatibility
**Scale/Scope**: Single utils/drawing.py module replacement, ~150 lines of code

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (main detection system only)
- Using framework directly? YES (supervision library used directly)
- Single data model? YES (detection results + OCR results)
- Avoiding patterns? YES (no unnecessary abstraction layers)

**Architecture**:
- EVERY feature as library? PARTIAL (utils module serves as library)
- Libraries listed: utils (visualization tools), infer_onnx (detection), supervision (external)
- CLI per library: YES (main.py provides CLI interface)
- Library docs: YES (CLAUDE.md format exists)

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? YES (will implement)
- Git commits show tests before implementation? YES (will follow)
- Order: Contract→Integration→E2E→Unit strictly followed? YES
- Real dependencies used? YES (actual supervision library)
- Integration tests for: contract changes, shared schemas? YES
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:
- Structured logging included? YES (existing logging_config.py)
- Frontend logs → backend? N/A (single application)
- Error context sufficient? YES (will maintain existing error handling)

**Versioning**:
- Version number assigned? NO (will add to requirements.txt)
- BUILD increments on every change? YES (through git commits)
- Breaking changes handled? YES (maintain backward compatibility)

## Project Structure

### Documentation (this feature)
```
specs/001-supervision-plate-box/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Single project structure (DEFAULT)
utils/
├── drawing.py           # TO BE MODIFIED - main target
├── pipeline.py          # INTEGRATION POINT
├── image_processing.py  # DEPENDENCY
└── __init__.py         # PUBLIC API

infer_onnx/
└── [detection engines]  # DATA SOURCE

tests/
├── contract/           # Contract tests for drawing API
├── integration/        # End-to-end visualization tests
└── unit/              # Unit tests for drawing functions
```

**Structure Decision**: Option 1 (single project) - matches existing codebase structure

## Phase 0: Outline & Research

### Research Tasks Identified:
1. **Supervision API Integration**: How to convert existing detection format to supervision.Detections
2. **OCR Text Overlay**: Best practices for custom text annotation with supervision
3. **Performance Comparison**: Benchmarking supervision vs current PIL-based approach
4. **Font Handling**: Chinese character support in supervision library
5. **Output Format Compatibility**: Maintaining cv2.imwrite and real-time display

### Research Agent Tasks:
```
Task: "Research supervision.Detections format and conversion from YOLO/RT-DETR outputs"
Task: "Find supervision text annotation patterns for OCR results overlay"
Task: "Benchmark supervision performance vs PIL-based drawing for 20+ objects"
Task: "Research Chinese font support in supervision RichLabelAnnotator"
Task: "Find supervision output format options for video and image saving"
```

**Output**: research.md with all technical decisions and integration patterns

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

### Data Model Extraction:
- DetectionResult: Convert from current tuple format to supervision.Detections
- OCRResult: Structure for plate text, color, layer information
- VisualizationConfig: Supervision annotator configuration
- OutputFormat: Image/video output handling

### API Contract Generation:
- draw_detections() function signature preservation
- Internal supervision integration layer
- Error handling and fallback mechanisms
- Performance monitoring interface

### Contract Tests:
- Input format validation (existing detection tuples)
- Output format verification (cv2 compatible images)
- OCR text positioning accuracy
- Performance benchmarks (<10ms constraint)

### Integration Scenarios:
- Replace utils.drawing.draw_detections() calls
- Maintain pipeline.py integration
- Preserve main.py CLI output options

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, updated CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load contracts from Phase 1 design
- Generate TDD-ordered tasks for supervision integration
- Each API change → contract test task [P]
- Each data model → conversion function task [P]
- Each user scenario → integration test task
- Implementation tasks to pass all tests

**Ordering Strategy**:
- Contract tests for draw_detections() API [P]
- Unit tests for format conversion utilities [P]
- Integration tests for pipeline compatibility
- Implementation: conversion utilities → annotator setup → integration
- Visual regression tests for output quality

**Estimated Output**: 15-20 numbered, ordered tasks in tasks.md

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation following TDD principles
**Phase 5**: Validation with visual regression and performance testing

## Complexity Tracking
*No constitutional violations identified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None      | N/A        | N/A                                 |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [x] Phase 4: Implementation complete
- [x] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/.specify/memory/constitution.md`*
