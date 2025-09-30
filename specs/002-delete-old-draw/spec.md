# Feature Specification: Remove Legacy Drawing Functions

**Feature Branch**: `002-delete-old-draw`
**Created**: 2025-09-30
**Status**: Draft
**Input**: User description: "delete old draw detection func, only preserve supervison lib funcs"

## Execution Flow (main)
```
1. Parse user description from Input
   → ✅ Feature description parsed
2. Extract key concepts from description
   → ✅ Identify: legacy code removal, supervision library retention
3. For each unclear aspect:
   → No clarifications needed - straightforward cleanup task
4. Fill User Scenarios & Testing section
   → ✅ Scenarios: code refactoring, backward compatibility
5. Generate Functional Requirements
   → ✅ Requirements defined for code removal
6. Identify Key Entities (if data involved)
   → Entities: drawing functions, supervision library integration
7. Run Review Checklist
   → ✅ No implementation details, focused on user value
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer maintaining the ONNX vehicle plate recognition codebase, I need to remove the deprecated PIL-based drawing implementation so that the codebase only uses the modern supervision library for visualization, reducing code duplication and maintenance burden.

### Acceptance Scenarios
1. **Given** the current codebase contains both PIL-based and supervision-based drawing implementations, **When** the legacy code is removed, **Then** all visualization functionality continues to work using only supervision library functions
2. **Given** existing code that calls drawing functions, **When** legacy functions are removed, **Then** the system gracefully uses supervision implementations without breaking existing functionality
3. **Given** the supervision library is not available in an environment, **When** drawing functions are called, **Then** the system provides clear error messages indicating the missing dependency

### Edge Cases
- What happens when the supervision library import fails? System should raise ImportError with clear installation instructions
- How does the system handle existing code that explicitly disabled supervision (`use_supervision=False`)? The parameter should be deprecated and ignored
- What if font files are missing? Supervision should handle this gracefully with fallback fonts

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST remove the PIL-based implementation from `draw_detections()` function (lines 43-152 in drawing.py)
- **FR-002**: System MUST retain only the supervision-based drawing implementation (`draw_detections_supervision()`)
- **FR-003**: System MUST remove the `use_supervision` parameter since only one implementation remains
- **FR-004**: System MUST remove the `SUPERVISION_AVAILABLE` fallback logic since supervision becomes a required dependency
- **FR-005**: System MUST update function documentation to reflect that supervision is the only supported method
- **FR-006**: System MUST preserve the benchmark function (`benchmark_drawing_performance()`) for performance validation
- **FR-007**: System MUST maintain backward compatibility for all existing function signatures (except removed parameters)
- **FR-008**: System MUST ensure all supervision helper modules remain intact (supervision_converter, supervision_labels, supervision_config)

### Key Entities *(include if feature involves data)*
- **Drawing Functions**: The main visualization interface - currently has dual implementation (PIL/supervision), will be simplified to supervision-only
- **Supervision Integration**: The modern visualization framework - includes converter, label creator, and annotator configuration modules
- **Detection Data**: The input format for drawing functions - remains unchanged to maintain compatibility

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---