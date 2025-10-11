# Feature Specification: OCR Metrics Evaluation Functions

**Feature Branch**: `006-make-ocr-metrics`
**Created**: 2025-10-10
**Status**: Draft
**Input**: User description: "make ocr metrics eval funcs, just like @infer_onnx/eval_coco.py, fetch github repo yyq19990828/PaddleOCR to see tools/calculate_plate_accuracy.py, modify it to this codebase."

## Clarifications

### Session 2025-10-10

- Q: OCR评估结果应该使用什么样的表格格式来显示指标? → A: 提供两种模式：默认表格对齐格式用于console，可选JSON格式用于导出和比较
- Q: OCR评估表格应该显示哪些列? → A: 分层显示：第一行显示总体指标(完全准确率、编辑距离相似度)，第二行显示详细统计(总样本、评估数、过滤数、跳过数)
- Q: OCR评估的中文列名应该使用什么宽度标准? → A: 第一列（指标名）使用宽度18（支持"完全准确率"等6个中文字），数值列使用宽度12（支持小数和百分比）
- Q: 在表格输出中应该显示哪种编辑距离指标? → A: 同时显示两者（提供完整信息，虽然增加列数）
- Q: 进度日志应该显示什么级别的详细信息? → A: 仅显示进度百分比：`"处理进度: 50/1000 (5.0%)"`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Basic OCR Accuracy Evaluation (Priority: P1)

A researcher or engineer has trained a new OCR model and needs to evaluate its accuracy on a labeled dataset of license plate images. They want to quickly assess how well the model performs by running a single evaluation command that produces standard accuracy metrics.

**Why this priority**: This is the core functionality - without basic accuracy evaluation, users cannot assess model performance at all. This represents the Minimum Viable Product (MVP).

**Independent Test**: Can be fully tested by providing a small test dataset (10-20 labeled images) and OCR model, running the evaluation function, and verifying that it outputs complete accuracy percentage. Delivers immediate value by answering "Is my model working?"

**Acceptance Scenarios**:

1. **Given** a dataset with 100 labeled license plate images and a trained OCR model, **When** user runs the evaluation function with default parameters, **Then** system outputs complete accuracy percentage (e.g., "Complete Accuracy: 92.5%")
2. **Given** a dataset where all predictions exactly match ground truth labels, **When** evaluation runs, **Then** complete accuracy is 100%
3. **Given** a dataset where no predictions match ground truth, **When** evaluation runs, **Then** complete accuracy is 0%

---

### User Story 2 - Detailed Character-Level Analysis (Priority: P2)

An engineer debugging OCR errors needs to understand which characters are frequently misrecognized. They want to see normalized edit distance metrics and character-level error analysis to identify patterns in recognition failures.

**Why this priority**: After knowing basic accuracy (P1), users need diagnostic information to improve their models. This helps answer "What is wrong with my model?"

**Independent Test**: Can be tested with a curated error dataset (e.g., 20 images with known character substitution errors), running evaluation with edit distance enabled, and verifying output includes normalized edit distance and similarity scores. Delivers value by identifying improvement opportunities.

**Acceptance Scenarios**:

1. **Given** predictions with partial character matches (e.g., "京A12345" vs "京A12346"), **When** evaluation calculates edit distance, **Then** normalized edit distance is proportional to character differences (e.g., 0.14 for 1/7 character error)
2. **Given** predictions with varying lengths (e.g., "京A123" vs "京A12345"), **When** edit distance calculation runs, **Then** normalization uses the longer string length
3. **Given** evaluation completes, **When** results are displayed, **Then** output includes both complete accuracy and average edit distance similarity

---

### User Story 3 - Confidence Threshold Filtering (Priority: P3)

A quality control engineer wants to evaluate model performance at different confidence levels to determine optimal threshold settings for production deployment. They need to filter predictions by confidence score and see how accuracy varies across thresholds.

**Why this priority**: Production optimization feature - helps users find the right balance between coverage and accuracy. Less critical than understanding base performance.

**Independent Test**: Can be tested by running evaluation multiple times with different confidence thresholds (e.g., 0.5, 0.7, 0.9) on the same dataset and verifying that lower thresholds include more predictions but may have lower accuracy. Delivers value for production tuning.

**Acceptance Scenarios**:

1. **Given** predictions with confidence scores ranging from 0.3 to 0.95, **When** user sets confidence threshold to 0.7, **Then** only predictions with confidence >= 0.7 are evaluated
2. **Given** 100 predictions where 20 have confidence < 0.5, **When** evaluation runs with threshold 0.5, **Then** those 20 predictions are excluded from accuracy calculation
3. **Given** evaluation with confidence filtering enabled, **When** results are displayed, **Then** output shows number of predictions filtered and remaining sample size

---

### User Story 4 - Dataset-Level Performance Reporting (Priority: P4)

A project manager needs comprehensive evaluation reports comparing multiple model versions across test datasets. They want detailed metrics exported in standard formats (JSON, CSV) for documentation and presentation purposes.

**Why this priority**: Nice-to-have feature for documentation and reporting. Basic console output (P1) is sufficient for development; structured export helps with communication and tracking.

**Independent Test**: Can be tested by running evaluation and verifying that results can be exported to files with structured format including all metrics (accuracy, edit distance, sample counts). Delivers value for team communication and tracking.

**Acceptance Scenarios**:

1. **Given** evaluation completes successfully, **When** user specifies output format as JSON, **Then** metrics are exported to structured JSON file with complete accuracy, average edit distance, and sample statistics
2. **Given** multiple model evaluations need comparison, **When** user runs evaluations sequentially, **Then** each result can be saved with unique identifiers for later comparison
3. **Given** evaluation dataset contains metadata (image names, categories), **When** detailed reporting is enabled, **Then** per-category accuracy statistics are included in output

---

### Edge Cases

- What happens when ground truth labels are missing or empty for some images?
- How does system handle predictions that are empty strings or whitespace-only?
- What if predicted text length differs significantly from ground truth (e.g., 3 chars vs 8 chars)?
- How are special characters or non-standard plate formats handled in accuracy calculation?
- What happens when the dataset path is invalid or contains no valid image files?
- How does system handle Unicode characters or multi-language plates?
- What if confidence scores are missing or invalid (NaN, negative, >1.0)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST calculate complete accuracy as the percentage of predictions that exactly match ground truth labels (character-by-character equality)
- **FR-002**: System MUST calculate normalized edit distance between predicted and ground truth text using Levenshtein distance algorithm, normalized by the length of the longer string
- **FR-003**: System MUST calculate edit distance similarity as (1 - normalized edit distance) to provide an intuitive 0-1 similarity score
- **FR-004**: System MUST accept a dataset with a label list file (e.g., train.txt, val.txt) where each line contains tab-separated image path and ground truth text
- **FR-005**: System MUST load OCR predictions from the model using the existing OCRONNX class interface
- **FR-006**: System MUST filter predictions based on configurable confidence threshold (default: 0.5)
- **FR-007**: System MUST handle empty predictions gracefully by treating them as zero-length strings in edit distance calculations
- **FR-008**: System MUST handle missing ground truth labels by skipping those samples with a warning logged
- **FR-009**: System MUST report the total number of samples evaluated, number filtered by confidence, and number skipped due to errors
- **FR-010**: System MUST display evaluation results to console in a two-row table format: first row showing overall metrics (完全准确率, 归一化编辑距离, 编辑距离相似度) and second row showing detailed statistics (总样本数, 评估数, 过滤数, 跳过数), with column widths of 18 for metric names (supporting 6 Chinese characters) and 12 for numeric values
- **FR-011**: System MUST follow the same architecture pattern as DatasetEvaluator in eval_coco.py, creating an OCRDatasetEvaluator class
- **FR-012**: System MUST support optional max_images parameter to limit evaluation size for quick testing
- **FR-013**: System MUST support two output modes: default table-aligned format for console display (with proper Chinese character alignment) and optional JSON format for export and comparison
- **FR-014**: System MUST log evaluation progress at regular intervals (e.g., every 50 images) showing only progress percentage in format: "处理进度: {current}/{total} ({percentage}%)"

### Key Entities

- **OCR Evaluation Result**: Contains complete accuracy (float 0-1), average normalized edit distance (float 0-1), average edit distance similarity (float 0-1), total samples (int), filtered samples (int), skipped samples (int)
- **Ground Truth Label**: Text string paired with image path in a tab-separated label list file (format: `<image_path>\t<ground_truth_text>`)
- **OCR Prediction**: Tuple of (predicted_text: str, confidence: float, char_confidences: list) returned by OCRONNX model
- **Sample Evaluation**: Individual comparison result containing image path, ground truth text, predicted text, confidence, edit distance, and whether it was a complete match

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can evaluate 1000 license plate images and receive complete accuracy results in under 5 minutes on a standard GPU
- **SC-002**: Evaluation output includes a two-row table with properly aligned Chinese labels: first row displays 完全准确率, 归一化编辑距离, 编辑距离相似度 (using width-18 columns for labels and width-12 for values), second row displays 总样本数, 评估数, 过滤数, 跳过数
- **SC-003**: System correctly identifies 100% complete matches (predictions that exactly equal ground truth) with no false positives or negatives
- **SC-004**: Edit distance calculations match reference implementation (Python's Levenshtein library or equivalent) within 0.001 tolerance
- **SC-005**: 95% of evaluation runs complete successfully without crashes or unhandled exceptions given valid dataset paths
- **SC-006**: Evaluation function can be imported and called in under 5 lines of code, similar to existing DatasetEvaluator usage pattern
- **SC-007**: Users receive clear error messages when dataset path is invalid, model fails to load, or other recoverable errors occur

## Assumptions *(include if relevant)*

- Ground truth labels are stored in a tab-separated text file where each line has format: `<image_path>\t<ground_truth_text>`
- Image paths in the label file can be relative or absolute paths
- Label file can be named train.txt, val.txt, test.txt, or any user-specified filename
- Text content uses standard character sets (no assumption about specific language, but assumes Unicode support)
- OCR model (OCRONNX) is already trained and loaded before evaluation begins
- Dataset base path is provided, and image paths in label file are resolved relative to this base path
- Confidence scores from OCRONNX are normalized to 0-1 range
- Evaluation is offline/batch processing (not real-time streaming)
- Edit distance calculation treats all characters with equal weight (no character-specific error costs)
- Console output assumes terminal supports UTF-8 encoding and East Asian Width properties (Chinese characters occupy 2 display columns, ASCII characters occupy 1)

## Out of Scope *(include if needed to set boundaries)*

- Training or fine-tuning OCR models (only evaluation)
- Real-time evaluation during training (only offline batch evaluation)
- Multi-model comparison or A/B testing workflows (single model evaluation only)
- Visualization of misrecognized characters or confusion matrices
- Per-character position accuracy analysis (only full-string metrics)
- Integration with MLOps platforms or experiment tracking tools
- Automated hyperparameter tuning based on evaluation results
- Support for other label file formats beyond tab-separated text (e.g., JSON, XML, COCO format)
