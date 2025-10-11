# Data Model: OCR Metrics Evaluation

**Feature**: 006-make-ocr-metrics | **Date**: 2025-10-10

## Entity Definitions

### 1. OCRDatasetEvaluator

**职责**: 管理OCR评估流程，协调模型推理和指标计算

**属性**:
```python
class OCRDatasetEvaluator:
    ocr_model: OCRONNX          # OCR模型实例
```

**方法**:
```python
def evaluate_dataset(
    self,
    label_file: str,
    dataset_base_path: str,
    conf_threshold: float = 0.5,
    max_images: Optional[int] = None,
    output_format: str = 'table'
) -> Dict[str, Any]:
    """评估OCR数据集，返回指标字典"""
```

**验证规则**:
- `label_file`必须存在且可读
- `dataset_base_path`必须是有效目录
- `conf_threshold`范围: [0.0, 1.0]
- `output_format`只能是 'table' 或 'json'

---

### 2. OCREvaluationResult

**职责**: 存储评估结果的所有指标和统计信息

**属性**:
```python
@dataclass
class OCREvaluationResult:
    # 核心指标
    accuracy: float                        # 完全准确率 [0, 1]
    normalized_edit_distance: float        # 归一化编辑距离 [0, 1]
    edit_distance_similarity: float        # 编辑距离相似度 [0, 1]

    # 样本统计
    total_samples: int                     # 标签文件中的总样本数
    evaluated_samples: int                 # 实际评估的样本数
    filtered_samples: int                  # 被置信度过滤的样本数
    skipped_samples: int                   # 跳过的样本数（错误、缺失等）

    # 性能统计
    evaluation_time: float                 # 总评估时间（秒）
    avg_inference_time_ms: float           # 平均推理时间（毫秒）

    # 详细信息（可选）
    per_sample_results: Optional[List['SampleEvaluation']] = None
```

**验证规则**:
- `accuracy`, `normalized_edit_distance`, `edit_distance_similarity` ∈ [0.0, 1.0]
- `total_samples` = `evaluated_samples` + `filtered_samples` + `skipped_samples`
- `evaluated_samples` >= 0
- `evaluation_time` > 0

**关系**:
- 包含多个 `SampleEvaluation` (1对多)

---

### 3. SampleEvaluation

**职责**: 存储单个样本的评估细节

**属性**:
```python
@dataclass
class SampleEvaluation:
    image_path: str                # 图像路径
    ground_truth: str              # 真实标签文本
    predicted_text: str            # 预测文本
    confidence: float              # 预测置信度 [0, 1]
    is_correct: bool               # 是否完全匹配
    edit_distance: int             # 编辑距离（整数）
    normalized_edit_distance: float  # 归一化编辑距离 [0, 1]
```

**验证规则**:
- `confidence` ∈ [0.0, 1.0]
- `edit_distance` >= 0
- `normalized_edit_distance` ∈ [0.0, 1.0]
- `is_correct` = True ⟺ `edit_distance` = 0

**关系**:
- 属于一个 `OCREvaluationResult` (多对1)

---

### 4. LabelEntry

**职责**: 表示标签文件中的一行数据

**属性**:
```python
@dataclass
class LabelEntry:
    image_path: str      # 图像路径（可以是相对或绝对路径）
    ground_truth: str    # 真实标签文本
    line_number: int     # 标签文件中的行号（用于错误定位）
```

**验证规则**:
- `image_path`非空字符串
- `ground_truth`非空字符串（允许特殊字符和Unicode）
- `line_number` > 0

**状态转换**:
```
[Raw Line] --parse--> [LabelEntry] --validate--> [Valid Entry] or [Skipped Entry]
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     OCRDatasetEvaluator                         │
│                                                                 │
│  1. Load label file → List[LabelEntry]                         │
│  2. For each entry:                                            │
│     - Load image                                               │
│     - OCR inference → (predicted_text, confidence)            │
│     - Calculate metrics → SampleEvaluation                     │
│  3. Aggregate → OCREvaluationResult                            │
│  4. Format output → table or JSON                              │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ Label File   │───>│ LabelEntry   │───>│ SampleEvaluation │
│ (train.txt)  │    │ (list)       │    │ (list)           │
└──────────────┘    └──────────────┘    └──────────────────┘
                                                │
                                                v
                                        ┌──────────────────┐
                                        │ OCREvaluation    │
                                        │ Result           │
                                        └──────────────────┘
                                                │
                                                v
                                        ┌──────────────────┐
                                        │ Output           │
                                        │ (table/JSON)     │
                                        └──────────────────┘
```

---

## Metrics Calculation

### 1. Complete Accuracy (完全准确率)

**定义**: 预测文本与真实标签完全匹配的样本比例

**公式**:
```
accuracy = (完全匹配样本数) / (评估样本总数)
```

**实现**:
```python
def calculate_accuracy(evaluations: List[SampleEvaluation]) -> float:
    if not evaluations:
        return 0.0
    correct_count = sum(1 for e in evaluations if e.is_correct)
    return correct_count / len(evaluations)
```

**范围**: [0.0, 1.0]

---

### 2. Normalized Edit Distance (归一化编辑距离)

**定义**: 编辑距离除以较长字符串的长度

**公式**:
```
normalized_ed = edit_distance(pred, gt) / max(len(pred), len(gt))
```

**实现**:
```python
import Levenshtein

def calculate_normalized_edit_distance(pred: str, gt: str) -> float:
    distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return distance / max_len if max_len > 0 else 0.0
```

**范围**: [0.0, 1.0]
- 0.0 = 完全匹配
- 1.0 = 完全不同

---

### 3. Edit Distance Similarity (编辑距离相似度)

**定义**: 1减去归一化编辑距离

**公式**:
```
similarity = 1.0 - normalized_edit_distance
```

**实现**:
```python
def calculate_edit_distance_similarity(pred: str, gt: str) -> float:
    return 1.0 - calculate_normalized_edit_distance(pred, gt)
```

**范围**: [0.0, 1.0]
- 1.0 = 完全匹配
- 0.0 = 完全不同

---

## Output Formats

### 1. Table Format (Console Display)

**示例输出**:
```
指标                  完全准确率        归一化编辑距离      编辑距离相似度
OCR评估              0.925           0.045           0.955

统计信息              总样本数          评估数            过滤数            跳过数
样本统计              1000            980             15              5
```

**列宽规范**:
- 指标名列: 18字符（支持6个中文字符）
- 数值列: 12字符（支持小数和整数）

---

### 2. JSON Format (Export)

**Schema**:
```json
{
  "accuracy": 0.925,
  "normalized_edit_distance": 0.045,
  "edit_distance_similarity": 0.955,
  "total_samples": 1000,
  "evaluated_samples": 980,
  "filtered_samples": 15,
  "skipped_samples": 5,
  "evaluation_time": 245.3,
  "avg_inference_time_ms": 12.5,
  "per_sample_results": [
    {
      "image_path": "images/train_word_1.png",
      "ground_truth": "京A12345",
      "predicted_text": "京A12345",
      "confidence": 0.98,
      "is_correct": true,
      "edit_distance": 0,
      "normalized_edit_distance": 0.0
    }
  ]
}
```

**可选字段**: `per_sample_results`仅在明确请求时包含（避免大文件）

---

## Error Handling

### 错误类型和处理策略

| 错误类型 | 处理策略 | 计数器 | 日志级别 |
|---------|---------|--------|---------|
| 标签文件不存在 | 抛出异常，终止评估 | N/A | ERROR |
| 图像文件不存在 | 跳过该样本，记录警告 | skipped_samples++ | WARNING |
| 标签行格式错误 | 跳过该行，记录警告 | skipped_samples++ | WARNING |
| OCR推理失败 | 跳过该样本，记录错误 | skipped_samples++ | ERROR |
| 置信度低于阈值 | 过滤该样本，记录调试 | filtered_samples++ | DEBUG |
| 空预测结果 | 记录为错误样本(ed=len(gt)) | evaluated_samples++ | DEBUG |

---

## State Transitions

```
┌─────────────┐
│   INIT      │ 初始化评估器
└──────┬──────┘
       │
       v
┌─────────────┐
│   LOADING   │ 加载标签文件
└──────┬──────┘
       │
       v
┌─────────────┐
│ EVALUATING  │ 逐样本评估（主循环）
└──────┬──────┘
       │
       v
┌─────────────┐
│ AGGREGATING │ 聚合指标
└──────┬──────┘
       │
       v
┌─────────────┐
│ FORMATTING  │ 格式化输出
└──────┬──────┘
       │
       v
┌─────────────┐
│  COMPLETE   │ 返回结果
└─────────────┘
```

---

## Performance Constraints

| 数据模型操作 | 时间复杂度 | 空间复杂度 | 注释 |
|------------|-----------|-----------|------|
| 加载标签文件 | O(n) | O(n) | n = 样本数 |
| 单样本评估 | O(m) | O(1) | m = OCR推理时间 |
| 编辑距离计算 | O(s*t) | O(s*t) | s,t = 字符串长度 |
| 指标聚合 | O(n) | O(1) | 累加操作 |
| 表格格式化 | O(1) | O(1) | 固定格式 |

**整体复杂度**: O(n*m) 其中n=样本数，m=单样本处理时间

---

## Validation Rules Summary

| 字段 | 类型 | 范围/约束 | 必需 |
|-----|------|----------|------|
| accuracy | float | [0.0, 1.0] | ✅ |
| normalized_edit_distance | float | [0.0, 1.0] | ✅ |
| edit_distance_similarity | float | [0.0, 1.0] | ✅ |
| total_samples | int | >= 0 | ✅ |
| evaluated_samples | int | >= 0, <= total_samples | ✅ |
| filtered_samples | int | >= 0 | ✅ |
| skipped_samples | int | >= 0 | ✅ |
| evaluation_time | float | > 0 | ✅ |
| conf_threshold | float | [0.0, 1.0] | ✅ |
| image_path | str | 非空, 文件存在 | ✅ |
| ground_truth | str | 非空 | ✅ |
| predicted_text | str | 允许空字符串 | ✅ |
