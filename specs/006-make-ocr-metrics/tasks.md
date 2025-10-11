# Implementation Tasks: OCR Metrics Evaluation Functions

**Feature**: 006-make-ocr-metrics | **Branch**: `006-make-ocr-metrics` | **Date**: 2025-10-10
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Task Summary

| Phase | User Story | Task Count | Parallel Opportunities | Status |
|-------|-----------|-----------|----------------------|--------|
| Phase 1 | Setup | 2 | 1 | ⏳ Pending |
| Phase 2 | Foundational | 4 | 3 | ⏳ Pending |
| Phase 3 | US1: Basic OCR Accuracy (P1 - MVP) | 8 | 4 | ⏳ Pending |
| Phase 4 | US2: Character-Level Analysis (P2) | 4 | 2 | ⏳ Pending |
| Phase 5 | US3: Confidence Filtering (P3) | 3 | 1 | ⏳ Pending |
| Phase 6 | US4: Performance Reporting (P4) | 3 | 1 | ⏳ Pending |
| Phase 7 | Polish & Integration | 3 | 2 | ⏳ Pending |
| **Total** | **4 User Stories** | **27 tasks** | **14 parallel** | - |

---

## Phase 1: Setup (项目初始化)

### T001 - 安装依赖: python-Levenshtein库
**Type**: Setup | **Priority**: P0 | **Parallel**: No
**Files**: `requirements.txt`, `pyproject.toml`

```bash
# 使用uv添加依赖
uv add python-Levenshtein>=0.25.0

# 或使用pip
pip install python-Levenshtein>=0.25.0
```

**验证**:
```python
import Levenshtein
assert Levenshtein.distance("test", "text") == 1
```

**Estimated Time**: 5分钟

---

### T002 - [P] 创建模块文件结构
**Type**: Setup | **Priority**: P0 | **Parallel**: Yes (与T001并行)
**Files**:
- `infer_onnx/eval_ocr.py` (新建)
- `utils/ocr_metrics.py` (新建)

```bash
touch infer_onnx/eval_ocr.py
touch utils/ocr_metrics.py
```

**Initial Content**:
```python
# infer_onnx/eval_ocr.py
"""OCR数据集评估模块"""

# utils/ocr_metrics.py
"""OCR指标计算函数"""
```

**Estimated Time**: 2分钟

---

## Phase 2: Foundational Tasks (所有User Story的前置条件)

### T003 - [P] 实现编辑距离计算函数 (utils/ocr_metrics.py)
**Type**: Core Function | **Priority**: P0 | **Parallel**: Yes
**Files**: `utils/ocr_metrics.py`
**Dependencies**: T001 (Levenshtein库)

**实现**:
```python
import Levenshtein
from typing import Tuple

def calculate_edit_distance_metrics(pred: str, gt: str) -> Tuple[int, float, float]:
    """计算编辑距离相关指标

    Args:
        pred: 预测文本
        gt: 真实标签文本

    Returns:
        (edit_distance, normalized_edit_distance, edit_distance_similarity)
    """
    distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    normalized_ed = distance / max_len if max_len > 0 else 0.0
    similarity = 1.0 - normalized_ed

    return distance, normalized_ed, similarity
```

**单元测试**:
```python
# tests/unit/test_ocr_metrics.py
def test_edit_distance_perfect_match():
    dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京A12345")
    assert dist == 0
    assert norm_ed == 0.0
    assert sim == 1.0

def test_edit_distance_partial_match():
    dist, norm_ed, sim = calculate_edit_distance_metrics("京A12345", "京A12346")
    assert dist == 1
    assert abs(norm_ed - 0.14) < 0.01  # 1/7
    assert abs(sim - 0.86) < 0.01
```

**Estimated Time**: 30分钟

---

### T004 - [P] 实现完全准确率计算函数 (utils/ocr_metrics.py)
**Type**: Core Function | **Priority**: P0 | **Parallel**: Yes (与T003并行)
**Files**: `utils/ocr_metrics.py`

**实现**:
```python
from typing import List, Tuple

def calculate_accuracy(predictions: List[Tuple[str, str]]) -> float:
    """计算完全准确率

    Args:
        predictions: [(predicted_text, ground_truth), ...]

    Returns:
        accuracy: 完全匹配的样本比例 [0, 1]
    """
    if not predictions:
        return 0.0

    correct_count = sum(1 for pred, gt in predictions if pred == gt)
    return correct_count / len(predictions)
```

**单元测试**:
```python
def test_accuracy_perfect():
    predictions = [("A", "A"), ("B", "B"), ("C", "C")]
    assert calculate_accuracy(predictions) == 1.0

def test_accuracy_partial():
    predictions = [("A", "A"), ("B", "X"), ("C", "C")]
    assert abs(calculate_accuracy(predictions) - 0.667) < 0.001
```

**Estimated Time**: 20分钟

---

### T005 - [P] 实现表格格式化函数 (utils/ocr_metrics.py)
**Type**: Core Function | **Priority**: P0 | **Parallel**: Yes (与T003/T004并行)
**Files**: `utils/ocr_metrics.py`
**Dependencies**: None

**实现**:
```python
from typing import Dict, Any

def print_ocr_metrics(results: Dict[str, Any]) -> None:
    """打印OCR评估指标（中文对齐）

    Args:
        results: 评估结果字典
    """
    # 第一行：总体指标
    header_line1 = f"{'指标':^18} {'完全准确率':^12} {'归一化编辑距离':^12} {'编辑距离相似度':^12}"
    print(header_line1)

    acc = results.get('accuracy', 0)
    norm_ed = results.get('normalized_edit_distance', 0)
    ed_sim = results.get('edit_distance_similarity', 0)

    metrics_line = f"{'OCR评估':^18} {acc:^12.3f} {norm_ed:^12.3f} {ed_sim:^12.3f}"
    print(metrics_line)
    print()  # 空行分隔

    # 第二行：详细统计
    header_line2 = f"{'统计信息':^18} {'总样本数':^12} {'评估数':^12} {'过滤数':^12} {'跳过数':^12}"
    print(header_line2)

    total = results.get('total_samples', 0)
    evaluated = results.get('evaluated_samples', 0)
    filtered = results.get('filtered_samples', 0)
    skipped = results.get('skipped_samples', 0)

    stats_line = f"{'样本统计':^18} {total:^12d} {evaluated:^12d} {filtered:^12d} {skipped:^12d}"
    print(stats_line)
```

**合约测试**:
```python
# tests/contract/test_ocr_evaluator_contract.py
def test_table_format_alignment(capsys):
    results = {
        'accuracy': 0.925,
        'normalized_edit_distance': 0.045,
        'edit_distance_similarity': 0.955,
        'total_samples': 1000,
        'evaluated_samples': 980,
        'filtered_samples': 15,
        'skipped_samples': 5
    }
    print_ocr_metrics(results)
    captured = capsys.readouterr()

    # 验证输出包含中文列名
    assert '完全准确率' in captured.out
    assert '归一化编辑距离' in captured.out
    assert '编辑距离相似度' in captured.out

    # 验证数值格式
    assert '0.925' in captured.out
    assert '980' in captured.out
```

**Estimated Time**: 45分钟

---

### T006 - [P] 实现JSON导出函数 (utils/ocr_metrics.py)
**Type**: Core Function | **Priority**: P0 | **Parallel**: Yes (与T005并行)
**Files**: `utils/ocr_metrics.py`

**实现**:
```python
import json
from typing import Dict, Any

def format_ocr_results_json(results: Dict[str, Any]) -> str:
    """格式化OCR评估结果为JSON

    Args:
        results: 评估结果字典

    Returns:
        JSON格式的结果字符串
    """
    return json.dumps(results, indent=2, ensure_ascii=False)
```

**单元测试**:
```python
def test_json_export():
    results = {'accuracy': 0.925, 'total_samples': 100}
    json_str = format_ocr_results_json(results)

    # 验证JSON有效
    parsed = json.loads(json_str)
    assert parsed['accuracy'] == 0.925

    # 验证中文不被转义
    assert 'accuracy' in json_str
```

**Estimated Time**: 15分钟

---

## Phase 3: User Story 1 - Basic OCR Accuracy Evaluation (P1 - MVP)

**Story Goal**: 研究员可以运行单个命令评估OCR模型准确率

**Independent Test**: 提供10-20张标注图像，运行评估，验证输出包含完全准确率百分比

### T007 - 实现标签文件加载函数 (infer_onnx/eval_ocr.py)
**Type**: [US1] Data Loading | **Priority**: P1 | **Parallel**: No
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T002

**实现**:
```python
from pathlib import Path
from typing import List, Tuple
import logging

def load_label_file(label_file: str, dataset_base_path: str) -> List[Tuple[str, str]]:
    """加载标签文件

    Args:
        label_file: 标签文件路径
        dataset_base_path: 数据集根目录

    Returns:
        [(image_path, ground_truth_text), ...]
    """
    dataset = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) != 2:
                logging.warning(f"跳过标签文件第{line_num}行：格式错误")
                continue

            image_path, gt_text = parts
            full_path = Path(dataset_base_path) / image_path

            if not full_path.exists():
                logging.warning(f"跳过图像：文件不存在 {full_path}")
                continue

            dataset.append((str(full_path), gt_text))

    return dataset
```

**单元测试**:
```python
def test_load_label_file_valid(tmp_path):
    # 创建测试标签文件
    label_file = tmp_path / "test.txt"
    label_file.write_text("img1.png\t京A12345\nimg2.png\t沪B67890")

    # 创建测试图像
    (tmp_path / "img1.png").touch()
    (tmp_path / "img2.png").touch()

    dataset = load_label_file(str(label_file), str(tmp_path))
    assert len(dataset) == 2
    assert dataset[0][1] == "京A12345"
```

**Estimated Time**: 40分钟

---

### T008 - [P] 实现OCRDatasetEvaluator类框架 (infer_onnx/eval_ocr.py)
**Type**: [US1] Core Class | **Priority**: P1 | **Parallel**: Yes (与T007并行，不同部分)
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T002

**实现**:
```python
from typing import Dict, Any, Optional
import logging
import time
import cv2

class OCRDatasetEvaluator:
    """OCR数据集评估器"""

    def __init__(self, ocr_model):
        """初始化评估器

        Args:
            ocr_model: OCRONNX实例
        """
        self.ocr_model = ocr_model

    def evaluate_dataset(
        self,
        label_file: str,
        dataset_base_path: str,
        conf_threshold: float = 0.5,
        max_images: Optional[int] = None,
        output_format: str = 'table'
    ) -> Dict[str, Any]:
        """评估OCR数据集

        Args:
            label_file: 标签文件路径
            conf_threshold: 置信度阈值
            max_images: 最大评估图像数
            output_format: 'table' 或 'json'

        Returns:
            评估结果字典
        """
        # Will be implemented in T009-T012
        pass
```

**Estimated Time**: 20分钟

---

### T009 - 实现评估主循环逻辑 (infer_onnx/eval_ocr.py)
**Type**: [US1] Core Logic | **Priority**: P1 | **Parallel**: No
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T007, T008

**实现** (在OCRDatasetEvaluator.evaluate_dataset中):
```python
# 加载数据集
dataset = load_label_file(label_file, dataset_base_path)
if max_images:
    dataset = dataset[:max_images]

logging.info(f"开始评估OCR数据集，共 {len(dataset)} 张图像")

# 初始化统计
evaluations = []
filtered_count = 0
skipped_count = 0
start_time = time.time()

# 主评估循环
for i, (image_path, gt_text) in enumerate(dataset):
    if i % 50 == 0:
        percentage = (i / len(dataset)) * 100
        logging.info(f"处理进度: {i}/{len(dataset)} ({percentage:.1f}%)")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"无法读取图像: {image_path}")
        skipped_count += 1
        continue

    # OCR推理
    try:
        result = self.ocr_model(image)
        if result is None:
            skipped_count += 1
            continue

        pred_text, confidence, _ = result

        # 置信度过滤
        if confidence < conf_threshold:
            filtered_count += 1
            continue

        # 记录评估结果
        evaluations.append((pred_text, gt_text))

    except Exception as e:
        logging.error(f"OCR推理失败: {image_path} - {e}")
        skipped_count += 1
        continue

evaluation_time = time.time() - start_time
```

**Estimated Time**: 60分钟

---

### T010 - [P] 实现指标聚合逻辑 (infer_onnx/eval_ocr.py)
**Type**: [US1] Aggregation | **Priority**: P1 | **Parallel**: Yes (独立函数)
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T003, T004

**实现** (在evaluate_dataset中继续):
```python
from utils.ocr_metrics import calculate_accuracy, calculate_edit_distance_metrics

# 聚合指标
if not evaluations:
    logging.warning("没有有效的评估样本")
    return {}

accuracy = calculate_accuracy(evaluations)

# 计算平均编辑距离
total_norm_ed = 0.0
total_ed_sim = 0.0
for pred, gt in evaluations:
    _, norm_ed, ed_sim = calculate_edit_distance_metrics(pred, gt)
    total_norm_ed += norm_ed
    total_ed_sim += ed_sim

avg_norm_ed = total_norm_ed / len(evaluations)
avg_ed_sim = total_ed_sim / len(evaluations)

# 构建结果字典
results = {
    'accuracy': accuracy,
    'normalized_edit_distance': avg_norm_ed,
    'edit_distance_similarity': avg_ed_sim,
    'total_samples': len(dataset),
    'evaluated_samples': len(evaluations),
    'filtered_samples': filtered_count,
    'skipped_samples': skipped_count,
    'evaluation_time': evaluation_time,
    'avg_inference_time_ms': (evaluation_time / len(evaluations) * 1000) if evaluations else 0
}

return results
```

**Estimated Time**: 30分钟

---

### T011 - [P] 实现输出格式化逻辑 (infer_onnx/eval_ocr.py)
**Type**: [US1] Output | **Priority**: P1 | **Parallel**: Yes (独立函数)
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T005, T006

**实现** (在evaluate_dataset最后):
```python
from utils.ocr_metrics import print_ocr_metrics, format_ocr_results_json

# 输出结果
if output_format == 'json':
    print(format_ocr_results_json(results))
else:  # table
    print_ocr_metrics(results)

return results
```

**Estimated Time**: 15分钟

---

### T012 - 导出OCRDatasetEvaluator到模块接口 (infer_onnx/__init__.py)
**Type**: [US1] Export | **Priority**: P1 | **Parallel**: No
**Files**: `infer_onnx/__init__.py`
**Dependencies**: T008-T011

**实现**:
```python
# 在infer_onnx/__init__.py添加
from .eval_ocr import OCRDatasetEvaluator

__all__ = [
    # ... existing exports
    'OCRDatasetEvaluator',
]
```

**Estimated Time**: 5分钟

---

### T013 - [P] 合约测试: 基础评估流程 (tests/contract/)
**Type**: [US1] Contract Test | **Priority**: P1 | **Parallel**: Yes
**Files**: `tests/contract/test_ocr_evaluator_contract.py`
**Dependencies**: T007-T012

**实现**:
```python
import pytest
from infer_onnx import OCRDatasetEvaluator, OCRONNX

def test_basic_evaluation_contract(tmp_path, ocr_model_fixture):
    """验证基础评估流程合约"""
    # 创建测试数据集
    label_file = tmp_path / "test.txt"
    label_file.write_text("img1.png\t京A12345\nimg2.png\t沪B67890")

    (tmp_path / "img1.png").write_bytes(create_test_image())
    (tmp_path / "img2.png").write_bytes(create_test_image())

    # 运行评估
    evaluator = OCRDatasetEvaluator(ocr_model_fixture)
    results = evaluator.evaluate_dataset(
        label_file=str(label_file),
        dataset_base_path=str(tmp_path)
    )

    # 验证返回格式
    assert 'accuracy' in results
    assert 'normalized_edit_distance' in results
    assert 'edit_distance_similarity' in results
    assert 'total_samples' in results
    assert 'evaluated_samples' in results

    # 验证数值范围
    assert 0 <= results['accuracy'] <= 1
    assert 0 <= results['normalized_edit_distance'] <= 1
    assert results['total_samples'] == 2
```

**Estimated Time**: 45分钟

---

### T014 - [P] 集成测试: 端到端评估 (tests/integration/)
**Type**: [US1] Integration Test | **Priority**: P1 | **Parallel**: Yes (与T013并行)
**Files**: `tests/integration/test_ocr_evaluation_integration.py`
**Dependencies**: T007-T012

**实现**:
```python
def test_end_to_end_evaluation(sample_ocr_dataset):
    """端到端评估测试"""
    from infer_onnx import OCRDatasetEvaluator, OCRONNX

    # 加载OCR模型
    ocr_model = OCRONNX('models/ocr.onnx', character=char_dict)

    # 评估
    evaluator = OCRDatasetEvaluator(ocr_model)
    results = evaluator.evaluate_dataset(
        label_file='data/ocr_rec_dataset_examples/val.txt',
        dataset_base_path='data/ocr_rec_dataset_examples',
        max_images=20  # 快速测试
    )

    # 验证评估完成
    assert results['evaluated_samples'] > 0
    assert results['total_samples'] == 20
```

**Estimated Time**: 30分钟

---

**✅ Checkpoint: User Story 1 Complete**
- [x] 基础OCR准确率评估功能实现
- [x] 可以评估数据集并输出表格格式结果
- [x] MVP已交付，用户可以回答"我的模型是否工作?"

---

## Phase 4: User Story 2 - Detailed Character-Level Analysis (P2)

**Story Goal**: 工程师可以看到归一化编辑距离和相似度，理解字符级错误

**Independent Test**: 使用已知字符替换错误的数据集，验证输出包含编辑距离指标

### T015 - 扩展SampleEvaluation数据结构 (infer_onnx/eval_ocr.py)
**Type**: [US2] Data Model | **Priority**: P2 | **Parallel**: No
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T009

**实现**:
```python
from dataclasses import dataclass

@dataclass
class SampleEvaluation:
    """单样本评估结果"""
    image_path: str
    ground_truth: str
    predicted_text: str
    confidence: float
    is_correct: bool
    edit_distance: int
    normalized_edit_distance: float
```

**修改T009中的评估循环**:
```python
# 在主循环中记录详细信息
dist, norm_ed, ed_sim = calculate_edit_distance_metrics(pred_text, gt_text)
sample_eval = SampleEvaluation(
    image_path=image_path,
    ground_truth=gt_text,
    predicted_text=pred_text,
    confidence=confidence,
    is_correct=(pred_text == gt_text),
    edit_distance=dist,
    normalized_edit_distance=norm_ed
)
detailed_evaluations.append(sample_eval)
```

**Estimated Time**: 30分钟

---

### T016 - [P] 添加per_sample_results到结果字典 (infer_onnx/eval_ocr.py)
**Type**: [US2] Feature | **Priority**: P2 | **Parallel**: Yes
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T015

**实现** (在T010的results构建中):
```python
results = {
    # ... existing fields
    'per_sample_results': [
        {
            'image_path': e.image_path,
            'ground_truth': e.ground_truth,
            'predicted_text': e.predicted_text,
            'confidence': e.confidence,
            'is_correct': e.is_correct,
            'edit_distance': e.edit_distance,
            'normalized_edit_distance': e.normalized_edit_distance
        }
        for e in detailed_evaluations
    ]
}
```

**Estimated Time**: 20分钟

---

### T017 - [P] 合约测试: 编辑距离指标 (tests/contract/)
**Type**: [US2] Contract Test | **Priority**: P2 | **Parallel**: Yes
**Files**: `tests/contract/test_ocr_evaluator_contract.py`
**Dependencies**: T015, T016

**实现**:
```python
def test_edit_distance_metrics_contract():
    """验证编辑距离指标合约"""
    evaluator = OCRDatasetEvaluator(mock_ocr_model)

    # 创建包含部分匹配的测试数据
    # "京A12345" vs "京A12346" (1个字符差异)
    results = evaluator.evaluate_dataset(...)

    # 验证编辑距离指标存在且合理
    assert 'normalized_edit_distance' in results
    assert 'edit_distance_similarity' in results

    # 验证per_sample_results包含编辑距离
    if 'per_sample_results' in results:
        sample = results['per_sample_results'][0]
        assert 'edit_distance' in sample
        assert 'normalized_edit_distance' in sample
```

**Estimated Time**: 30分钟

---

### T018 - [P] 单元测试: 编辑距离边界情况 (tests/unit/)
**Type**: [US2] Unit Test | **Priority**: P2 | **Parallel**: Yes (与T017并行)
**Files**: `tests/unit/test_ocr_metrics.py`
**Dependencies**: T003

**实现**:
```python
def test_edit_distance_empty_strings():
    """测试空字符串边界情况"""
    dist, norm_ed, sim = calculate_edit_distance_metrics("", "")
    assert dist == 0
    assert norm_ed == 0.0
    assert sim == 1.0

def test_edit_distance_length_difference():
    """测试长度差异"""
    dist, norm_ed, sim = calculate_edit_distance_metrics("京A123", "京A12345")
    assert dist == 2
    assert abs(norm_ed - 0.286) < 0.01  # 2/7
```

**Estimated Time**: 20分钟

---

**✅ Checkpoint: User Story 2 Complete**
- [x] 编辑距离和相似度指标已添加
- [x] per_sample_results提供详细分析
- [x] 工程师可以诊断字符级错误

---

## Phase 5: User Story 3 - Confidence Threshold Filtering (P3)

**Story Goal**: 质量工程师可以测试不同置信度阈值的影响

**Independent Test**: 使用不同阈值(0.5, 0.7, 0.9)运行评估，验证过滤样本数变化

### T019 - 验证置信度过滤逻辑 (infer_onnx/eval_ocr.py)
**Type**: [US3] Feature Verification | **Priority**: P3 | **Parallel**: No
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T009 (已在T009中实现)

**验证** (确认T009中的过滤逻辑正确):
```python
# 在T009主循环中已有:
if confidence < conf_threshold:
    filtered_count += 1
    continue
```

**添加验证日志**:
```python
logging.debug(f"过滤低置信度样本: {image_path} (conf={confidence:.3f} < {conf_threshold})")
```

**Estimated Time**: 15分钟

---

### T020 - [P] 合约测试: 置信度过滤 (tests/contract/)
**Type**: [US3] Contract Test | **Priority**: P3 | **Parallel**: Yes
**Files**: `tests/contract/test_ocr_evaluator_contract.py`
**Dependencies**: T019

**实现**:
```python
def test_confidence_threshold_filtering_contract():
    """验证置信度过滤合约"""
    evaluator = OCRDatasetEvaluator(mock_ocr_model_with_varying_conf)

    # 低阈值: 应包含更多样本
    results_low = evaluator.evaluate_dataset(..., conf_threshold=0.3)

    # 高阈值: 应过滤更多样本
    results_high = evaluator.evaluate_dataset(..., conf_threshold=0.9)

    # 验证过滤行为
    assert results_high['filtered_samples'] > results_low['filtered_samples']
    assert results_high['evaluated_samples'] < results_low['evaluated_samples']

    # 验证样本总数守恒
    def verify_sample_count(r):
        total = r['evaluated_samples'] + r['filtered_samples'] + r['skipped_samples']
        assert total == r['total_samples']

    verify_sample_count(results_low)
    verify_sample_count(results_high)
```

**Estimated Time**: 40分钟

---

### T021 - [P] 集成测试: 阈值扫描 (tests/integration/)
**Type**: [US3] Integration Test | **Priority**: P3 | **Parallel**: Yes (与T020并行)
**Files**: `tests/integration/test_ocr_evaluation_integration.py`
**Dependencies**: T019

**实现**:
```python
def test_threshold_sweep():
    """测试不同置信度阈值"""
    evaluator = OCRDatasetEvaluator(ocr_model)
    thresholds = [0.3, 0.5, 0.7, 0.9]

    results_list = []
    for threshold in thresholds:
        results = evaluator.evaluate_dataset(
            label_file='data/ocr_rec_dataset_examples/val.txt',
            dataset_base_path='data/ocr_rec_dataset_examples',
            conf_threshold=threshold,
            max_images=50
        )
        results_list.append((threshold, results))

    # 验证趋势: 高阈值 -> 更少评估样本
    for i in range(len(results_list) - 1):
        curr_thresh, curr_res = results_list[i]
        next_thresh, next_res = results_list[i + 1]

        assert next_res['evaluated_samples'] <= curr_res['evaluated_samples'], \
            f"阈值{next_thresh}应该比{curr_thresh}过滤更多样本"
```

**Estimated Time**: 30分钟

---

**✅ Checkpoint: User Story 3 Complete**
- [x] 置信度过滤逻辑已验证
- [x] 质量工程师可以优化生产阈值

---

## Phase 6: User Story 4 - Dataset-Level Performance Reporting (P4)

**Story Goal**: 项目经理可以导出JSON格式的详细报告用于比较

**Independent Test**: 运行评估并导出JSON，验证文件包含所有指标

### T022 - 添加JSON导出选项验证 (infer_onnx/eval_ocr.py)
**Type**: [US4] Feature Verification | **Priority**: P4 | **Parallel**: No
**Files**: `infer_onnx/eval_ocr.py`
**Dependencies**: T011 (已在T011中实现)

**验证** (确认T011中的JSON模式正确):
```python
# 在T011中已有:
if output_format == 'json':
    print(format_ocr_results_json(results))
```

**添加输入验证**:
```python
def evaluate_dataset(self, ..., output_format: str = 'table'):
    if output_format not in ['table', 'json']:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'table' or 'json'")
```

**Estimated Time**: 10分钟

---

### T023 - [P] 合约测试: JSON导出格式 (tests/contract/)
**Type**: [US4] Contract Test | **Priority**: P4 | **Parallel**: Yes
**Files**: `tests/contract/test_ocr_evaluator_contract.py`
**Dependencies**: T022

**实现**:
```python
import json

def test_json_export_format_contract(capsys):
    """验证JSON导出格式合约"""
    evaluator = OCRDatasetEvaluator(mock_ocr_model)

    results = evaluator.evaluate_dataset(
        label_file='test.txt',
        dataset_base_path='/tmp',
        output_format='json'
    )

    captured = capsys.readouterr()

    # 验证输出是有效JSON
    parsed = json.loads(captured.out)

    # 验证必需字段
    required_fields = [
        'accuracy', 'normalized_edit_distance', 'edit_distance_similarity',
        'total_samples', 'evaluated_samples', 'filtered_samples', 'skipped_samples'
    ]
    for field in required_fields:
        assert field in parsed, f"Missing required field: {field}"

    # 验证中文不被转义
    assert '\\u' not in captured.out  # ensure_ascii=False
```

**Estimated Time**: 30分钟

---

### T024 - [P] 集成测试: 模型比较场景 (tests/integration/)
**Type**: [US4] Integration Test | **Priority**: P4 | **Parallel**: Yes (与T023并行)
**Files**: `tests/integration/test_ocr_evaluation_integration.py`
**Dependencies**: T022

**实现**:
```python
def test_model_comparison_workflow(tmp_path):
    """测试模型比较工作流"""
    import json

    # 模拟两个模型
    model_a = OCRONNX('models/ocr_v1.onnx', character=char_dict)
    model_b = OCRONNX('models/ocr_v2.onnx', character=char_dict)

    evaluator_a = OCRDatasetEvaluator(model_a)
    evaluator_b = OCRDatasetEvaluator(model_b)

    # 评估并保存结果
    for model_name, evaluator in [('v1', evaluator_a), ('v2', evaluator_b)]:
        results = evaluator.evaluate_dataset(
            label_file='data/ocr_rec_dataset_examples/val.txt',
            dataset_base_path='data/ocr_rec_dataset_examples',
            output_format='json',
            max_images=50
        )

        # 保存到文件
        output_file = tmp_path / f"results_{model_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 验证文件存在且可读
    results_v1 = json.loads((tmp_path / "results_v1.json").read_text())
    results_v2 = json.loads((tmp_path / "results_v2.json").read_text())

    # 比较结果
    improvement = results_v2['accuracy'] - results_v1['accuracy']
    print(f"Model v2 improvement: {improvement:+.2%}")
```

**Estimated Time**: 40分钟

---

**✅ Checkpoint: User Story 4 Complete**
- [x] JSON导出功能已验证
- [x] 项目经理可以比较多个模型版本

---

## Phase 7: Polish & Integration (跨用户故事的集成任务)

### T025 - [P] 更新模块文档和docstring (infer_onnx/eval_ocr.py, utils/ocr_metrics.py)
**Type**: Documentation | **Priority**: P5 | **Parallel**: Yes
**Files**:
- `infer_onnx/eval_ocr.py`
- `utils/ocr_metrics.py`
**Dependencies**: All previous tasks

**添加模块级docstring**:
```python
# infer_onnx/eval_ocr.py
"""OCR数据集评估模块

提供OCR模型性能评估功能，支持：
- 完全准确率计算
- 归一化编辑距离和相似度
- 置信度过滤
- 表格和JSON输出

Example:
    >>> from infer_onnx import OCRDatasetEvaluator, OCRONNX
    >>> ocr_model = OCRONNX('models/ocr.onnx', character=char_dict)
    >>> evaluator = OCRDatasetEvaluator(ocr_model)
    >>> results = evaluator.evaluate_dataset(
    ...     label_file='dataset/val.txt',
    ...     dataset_base_path='dataset'
    ... )
"""
```

**完善函数docstring** (Google Style):
确保所有公共函数包含完整的Args/Returns/Raises/Example

**Estimated Time**: 45分钟

---

### T026 - [P] 创建命令行脚本 (tools/eval_ocr.py或main脚本)
**Type**: CLI Tool | **Priority**: P5 | **Parallel**: Yes (与T025并行)
**Files**: 新建 `tools/eval_ocr.py` 或添加到main.py
**Dependencies**: T012

**实现**:
```python
#!/usr/bin/env python
"""OCR评估命令行工具"""
import argparse
from infer_onnx import OCRDatasetEvaluator, OCRONNX

def main():
    parser = argparse.ArgumentParser(description='评估OCR模型性能')
    parser.add_argument('--label-file', required=True, help='标签文件路径')
    parser.add_argument('--dataset-base', required=True, help='数据集根目录')
    parser.add_argument('--ocr-model', required=True, help='OCR模型路径')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--max-images', type=int, help='最大评估图像数')
    parser.add_argument('--output-format', choices=['table', 'json'], default='table')

    args = parser.parse_args()

    # 加载OCR模型
    ocr_model = OCRONNX(args.ocr_model, character=load_char_dict())

    # 评估
    evaluator = OCRDatasetEvaluator(ocr_model)
    results = evaluator.evaluate_dataset(
        label_file=args.label_file,
        dataset_base_path=args.dataset_base,
        conf_threshold=args.conf_threshold,
        max_images=args.max_images,
        output_format=args.output_format
    )

if __name__ == '__main__':
    main()
```

**Estimated Time**: 30分钟

---

### T027 - 更新根目录CLAUDE.md和quickstart文档
**Type**: Documentation | **Priority**: P5 | **Parallel**: No
**Files**:
- `CLAUDE.md`
- `specs/006-make-ocr-metrics/quickstart.md`
**Dependencies**: All previous tasks

**更新CLAUDE.md**:
```markdown
## 变更日志 (Changelog)

**2025-10-10** - 完成OCR指标评估功能 (006-make-ocr-metrics)
- ✅ 添加OCRDatasetEvaluator类到infer_onnx模块
- ✅ 支持完全准确率、归一化编辑距离、编辑距离相似度计算
- ✅ 双输出模式：表格对齐(中文支持) + JSON导出
- ✅ 置信度过滤和进度日志功能
- ✅ 27个任务完成,14个可并行,4个用户故事交付
```

**更新quickstart.md**:
已在Phase 1生成，确保示例代码与最终实现一致

**Estimated Time**: 30分钟

---

## Dependencies & Parallel Execution

### Dependency Graph (User Story Level)

```
Setup (Phase 1) → Foundational (Phase 2) → US1 (P1 - MVP) → US2 (P2) → US3 (P3) → US4 (P4) → Polish (Phase 7)
                                              ↓
                                     ✅ MVP Checkpoint
                                     (可独立发布)
```

### Task Dependencies (Critical Path)

```
T001 (安装Levenshtein)
  └─> T003 (编辑距离函数) ────┐
                              ├─> T007 (加载标签) ────> T009 (主循环) ────> T010 (聚合) ────> T015 (详细数据)
T002 (创建文件) ────────> T008 (Evaluator类) ───┘                            ↓
                              ├─> T004 (准确率函数) ────────────────────────┘
                              ├─> T005 (表格格式) ────> T011 (输出)
                              └─> T006 (JSON函数) ────> T011 (输出)
```

### Parallel Execution Opportunities

**Phase 1**:
- T001 (安装依赖) || T002 (创建文件) - **并行执行**

**Phase 2**:
- T003 (编辑距离) || T004 (准确率) || T005 (表格) || T006 (JSON) - **4任务并行**

**Phase 3 (US1)**:
- T008 (类框架) || T007 (加载函数) - **并行编写不同函数**
- T013 (合约测试) || T014 (集成测试) - **测试并行**

**Phase 4 (US2)**:
- T016 (添加字段) || T017 (合约测试) || T018 (单元测试) - **3任务并行**

**Phase 5 (US3)**:
- T020 (合约测试) || T021 (集成测试) - **测试并行**

**Phase 6 (US4)**:
- T023 (合约测试) || T024 (集成测试) - **测试并行**

**Phase 7**:
- T025 (文档) || T026 (CLI) - **文档和工具并行**

---

## Implementation Strategy

### MVP-First Approach

**阶段1: MVP (Phase 1-3, US1)**
- 目标: 基础OCR准确率评估
- 交付物: 可运行的评估器,输出准确率
- 预计时间: 4-6小时
- 验证: 用户可以评估数据集并得到有意义的结果

**阶段2: 增强分析 (Phase 4, US2)**
- 目标: 添加编辑距离分析
- 交付物: 详细的字符级错误信息
- 预计时间: 2-3小时
- 验证: 工程师可以诊断OCR错误

**阶段3: 生产优化 (Phase 5, US3)**
- 目标: 置信度阈值优化
- 交付物: 过滤功能和统计
- 预计时间: 1.5-2小时
- 验证: 质量工程师可以调优阈值

**阶段4: 报告与集成 (Phase 6-7, US4 + Polish)**
- 目标: JSON导出和文档完善
- 交付物: 完整功能和文档
- 预计时间: 2-3小时
- 验证: 所有用户故事完成,文档齐全

### Incremental Delivery

每个User Story完成后立即可用:
- **US1 (P1)**: 研究员可以评估模型准确率 → 立即发布MVP
- **US2 (P2)**: 工程师可以诊断错误 → 增量更新
- **US3 (P3)**: 质量工程师可以优化阈值 → 增量更新
- **US4 (P4)**: 项目经理可以比较模型 → 最终版本

### Test Strategy

- **TDD (Test-Driven Development)**: 所有核心函数(T003-T006)先写测试后实现
- **合约测试优先**: 在实现前定义API合约(T013, T017, T020, T023)
- **独立测试**: 每个User Story有独立的测试用例
- **持续验证**: 每个Checkpoint后运行完整测试套件

---

## Estimation Summary

| Phase | Tasks | Sequential Time | Parallel Time | Actual Time (Conservative) |
|-------|-------|----------------|---------------|---------------------------|
| Phase 1: Setup | 2 | 7 min | 5 min | 10 min |
| Phase 2: Foundational | 4 | 110 min | 45 min | 60 min |
| Phase 3: US1 (MVP) | 8 | 285 min | 180 min | 240 min (4h) |
| Phase 4: US2 | 4 | 100 min | 60 min | 90 min (1.5h) |
| Phase 5: US3 | 3 | 85 min | 50 min | 75 min (1.25h) |
| Phase 6: US4 | 3 | 80 min | 45 min | 70 min (1.2h) |
| Phase 7: Polish | 3 | 105 min | 60 min | 90 min (1.5h) |
| **Total** | **27** | **~12h** | **~7.5h** | **~10h (with slack)** |

**Recommended Approach**:
- Day 1: Phase 1-3 (MVP, ~5-6h)
- Day 2: Phase 4-7 (Enhancements, ~4-5h)

---

## Success Criteria Mapping

| Success Criterion | Verified By | Status |
|------------------|-------------|--------|
| SC-001: 1000图<5分钟 | T014集成测试 + 性能profiling | ⏳ |
| SC-002: 两行表格对齐 | T005实现 + T013合约测试 | ⏳ |
| SC-003: 100%准确匹配 | T004实现 + T013合约测试 | ⏳ |
| SC-004: 编辑距离精度 | T003实现 + T018单元测试 | ⏳ |
| SC-005: 95%成功率 | T014, T021, T024集成测试 | ⏳ |
| SC-006: ≤5行代码调用 | T012 API设计 + quickstart示例 | ⏳ |
| SC-007: 清晰错误信息 | T007-T010错误处理 + T014测试 | ⏳ |

---

**最后更新**: 2025-10-10
**维护者**: ONNX Vehicle Plate Recognition Team
**下一步**: 开始执行 Phase 1 - Setup Tasks
