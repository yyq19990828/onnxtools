# Research: OCR Metrics Evaluation Functions

**Feature**: 006-make-ocr-metrics | **Date**: 2025-10-10

## Overview

本文档记录OCR指标评估功能的技术决策研究，包括编辑距离计算、中文对齐显示、表格格式化和性能优化等关键技术选型。

## Decision 1: 编辑距离库选择

### Decision
使用 **python-Levenshtein** 库计算编辑距离

### Rationale
1. **性能优异**: C扩展实现，比纯Python实现快100-1000倍
2. **API简洁**: `Levenshtein.distance(s1, s2)` 直接返回编辑距离
3. **广泛使用**: PaddleOCR等主流OCR评估工具的标准选择
4. **维护活跃**: 最新版本0.25.1 (2024)，持续维护
5. **安装简单**: `pip install python-Levenshtein` 或 `uv add python-Levenshtein`

### Alternatives Considered

| 库名 | 优点 | 缺点 | 拒绝原因 |
|-----|------|------|---------|
| editdistance | 纯Python实现 | 性能差(慢50-100倍) | 大数据集评估慢 |
| rapidfuzz | 模糊匹配功能丰富 | 依赖较重，API复杂 | 过度设计，仅需基本编辑距离 |
| jellyfish | 多种字符串距离算法 | 性能中等，API冗余 | 不需要额外算法 |
| 标准库difflib | 无额外依赖 | 性能差，API不直观 | 需要手动计算编辑距离 |

### Implementation
```python
import Levenshtein

def calculate_normalized_edit_distance(pred: str, gt: str) -> float:
    """计算归一化编辑距离"""
    distance = Levenshtein.distance(pred, gt)
    max_len = max(len(pred), len(gt))
    return distance / max_len if max_len > 0 else 0.0

def calculate_edit_distance_similarity(pred: str, gt: str) -> float:
    """计算编辑距离相似度 (1 - normalized_distance)"""
    return 1.0 - calculate_normalized_edit_distance(pred, gt)
```

---

## Decision 2: 中文字符宽度计算

### Decision
使用内置 **字符串格式化** + **手动宽度调整** 方案

### Rationale
1. **简单直接**: 参考现有`detection_metrics.py::print_metrics()`实现
2. **零依赖**: 不引入wcwidth等额外库，减少依赖
3. **足够准确**: 固定列宽(18和12)适配"完全准确率"等常见6字中文术语
4. **性能最优**: 字符串format操作无额外计算开销
5. **终端兼容**: UTF-8终端自动处理东亚字符宽度

### Alternatives Considered

| 方案 | 优点 | 缺点 | 拒绝原因 |
|-----|------|------|---------|
| wcwidth库 | 精确计算字符显示宽度 | 引入额外依赖 | 固定列宽足够，无需动态计算 |
| unicodedata.east_asian_width() | 标准库，无依赖 | 需要手动遍历字符 | 性能差，代码复杂 |
| rich库 | 自动处理对齐和样式 | 重依赖(10+ MB) | 过度设计，仅需基本对齐 |

### Implementation
参考`utils/detection_metrics.py:581-655`:
```python
def print_ocr_metrics(results: Dict[str, Any]):
    """打印OCR评估指标（中文对齐）"""
    # 第一行：总体指标
    header_line1 = f"{'指标':^18} {'完全准确率':^12} {'归一化编辑距离':^12} {'编辑距离相似度':^12}"
    print(header_line1)

    acc = results.get('accuracy', 0)
    norm_ed = results.get('normalized_edit_distance', 0)
    ed_sim = results.get('edit_distance_similarity', 0)

    metrics_line = f"{'OCR评估':^18} {acc:^12.3f} {norm_ed:^12.3f} {ed_sim:^12.3f}"
    print(metrics_line)

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

**列宽选择**:
- 指标名列: 18字符宽度 = 6个中文字符×3 (考虑对齐填充)
- 数值列: 12字符宽度 = 小数(0.123)或整数(1000) + 对齐空间

---

## Decision 3: 表格格式化模式

### Decision
实现 **两行分层表格** + **可选JSON导出** 模式

### Rationale
1. **用户体验**: 第一行关注核心指标(准确率、编辑距离)，第二行显示统计细节
2. **对齐一致**: 与`detection_metrics.py::print_metrics()`风格统一
3. **灵活性**: JSON模式支持后续可视化和模型比较
4. **可扩展**: 未来可添加更多输出格式(CSV、Markdown)而不破坏现有接口

### Implementation
```python
def format_ocr_results(results: Dict[str, Any], output_format: str = 'table') -> str:
    """格式化OCR评估结果

    Args:
        results: 评估结果字典
        output_format: 'table' (默认) 或 'json'

    Returns:
        格式化的结果字符串
    """
    if output_format == 'json':
        import json
        return json.dumps(results, indent=2, ensure_ascii=False)
    else:  # table
        return _format_table_aligned(results)
```

**JSON Schema**:
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
  "avg_inference_time_ms": 12.5
}
```

---

## Decision 4: OCRDatasetEvaluator架构

### Decision
继承 **DatasetEvaluator模式**，创建独立的OCRDatasetEvaluator类

### Rationale
1. **架构一致性**: 与`eval_coco.py::DatasetEvaluator`保持相同的设计模式
2. **易用性**: 用户只需实例化evaluator并调用`evaluate_dataset()`方法
3. **可测试性**: 独立类便于单元测试和合约测试
4. **可维护性**: 清晰的职责边界，不污染其他模块

### Implementation
参考`infer_onnx/eval_coco.py:21-243`:
```python
class OCRDatasetEvaluator:
    """OCR数据集评估器"""

    def __init__(self, ocr_model: OCRONNX):
        """初始化评估器

        Args:
            ocr_model: OCR模型实例
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
            label_file: 标签文件路径 (train.txt, val.txt)
            dataset_base_path: 数据集根目录
            conf_threshold: 置信度阈值
            max_images: 最大评估图像数
            output_format: 输出格式 ('table' 或 'json')

        Returns:
            评估结果字典
        """
        # 实现逻辑...
```

---

## Decision 5: 进度日志实现

### Decision
使用 **计数器取模** + **logging模块** 实现高效进度日志

### Rationale
1. **性能最优**: `if i % 50 == 0` 仅在整数倍时触发，零性能损失
2. **可配置**: 通过参数控制日志间隔(默认50)
3. **标准化**: 使用logging.info()统一日志格式
4. **清晰度**: 显示"处理进度: {current}/{total} ({percentage}%)"格式

### Implementation
参考`infer_onnx/eval_coco.py:180-182`:
```python
def evaluate_dataset(self, ...):
    total_images = len(image_list)
    logging.info(f"开始评估OCR数据集，共 {total_images} 张图像")

    for i, (image_path, gt_text) in enumerate(dataset):
        if i % 50 == 0:
            percentage = (i / total_images) * 100
            logging.info(f"处理进度: {i}/{total_images} ({percentage:.1f}%)")

        # 处理单张图像...
```

---

## Decision 6: 数据加载策略

### Decision
使用 **逐行解析** + **路径验证** 的渐进式加载

### Rationale
1. **内存高效**: 不需要一次性加载所有图像到内存
2. **错误容忍**: 跳过损坏的图像和标签，记录警告日志
3. **路径灵活**: 支持相对路径和绝对路径
4. **早期验证**: 在开始评估前验证文件存在性

### Implementation
```python
def load_label_file(label_file: str, dataset_base_path: str) -> List[Tuple[str, str]]:
    """加载标签文件

    Args:
        label_file: 标签文件路径
        dataset_base_path: 数据集根目录

    Returns:
        (image_path, ground_truth_text)元组列表
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

---

## Performance Considerations

### 预期性能特征

| 操作 | 预期时间 | 优化措施 |
|-----|---------|---------|
| 标签文件解析 | <1秒 (10k行) | 逐行解析，不加载整个文件到内存 |
| 单张图像OCR推理 | ~10-20ms (GPU) | OCRONNX已优化 |
| 编辑距离计算 | <0.1ms (20字符) | C扩展Levenshtein库 |
| 表格格式化 | <1ms | 字符串format操作 |
| **总计 (1000图像)** | **~3-4分钟** | 满足<5分钟要求 |

### 潜在瓶颈
1. **图像加载**: cv2.imread()可能成为瓶颈
   - **缓解**: OCRONNX内部已使用优化的加载流程
2. **内存占用**: 存储1000个预测结果
   - **缓解**: 仅存储文本字符串和置信度，不存储图像数据

---

## Dependencies Summary

### New Dependencies
```toml
[project.dependencies]
python-Levenshtein = ">=0.25.0"  # 编辑距离计算
```

### Existing Dependencies (Already in project)
- numpy >=2.2.6
- opencv-python >=4.12.0
- logging (标准库)
- pathlib (标准库)
- json (标准库)

### Installation
```bash
# 使用uv (推荐)
uv add python-Levenshtein

# 或使用pip
pip install python-Levenshtein
```

---

## Risk Mitigation

| 风险 | 缓解措施 | 负责人 |
|-----|---------|-------|
| Levenshtein库安装失败 | 提供纯Python回退实现(editdistance) | 开发者 |
| 中文对齐在某些终端错误 | 文档说明UTF-8终端要求 | 文档维护者 |
| 大数据集内存溢出 | 实现max_images参数，分批评估 | 开发者 |
| 编辑距离精度问题 | 单元测试验证与参考实现一致 | 测试工程师 |

---

## Next Steps

1. ✅ 完成技术决策研究 (本文档)
2. ⏭️ Phase 1: 生成data-model.md (数据模型定义)
3. ⏭️ Phase 1: 生成contracts/ocr_evaluator_api.yaml (API合约)
4. ⏭️ Phase 1: 生成quickstart.md (快速入门指南)
5. ⏭️ Phase 2: 生成tasks.md (实施任务清单)
