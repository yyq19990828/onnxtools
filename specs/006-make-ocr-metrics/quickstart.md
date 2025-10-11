# Quick Start: OCR Metrics Evaluation

**Feature**: 006-make-ocr-metrics | **Date**: 2025-10-10

## 5分钟快速上手

### 1. 安装依赖

```bash
# 使用uv（推荐）
uv add python-Levenshtein

# 或使用pip
pip install python-Levenshtein
```

### 2. 准备数据集

确保您的数据集遵循以下结构：

```
dataset/
├── images/
│   ├── train_word_1.png
│   ├── train_word_2.png
│   └── ...
├── train.txt          # 标签文件
└── val.txt            # 可选：验证集标签
```

**标签文件格式** (train.txt):
```
images/train_word_1.png	京A12345
images/train_word_2.png	沪B67890
images/train_word_3.png	粤C11111
```

> **注意**: 每行用Tab（`\t`）分隔图像路径和标签文本

### 3. 基础使用

```python
from infer_onnx import OCRONNX, OCRDatasetEvaluator

# 加载OCR模型
ocr_model = OCRONNX(
    onnx_path='models/ocr.onnx',
    character=character_dict,  # 字符字典
    conf_thres=0.5
)

# 创建评估器
evaluator = OCRDatasetEvaluator(ocr_model)

# 评估数据集（默认表格输出）
results = evaluator.evaluate_dataset(
    label_file='/path/to/dataset/train.txt',
    dataset_base_path='/path/to/dataset',
    conf_threshold=0.5
)
```

**输出示例**:
```
开始评估OCR数据集，共 1000 张图像
处理进度: 50/1000 (5.0%)
处理进度: 100/1000 (10.0%)
...
处理进度: 1000/1000 (100.0%)

指标                  完全准确率        归一化编辑距离      编辑距离相似度
OCR评估              0.925           0.045           0.955

统计信息              总样本数          评估数            过滤数            跳过数
样本统计              1000            980             15              5
```

---

## 高级用法

### 1. JSON导出模式

```python
# 导出为JSON格式
results = evaluator.evaluate_dataset(
    label_file='/path/to/dataset/val.txt',
    dataset_base_path='/path/to/dataset',
    output_format='json'  # 'table' 或 'json'
)

# 保存到文件
import json
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

**JSON输出示例**:
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

### 2. 置信度阈值优化

```python
# 测试不同置信度阈值
thresholds = [0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    results = evaluator.evaluate_dataset(
        label_file='/path/to/dataset/val.txt',
        dataset_base_path='/path/to/dataset',
        conf_threshold=threshold
    )
    print(f"Threshold {threshold}: Accuracy={results['accuracy']:.3f}")
```

### 3. 快速测试（限制图像数量）

```python
# 仅评估前100张图像
results = evaluator.evaluate_dataset(
    label_file='/path/to/dataset/train.txt',
    dataset_base_path='/path/to/dataset',
    max_images=100  # 快速测试
)
```

### 4. 命令行工具（推荐）

```bash
# 使用eval_ocr.py脚本
python -m infer_onnx.eval_ocr \
    --label-file /path/to/dataset/train.txt \
    --dataset-base /path/to/dataset \
    --ocr-model models/ocr.onnx \
    --conf-threshold 0.5 \
    --output-format table
```

---

## 常见场景

### 场景1: 模型A/B测试

```python
from infer_onnx import OCRONNX, OCRDatasetEvaluator

# 评估模型A
model_a = OCRONNX('models/ocr_v1.onnx', character=char_dict)
evaluator_a = OCRDatasetEvaluator(model_a)
results_a = evaluator_a.evaluate_dataset(
    label_file='dataset/val.txt',
    dataset_base_path='dataset'
)

# 评估模型B
model_b = OCRONNX('models/ocr_v2.onnx', character=char_dict)
evaluator_b = OCRDatasetEvaluator(model_b)
results_b = evaluator_b.evaluate_dataset(
    label_file='dataset/val.txt',
    dataset_base_path='dataset'
)

# 比较结果
print(f"Model A accuracy: {results_a['accuracy']:.3f}")
print(f"Model B accuracy: {results_b['accuracy']:.3f}")
improvement = (results_b['accuracy'] - results_a['accuracy']) * 100
print(f"Improvement: {improvement:+.2f}%")
```

### 场景2: 跨数据集评估

```python
datasets = {
    'train': 'dataset/train.txt',
    'val': 'dataset/val.txt',
    'test': 'dataset/test.txt'
}

for split_name, label_file in datasets.items():
    results = evaluator.evaluate_dataset(
        label_file=label_file,
        dataset_base_path='dataset'
    )
    print(f"{split_name}: Accuracy={results['accuracy']:.3f}, "
          f"Edit Distance Similarity={results['edit_distance_similarity']:.3f}")
```

### 场景3: 错误分析（保存详细结果）

```python
# 导出每个样本的详细结果
results = evaluator.evaluate_dataset(
    label_file='dataset/val.txt',
    dataset_base_path='dataset',
    output_format='json'
)

# 分析错误样本
import json
with open('detailed_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 找出识别错误的样本
if 'per_sample_results' in results:
    errors = [s for s in results['per_sample_results'] if not s['is_correct']]
    print(f"Found {len(errors)} errors")
    for e in errors[:10]:  # 显示前10个错误
        print(f"GT: {e['ground_truth']} -> Pred: {e['predicted_text']} "
              f"(ED: {e['edit_distance']})")
```

---

## 性能优化建议

### 1. GPU加速

确保OCR模型使用GPU推理：

```python
import onnxruntime as ort

# 检查可用的execution providers
print(ort.get_available_providers())  # 应包含 'CUDAExecutionProvider'

# OCRONNX会自动使用GPU（如果可用）
ocr_model = OCRONNX('models/ocr.onnx', character=char_dict)
```

### 2. 批量评估

对于大数据集，建议分批评估：

```python
# 分批评估（避免内存溢出）
batch_size = 500
total_samples = 5000

for i in range(0, total_samples, batch_size):
    results = evaluator.evaluate_dataset(
        label_file='dataset/train.txt',
        dataset_base_path='dataset',
        max_images=batch_size,
        skip_samples=i  # 跳过前i个样本
    )
    # 累积结果...
```

### 3. 进度监控

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

# 评估时会自动显示进度
results = evaluator.evaluate_dataset(
    label_file='dataset/train.txt',
    dataset_base_path='dataset'
)
# 输出: 处理进度: 50/1000 (5.0%)
#       处理进度: 100/1000 (10.0%)
#       ...
```

---

## 故障排除

### 问题1: "Label file not found"

**原因**: 标签文件路径不正确

**解决**:
```python
from pathlib import Path

label_file = Path('/path/to/dataset/train.txt')
assert label_file.exists(), f"Label file not found: {label_file}"
```

### 问题2: 中文显示乱码

**原因**: 终端不支持UTF-8编码

**解决**:
```bash
# Linux/Mac
export LANG=en_US.UTF-8

# Windows (PowerShell)
chcp 65001
```

### 问题3: 评估速度慢

**可能原因**:
1. CPU推理（应使用GPU）
2. 图像加载IO瓶颈
3. 数据集过大

**解决**:
```python
# 1. 确认GPU可用
import onnxruntime as ort
assert 'CUDAExecutionProvider' in ort.get_available_providers()

# 2. 使用max_images限制测试规模
results = evaluator.evaluate_dataset(..., max_images=100)

# 3. 检查图像分辨率（过大的图像会慢）
```

### 问题4: "KeyError: 'accuracy'"

**原因**: 评估失败或返回空结果

**解决**:
```python
results = evaluator.evaluate_dataset(...)

# 安全访问结果
accuracy = results.get('accuracy', 0.0)
if accuracy == 0.0:
    print("Warning: No valid evaluations performed")
```

---

## 与目标检测评估对比

| 特性 | 目标检测 (eval_coco.py) | OCR评估 (eval_ocr.py) |
|-----|------------------------|---------------------|
| **输入格式** | YOLO格式 (images/, labels/) | Tab分隔的label list (train.txt) |
| **核心指标** | mAP, Precision, Recall | 完全准确率, 编辑距离 |
| **输出格式** | 表格对齐（中文支持） | 表格对齐 + JSON导出 |
| **架构模式** | DatasetEvaluator类 | OCRDatasetEvaluator类 |
| **性能目标** | <5分钟/1000图（GPU） | <5分钟/1000图（GPU） |
| **日志进度** | 每100张 | 每50张 |

---

## API参考

### OCRDatasetEvaluator

```python
class OCRDatasetEvaluator:
    def __init__(self, ocr_model: OCRONNX):
        """初始化评估器"""

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
            dataset_base_path: 数据集根目录
            conf_threshold: 置信度阈值 [0, 1]
            max_images: 最大评估图像数
            output_format: 'table' 或 'json'

        Returns:
            评估结果字典
        """
```

### 返回值结构

```python
{
    'accuracy': float,                    # 完全准确率 [0, 1]
    'normalized_edit_distance': float,    # 归一化编辑距离 [0, 1]
    'edit_distance_similarity': float,    # 编辑距离相似度 [0, 1]
    'total_samples': int,                 # 总样本数
    'evaluated_samples': int,             # 评估样本数
    'filtered_samples': int,              # 过滤样本数
    'skipped_samples': int,               # 跳过样本数
    'evaluation_time': float,             # 评估时间（秒）
    'avg_inference_time_ms': float        # 平均推理时间（毫秒）
}
```

---

## 下一步

- 📖 阅读 [data-model.md](./data-model.md) 了解数据模型
- 📋 查看 [contracts/ocr_evaluator_api.yaml](./contracts/ocr_evaluator_api.yaml) API合约
- 🧪 运行合约测试: `pytest tests/contract/test_ocr_evaluator_contract.py`
- 📊 查看 [research.md](./research.md) 技术决策

---

**最后更新**: 2025-10-10 | **维护者**: ONNX Vehicle Plate Recognition Team
