[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **eval**

# 评估子模块 (onnxtools.eval)

## 模块职责

提供COCO格式检测数据集评估和OCR数据集评估功能,支持mAP计算、编辑距离指标、置信度过滤和多种输出格式(表格/JSON)。

## 入口和启动

- **模块入口**: `onnxtools/eval/__init__.py`
- **COCO评估**: `eval_coco.py` - DatasetEvaluator类
- **OCR评估**: `eval_ocr.py` - OCRDatasetEvaluator类和SampleEvaluation数据类

### 快速开始

```python
# COCO数据集评估
from onnxtools import DatasetEvaluator, create_detector

detector = create_detector('rtdetr', 'models/rtdetr.onnx')
evaluator = DatasetEvaluator(detector)
results = evaluator.evaluate_dataset(
    dataset_path='/path/to/coco',
    conf_threshold=0.25,
    iou_threshold=0.7,
    max_images=100
)

# OCR数据集评估
from onnxtools import OCRDatasetEvaluator, OcrORT

ocr_model = OcrORT('models/ocr.onnx', character=char_dict)
evaluator = OCRDatasetEvaluator(ocr_model)
results = evaluator.evaluate_dataset(
    label_file='data/val.txt',
    dataset_base_path='data/',
    conf_threshold=0.5,
    output_format='table'
)
```

## 外部接口

### 1. DatasetEvaluator - COCO数据集评估器

```python
from onnxtools.eval import DatasetEvaluator

class DatasetEvaluator:
    """通用YOLO格式数据集评估器"""

    def __init__(self, detector: BaseORT):
        """初始化评估器

        Args:
            detector: 检测器实例(BaseORT子类)
        """
        pass

    def evaluate_dataset(
        self,
        dataset_path: str,
        output_transform: Optional[Callable] = None,
        conf_threshold: float = 0.25,  # 与Ultralytics验证模式对齐
        iou_threshold: float = 0.7,
        max_images: Optional[int] = None,
        exclude_files: Optional[List[str]] = None,
        exclude_labels_containing: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """在YOLO格式数据集上评估模型性能

        Returns:
            Dict包含:
            - map50: mAP@0.5
            - map50_95: mAP@0.5:0.95
            - precision: 精确率
            - recall: 召回率
            - speed_preprocess: 预处理平均时间(ms)
            - speed_inference: 推理平均时间(ms)
            - speed_postprocess: 后处理平均时间(ms)
        """
        pass
```

**重要说明**: `conf_threshold`默认值从0.001改为0.25,与Ultralytics验证模式保持一致,避免指标差异。

### 2. OCRDatasetEvaluator - OCR数据集评估器

```python
from onnxtools.eval import OCRDatasetEvaluator, SampleEvaluation, load_label_file

class OCRDatasetEvaluator:
    """OCR数据集评估器,提供完全匹配率、编辑距离和相似度指标"""

    def __init__(self, ocr_model):
        """初始化评估器

        Args:
            ocr_model: OcrORT实例
        """
        pass

    def evaluate_dataset(
        self,
        label_file: str,
        dataset_base_path: str,
        conf_threshold: float = 0.5,
        max_images: Optional[int] = None,
        output_format: str = 'table',  # 'table' or 'json'
        min_width: int = 40
    ) -> Dict[str, Any]:
        """评估OCR模型性能

        Returns:
            Dict包含:
            - accuracy: 完全匹配准确率 [0, 1]
            - normalized_edit_distance: 归一化编辑距离 [0, 1]
            - edit_distance_similarity: 编辑距离相似度 [0, 1]
            - total_samples: 总样本数
            - evaluated_samples: 评估样本数
            - filtered_samples: 过滤样本数
            - skipped_samples: 跳过样本数
            - evaluation_time: 总评估时间(秒)
            - avg_inference_time_ms: 平均推理时间(毫秒)
            - per_sample_results: 每个样本的详细结果列表
        """
        pass

@dataclass
class SampleEvaluation:
    """单个样本评估结果

    Attributes:
        image_path: 图像路径
        ground_truth: 真实标签文本
        predicted_text: 预测文本
        confidence: 置信度
        is_correct: 是否完全匹配
        edit_distance: 编辑距离
        normalized_edit_distance: 归一化编辑距离
    """
    pass

def load_label_file(label_file: str, dataset_base_path: str) -> List[Tuple[str, str]]:
    """加载标签文件

    支持两种格式:
    1. 单张图像: image_path<TAB>ground_truth
    2. 多张图像(JSON): ["img1.jpg", "img2.jpg"]<TAB>ground_truth

    Returns:
        (图像路径, 真实文本)元组列表
    """
    pass
```

## 模块结构

```
onnxtools/eval/
├── __init__.py           # 导出DatasetEvaluator, OCRDatasetEvaluator, SampleEvaluation
├── eval_coco.py          # COCO/YOLO数据集评估器
└── eval_ocr.py           # OCR数据集评估器
```

## 关键依赖和配置

### 核心依赖
- `onnxtools.infer_onnx.BaseORT` - 检测器基类
- `onnxtools.utils.detection_metrics` - mAP计算
- `onnxtools.utils.ocr_metrics` - OCR指标计算
- `python-levenshtein>=0.25.0` - 编辑距离计算

### 数据集格式要求

**COCO/YOLO数据集结构**:
```
dataset_path/
├── images/
│   ├── test/      # 优先级1
│   ├── val/       # 优先级2
│   └── train/     # 优先级3(回退)
├── labels/
│   ├── test/
│   ├── val/
│   └── train/
└── classes.yaml   # 类别名称映射
```

**YOLO标签格式** (labels/*.txt):
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.2
1 0.7 0.3 0.1 0.05
```

**OCR标签文件格式** (val.txt/train.txt):
```
image_path<TAB>ground_truth
plates/001.jpg<TAB>京A12345
["img1.jpg", "img2.jpg"]<TAB>京B54321
```

## 数据模型

### COCO评估结果
```python
coco_results = {
    'map50': float,              # mAP@IoU=0.5
    'map50_95': float,           # mAP@IoU=0.5:0.95
    'map75': float,              # mAP@IoU=0.75
    'precision': float,          # 平均精确率
    'recall': float,             # 平均召回率
    'f1_score': float,           # F1分数
    'per_class_ap': Dict[int, float],  # 每个类别的AP
    'speed_preprocess': float,   # 预处理平均时间(ms)
    'speed_inference': float,    # 推理平均时间(ms)
    'speed_postprocess': float,  # 后处理平均时间(ms)
    'total_images': int,         # 评估图像总数
}
```

### OCR评估结果
```python
ocr_results = {
    'accuracy': float,                      # 完全匹配准确率 [0, 1]
    'normalized_edit_distance': float,      # 归一化编辑距离 [0, 1]
    'edit_distance_similarity': float,      # 1 - normalized_edit_distance
    'total_samples': int,                   # 总样本数
    'evaluated_samples': int,               # 实际评估样本数
    'filtered_samples': int,                # 置信度过滤样本数
    'skipped_samples': int,                 # 跳过样本数(读取失败等)
    'evaluation_time': float,               # 总评估时间(秒)
    'avg_inference_time_ms': float,         # 平均推理时间(毫秒)
    'per_sample_results': List[Dict],       # 每个样本的详细结果
}

# per_sample_results中的单个样本格式
sample_result = {
    'image_path': str,
    'ground_truth': str,
    'predicted_text': str,
    'confidence': float,
    'is_correct': bool,
    'edit_distance': int,
    'normalized_edit_distance': float
}
```

## 测试和质量

### 测试覆盖
- **集成测试**: `tests/integration/test_ocr_evaluation_integration.py` (8个测试)
  - 端到端评估流程(table和JSON格式)
  - 参数验证(max_images、置信度阈值扫描)
  - 边界情况处理(缺失图像、损坏图像)
  - 性能测试(<1秒处理5张图像)

- **合约测试**: `tests/contract/test_ocr_evaluator_contract.py` (11个测试)
  - 基础评估流程合约(返回格式、数值范围)
  - 编辑距离指标合约(完美匹配、部分匹配、per_sample_results)
  - 置信度过滤合约(阈值行为、样本守恒)
  - JSON导出格式合约(有效性、必需字段)
  - 表格对齐合约(中文列名、数值格式)

- **单元测试**: `tests/unit/test_ocr_metrics.py` (23个测试)
  - 编辑距离边界情况
  - 中文字符处理
  - 真实OCR场景(常见混淆、噪声)

### 测试通过率
- 集成测试: 100% (8/8)
- 合约测试: 100% (11/11)
- 单元测试: 100% (23/23)

### 运行测试
```bash
# OCR评估集成测试
pytest tests/integration/test_ocr_evaluation_integration.py -v

# OCR评估合约测试
pytest tests/contract/test_ocr_evaluator_contract.py -v

# OCR指标单元测试
pytest tests/unit/test_ocr_metrics.py -v
```

## 常见问题 (FAQ)

### Q: 为什么COCO评估的conf_threshold默认值是0.25而不是0.001?
A: 为了与Ultralytics验证模式保持一致。Ultralytics在验证过程中会自动将过低的置信度阈值(0.001)重置为0.25,如果使用0.001会导致评估结果与Ultralytics存在显著差异。详见`eval_coco.py:61-73`注释。

### Q: OCR评估支持哪些输出格式?
A: 支持两种格式:
- `table`: 终端表格输出,中文列名,数值对齐
- `json`: JSON格式输出,包含per_sample_results详细信息

### Q: 如何处理OCR标签文件中的多张图像?
A: 支持JSON数组格式,例如`["img1.jpg", "img2.jpg"]<TAB>京A12345`,会自动展开为两条独立记录。参见`load_label_file`函数。

### Q: 如何排除特定的样本?
A: COCO评估器支持:
- `exclude_files`: 排除特定文件名列表
- `exclude_labels_containing`: 排除标签包含特定内容的样本

OCR评估器支持:
- `conf_threshold`: 过滤低置信度样本
- `min_width`: 过滤宽度小于阈值的图像

### Q: 评估时如何优化性能?
A:
1. 使用`max_images`参数限制评估样本数量
2. 使用TensorRT引擎代替ONNX模型
3. 调整`conf_threshold`和`iou_threshold`减少后处理开销
4. 考虑批处理推理(需要修改评估器代码)

### Q: OCR评估指标的含义是什么?
A:
- **accuracy**: 完全匹配率,预测文本与真实文本完全一致的样本比例
- **normalized_edit_distance**: 归一化编辑距离,值越小越好(0表示完美匹配)
- **edit_distance_similarity**: 1 - normalized_edit_distance,值越大越好(1表示完美匹配)

## 相关文件列表

### 核心模块文件
- `onnxtools/eval/__init__.py` - 模块入口,导出公共API
- `onnxtools/eval/eval_coco.py` - COCO数据集评估器实现(273行)
- `onnxtools/eval/eval_ocr.py` - OCR数据集评估器实现(359行)

### 工具脚本
- `tools/eval.py` - COCO评估命令行工具
- `tools/eval_ocr.py` - OCR评估命令行工具

### 依赖模块
- `onnxtools/utils/detection_metrics.py` - mAP计算和打印
- `onnxtools/utils/ocr_metrics.py` - OCR指标计算(编辑距离、相似度)

### 测试文件
- `tests/integration/test_ocr_evaluation_integration.py` - OCR评估集成测试
- `tests/contract/test_ocr_evaluator_contract.py` - OCR评估合约测试
- `tests/unit/test_ocr_metrics.py` - OCR指标单元测试

### 配置文件
- `configs/det_config.yaml` - 检测类别和颜色配置
- `configs/plate.yaml` - OCR字典和映射配置

## 使用示例

### COCO数据集评估示例
```bash
# 命令行评估
python tools/eval.py \
    --model-type rtdetr \
    --model-path models/rtdetr.onnx \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25 \
    --iou-threshold 0.7 \
    --max-images 1000

# Python API
from onnxtools import create_detector, DatasetEvaluator

detector = create_detector('yolo', 'models/yolo11n.onnx', conf_thres=0.25)
evaluator = DatasetEvaluator(detector)
results = evaluator.evaluate_dataset(
    dataset_path='/data/coco',
    max_images=500,
    exclude_files=['corrupt.jpg', 'invalid.jpg']
)

print(f"mAP@0.5: {results['map50']:.3f}")
print(f"mAP@0.5:0.95: {results['map50_95']:.3f}")
print(f"Inference: {results['speed_inference']:.1f}ms")
```

### OCR数据集评估示例
```bash
# 命令行评估(表格输出)
python -m onnxtools.eval.eval_ocr \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5 \
    --output-format table

# 命令行评估(JSON输出)
python tools/eval_ocr.py \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --output-format json > results.json

# Python API
from onnxtools import OcrORT, OCRDatasetEvaluator
import yaml

with open('configs/plate.yaml') as f:
    config = yaml.safe_load(f)

ocr_model = OcrORT('models/ocr.onnx', character=config['ocr_dict'])
evaluator = OCRDatasetEvaluator(ocr_model)

results = evaluator.evaluate_dataset(
    label_file='data/val.txt',
    dataset_base_path='data/',
    conf_threshold=0.7,
    max_images=100,
    output_format='json'
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Normalized Edit Distance: {results['normalized_edit_distance']:.3f}")
print(f"Evaluated: {results['evaluated_samples']}/{results['total_samples']}")

# 访问每个样本的详细结果
for sample in results['per_sample_results'][:5]:
    print(f"{sample['image_path']}: {sample['predicted_text']} vs {sample['ground_truth']}")
```

## 变更日志 (Changelog)

**2025-11-07** - 创建评估子模块文档
- 初始化完整的eval子模块文档
- 记录COCO和OCR评估器的API和使用方法
- 补充测试覆盖信息和常见问题
- 添加数据集格式和输出模型说明

**2025-10-11** - Bug修复和改进
- 修复OCR评估器JSON数组格式支持
- 新增12个单元测试用例覆盖JSON数组边界情况

**2025-10-10** - OCR评估功能完成
- OCRDatasetEvaluator完整实现
- 三大指标: 完全匹配率、归一化编辑距离、编辑距离相似度
- 表格对齐终端输出 + JSON导出格式
- 42个测试用例(11个合约 + 8个集成 + 23个单元)

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/onnxtools/eval/`*
*最后更新: 2025-11-13 20:30:00*
