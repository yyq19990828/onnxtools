# 目标检测模型评估指南

本系统提供了完整的目标检测模型评估功能，支持YOLO格式数据集的标准指标计算。

## 系统架构

```
onnx_vehicle_plate_recognition/
├── onnxtools/
│   ├── eval/
│   │   ├── eval_coco.py          # COCO数据集评估器 (DatasetEvaluator)
│   │   ├── eval_ocr.py           # OCR数据集评估器
│   │   └── eval_cls.py           # 分类数据集评估器 (ClsDatasetEvaluator)
│   ├── infer_onnx/
│   │   ├── onnx_base.py          # BaseORT 检测基类
│   │   ├── onnx_cls.py           # BaseClsORT 分类基类
│   │   ├── onnx_yolo.py          # YoloORT
│   │   ├── onnx_rtdetr.py        # RtdetrORT
│   │   └── onnx_rfdetr.py        # RfdetrORT
│   └── utils/
│       └── ocr_metrics.py        # OCR评估指标
├── tools/
│   ├── eval.py                   # 检测模型评估CLI
│   ├── eval_cls.py               # 分类模型评估CLI
│   └── eval_ocr.py               # OCR模型评估CLI
└── tests/                        # 测试套件
```

## 主要特性

1. **标准检测指标**: mAP@0.5、mAP@0.5:0.95、精度、召回率、F1分数
2. **多模型支持**: 原生支持YOLO、RT-DETR、RF-DETR，可通过转换函数支持任意模型
3. **分类模型评估**: 支持CSV和ImageFolder格式，多分支模型，Precision/Recall/F1指标
4. **OCR评估**: 完全准确率、归一化编辑距离、编辑距离相似度
5. **YOLO数据集格式**: 完全兼容YOLO训练数据集格式
6. **输出格式转换**: 内置转换函数，支持不同模型的输出格式

## 快速开始

### 1. 基本评估（检测模型）

```python
from onnxtools import create_detector, DatasetEvaluator

# 初始化检测器 (支持 'yolo', 'rtdetr', 'rfdetr')
detector = create_detector(
    model_type='yolo',
    onnx_path='models/yolo_model.onnx',
    conf_thres=0.25
)

# 使用统一评估器
evaluator = DatasetEvaluator(detector)

# 运行评估
results = evaluator.evaluate_dataset(
    dataset_path="path/to/dataset",  # 包含images/和labels/文件夹
    conf_threshold=0.001,            # 评估时的置信度阈值
    max_images=1000                  # 限制评估图像数量（可选）
)

print(f"mAP@0.5: {results['map50']:.3f}")
print(f"mAP@0.5:0.95: {results['map']:.3f}")
```

#### CLI方式
```bash
python tools/eval.py \
    --model-type yolo \
    --model-path models/yolo_model.onnx \
    --dataset-path /path/to/coco \
    --conf-threshold 0.25
```

### 2. 分类模型评估

```python
from onnxtools import HelmetORT, ClsDatasetEvaluator
from onnxtools.eval.eval_cls import BranchConfig

# 初始化分类器
model = HelmetORT('models/helmet.onnx')
evaluator = ClsDatasetEvaluator(model)

# CSV格式评估
branch = BranchConfig(
    branch_index=0,
    column_name='helmet_missing',
    label_map={0: 'normal', 1: 'helmet_missing'},
    branch_name='helmet_missing'
)
results = evaluator.evaluate_dataset(
    csv_path='data/val.csv',
    image_dir='data/images/',
    branches=[branch],
    output_format='table'
)

# ImageFolder格式评估
results = evaluator.evaluate_dataset(
    dataset_dir='data/helmet_val/',  # dataset_dir/class_name/image.jpg
    output_format='table'
)
```

#### CLI方式
```bash
python tools/eval_cls.py \
    --model-type helmet \
    --model-path models/helmet.onnx \
    --csv-path data/val.csv \
    --image-dir data/images/ \
    --branches "helmet_missing:0:0=normal,1=helmet_missing" \
    --output-format table
```

### 3. OCR模型评估

```bash
python -m onnxtools.eval.eval_ocr \
    --label-file data/val.txt \
    --dataset-base data/ \
    --ocr-model models/ocr.onnx \
    --config configs/plate.yaml \
    --conf-threshold 0.5
```

指标：完全准确率、归一化编辑距离、编辑距离相似度。

## 数据集格式

### 目录结构（按优先级）

系统会按以下优先级自动检测YOLO数据集结构：

**1. Test数据集（最高优先级）**
```
dataset/
├── images/
│   └── test/            # 测试图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── test/            # 测试标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**2. Validation数据集（中等优先级）**
```
dataset/
├── images/
│   └── val/             # 验证图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── val/             # 验证标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**3. Train数据集（较低优先级）**
```
dataset/
├── images/
│   └── train/           # 训练图像文件
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   └── train/           # 训练标签文件
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**4. 根目录结构（最低优先级）**
```
dataset/
├── images/              # 图像文件（根目录）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/              # YOLO格式标签文件（根目录）
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── data.yaml           # 数据集配置文件（可选）
```

**注意**: 系统会自动按优先级检测，优先使用test数据集进行评估，这符合标准的模型评估实践。

### 标签格式
每个标签文件包含多行，每行格式为：
```
class_id center_x center_y width height
```
- 所有坐标值都是归一化的（0-1之间）
- `center_x, center_y`: 边界框中心点的归一化坐标
- `width, height`: 边界框的归一化宽度和高度

### 数据配置文件（可选）
```yaml
# data.yaml
names:
  0: person
  1: bicycle
  2: car
  # ... 更多类别
```

## 评估指标说明

### 核心指标

- **mAP@0.5**: IoU阈值为0.5时的平均精度均值
- **mAP@0.5:0.95**: IoU阈值从0.5到0.95（步长0.05）的平均精度均值
- **Precision (P)**: 精确度，预测为正例中实际为正例的比例
- **Recall (R)**: 召回率，实际正例中被正确预测的比例
- **F1-Score**: 精确度和召回率的调和平均数

### 计算过程

1. **IoU计算**: 计算预测框与真实框之间的交并比
2. **匹配策略**: 在不同IoU阈值下匹配预测和真实标签
3. **TP/FP统计**: 统计真正例和假正例
4. **AP计算**: 基于Precision-Recall曲线计算平均精度
5. **mAP汇总**: 对所有类别求平均得到mAP

## 支持的模型类型

### 检测模型 (tools/eval.py)

| 模型类型 | 推理类 | 默认输入尺寸 | 创建方式 |
|---------|--------|------------|---------|
| `yolo` | `YoloORT` | 640×640 | `create_detector('yolo', ...)` |
| `rtdetr` | `RtdetrORT` | 640×640 | `create_detector('rtdetr', ...)` |
| `rfdetr` | `RfdetrORT` | 576×576 | `create_detector('rfdetr', ...)` |

### 分类模型 (tools/eval_cls.py)

| 模型类型 | 推理类 | 默认输入尺寸 | Batch处理 |
|---------|--------|------------|----------|
| `helmet` | `HelmetORT` | 128×128 | 自动适配固定/动态batch |
| `color_layer` | `ColorLayerORT` | 48×168 | 自动适配 |
| `vehicle_attribute` | `VehicleAttributeORT` | 224×224 | 自动适配 |

> **Batch自动适配**: `BaseClsORT` 基类自动检测ONNX模型的batch维度。固定batch模型（如batch=4）会自动补全输入并截取结果；动态batch模型直接以batch=1推理。无需修改代码即可切换。

## 性能优化建议

1. **批量处理**: 系统支持批量处理多张图像
2. **置信度过滤**: 合理设置`conf_threshold`以平衡精度和召回率
3. **图像数量限制**: 使用`max_images`参数控制评估范围
4. **内存管理**: 大数据集评估时注意内存使用情况

## 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   ValueError: 图像目录不存在: /path/to/images
   ```
   解决：检查数据集路径是否正确，确保包含images/和labels/文件夹

2. **标签文件缺失**
   ```
   Warning: 无法找到标签文件
   ```
   解决：确保每个图像都有对应的标签文件，文件名相同（不含扩展名）

3. **模型输出格式不匹配**
   ```
   Error: 检测结果格式错误
   ```
   解决：检查输出转换函数是否正确，确保输出为[x1,y1,x2,y2,conf,class_id]格式

4. **内存不足**
   ```
   MemoryError: 内存不足
   ```
   解决：使用`max_images`参数限制评估图像数量，或减小batch size

### 调试技巧

1. **检查检测结果**：
   ```python
   result = detector(image)
   print(f"检测到 {len(result.boxes)} 个目标")
   print(f"类别: {result.class_ids}, 置信度: {result.scores}")
   ```

2. **检查分类结果**：
   ```python
   result = classifier(image)
   print(f"标签: {result.labels}, 置信度: {result.confidences}")
   print(f"Batch size: {classifier._expected_batch_size}")  # 查看自动检测的batch
   ```

3. **小规模测试**：
   ```bash
   python tools/eval.py --model-type yolo --model-path models/yolo.onnx \
       --dataset-path /path/to/dataset --max-images 10
   ```

## 扩展开发

### 添加新的检测模型评估支持

1. 在 `onnxtools/infer_onnx/` 中实现新推理类（继承 `BaseORT`）
2. 在 `onnxtools/__init__.py` 的 `create_detector()` 中注册
3. 在 `tools/eval.py` 中添加 model-type 选项

### 添加新的分类模型评估支持

1. 在 `onnxtools/infer_onnx/onnx_cls.py` 中实现新分类类（继承 `BaseClsORT`）
2. 在 `tools/eval_cls.py` 的 `create_model()` 中注册
3. Batch维度会由 `BaseClsORT` 自动处理

### 相关文档

- [模型支持列表](model_support_list.md) - 所有模型输入/输出规格
- [项目总览](../README.md)
- [推理引擎文档](../onnxtools/infer_onnx/CLAUDE.md)
- [评估模块文档](../onnxtools/eval/CLAUDE.md)
