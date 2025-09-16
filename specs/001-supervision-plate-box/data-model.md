# Data Model: Supervision库可视化数据模型

**Date**: 2025-09-15
**Phase**: Phase 1 - Design
**Feature**: 使用Supervision库增强可视化功能

## 数据模型概览

本文档定义了supervision库集成所需的核心数据结构，确保与现有detection pipeline的兼容性，同时支持enhanced可视化功能。

## 1. 检测结果数据模型

### 1.1 现有检测格式 (Legacy Format)
```python
# 当前项目使用的检测格式
DetectionTuple = Tuple[float, float, float, float, float, int]
# 格式: (x1, y1, x2, y2, confidence, class_id)

DetectionBatch = List[List[DetectionTuple]]
# 格式: [[detection1, detection2, ...]]  # 批次检测结果
```

### 1.2 Supervision检测格式 (Target Format)
```python
import supervision as sv
import numpy as np
from typing import Optional, Dict, Any

# supervision.Detections标准格式
@dataclass
class SupervisionDetections:
    xyxy: np.ndarray          # shape: (n, 4), bbox coordinates
    confidence: np.ndarray    # shape: (n,), confidence scores
    class_id: np.ndarray      # shape: (n,), class identifiers
    data: Dict[str, Any]      # Additional metadata
```

### 1.3 格式转换映射
```python
# 转换规则
LegacyToSupervision = {
    'xyxy': 'detections[:4]',      # 坐标直接映射
    'confidence': 'detections[4]',  # 置信度直接映射
    'class_id': 'detections[5]',   # 类别ID直接映射
    'data': {                       # 元数据生成
        'class_name': 'class_names[class_id]',
        'detection_index': 'range(len(detections))'
    }
}
```

## 2. OCR结果数据模型

### 2.1 车牌OCR结果结构
```python
@dataclass
class PlateOCRResult:
    """车牌OCR识别结果"""
    plate_text: str                    # 车牌号码文本
    color: str                         # 车牌颜色 (蓝牌/绿牌/黄牌等)
    layer: str                         # 车牌层级 (单层/双层)
    confidence: float                  # OCR置信度
    should_display_ocr: bool          # 是否显示OCR结果

    # 位置信息 (可选)
    text_position: Optional[Tuple[int, int]] = None
    text_bbox: Optional[Tuple[int, int, int, int]] = None

# OCR结果批次
PlateOCRBatch = List[Optional[PlateOCRResult]]
```

### 2.2 OCR结果验证规则
```python
class PlateOCRValidator:
    """车牌OCR结果验证"""

    @staticmethod
    def is_valid_plate_text(text: str) -> bool:
        """验证车牌号码格式"""
        # 中国车牌格式验证逻辑
        return len(text) >= 6 and len(text) <= 8

    @staticmethod
    def is_valid_color(color: str) -> bool:
        """验证车牌颜色"""
        valid_colors = {'蓝牌', '绿牌', '黄牌', '白牌', '黑牌', 'unknown'}
        return color in valid_colors

    @staticmethod
    def is_valid_layer(layer: str) -> bool:
        """验证车牌层级"""
        valid_layers = {'单层', '双层', 'unknown'}
        return layer in valid_layers
```

## 3. 可视化配置数据模型

### 3.1 Supervision注释器配置
```python
@dataclass
class BoxAnnotatorConfig:
    """边界框注释器配置 (基于最新API)"""
    color: Union[sv.Color, sv.ColorPalette] = sv.ColorPalette.DEFAULT
    thickness: int = 2                              # 默认厚度为2
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS

@dataclass
class RichLabelAnnotatorConfig:
    """RichLabel注释器配置 (基于最新API)"""
    color: Union[sv.Color, sv.ColorPalette] = sv.ColorPalette.DEFAULT
    text_color: Union[sv.Color, sv.ColorPalette] = sv.Color.WHITE
    font_path: Optional[str] = "SourceHanSans-VF.ttf"
    font_size: int = 10                             # 默认字体大小为10
    text_padding: int = 10                          # 默认内边距为10
    text_position: sv.Position = sv.Position.TOP_LEFT
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS
    border_radius: int = 0                          # 默认无圆角
    smart_position: bool = False                    # 默认关闭智能位置

@dataclass
class VisualizationConfig:
    """整体可视化配置 (Updated)"""
    box_config: BoxAnnotatorConfig
    label_config: RichLabelAnnotatorConfig        # 使用RichLabelAnnotator
    enable_ocr_display: bool = True
    fallback_to_pil: bool = True
    performance_logging: bool = False
    use_smart_position: bool = True                # 启用智能位置
    metadata_support: bool = True                  # 支持metadata属性
```

### 3.2 输出格式配置
```python
@dataclass
class OutputFormatConfig:
    """输出格式配置"""
    output_mode: str = "show"  # show/save/stream
    save_path: Optional[str] = None
    video_fps: int = 30
    video_codec: str = "mp4v"
    image_format: str = "jpg"
    quality: int = 95

# 输出模式枚举
class OutputMode(Enum):
    SHOW = "show"      # 实时显示
    SAVE = "save"      # 保存文件
    STREAM = "stream"  # 视频流
```

## 4. 性能监控数据模型

### 4.1 性能指标结构
```python
@dataclass
class DrawingPerformanceMetrics:
    """绘制性能指标"""
    draw_time_ms: float              # 绘制耗时(毫秒)
    detection_count: int             # 检测对象数量
    ocr_count: int                   # OCR对象数量
    image_resolution: Tuple[int, int] # 图像分辨率
    backend_used: str                # 使用的后端 (supervision/pil)

    # 计算属性
    @property
    def objects_per_second(self) -> float:
        """每秒处理对象数"""
        if self.draw_time_ms > 0:
            return (self.detection_count * 1000) / self.draw_time_ms
        return 0.0

@dataclass
class PerformanceBenchmark:
    """性能基准测试结果"""
    pil_metrics: DrawingPerformanceMetrics
    supervision_metrics: DrawingPerformanceMetrics
    improvement_ratio: float
    test_timestamp: str
    test_conditions: Dict[str, Any]
```

## 5. 错误处理数据模型

### 5.1 异常类定义
```python
class SupervisionIntegrationError(Exception):
    """Supervision集成相关异常基类"""
    pass

class DetectionFormatError(SupervisionIntegrationError):
    """检测格式转换异常"""
    def __init__(self, detection_data, message="Invalid detection format"):
        self.detection_data = detection_data
        super().__init__(f"{message}: {detection_data}")

class FontLoadError(SupervisionIntegrationError):
    """字体加载异常"""
    def __init__(self, font_path, message="Failed to load font"):
        self.font_path = font_path
        super().__init__(f"{message}: {font_path}")

class OCRDisplayError(SupervisionIntegrationError):
    """OCR显示异常"""
    def __init__(self, ocr_data, message="Failed to display OCR"):
        self.ocr_data = ocr_data
        super().__init__(f"{message}: {ocr_data}")
```

### 5.2 回退机制数据
```python
@dataclass
class FallbackContext:
    """回退上下文信息"""
    original_backend: str           # 原始后端
    fallback_backend: str           # 回退后端
    error_message: str              # 错误信息
    fallback_timestamp: str         # 回退时间
    success_after_fallback: bool    # 回退后是否成功
```

## 6. API合约数据模型

### 6.1 函数签名规范
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DrawingFunction(Protocol):
    """绘制函数协议"""

    def __call__(
        self,
        image: np.ndarray,
        detections: List[List[DetectionTuple]],
        class_names: List[str],
        colors: List[Tuple[int, int, int]],
        plate_results: Optional[PlateOCRBatch] = None,
        font_path: str = "SourceHanSans-VF.ttf"
    ) -> np.ndarray:
        """
        绘制检测结果到图像上

        Args:
            image: 输入图像 (BGR格式)
            detections: 检测结果批次
            class_names: 类别名称列表
            colors: 颜色列表
            plate_results: 车牌OCR结果 (可选)
            font_path: 字体文件路径

        Returns:
            np.ndarray: 绘制后的图像 (BGR格式)
        """
        ...
```

### 6.2 返回值规范
```python
@dataclass
class DrawingResult:
    """绘制结果"""
    annotated_image: np.ndarray              # 标注后图像
    performance_metrics: DrawingPerformanceMetrics  # 性能指标
    backend_used: str                        # 使用的后端
    errors: List[str]                        # 错误信息列表
    warnings: List[str]                      # 警告信息列表

    @property
    def is_success(self) -> bool:
        """是否成功绘制"""
        return len(self.errors) == 0
```

## 7. 配置文件数据模型

### 7.1 YAML配置结构
```yaml
# supervision_config.yaml
supervision:
  enabled: true
  fallback_to_pil: true
  performance_logging: true

  box_annotator:
    thickness: 3
    color_lookup: "CLASS"  # CLASS/INDEX/TRACK

  label_annotator:
    font_path: "SourceHanSans-VF.ttf"
    font_size: 16
    text_position: "TOP_LEFT"  # TOP_LEFT/CENTER/BOTTOM_RIGHT等
    smart_position: true
    border_radius: 3

  colors:
    vehicle: "#FF0000"  # 红色
    plate: "#00FF00"    # 绿色

  performance:
    benchmark_iterations: 100
    log_performance: false
    target_fps: 30
```

### 7.2 配置加载器
```python
@dataclass
class SupervisionConfigLoader:
    """Supervision配置加载器"""

    @staticmethod
    def load_from_yaml(config_path: str) -> VisualizationConfig:
        """从YAML文件加载配置"""
        # 实现配置加载逻辑
        pass

    @staticmethod
    def get_default_config() -> VisualizationConfig:
        """获取默认配置"""
        return VisualizationConfig(
            box_config=BoxAnnotatorConfig(),
            label_config=LabelAnnotatorConfig(),
            enable_ocr_display=True,
            fallback_to_pil=True,
            performance_logging=False
        )
```

## 8. 数据流验证

### 8.1 输入验证规则
```python
class InputValidator:
    """输入数据验证器"""

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """验证输入图像格式"""
        return (
            isinstance(image, np.ndarray) and
            len(image.shape) == 3 and
            image.shape[2] == 3 and
            image.dtype == np.uint8
        )

    @staticmethod
    def validate_detections(detections: List[List[DetectionTuple]]) -> bool:
        """验证检测结果格式"""
        if not isinstance(detections, list) or len(detections) == 0:
            return False

        for batch in detections:
            for det in batch:
                if not (isinstance(det, (list, tuple)) and len(det) == 6):
                    return False
        return True

    @staticmethod
    def validate_class_names(class_names: List[str]) -> bool:
        """验证类别名称"""
        return (
            isinstance(class_names, list) and
            len(class_names) > 0 and
            all(isinstance(name, str) for name in class_names)
        )
```

### 8.2 输出验证规则
```python
class OutputValidator:
    """输出数据验证器"""

    @staticmethod
    def validate_annotated_image(image: np.ndarray, original_shape: Tuple[int, int, int]) -> bool:
        """验证标注后图像"""
        return (
            isinstance(image, np.ndarray) and
            image.shape == original_shape and
            image.dtype == np.uint8
        )

    @staticmethod
    def validate_performance_metrics(metrics: DrawingPerformanceMetrics) -> bool:
        """验证性能指标"""
        return (
            metrics.draw_time_ms >= 0 and
            metrics.detection_count >= 0 and
            metrics.ocr_count >= 0 and
            len(metrics.image_resolution) == 2
        )
```

## 9. 数据模型总结

### 9.1 关键实体关系
```
DetectionTuple → SupervisionDetections (转换)
PlateOCRResult → LabelAnnotation (映射)
VisualizationConfig → Annotators (配置)
DrawingResult ← Performance + Error (组合)
```

### 9.2 数据流向
```
Input: [Image + Detections + OCR]
  ↓ (Validation)
Transform: [Format Conversion + Config Loading]
  ↓ (Supervision Processing)
Process: [Box Drawing + Label Annotation]
  ↓ (Quality Check)
Output: [Annotated Image + Metrics + Status]
```

### 9.3 兼容性保证
1. **向后兼容**: 保持现有API签名不变
2. **格式兼容**: 支持原有tuple格式检测结果
3. **输出兼容**: 保持BGR numpy数组输出格式
4. **性能兼容**: 提供fallback机制确保稳定性

---

**数据模型设计完成**: 所有核心数据结构已定义，支持完整的supervision集成流程。