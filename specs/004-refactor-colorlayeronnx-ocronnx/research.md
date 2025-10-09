# Research Report: 重构ColorLayerONNX和OCRONNX以继承BaseOnnx

**Feature**: 004-refactor-colorlayeronnx-ocronnx
**Date**: 2025-10-09
**Status**: Completed

## Executive Summary

本研究报告针对ColorLayerONNX和OCRONNX重构项目的5个关键研究任务进行了深入调查，解决了技术不确定性，验证了测试覆盖充分性，并为Phase 1设计提供了坚实基础。

**关键发现**:
1. 🔴 **测试覆盖不足**（R1）：当前无OCR和颜色分类的单元测试，存在高风险
2. 🔴 **依赖范围扩大**（R2）：除utils/pipeline.py外，还有MCP模块和tools目录依赖OCR函数
3. ✅ **混合模式明确**（R3）：YoloOnnx提供了清晰的实现模式参考
4. ✅ **类型提示标准**（R4）：定义了numpy数组和复杂返回值的类型注解规范
5. ⚠️ **拆分验证需求**（R5）：双层车牌逻辑复杂，需golden test验证

---

## R1: 测试覆盖充分性审查 🔴 高优先级

### 研究目标
确认OCR和颜色分类是否有充分测试，为SC-001（100%现有测试通过）提供可行性评估。

### 调查结果

#### 1.1 测试目录结构
```
tests/
├── integration/        # 13个集成测试文件（主要针对Annotator功能）
│   ├── test_ocr_integration.py         # ✅ 存在OCR集成测试
│   ├── test_pipeline_integration.py    # ✅ 存在pipeline测试
│   └── ... (其他annotator测试)
├── performance/        # 2个性能测试文件（仅annotator基准测试）
└── contract/           # 3个合约测试文件（仅annotator合约）
```

#### 1.2 OCR相关测试现状
**找到的测试文件**:
- `tests/integration/test_ocr_integration.py` - OCR集成测试（存在）
- `tests/integration/test_pipeline_integration.py` - 端到端管道测试（可能包含OCR）

**缺失的测试**:
- ❌ **无`tests/unit/`目录** - 缺少单元测试层
- ❌ **无ColorLayerONNX单元测试** - 颜色/层级分类无独立测试
- ❌ **无OCRONNX单元测试** - OCR推理无独立测试
- ❌ **无OCR预处理测试** - `process_plate_image`, `resize_norm_img`无测试
- ❌ **无OCR后处理测试** - `decode`, `get_ignored_tokens`无测试

#### 1.3 风险评估

| 风险类别 | 级别 | 描述 | 影响 |
|---------|------|------|------|
| 缺少单元测试 | 🔴 关键 | OCR和颜色分类无单元测试覆盖 | SC-001无法验证，重构风险极高 |
| 集成测试不全面 | 🟡 中 | 仅有pipeline端到端测试，无法定位具体函数问题 | 难以快速定位回归 |
| 无性能基准 | 🟡 中 | 无OCR推理时间基准，SC-003/SC-006无法验证 | 性能回归不可检测 |
| 无合约测试 | 🟡 中 | 无OCRONNX/ColorLayerONNX API合约测试 | 接口变更无法检测 |

### 决策与建议

**决策**: **必须在重构前补充测试**

**行动计划**:
1. **Phase 0补救**（必须）:
   - 创建`tests/unit/`目录
   - 编写`test_ocr_onnx.py`基线测试（至少覆盖`infer()`方法）
   - 编写`test_color_layer_onnx.py`基线测试
   - 使用现有模型文件运行基线测试，记录golden outputs

2. **Phase 1测试扩展**（推荐）:
   - 为所有迁移函数编写单元测试（`_process_plate_image_static`, `_decode_static`等）
   - 创建合约测试验证API不变性
   - 添加性能基准测试（pytest-benchmark）

3. **测试数据准备**:
   - 收集5-10张真实车牌图像（单层+双层）
   - 记录重构前的OCR输出作为golden reference
   - 准备边界情况图像（倾斜、模糊、遮挡）

**备选方案被拒**:
- ❌ **跳过测试直接重构**：风险太高，违反宪法原则V（TDD）
- ❌ **仅依赖集成测试**：无法快速定位问题根源，调试效率低

---

## R2: pipeline.py依赖识别 🔴 高优先级

### 研究目标
全面识别utils/ocr_*.py的所有调用者，确保FR-018（同步修改所有调用者）的完整性。

### 调查结果

#### 2.1 依赖文件清单（通过代码搜索）

**主要调用者**:
1. **utils/pipeline.py** (主要)
   - 行6-12: 导入`process_plate_image`, `image_pretreatment`, `resize_norm_img`, `decode`
   - 行224: `image_pretreatment(img_rgb)`
   - 行237: `process_plate_image(plate_img, is_double_layer=is_double)`
   - 行238: `resize_norm_img(processed_plate)`
   - 行242: `decode(character, preds_idx, preds_prob, is_remove_duplicate=True)`

2. **utils/__init__.py**
   - 导出这些函数供外部使用
   - 需要移除或重定向导出

3. **MCP模块** (发现的额外依赖)
   - `mcp_vehicle_detection/services/detection_service.py`
   - `mcp_vehicle_detection/mcp_utils/image_processor.py`
   - `mcp_vehicle_detection/mcp_utils/__init__.py`
   - `mcp_vehicle_detection/mcp_utils/validation.py`
   - **影响**: MCP模块也依赖utils函数，需同步修改

4. **tools/network_postprocess.py**
   - 可能用于模型调试和后处理验证
   - 需要确认具体依赖

5. **mcp_vehicle_detection/server.py**
   - MCP服务器可能间接依赖

#### 2.2 依赖类型分析

| 依赖文件 | 依赖函数 | 调用次数 | 修改复杂度 |
|---------|---------|---------|-----------|
| utils/pipeline.py | process_plate_image, resize_norm_img, image_pretreatment, decode | 约4处 | 🔴 高 - 核心逻辑 |
| utils/__init__.py | 导出所有OCR函数 | 导出声明 | 🟢 低 - 仅删除导出 |
| mcp_vehicle_detection/* | 可能通过utils导入 | 未知 | 🟡 中 - 需验证 |
| tools/network_postprocess.py | 可能用于调试 | 未知 | 🟢 低 - 工具脚本 |

### 决策与建议

**决策**: **分阶段修改，先core后MCP**

**修改策略**:

#### 策略1: pipeline.py重构（优先）
```python
# Before (使用utils函数)
from utils import process_plate_image, resize_norm_img, image_pretreatment, decode

color_input = image_pretreatment(img_rgb)
processed_plate = process_plate_image(plate_img, is_double_layer=is_double)
ocr_input = resize_norm_img(processed_plate)
ocr_result = decode(character, preds_idx, preds_prob, is_remove_duplicate=True)

# After (使用OCRONNX和ColorLayerONNX方法)
# 方案A: 直接调用类方法（推荐）
from infer_onnx import OCRONNX, ColorLayerONNX

color_input = ColorLayerONNX._image_pretreatment_static(img_rgb)
processed_plate = OCRONNX._process_plate_image_static(plate_img, is_double_layer=is_double)
ocr_input = OCRONNX._resize_norm_img_static(processed_plate)
preds_idx, preds_prob = ... # 从OCR推理输出获取
ocr_result = OCRONNX._decode_static(character, preds_idx, preds_prob, is_remove_duplicate=True)

# 方案B: 封装为pipeline辅助函数（可选）
# 创建PlateProcessor类（如plan.md中的Entity 3）
plate_processor = PlateProcessor(color_layer_model, ocr_model, character, plate_yaml)
result = plate_processor.process(plate_img)
```

**推荐方案A**，原因：
- 更直接，减少抽象层
- 符合"删除文件"要求，不引入新工具类
- 静态方法调用清晰明确

#### 策略2: MCP模块修改（次要）
- 先完成core模块重构并验证
- 再同步修改MCP模块的导入路径
- 如果MCP模块仅通过`from utils import`导入，修改成本低

#### 策略3: utils/__init__.py清理
```python
# Before
__all__ = [
    "process_plate_image",
    "resize_norm_img",
    "image_pretreatment",
    "decode",
    ...
]

from .ocr_image_processing import process_plate_image, resize_norm_img, image_pretreatment
from .ocr_post_processing import decode

# After
__all__ = [
    # 移除OCR相关函数导出
    ...
]

# 删除import语句
```

### 实施顺序

1. ✅ **Step 1**: 重构`infer_onnx/ocr_onnx.py`（添加静态方法）
2. ✅ **Step 2**: 修改`utils/pipeline.py`（改用静态方法）
3. ✅ **Step 3**: 测试pipeline端到端功能
4. ✅ **Step 4**: 修改`utils/__init__.py`（移除导出）
5. ✅ **Step 5**: 删除`utils/ocr_*.py`文件
6. ⏭️ **Step 6**: 验证MCP模块（如有问题，同步修改）
7. ⏭️ **Step 7**: 清理`tools/`脚本（如需要）

**备选方案被拒**:
- ❌ **保留utils函数作为包装器**：违反"删除文件"要求，增加维护负担
- ❌ **创建过渡兼容层**：与"无渐进式迁移"冲突

---

## R3: BaseOnnx混合模式最佳实践 🟡 中优先级

### 研究目标
参考yolo_onnx.py的实现模式，确保OCRONNX和ColorLayerONNX的实现一致性。

### 调查结果

#### 3.1 YoloOnnx混合模式分析

**核心模式**（基于yolo_onnx.py:50-88）:

```python
class YoloOnnx(BaseOnnx):
    # 1. 实例方法：对外接口，调用静态方法
    def _preprocess(self, image: np.ndarray) -> Tuple[...]:
        """实例方法，向后兼容，传递self的配置参数"""
        return self._preprocess_static(
            image,
            self.input_shape,              # 实例属性
            self.use_ultralytics_preprocess # 实例属性
        )

    # 2. 静态方法：无状态，可被TensorRT数据加载器复用
    @staticmethod
    def _preprocess_static(
        image: np.ndarray,
        input_shape: Tuple[int, int],
        use_ultralytics_preprocess: bool = False
    ) -> Tuple[...]:
        """静态方法，所有参数显式传递，无依赖self"""
        # 实际预处理逻辑
        if use_ultralytics_preprocess:
            ...
        else:
            ...
        return input_tensor, scale, original_shape, ratio_pad
```

**关键设计要点**:
1. **职责分离**:
   - 实例方法：封装实例状态（`self.input_shape`等），提供便捷接口
   - 静态方法：纯函数，接受所有参数，可独立测试和复用

2. **参数传递**:
   - 实例方法从`self`获取配置参数
   - 静态方法所有参数显式声明，无隐式依赖

3. **TensorRT支持**:
   - `engine_dataloader.py`直接调用`_preprocess_static()`
   - 无需创建实例，避免不必要的模型加载

#### 3.2 BaseOnnx抽象方法签名

**必须实现的抽象方法**:
```python
@abstractmethod
def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[np.ndarray]:
    """后处理抽象方法，子类需要实现"""
    pass
```

**可选覆盖的方法**:
```python
def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, tuple]:
    """预处理（实例方法，向后兼容）"""
    return self._preprocess_static(image, self.input_shape)

@staticmethod
def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple[...]:
    """预处理静态方法（默认实现）"""
    return preprocess_image(image, input_shape)
```

### 决策与建议

**决策**: **严格遵循YoloOnnx模式**

**OCRONNX实现模板**:
```python
class OCRONNX(BaseOnnx):
    def __init__(self, onnx_path: str, character: List[str],
                 input_shape: Tuple[int, int] = (48, 168), ...):
        super().__init__(onnx_path, input_shape, ...)
        self.character = character  # OCR字典

    # ===== 预处理 =====
    def _preprocess(self, image: np.ndarray, is_double_layer: bool = False) -> Tuple[...]:
        """实例方法：封装实例配置"""
        processed_plate = self._process_plate_image_static(image, is_double_layer)
        return self._resize_norm_img_static(processed_plate, [3, *self.input_shape])

    @staticmethod
    def _process_plate_image_static(img: np.ndarray, is_double_layer: bool) -> np.ndarray:
        """静态方法：双层车牌处理主逻辑"""
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = OCRONNX._detect_skew_angle(gray_img)
        corrected = OCRONNX._correct_skew(img, angle)

        if not is_double_layer:
            return corrected

        # 双层车牌分割拼接
        enhanced_gray = CLAHE_enhance(corrected)
        split_point = OCRONNX._find_optimal_split_line(enhanced_gray)
        return OCRONNX._stitch_double_layer(corrected, split_point)

    @staticmethod
    def _detect_skew_angle(gray_img: np.ndarray) -> float:
        """辅助方法：倾斜检测"""
        ...

    @staticmethod
    def _correct_skew(img: np.ndarray, angle: float) -> np.ndarray:
        """辅助方法：倾斜校正"""
        ...

    @staticmethod
    def _find_optimal_split_line(gray_img: np.ndarray) -> int:
        """辅助方法：分割线定位"""
        ...

    @staticmethod
    def _resize_norm_img_static(img: np.ndarray, image_shape: List[int]) -> np.ndarray:
        """静态方法：resize+归一化"""
        ...

    # ===== 后处理 =====
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[...]:
        """实例方法：调用静态decode方法"""
        preds_idx = np.argmax(prediction, axis=2)
        preds_prob = np.max(prediction, axis=2)
        return self._decode_static(self.character, preds_idx, preds_prob, is_remove_duplicate=True)

    @staticmethod
    def _decode_static(character: List[str], text_index: np.ndarray,
                       text_prob: Optional[np.ndarray], is_remove_duplicate: bool) -> List[...]:
        """静态方法：OCR解码"""
        ...

    @staticmethod
    def _get_ignored_tokens() -> List[int]:
        """辅助方法：忽略token"""
        return [0]
```

**关键模式遵循**:
- ✅ 实例方法调用静态方法（`_preprocess` → `_preprocess_static`）
- ✅ 静态方法接收所有必要参数（如`is_double_layer`）
- ✅ 辅助方法也是静态方法（`_detect_skew_angle`等）
- ✅ 实例方法传递`self`的属性（如`self.character`, `self.input_shape`）

---

## R4: 类型提示策略 🟡 中优先级

### 研究目标
定义完整的类型提示标准，满足宪法原则IV要求，提高代码可维护性。

### 调查结果

#### 4.1 Numpy数组类型提示最佳实践

**推荐方案**（基于PEP 484和numpy 1.20+）:
```python
from typing import List, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray  # numpy 1.20+

# 基础numpy数组类型
def process_image(img: NDArray[np.uint8]) -> NDArray[np.float32]:
    """明确指定dtype"""
    ...

# 复杂形状注解（使用注释）
def resize_norm_img(
    img: NDArray[np.uint8],  # shape: [H, W, C]
    image_shape: List[int]   # [C, H, W]
) -> NDArray[np.float32]:    # shape: [1, C, H, W]
    """
    Resize and normalize image

    Args:
        img: Input image (H, W, C) uint8
        image_shape: Target shape [C, H, W]

    Returns:
        Normalized tensor (1, C, H, W) float32
    """
    ...
```

**类型别名定义**:
```python
# 为复杂类型创建别名
from typing import TypeAlias

# OCR相关类型
OCRResult: TypeAlias = Tuple[str, float, List[float]]  # (text, avg_conf, char_confs)
OCRBatchResult: TypeAlias = List[OCRResult]

# 检测相关类型
BBox: TypeAlias = List[float]  # [x1, y1, x2, y2]
Detection: TypeAlias = NDArray[np.float32]  # shape: [N, 6] (x1,y1,x2,y2,conf,cls)

# 预处理输出
PreprocessResult: TypeAlias = Tuple[
    NDArray[np.float32],  # input_tensor
    float,                 # scale
    Tuple[int, int, int],  # original_shape (H, W, C)
    Optional[Tuple[Tuple[float, float], Tuple[float, float]]]  # ratio_pad
]
```

#### 4.2 OCRONNX和ColorLayerONNX类型提示标准

**完整类型提示示例**:
```python
class OCRONNX(BaseOnnx):
    def __init__(
        self,
        onnx_path: str,
        character: List[str],
        input_shape: Tuple[int, int] = (48, 168),
        conf_thres: float = 0.5,
        providers: Optional[List[str]] = None
    ) -> None:
        ...

    def _preprocess(
        self,
        image: NDArray[np.uint8],  # [H, W, 3]
        is_double_layer: bool = False
    ) -> Tuple[NDArray[np.float32], float, Tuple[int, int, int]]:
        ...

    @staticmethod
    def _process_plate_image_static(
        img: NDArray[np.uint8],  # [H, W, 3] BGR
        is_double_layer: bool = False,
        verbose: bool = False
    ) -> Optional[NDArray[np.uint8]]:  # [H', W', 3] or None
        ...

    @staticmethod
    def _resize_norm_img_static(
        img: NDArray[np.uint8],  # [H, W, 3]
        image_shape: List[int] = [3, 48, 168]
    ) -> NDArray[np.float32]:  # [1, C, H, W]
        ...

    @staticmethod
    def _decode_static(
        character: List[str],
        text_index: NDArray[np.int_],     # [B, seq_len]
        text_prob: Optional[NDArray[np.float32]] = None,  # [B, seq_len]
        is_remove_duplicate: bool = False
    ) -> List[OCRResult]:  # List[(text, avg_conf, char_confs)]
        ...

    def __call__(
        self,
        image: NDArray[np.uint8],  # [H, W, 3]
        is_double_layer: bool = False,
        conf_thres: Optional[float] = None
    ) -> Tuple[List[OCRResult], Tuple[int, int, int]]:
        ...
```

#### 4.3 mypy配置建议

**项目mypy配置** (`pyproject.toml` or `mypy.ini`):
```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True  # 严格模式：所有函数必须有类型提示
disallow_any_unimported = False  # 允许导入的第三方库无类型
ignore_missing_imports = True  # 忽略缺少类型的第三方库

# numpy相关
[mypy-numpy.*]
ignore_missing_imports = False  # numpy有类型提示

# onnxruntime无类型提示
[mypy-onnxruntime.*]
ignore_missing_imports = True

# cv2无类型提示
[mypy-cv2.*]
ignore_missing_imports = True
```

### 决策与建议

**决策**: **采用严格类型提示，使用TypeAlias简化**

**实施计划**:
1. **Phase 1重构时**:
   - 为所有新增方法添加完整类型提示
   - 使用`NDArray`而非裸`np.ndarray`
   - 在docstring中注释形状信息

2. **类型别名定义**:
   - 在`infer_onnx/type_aliases.py`创建类型别名文件
   - 导出`OCRResult`, `PreprocessResult`等常用类型

3. **mypy验证**:
   - 添加`mypy`到开发依赖
   - 在CI/CD中运行`mypy infer_onnx/`
   - 逐步提高严格性（先warn后error）

**备选方案被拒**:
- ❌ **使用旧式注释（# type: ...）**：Python 3.10+应使用PEP 484语法
- ❌ **跳过类型提示**：违反宪法原则IV

---

## R5: 双层车牌处理逻辑验证 🟢 低优先级

### 研究目标
验证拆分复杂逻辑（FR-019）的正确性，确保辅助方法边界清晰。

### 调查结果

#### 5.1 process_plate_image拆分设计

**现有代码结构**（utils/ocr_image_processing.py:58-94）:
```python
def process_plate_image(img, is_double_layer=False, verbose=False):
    """80+行单一函数"""
    # 1. 灰度转换
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 倾斜检测
    skew_angle = detect_skew_angle(gray_img)  # 已独立

    # 3. 倾斜校正
    corrected_img = correct_skew(img, skew_angle)  # 已独立

    if not is_double_layer:
        return corrected_img

    # 4. 对比度增强（双层特有）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray_img = clahe.apply(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))

    # 5. 分割线定位
    split_point = find_optimal_split_line(enhanced_gray_img)  # 已独立

    # 6. 分割上下两部分
    top_part = corrected_img[0:split_point, :]
    bottom_part = corrected_img[split_point:, :]

    # 7. 上层缩小50%宽度
    target_height = bottom_part.shape[0]
    top_resized = cv2.resize(top_part, (int(top_w * 0.5), target_height))

    # 8. 拼接
    stitched_plate = cv2.hconcat([top_resized, bottom_part])

    return stitched_plate
```

**拆分后的方法结构**:
```python
class OCRONNX:
    @staticmethod
    def _process_plate_image_static(img: np.ndarray, is_double_layer: bool) -> Optional[np.ndarray]:
        """主方法：编排子步骤"""
        # 1-3: 倾斜处理（单层+双层）
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = OCRONNX._detect_skew_angle(gray_img)
        corrected = OCRONNX._correct_skew(img, angle)

        if not is_double_layer:
            return corrected

        # 4-8: 双层处理
        enhanced_gray = OCRONNX._enhance_contrast(corrected)
        split_point = OCRONNX._find_optimal_split_line(enhanced_gray)
        return OCRONNX._stitch_double_layer(corrected, split_point)

    @staticmethod
    def _detect_skew_angle(gray_img: np.ndarray) -> float:
        """倾斜检测（18行）- 已独立"""
        ...

    @staticmethod
    def _correct_skew(img: np.ndarray, angle: float) -> np.ndarray:
        """倾斜校正（8行）- 已独立"""
        ...

    @staticmethod
    def _enhance_contrast(img: np.ndarray) -> np.ndarray:
        """对比度增强（3行）- 新拆分"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    @staticmethod
    def _find_optimal_split_line(gray_img: np.ndarray) -> int:
        """分割线定位（56行）- 已独立"""
        ...

    @staticmethod
    def _stitch_double_layer(img: np.ndarray, split_point: int) -> Optional[np.ndarray]:
        """双层拼接（15行）- 新拆分"""
        top_part = img[0:split_point, :]
        bottom_part = img[split_point:, :]

        # 验证分割有效性
        if top_part.size == 0 or bottom_part.size == 0:
            return None

        # 上层缩小50%宽度
        target_height = bottom_part.shape[0]
        top_h, top_w = top_part.shape[:2]
        target_top_width = int(top_w * top_h / target_height * 0.5)
        top_resized = cv2.resize(top_part, (target_top_width, target_height))

        return cv2.hconcat([top_resized, bottom_part])
```

**方法边界和职责**:
| 方法名 | 行数 | 职责 | 输入 | 输出 |
|--------|------|------|------|------|
| `_detect_skew_angle` | 18 | 使用Hough线检测倾斜角度 | 灰度图 | 角度(float) |
| `_correct_skew` | 8 | 应用仿射变换校正倾斜 | BGR图+角度 | 校正后BGR图 |
| `_enhance_contrast` | 3 | CLAHE对比度增强 | BGR图 | 增强后灰度图 |
| `_find_optimal_split_line` | 56 | 水平投影+高斯平滑定位分割线 | 灰度图 | 分割点y坐标 |
| `_stitch_double_layer` | 15 | 分割+缩放+拼接 | BGR图+分割点 | 拼接后BGR图 |
| `_process_plate_image_static` | 约15 | 编排上述5个方法 | BGR图+is_double | 处理后BGR图 |

#### 5.2 Golden Test验证计划

**测试用例设计**:
```python
def test_process_plate_image_golden():
    """Golden test: 验证重构前后输出一致"""
    # 1. 准备测试图像
    single_layer_plate = cv2.imread("test_data/single_layer.jpg")
    double_layer_plate = cv2.imread("test_data/double_layer.jpg")

    # 2. 记录重构前的输出（baseline）
    baseline_single = process_plate_image(single_layer_plate, is_double_layer=False)
    baseline_double = process_plate_image(double_layer_plate, is_double_layer=True)

    # 3. 重构后的输出
    refactored_single = OCRONNX._process_plate_image_static(single_layer_plate, False)
    refactored_double = OCRONNX._process_plate_image_static(double_layer_plate, True)

    # 4. 像素级比较
    np.testing.assert_array_equal(baseline_single, refactored_single)
    np.testing.assert_array_equal(baseline_double, refactored_double)
```

**中间状态验证**:
```python
def test_double_layer_intermediate_states():
    """验证双层车牌处理的中间状态"""
    img = cv2.imread("test_data/double_layer.jpg")

    # 倾斜检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = OCRONNX._detect_skew_angle(gray)
    assert -45 < angle < 45, f"异常倾斜角度: {angle}"

    # 倾斜校正
    corrected = OCRONNX._correct_skew(img, angle)
    assert corrected.shape == img.shape

    # 对比度增强
    enhanced = OCRONNX._enhance_contrast(corrected)
    assert enhanced.dtype == np.uint8
    assert len(enhanced.shape) == 2  # 灰度图

    # 分割线定位
    split_point = OCRONNX._find_optimal_split_line(enhanced)
    assert 0 < split_point < enhanced.shape[0]

    # 拼接
    stitched = OCRONNX._stitch_double_layer(corrected, split_point)
    assert stitched is not None
    assert stitched.shape[0] == corrected.shape[0] - split_point  # 高度=下层高度
```

### 决策与建议

**决策**: **采用拆分方案，通过golden test验证**

**拆分优势**:
1. **可测试性**: 每个辅助方法可独立测试（如`test_detect_skew_angle`）
2. **可维护性**: 修改对比度增强算法时仅改`_enhance_contrast`
3. **可读性**: `_process_plate_image_static`成为编排方法，逻辑清晰
4. **可复用性**: 其他模块可能复用`_detect_skew_angle`

**Golden Test必要性**:
- 双层车牌处理是OCR的核心功能，像素级差异可能导致OCR失败
- 拆分可能引入边界条件bug（如分割点计算误差）
- Golden test提供回归检测保障

**实施建议**:
1. **Phase 0**: 收集5张双层车牌图像，运行现有代码记录输出
2. **Phase 1**: 实现拆分方法，通过golden test验证
3. **Phase 1**: 添加中间状态单元测试
4. **Phase 2**: 扩展边界情况测试（倾斜极端角度、分割线失败等）

**备选方案被拒**:
- ❌ **保持单一方法**：违反单一职责原则，难以测试和维护
- ❌ **过度拆分**（如每5行拆一个方法）：过度设计，降低可读性

---

## Final Recommendations

### 关键行动项（Phase 1前必须完成）

1. 🔴 **R1补救**（最高优先级）:
   - [ ] 创建`tests/unit/`目录
   - [ ] 编写`test_ocr_onnx.py`和`test_color_layer_onnx.py`基线测试
   - [ ] 运行测试并记录golden outputs
   - [ ] 确保基线测试100%通过

2. 🔴 **R2验证**（高优先级）:
   - [ ] 使用grep全面搜索MCP模块的OCR函数依赖
   - [ ] 制定MCP模块修改计划（如需要）
   - [ ] 预估总体修改工作量（utils + MCP）

3. ✅ **R3/R4应用**（Phase 1实施）:
   - [ ] 严格遵循YoloOnnx混合模式
   - [ ] 为所有方法添加完整类型提示
   - [ ] 创建`type_aliases.py`文件

4. ⏭️ **R5验证**（Phase 1后半段）:
   - [ ] 收集双层车牌测试图像
   - [ ] 实施golden test
   - [ ] 验证拆分正确性

### 风险矩阵

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 无单元测试导致回归 | 🔴 严重 | 高 | R1补救：立即创建基线测试 |
| MCP模块同步修改复杂 | 🟡 中 | 中 | R2预研：先修改core，再同步MCP |
| 双层车牌拆分引入bug | 🟡 中 | 低 | R5验证：golden test保障 |
| 类型提示不完整 | 🟢 低 | 低 | R4标准：强制mypy检查 |

### Phase 1就绪状态

**✅ 可以进入Phase 1的条件**:
- ✅ 混合模式实现模板明确（R3）
- ✅ 类型提示标准定义清晰（R4）
- ✅ 双层车牌拆分方案可行（R5）
- ⚠️ **需完成R1补救**（创建基线测试）
- ⚠️ **需完成R2验证**（MCP依赖确认）

**建议**: 并行执行R1补救和Phase 1设计，MCP修改可延后到Phase 2

---

*研究报告完成 | Phase 0 Ready for Phase 1 | 2025-10-09*
