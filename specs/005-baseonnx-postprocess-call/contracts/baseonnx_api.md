# API Contract: BaseOnnx推理基类

**Branch**: `005-baseonnx-postprocess-call` | **Date**: 2025-10-09 | **Spec**: [spec.md](../spec.md)

## Overview

本文档定义BaseOnnx推理基类的API合约,包括抽象方法、模板方法和阶段方法的接口规范、行为约定、错误处理和性能保证。

**合约版本**: 1.0.0
**生效日期**: 2025-10-09
**适用范围**: 所有BaseOnnx子类 (YoloOnnx, RTDETROnnx, RFDETROnnx, ColorLayerONNX, OCRONNX)

## Contract Principles

### 1. 接口稳定性
- **抽象方法签名**: 一旦定义,不可修改 (向后兼容)
- **模板方法**: `__call__()`签名保持稳定,仅内部优化
- **阶段方法**: 默认实现可修改,但签名保持兼容

### 2. 强制实现保证
- 所有子类**必须**实现`_postprocess()`和`_preprocess_static()`
- 实例化未实现抽象方法的子类时,立即抛出`TypeError`
- 错误消息格式统一,提供清晰的实现指导

### 3. 行为一致性
- 所有子类的`__call__()`方法行为一致 (模板方法模式)
- 3个阶段方法按固定顺序执行: prepare -> execute -> finalize
- 推理结果格式统一: `(detections, original_shape)` 元组

## API Contracts

### Contract 1: Abstract Methods (抽象方法合约)

#### 1.1 `_postprocess()`

**接口签名**:
```python
@abstractmethod
def _postprocess(
    self,
    prediction: List[np.ndarray],
    conf_thres: float,
    **kwargs
) -> List[np.ndarray]:
    """
    Post-process model outputs into final detection/classification results.

    Args:
        prediction: Raw model outputs, list of numpy arrays
        conf_thres: Confidence threshold for filtering results
        **kwargs: Additional parameters (e.g., iou_thres, max_det)

    Returns:
        List of post-processed results, each element is a numpy array

    Raises:
        NotImplementedError: If not implemented by subclass
    """
    raise NotImplementedError(
        f"{self.__class__.__name__}._postprocess() must be implemented by subclass. "
        "This method is responsible for post-processing model outputs."
    )
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **输入验证** | `prediction`必须是非空列表,元素为`np.ndarray` | `assert isinstance(prediction, list) and len(prediction) > 0` |
| **置信度范围** | `conf_thres`必须在[0, 1]范围内 | `assert 0 <= conf_thres <= 1` |
| **返回值格式** | 返回`List[np.ndarray]`,长度与`prediction`相同或过滤后 | `assert isinstance(result, list)` |
| **空结果处理** | 无有效检测时,返回空数组`[np.empty((0, 6))]` | `assert result[0].shape[1] == 6` (检测任务) |
| **性能要求** | 后处理时间 < 10ms (640x640输入, 100个对象) | Pytest benchmark测试 |
| **异常处理** | 数据格式错误时抛出`ValueError`,类型错误抛出`TypeError` | 合约测试验证 |

**子类实现示例**:
```python
class YoloOnnx(BaseOnnx):
    def _postprocess(self, prediction, conf_thres, **kwargs):
        """YOLO后处理: NMS + 置信度过滤"""
        # 1. 输入验证
        if not isinstance(prediction, list):
            raise TypeError(f"Expected list, got {type(prediction)}")
        if not (0 <= conf_thres <= 1):
            raise ValueError(f"conf_thres must be in [0, 1], got {conf_thres}")

        # 2. NMS处理
        iou_thres = kwargs.get('iou_thres', self.iou_thres)
        results = []
        for pred in prediction:
            detections = non_max_suppression(pred, conf_thres, iou_thres)
            results.append(detections if detections.size > 0 else np.empty((0, 6)))

        return results
```

**合约测试**:
```python
def test_postprocess_contract(detector_instance):
    """验证_postprocess()合约"""
    # 1. 正常输入
    prediction = [np.random.rand(100, 85)]  # YOLO格式
    result = detector_instance._postprocess(prediction, conf_thres=0.5)
    assert isinstance(result, list)
    assert result[0].shape[1] == 6  # xyxy + conf + class

    # 2. 边界值
    empty_result = detector_instance._postprocess([np.empty((0, 85))], 0.5)
    assert empty_result[0].shape == (0, 6)

    # 3. 异常情况
    with pytest.raises(TypeError):
        detector_instance._postprocess("invalid", 0.5)

    with pytest.raises(ValueError):
        detector_instance._postprocess(prediction, 1.5)  # 越界
```

#### 1.2 `_preprocess_static()`

**接口签名**:
```python
@staticmethod
@abstractmethod
def _preprocess_static(
    image: np.ndarray,
    input_shape: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple]:
    """
    Static preprocessing method for image transformation.

    Args:
        image: Input image in BGR format, shape [H, W, C]
        input_shape: Target input size (height, width)

    Returns:
        Tuple containing:
            - input_tensor: Preprocessed tensor, shape [1, 3, H, W], range [0, 1]
            - scale: Scaling information (scale_x, scale_y) or padding info

    Raises:
        NotImplementedError: If not implemented by subclass
        ValueError: If image dimensions are invalid
    """
    raise NotImplementedError(
        f"BaseOnnx._preprocess_static() must be implemented by subclass. "
        "This static method is responsible for image preprocessing."
    )
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **装饰器顺序** | 必须是`@staticmethod` -> `@abstractmethod` (外 -> 内) | 代码审查 |
| **输入图像格式** | `image.shape` 为 `(H, W, 3)`,BGR格式,`dtype=uint8` | `assert image.ndim == 3 and image.shape[2] == 3` |
| **输入尺寸格式** | `input_shape`为2元素元组 `(height, width)` | `assert len(input_shape) == 2` |
| **返回张量形状** | `input_tensor.shape == (1, 3, H, W)`,批次维度为1 | `assert result[0].shape[0] == 1` |
| **返回张量范围** | `input_tensor`值域为[0, 1]或[-1, 1] (归一化) | `assert 0 <= result[0].max() <= 1` |
| **返回缩放信息** | `scale`为元组,包含缩放比例或填充信息 | `assert isinstance(result[1], tuple)` |
| **性能要求** | 预处理时间 < 5ms (1920x1080 -> 640x640) | Pytest benchmark测试 |
| **异常处理** | 图像维度错误时抛出`ValueError` | 合约测试验证 |

**子类实现示例**:
```python
class RTDETROnnx(BaseOnnx):
    @staticmethod
    @abstractmethod
    def _preprocess_static(image, input_shape):
        """RT-DETR预处理: letterbox + BGR2RGB + 归一化"""
        # 1. 输入验证
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_shape (H, W), got {input_shape}")

        # 2. Letterbox resize (保持宽高比)
        resized, scale = letterbox_resize(image, input_shape)

        # 3. BGR -> RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 4. 归一化到 [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0

        # 5. NCHW格式
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        return input_tensor, scale
```

**合约测试**:
```python
def test_preprocess_static_contract():
    """验证_preprocess_static()合约"""
    # 1. 正常输入
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    input_tensor, scale = RTDETROnnx._preprocess_static(image, (640, 640))

    assert input_tensor.shape == (1, 3, 640, 640)
    assert 0 <= input_tensor.max() <= 1
    assert isinstance(scale, tuple)

    # 2. 异常情况
    with pytest.raises(ValueError):
        RTDETROnnx._preprocess_static(np.random.rand(1080, 1920), (640, 640))  # 缺少通道维度

    with pytest.raises(ValueError):
        RTDETROnnx._preprocess_static(image, (640,))  # 输入尺寸错误
```

### Contract 2: Template Method (模板方法合约)

#### 2.1 `__call__()`

**接口签名**:
```python
def __call__(
    self,
    image: np.ndarray,
    conf_thres: Optional[float] = None,
    **kwargs
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Main inference entry point (template method).

    This method defines the inference pipeline skeleton and should NOT be
    overridden by subclasses unless there's a very special need.

    Args:
        image: Input image in BGR format, shape [H, W, C]
        conf_thres: Confidence threshold, overrides instance default if provided
        **kwargs: Additional parameters passed to _postprocess()

    Returns:
        Tuple containing:
            - detections: List of detection results (numpy arrays)
            - original_shape: Original image size (height, width)

    Raises:
        TypeError: If image is not a numpy array
        ValueError: If conf_thres is out of range [0, 1]
        RuntimeError: If model inference fails
    """
    # Stage 1: Prepare
    self._prepare_inference(image, conf_thres, **kwargs)

    # Stage 2: Execute
    self._execute_inference(self._context.input_tensor)

    # Stage 3: Finalize
    detections = self._finalize_inference(
        self._context.raw_outputs,
        self._context.scale,
        self._context.original_shape,
        self._context.conf_thres,
        **kwargs
    )

    return detections, self._context.original_shape
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **重写限制** | 子类**不应该**重写此方法 (除非极特殊需求) | 代码审查 + 合约测试 |
| **执行顺序** | 必须按顺序调用3个阶段: prepare -> execute -> finalize | 合约测试验证 |
| **输入验证** | `image`必须是`np.ndarray`,`conf_thres`可选但必须在[0, 1] | `assert isinstance(image, np.ndarray)` |
| **返回格式** | 返回2元素元组: `(List[np.ndarray], Tuple[int, int])` | `assert len(result) == 2` |
| **性能保证** | 总延迟 < 50ms (640x640输入, GPU推理) | 性能测试验证 |
| **异常处理** | 捕获并重新抛出清晰的异常消息 | 合约测试验证 |
| **状态管理** | 每次调用自动重置推理上下文 | 多次推理测试 |

**合约测试**:
```python
def test_call_template_contract(detector_instance):
    """验证__call__()模板方法合约"""
    # 1. 正常推理
    image = cv2.imread("test.jpg")
    detections, original_shape = detector_instance(image, conf_thres=0.5)

    assert isinstance(detections, list)
    assert isinstance(original_shape, tuple)
    assert len(original_shape) == 2
    assert original_shape == (image.shape[0], image.shape[1])

    # 2. 参数覆盖
    detections2, _ = detector_instance(image, conf_thres=0.7, iou_thres=0.5)
    assert len(detections2) <= len(detections)  # 更高阈值,结果更少

    # 3. 异常情况
    with pytest.raises(TypeError):
        detector_instance("invalid_image", 0.5)

    with pytest.raises(ValueError):
        detector_instance(image, conf_thres=1.5)

    # 4. 性能保证
    import time
    start = time.perf_counter()
    detector_instance(image)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 50, f"Inference took {elapsed:.2f}ms, expected < 50ms"
```

**子类重写限制验证**:
```python
def test_call_not_overridden():
    """验证子类不应重写__call__"""
    # 检查方法来源
    assert YoloOnnx.__call__ is BaseOnnx.__call__, "YoloOnnx should not override __call__"
    assert RTDETROnnx.__call__ is BaseOnnx.__call__, "RTDETROnnx should not override __call__"
    # ... 对所有子类验证
```

### Contract 3: Stage Methods (阶段方法合约)

#### 3.1 `_prepare_inference()`

**接口签名**:
```python
def _prepare_inference(
    self,
    image: np.ndarray,
    conf_thres: Optional[float],
    **kwargs
) -> None:
    """
    Stage 1: Prepare inference (model initialization, preprocessing, validation).

    BaseOnnx provides a default implementation. Subclasses can override to add
    custom preparation logic.

    Args:
        image: Input image in BGR format
        conf_thres: Confidence threshold
        **kwargs: Additional parameters

    Raises:
        ValueError: If input validation fails
        RuntimeError: If model initialization fails
    """
    # Default implementation in BaseOnnx
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **默认实现** | BaseOnnx提供默认实现,子类可选重写 | 文档验证 |
| **职责范围** | 仅负责准备,不执行推理 | 合约测试验证 |
| **状态保存** | 必须保存`original_shape`, `input_tensor`, `scale`, `conf_thres`到context | 断点调试验证 |
| **模型初始化** | 必须调用`_ensure_initialized()`确保模型加载 | 合约测试验证 |
| **预处理调用** | 必须调用`_preprocess_static()`进行图像预处理 | 合约测试验证 |
| **异常传播** | 预处理错误应向上传播,不吞没异常 | 异常测试 |

**默认实现** (BaseOnnx):
```python
def _prepare_inference(self, image, conf_thres, **kwargs):
    """BaseOnnx默认实现"""
    # 1. 确保模型已初始化
    self._ensure_initialized()

    # 2. 重置推理上下文
    if hasattr(self, '_context'):
        self._context.reset()

    # 3. 保存原始图像尺寸
    original_shape = (image.shape[0], image.shape[1])
    if hasattr(self, '_context'):
        self._context.original_shape = original_shape
    else:
        self._original_shape = original_shape

    # 4. 执行预处理
    input_tensor, scale = self._preprocess_static(image, self.input_shape)

    # 5. 保存预处理结果
    if hasattr(self, '_context'):
        self._context.input_tensor = input_tensor
        self._context.scale = scale
        self._context.conf_thres = conf_thres or self.conf_thres
    else:
        self._input_tensor = input_tensor
        self._scale = scale
        self._conf_thres = conf_thres or self.conf_thres
```

**子类重写示例** (OCRONNX):
```python
class OCRONNX(BaseOnnx):
    def _prepare_inference(self, image, conf_thres, **kwargs):
        """OCR特殊准备: 双层车牌检测"""
        # 1. 调用父类默认实现
        super()._prepare_inference(image, conf_thres, **kwargs)

        # 2. OCR特殊逻辑
        is_double_layer = kwargs.get('is_double_layer', False)
        if is_double_layer:
            # 双层车牌需要特殊预处理
            self._context.input_tensor = self._preprocess_double_layer(image)
```

**合约测试**:
```python
def test_prepare_inference_contract(detector_instance):
    """验证_prepare_inference()合约"""
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # 1. 默认实现正常执行
    detector_instance._prepare_inference(image, 0.5)

    # 2. 验证状态保存
    if hasattr(detector_instance, '_context'):
        assert detector_instance._context.original_shape == (1080, 1920)
        assert detector_instance._context.input_tensor is not None
        assert detector_instance._context.scale is not None
        assert detector_instance._context.conf_thres == 0.5

    # 3. 验证模型初始化
    assert detector_instance._runner is not None

    # 4. 验证预处理调用
    # (通过mock验证_preprocess_static被调用)
```

#### 3.2 `_execute_inference()`

**接口签名**:
```python
def _execute_inference(self, input_tensor: np.ndarray) -> None:
    """
    Stage 2: Execute ONNX inference using Polygraphy.

    BaseOnnx provides a default implementation. Subclasses rarely need to override
    unless there's special inference logic (e.g., multi-scale inference).

    Args:
        input_tensor: Preprocessed input tensor, shape [1, 3, H, W]

    Raises:
        RuntimeError: If Polygraphy inference fails
    """
    # Default implementation in BaseOnnx
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **默认实现** | BaseOnnx提供默认实现,子类极少重写 | 文档验证 |
| **职责范围** | 仅负责推理执行,不处理预处理和后处理 | 合约测试验证 |
| **状态保存** | 必须保存`raw_outputs`到context | 断点调试验证 |
| **Polygraphy调用** | 必须使用`self._runner.infer()`执行推理 | 合约测试验证 |
| **batch维度处理** | 支持单batch和多batch,自动处理维度 | 合约测试验证 |
| **异常传播** | Polygraphy错误应向上传播,不吞没异常 | 异常测试 |

**默认实现** (BaseOnnx):
```python
def _execute_inference(self, input_tensor):
    """BaseOnnx默认实现"""
    # 1. Polygraphy推理调用
    raw_outputs = self._runner.infer(feed_dict={self._runner.input_names[0]: input_tensor})

    # 2. 提取输出张量
    outputs = [raw_outputs[name] for name in self._runner.output_names]

    # 3. 处理batch维度
    processed_outputs = []
    for output in outputs:
        if output.shape[0] == 1:
            # 单batch: 移除batch维度
            processed_outputs.append(output.squeeze(0))
        else:
            # 多batch: 保持原样
            processed_outputs.append(output)

    # 4. 保存输出结果
    if hasattr(self, '_context'):
        self._context.raw_outputs = processed_outputs
    else:
        self._raw_outputs = processed_outputs
```

**合约测试**:
```python
def test_execute_inference_contract(detector_instance):
    """验证_execute_inference()合约"""
    # 1. 准备输入张量
    input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # 2. 确保模型已初始化
    detector_instance._ensure_initialized()

    # 3. 执行推理
    detector_instance._execute_inference(input_tensor)

    # 4. 验证状态保存
    if hasattr(detector_instance, '_context'):
        assert detector_instance._context.raw_outputs is not None
        assert isinstance(detector_instance._context.raw_outputs, list)

    # 5. 验证batch维度处理
    # (通过检查输出形状验证)
```

#### 3.3 `_finalize_inference()`

**接口签名**:
```python
def _finalize_inference(
    self,
    outputs: List[np.ndarray],
    scale: Tuple,
    original_shape: Tuple[int, int],
    conf_thres: float,
    **kwargs
) -> List[np.ndarray]:
    """
    Stage 3: Finalize inference (post-processing, coordinate transformation, filtering).

    BaseOnnx provides a default implementation. Subclasses can override to add
    custom post-processing logic.

    Args:
        outputs: Raw model outputs
        scale: Scaling information from preprocessing
        original_shape: Original image size (height, width)
        conf_thres: Confidence threshold
        **kwargs: Additional parameters for _postprocess()

    Returns:
        List of finalized detections

    Raises:
        ValueError: If post-processing fails
    """
    # Default implementation in BaseOnnx
```

**合约条款**:

| 项目 | 约定 | 验证方法 |
|------|------|----------|
| **默认实现** | BaseOnnx提供默认实现,子类可重写定制后处理 | 文档验证 |
| **职责范围** | 负责后处理、坐标转换、结果过滤 | 合约测试验证 |
| **后处理调用** | 必须调用`_postprocess()`获取检测结果 | 合约测试验证 |
| **坐标转换** | 必须将检测框从模型空间转换到原图空间 | 合约测试验证 |
| **结果过滤** | 必须过滤空检测或无效结果 | 合约测试验证 |
| **返回格式** | 返回`List[np.ndarray]`,空结果返回`[np.empty((0, 6))]` | 合约测试验证 |

**默认实现** (BaseOnnx):
```python
def _finalize_inference(self, outputs, scale, original_shape, conf_thres, **kwargs):
    """BaseOnnx默认实现"""
    # 1. 调用子类后处理
    detections = self._postprocess(outputs, conf_thres, **kwargs)

    # 2. 坐标转换: 模型空间 -> 原图空间
    scaled_detections = []
    for det in detections:
        if det.size > 0:
            # 缩放边界框坐标
            det[:, :4] = self._rescale_boxes(det[:, :4], scale, original_shape)
        scaled_detections.append(det)

    # 3. 结果过滤
    filtered_detections = [det for det in scaled_detections if det.size > 0]

    return filtered_detections if filtered_detections else [np.empty((0, 6))]
```

**子类重写示例** (ColorLayerONNX):
```python
class ColorLayerONNX(BaseOnnx):
    def _finalize_inference(self, outputs, scale, original_shape, conf_thres, **kwargs):
        """颜色分类特殊完成: 返回颜色和层级"""
        # 1. 调用分类后处理
        color_logits, layer_logits = self._postprocess(outputs, conf_thres, **kwargs)

        # 2. Softmax + argmax
        color_idx = np.argmax(color_logits)
        layer_idx = np.argmax(layer_logits)

        # 3. 映射到类别名称
        color = self.color_map[color_idx]
        layer = self.layer_map[layer_idx]

        # 4. 计算置信度
        color_conf = np.max(color_logits)
        layer_conf = np.max(layer_logits)

        return [(color, layer, color_conf, layer_conf)]
```

**合约测试**:
```python
def test_finalize_inference_contract(detector_instance):
    """验证_finalize_inference()合约"""
    # 1. 准备模拟输出
    outputs = [np.random.rand(100, 85)]  # YOLO格式
    scale = (0.5, 0.5)
    original_shape = (1080, 1920)

    # 2. 执行完成阶段
    detections = detector_instance._finalize_inference(outputs, scale, original_shape, 0.5)

    # 3. 验证返回格式
    assert isinstance(detections, list)
    assert detections[0].shape[1] == 6  # xyxy + conf + class

    # 4. 验证坐标转换 (检测框应在原图范围内)
    if detections[0].size > 0:
        boxes = detections[0][:, :4]
        assert boxes[:, 0].min() >= 0  # x1 >= 0
        assert boxes[:, 1].min() >= 0  # y1 >= 0
        assert boxes[:, 2].max() <= original_shape[1]  # x2 <= W
        assert boxes[:, 3].max() <= original_shape[0]  # y2 <= H
```

## Error Handling Contracts

### 错误消息格式

所有抽象方法的`NotImplementedError`必须遵循统一格式:

**格式模板**:
```python
"{ClassName}.{method_name}() must be implemented by subclass. {Responsibility Description}"
```

**示例**:
```python
# _postprocess未实现
raise NotImplementedError(
    "YoloOnnx._postprocess() must be implemented by subclass. "
    "This method is responsible for post-processing model outputs."
)

# _preprocess_static未实现
raise NotImplementedError(
    "RTDETROnnx._preprocess_static() must be implemented by subclass. "
    "This static method is responsible for image preprocessing."
)
```

### 异常层次

| 异常类型 | 触发场景 | 示例 |
|---------|---------|------|
| `TypeError` | 实例化未实现抽象方法的子类 | `Can't instantiate abstract class Foo with abstract methods _postprocess` |
| `TypeError` | 参数类型错误 | `Expected np.ndarray, got <class 'str'>` |
| `ValueError` | 参数值超出范围 | `conf_thres must be in [0, 1], got 1.5` |
| `ValueError` | 图像维度错误 | `Expected image shape (H, W, 3), got (H, W)` |
| `RuntimeError` | 模型推理失败 | `Polygraphy inference failed: ...` |
| `RuntimeError` | 模型未初始化 | `Model not initialized, call _ensure_initialized() first` |

### 异常传播规则

1. **不吞没异常**: 所有方法必须向上传播异常,不使用裸`except:`
2. **添加上下文**: 重新抛出异常时添加有用的上下文信息
3. **清理资源**: 异常发生时,确保释放GPU内存和文件句柄

**示例**:
```python
def _execute_inference(self, input_tensor):
    try:
        raw_outputs = self._runner.infer(feed_dict={...})
    except Exception as e:
        # 添加上下文后重新抛出
        raise RuntimeError(f"Polygraphy inference failed for {self.onnx_path}: {e}") from e
```

## Performance Contracts

### 性能基准

| 指标 | 基准值 | 测试条件 | 合约要求 |
|------|--------|----------|----------|
| `__call__()` 总延迟 | < 50ms | 640x640输入, GPU, batch=1 | 强制 (合约测试验证) |
| `_prepare_inference()` | < 5ms | 同上 | 推荐 (性能测试) |
| `_execute_inference()` | < 30ms | 同上 | 推荐 (性能测试) |
| `_finalize_inference()` | < 10ms | 同上 | 推荐 (性能测试) |
| GPU内存占用 | < 2GB | 同上 | 强制 (集成测试验证) |

### 性能合约测试

```python
@pytest.mark.benchmark
def test_performance_contract(detector_instance, benchmark_image):
    """验证性能合约"""
    import time

    # 1. 总延迟测试
    start = time.perf_counter()
    detections, _ = detector_instance(benchmark_image, conf_thres=0.5)
    total_time = (time.perf_counter() - start) * 1000

    assert total_time < 50, f"Total inference time {total_time:.2f}ms exceeds 50ms"

    # 2. 分阶段测试
    start = time.perf_counter()
    detector_instance._prepare_inference(benchmark_image, 0.5)
    prepare_time = (time.perf_counter() - start) * 1000
    assert prepare_time < 5, f"Prepare time {prepare_time:.2f}ms exceeds 5ms"

    # ... (execute和finalize测试类似)

    # 3. GPU内存测试
    import subprocess
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    gpu_mem_mb = int(result.decode().strip())
    assert gpu_mem_mb < 2048, f"GPU memory {gpu_mem_mb}MB exceeds 2GB"
```

## Backward Compatibility

### 向后兼容性保证

| 项目 | 保证级别 | 说明 |
|------|---------|------|
| **抽象方法签名** | 强保证 | 一旦发布,签名不可修改 |
| **模板方法签名** | 强保证 | `__call__()`签名保持稳定 |
| **阶段方法签名** | 弱保证 | 默认实现可优化,但签名兼容 |
| **返回值格式** | 强保证 | `(detections, original_shape)`格式不变 |
| **异常类型** | 强保证 | 已定义的异常类型保持一致 |

### 废弃策略

**废弃方法标记** (如需废弃旧接口):
```python
import warnings

def infer(self, image, **kwargs):
    """旧版推理方法 (已废弃,使用__call__替代)"""
    warnings.warn(
        "infer() is deprecated, use __call__() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self(image, **kwargs)
```

**废弃周期**:
1. **v1.0**: 标记为deprecated,发出警告
2. **v1.1**: 保持兼容,文档标注废弃
3. **v2.0**: 移除废弃方法 (主版本升级)

## Contract Testing

### 合约测试套件

所有BaseOnnx子类必须通过以下合约测试:

```python
# tests/contract/test_baseonnx_contract.py

class TestBaseOnnxContract:
    """BaseOnnx合约测试基类"""

    @pytest.fixture
    def detector(self):
        """子类必须提供此fixture"""
        raise NotImplementedError("Subclass must provide detector fixture")

    def test_abstract_methods_implemented(self, detector):
        """合约1: 验证抽象方法已实现"""
        assert hasattr(detector, '_postprocess')
        assert hasattr(detector, '_preprocess_static')
        assert callable(detector._postprocess)
        assert callable(detector._preprocess_static)

    def test_call_signature(self, detector):
        """合约2: 验证__call__签名"""
        import inspect
        sig = inspect.signature(detector.__call__)
        params = list(sig.parameters.keys())
        assert params[:3] == ['self', 'image', 'conf_thres']

    def test_call_returns_tuple(self, detector, test_image):
        """合约3: 验证__call__返回格式"""
        result = detector(test_image, conf_thres=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], tuple)

    def test_performance_contract(self, detector, benchmark_image):
        """合约4: 验证性能要求"""
        import time
        start = time.perf_counter()
        detector(benchmark_image, conf_thres=0.5)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"Inference took {elapsed:.2f}ms"

    def test_error_handling(self, detector):
        """合约5: 验证异常处理"""
        with pytest.raises(TypeError):
            detector("invalid_image", 0.5)

        with pytest.raises(ValueError):
            detector(np.random.rand(100, 100, 3), conf_thres=1.5)

    def test_stage_methods_exist(self, detector):
        """合约6: 验证阶段方法存在"""
        assert hasattr(detector, '_prepare_inference')
        assert hasattr(detector, '_execute_inference')
        assert hasattr(detector, '_finalize_inference')

# 子类合约测试示例
class TestYoloOnnxContract(TestBaseOnnxContract):
    """YOLO合约测试"""

    @pytest.fixture
    def detector(self):
        return YoloOnnx(onnx_path='models/yolo11n.onnx', conf_thres=0.5)

# ... (其他子类合约测试类似)
```

## Contract Versioning

**当前版本**: 1.0.0
**发布日期**: 2025-10-09

### 版本变更规则

- **主版本 (1.x.x)**: 破坏性变更 (如修改抽象方法签名)
- **次版本 (x.1.x)**: 新增功能 (如新增阶段方法)
- **补丁版本 (x.x.1)**: Bug修复和性能优化

### 合约变更日志

**v1.0.0** (2025-10-09):
- 初始版本
- 定义`_postprocess()`和`_preprocess_static()`抽象方法合约
- 定义`__call__()`模板方法合约
- 定义3个阶段方法合约
- 定义错误处理和性能合约

---

*Contract generated: 2025-10-09*
*Next steps: 创建quickstart.md提供快速入门指南*
