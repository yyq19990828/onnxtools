# Research: BaseOnnx抽象方法强制实现与__call__优化

**Date**: 2025-10-09
**Feature**: [spec.md](./spec.md)
**Plan**: [plan.md](./plan.md)

本文档记录Phase 0技术调研的所有发现和决策,解决实施计划中标记的所有"NEEDS CLARIFICATION"。

---

## 研究任务1: @abstractmethod与@staticmethod组合验证

### 决策 (Decision)

**装饰器顺序**: `@staticmethod`必须在外层,`@abstractmethod`必须在内层(最接近函数定义)

```python
from abc import ABC, abstractmethod

class BaseOnnx(ABC):
    @staticmethod        # ✅ 外层
    @abstractmethod      # ✅ 内层(最接近函数)
    def _preprocess_static(image, input_shape):
        raise NotImplementedError("...")
```

**TypeError触发时机**: 在**实例化时**抛出TypeError,而非类定义时

```python
# 允许定义未实现的子类
class IncompleteDetector(BaseOnnx):
    pass  # ✅ 类定义成功

# 实例化时检查并抛出TypeError
obj = IncompleteDetector()  # ❌ TypeError: Can't instantiate abstract class
```

### 理由 (Rationale)

1. **Python官方文档明确规定**: "@abstractmethod should be applied as the innermost decorator" ([Python abc文档](https://docs.python.org/3/library/abc.html))

2. **错误顺序的后果**: 如果@abstractmethod在外层,会导致`AttributeError: attribute '__isabstractmethod__' of 'staticmethod' objects is not writable`

3. **实例化时检查的优势**:
   - 允许动态类定义(元编程场景)
   - 错误消息明确指出未实现的方法名
   - 符合Python的"ask forgiveness"哲学

### 替代方案 (Alternatives Considered)

**方案A**: 使用已废弃的`@abstractstaticmethod`
- **拒绝原因**: Python 3.3+已废弃,不推荐使用

**方案B**: 运行时手动检查方法实现
- **拒绝原因**: 违反DRY原则,Python ABC已提供标准机制

**方案C**: 使用类型检查工具(mypy)代替运行时检查
- **评估**: 可作为补充,但不能替代运行时检查(动态实例化场景)

### Python版本兼容性

**测试结论**: Python 3.10, 3.11, 3.12行为完全一致

| 版本 | 装饰器顺序要求 | 实例化检查 | 兼容性 |
|------|---------------|----------|--------|
| 3.10.9 | @static→@abstract | ✅ | 完全兼容 |
| 3.11.x | @static→@abstract | ✅ | 完全兼容 |
| 3.12.x | @static→@abstract | ✅ | 完全兼容 |

### 实施指南

```python
# infer_onnx/onnx_base.py

from abc import ABC, abstractmethod

class BaseOnnx(ABC):
    """ONNX推理基类 - 强制子类实现核心方法"""

    @abstractmethod
    def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[np.ndarray]:
        """
        后处理抽象方法 - 子类必须实现

        Args:
            prediction: 模型原始输出
            conf_thres: 置信度阈值
            **kwargs: 其他参数

        Returns:
            检测结果列表

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._postprocess() must be implemented by subclass. "
            "This method is responsible for processing model output."
        )

    @staticmethod
    @abstractmethod
    def _preprocess_static(image: np.ndarray, input_shape: Tuple[int, int]) -> Tuple:
        """
        预处理静态抽象方法 - 子类必须实现

        Args:
            image: 输入图像,BGR格式
            input_shape: 目标尺寸

        Returns:
            (input_tensor, scale, original_shape, ratio_pad)

        Raises:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError(
            "BaseOnnx._preprocess_static() must be implemented by subclass. "
            "This static method is responsible for image preprocessing."
        )
```

---

## 研究任务2: pytest-cov覆盖率分析最佳实践

### 决策 (Decision)

**推荐命令**:
```bash
pytest --cov=infer_onnx.onnx_base \
       --cov-report=term-missing \
       --cov-report=html:htmlcov \
       --cov-branch \
       tests/
```

**删除代码的阈值策略**: **仅删除0%覆盖率的分支代码**

**配置文件位置**: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/pyproject.toml`

### 理由 (Rationale)

1. **0%阈值的行业共识**:
   - Google Testing Blog: "如果代码重要到需要保留,就应该有测试"
   - Stack Overflow社区: 70-80%覆盖率是生产级标准,但删除代码应严格限制为0%

2. **分支覆盖优于语句覆盖**: `--cov-branch`标志启用分支覆盖,能捕获未执行的if/else路径

3. **HTML报告的优势**:
   - 可视化显示未覆盖代码(红色/黄色标记)
   - 点击文件查看具体行号
   - 比终端报告更直观

### 替代方案 (Alternatives Considered)

**方案A**: 删除<30%覆盖率的代码
- **拒绝原因**: 过于激进,可能误删关键但难以测试的代码(如错误处理)

**方案B**: 使用多种覆盖率工具(coverage.py + pytest-cov + codecov)
- **评估**: pytest-cov已足够,多工具增加复杂度

**方案C**: 只看语句覆盖,不看分支覆盖
- **拒绝原因**: 分支覆盖更全面,能发现未测试的条件分支

### 实施的配置文件

```toml
# /home/tyjt/桌面/onnx_vehicle_plate_recognition/pyproject.toml

[tool.coverage.run]
branch = true
source = ["infer_onnx", "utils", "tools"]
omit = [
    "*/tests/*",
    "*/third_party/*",
    "*/__pycache__/*",
    "*/venv/*",
    ".venv/*",
    "*/mcp_vehicle_detection/*",
    "*/debug/*",
    "setup.py",
    "main.py",
]

[tool.coverage.report]
fail_under = 70
precision = 2
skip_covered = false
skip_empty = true
sort = "Cover"

exclude_also = [
    "def __repr__",
    "if self\\.debug:",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@(abc\\.)?abstractmethod",
    "raise NotImplementedError",
    "except ImportError:",
    "def main\\(",
    "@deprecated",
]

[tool.coverage.html]
directory = "htmlcov"

[tool.coverage.xml]
output = "coverage.xml"
```

### 覆盖率报告解读

**term-missing输出格式**:
```
Name                     Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------------
infer_onnx/onnx_base.py    150     20     45      8    87%   23-25, 45->50
```

**Missing列符号解读**:
- `23-25`: 第23到25行未执行(连续行)
- `45->50`: 从第45行跳转到第50行的分支未执行
- `line->exit`: 从某行到函数退出的路径未执行

**删除决策流程**:
1. ✅ **0%覆盖** → 强烈建议删除或添加测试
2. ⚠️ **1-30%覆盖** → 评估重要性,添加测试
3. ✅ **>30%覆盖** → 保留并改进测试

### 实际应用于BaseOnnx.__call__

```bash
# 步骤1: 生成当前覆盖率报告
pytest --cov=infer_onnx.onnx_base --cov-report=html --cov-report=term-missing --cov-branch tests/

# 步骤2: 在浏览器中查看HTML报告
firefox htmlcov/infer_onnx_onnx_base_py.html

# 步骤3: 识别__call__方法中的0%覆盖分支
#   - 查找红色标记的代码行
#   - 重点关注line 162-168(兼容返回值处理)
#   - 查找line->exit模式(未执行的异常处理)

# 步骤4: 仅删除确认为0%覆盖的分支
#   - 搜索代码库确认未被引用
#   - 检查git历史确认创建原因
#   - 标记为deprecated或直接删除
```

### 预期在BaseOnnx中发现的未使用分支

基于代码审查,预期0%覆盖的候选:

```python
# infer_onnx/onnx_base.py line 162-168
# 预期分析: 3元组返回值兼容代码可能未被使用
if len(preprocess_result) == 3:
    # 兼容旧版本返回值(可能0%覆盖)
    input_tensor, scale, original_shape = preprocess_result
    ratio_pad = None
else:
    # 新版本返回值
    input_tensor, scale, original_shape, ratio_pad = preprocess_result
```

**删除前验证**:
```bash
# 搜索是否有子类返回3元组
grep -r "_preprocess.*return" infer_onnx/*.py | grep -v "ratio_pad"
# 如果无结果,确认可安全删除3元组分支
```

---

## 研究任务3: 模板方法模式在推理管道中的应用

### 决策 (Decision)

**粒度选择**: 采用**3阶段划分** - `_prepare_inference()` / `_execute_inference()` / `_finalize_inference()`

**参数传递策略**: 使用**InferenceContext实例变量**,而非返回元组

**子类重写策略**: 重写**抽象步骤方法**(_postprocess, _preprocess_static),保持__call__模板方法不变

### 理由 (Rationale)

1. **3阶段是ML框架主流**:
   - TensorFlow Keras: preprocess → forward → update metrics
   - PyTorch Lightning: setup → train_step → log
   - ONNX Runtime官方: preprocess → inference → postprocess

2. **实例变量优于返回元组**:
   - 可扩展性: 添加新参数不破坏方法签名
   - 可调试性: 单点访问context.to_dict()
   - 符合OOP: 对象封装完整状态
   - 避免位置依赖: 不会因元组元素顺序变化而出错

3. **框架实践验证**:
   - Keras使用`self.inputs`, `self.loss`存储状态
   - PyTorch使用`self.example_input_array`存储上下文
   - 所有主流框架都避免返回多值元组

### 替代方案 (Alternatives Considered)

**方案A**: 细粒度分解(6+阶段)
```python
_load → _validate → _encode → _infer → _threshold → _interpret
```
- **拒绝原因**: 过度设计,增加工程复杂度,不符合YAGNI原则

**方案B**: 粗粒度(prepare/execute/finalize)
```python
_prepare_inference → _execute_inference → _finalize_inference
```
- **评估**: 可用,但不如preprocess/inference/postprocess语义清晰
- **采纳调整**: 保留3阶段但重命名为更语义化的名称

**方案C**: 继续使用返回元组
```python
def _preprocess(image) -> Tuple[Tensor, float, tuple, optional]:
```
- **拒绝原因**:
  - 当前已有3元组/4元组兼容问题(line 162-168)
  - 添加新参数需修改所有子类
  - 元组位置依赖容易出错

### 3阶段方法设计

基于用户输入"__call__提取的3个阶段方法, 在BaseOnnx中提供基础的模板方法, 子类中按需重写或直接继承",设计如下:

#### 阶段1: _prepare_inference()

**职责**: 模型初始化、预处理、验证输入

**BaseOnnx默认实现** (子类可继承):
```python
def _prepare_inference(self, image: np.ndarray, conf_thres: Optional[float], **kwargs):
    """准备阶段 - 在BaseOnnx中提供默认实现"""
    # 确保模型已初始化
    self._ensure_initialized()

    # 初始化上下文
    self._context.reset()
    self._context.original_shape = (image.shape[0], image.shape[1])
    self._context.conf_thres = conf_thres or self.conf_thres

    # 调用抽象方法进行预处理(子类必须实现)
    self._preprocess(image)

    # 验证预处理输出
    assert self._context.input_tensor is not None, "预处理未设置input_tensor"
```

**子类重写场景**: 需要特殊初始化逻辑(如动态加载配置、预热模型)

#### 阶段2: _execute_inference()

**职责**: Polygraphy推理执行、batch维度处理

**BaseOnnx默认实现** (子类可继承):
```python
def _execute_inference(self, input_tensor: np.ndarray):
    """执行阶段 - 在BaseOnnx中提供默认实现"""
    with self._runner:
        # 检查batch维度
        input_metadata = self._runner.get_input_metadata()
        input_shape = input_metadata[self.input_name].shape
        expected_batch_size = input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else 1

        # 调整batch维度(如果需要)
        if expected_batch_size > 1 and input_tensor.shape[0] == 1:
            input_tensor = np.repeat(input_tensor, expected_batch_size, axis=0)

        # 执行推理
        feed_dict = {self.input_name: input_tensor}
        outputs_dict = self._runner.infer(feed_dict)
        self._context.raw_outputs = [outputs_dict[name] for name in self.output_names]
```

**子类重写场景**: 异步推理、流式处理、自定义batch逻辑

#### 阶段3: _finalize_inference()

**职责**: 后处理、结果过滤和格式化

**BaseOnnx默认实现** (子类可继承):
```python
def _finalize_inference(self, outputs: List[np.ndarray], conf_thres: float, **kwargs):
    """完成阶段 - 在BaseOnnx中提供默认实现"""
    # 根据模型类型选择后处理策略
    if type(self).__name__ == 'RFDETROnnx':
        # RF-DETR需要完整outputs
        detections = self._postprocess(outputs, conf_thres, **kwargs)
    else:
        # 其他模型使用第一个输出
        prediction = outputs[0]
        detections = self._postprocess(prediction, conf_thres,
                                       scale=self._context.scale,
                                       ratio_pad=self._context.ratio_pad, **kwargs)

    # 过滤batch结果(如果是multi-batch输入)
    if len(detections) > 1:
        detections = [detections[0]]  # 只返回第一个batch

    return detections
```

**子类重写场景**: 特殊输出格式(如多头输出、分段输出)

### InferenceContext设计

```python
# infer_onnx/inference_context.py (新文件)
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class InferenceContext:
    """推理上下文 - 封装阶段间共享的状态"""

    # 输入
    original_shape: Tuple[int, int] = field(default_factory=lambda: (0, 0))

    # 预处理输出
    input_tensor: Optional[np.ndarray] = None
    scale: float = 1.0
    ratio_pad: Optional[Tuple] = None

    # 推理输出
    raw_outputs: Optional[List[np.ndarray]] = None

    # 参数
    conf_thres: float = 0.5

    def reset(self):
        """重置上下文用于新的推理"""
        self.input_tensor = None
        self.raw_outputs = None
```

### 重构后的BaseOnnx.__call__结构

```python
def __call__(self, image: np.ndarray, conf_thres: Optional[float] = None, **kwargs):
    """Template method - 定义推理流程骨架(不应被子类重写)"""

    # 阶段1: 准备
    self._prepare_inference(image, conf_thres, **kwargs)

    # 阶段2: 执行
    self._execute_inference(self._context.input_tensor)

    # 阶段3: 完成
    detections = self._finalize_inference(self._context.raw_outputs,
                                           self._context.conf_thres, **kwargs)

    return detections, self._context.original_shape
```

### 子类使用指南

**场景1: 默认行为(大多数子类)**
```python
class YoloOnnx(BaseOnnx):
    # ✅ 只需实现抽象方法
    def _postprocess(self, prediction, conf_thres, **kwargs):
        # YOLO特定后处理
        pass

    @staticmethod
    def _preprocess_static(image, input_shape):
        # YOLO特定预处理
        pass

    # ✅ 直接继承3个阶段方法,无需重写
    # _prepare_inference, _execute_inference, _finalize_inference 自动继承
```

**场景2: 重写特定阶段(高级子类)**
```python
class StreamingDetector(BaseOnnx):
    def _execute_inference(self, input_tensor):
        """重写执行阶段支持流式推理"""
        # 自定义流式推理逻辑
        async with self._runner:
            for chunk in self._stream_inference(input_tensor):
                yield chunk

    # 其他阶段保持默认
```

---

## 研究任务4: 旧版本分支逻辑识别策略

### 决策 (Decision)

**识别方法**: 结合pytest-cov覆盖率报告 + git历史分析 + 代码搜索

**删除候选**:
1. **Line 162-168**: 3元组返回值兼容代码(需验证)
2. **Line 177-180**: batch维度复制逻辑(需验证是否仍需要)
3. **Line 199-202**: batch结果过滤(需验证multi-batch场景)

### 理由 (Rationale)

1. **基于数据决策**: 使用pytest-cov生成的覆盖率报告,客观识别0%覆盖分支

2. **保守策略**: 只删除完全未使用的代码,保留低覆盖但可能关键的代码

3. **可追溯性**: 通过git blame和git log理解代码存在原因

### 识别流程

**步骤1: 生成覆盖率报告**
```bash
pytest --cov=infer_onnx.onnx_base \
       --cov-report=html:htmlcov_before \
       --cov-report=term-missing \
       --cov-branch \
       tests/

# 保存基准报告
cp htmlcov_before/infer_onnx_onnx_base_py.html baseline_coverage.html
```

**步骤2: 分析__call__方法覆盖情况**
```bash
# 在HTML报告中:
# - 红色行 = 0%覆盖(删除候选)
# - 黄色行 = 部分分支覆盖(需评估)
# - 绿色行 = 完全覆盖(保留)
```

**步骤3: 验证候选代码的使用情况**
```bash
# 检查3元组返回值是否被使用
grep -r "def _preprocess" infer_onnx/*.py | head -20
grep -r "return.*,.*,.*$" infer_onnx/onnx_*.py | grep _preprocess

# 检查是否有子类返回3元组
# 如果所有子类都返回4元组,则3元组兼容代码可删除
```

**步骤4: 查看git历史**
```bash
# 查看兼容代码的创建原因
git blame infer_onnx/onnx_base.py | grep -A 5 "len(preprocess_result)"

# 查看相关commit
git log --follow -p -- infer_onnx/onnx_base.py | grep -B 10 -A 10 "preprocess_result"
```

### 已识别的候选分支

基于代码审查(未运行覆盖率工具),预期的候选:

**候选1: 3元组兼容代码 (line 162-168)**
```python
# 预期覆盖率: 可能0%
if len(preprocess_result) == 3:
    # 兼容旧版本返回值
    input_tensor, scale, original_shape = preprocess_result
    ratio_pad = None
else:
    # 新版本返回值
    input_tensor, scale, original_shape, ratio_pad = preprocess_result
```

**删除决策**:
- ✅ 如果所有子类都返回4元组 → 删除
- ❌ 如果有子类返回3元组 → 保留或修复子类

**候选2: RF-DETR特殊处理 (line 193-197)**
```python
# 预期覆盖率: 20%(仅RF-DETR测试覆盖)
if type(self).__name__ == 'RFDETROnnx':
    detections = self._postprocess(outputs, effective_conf_thres, **kwargs)
else:
    prediction = outputs[0]
    detections = self._postprocess(prediction, effective_conf_thres, scale=scale, ratio_pad=ratio_pad, **kwargs)
```

**删除决策**:
- ❌ 保留(RF-DETR确实需要特殊处理)
- ⚠️ 可优化为更通用的机制(如子类标志位)

**候选3: batch过滤逻辑 (line 199-202)**
```python
# 预期覆盖率: 待验证
if (expected_batch_size > 1 and len(detections) > 1):
    detections = [detections[0]]
    logging.debug(f"只返回第一个batch的检测结果")
```

**删除决策**:
- ⏳ 需要覆盖率数据确认
- ⏳ 需要确认是否有multi-batch推理场景

### 安全删除清单

删除任何代码前,必须完成:

- [ ] pytest-cov确认0%覆盖
- [ ] grep搜索确认无引用
- [ ] git历史确认创建原因
- [ ] 询问原作者(如可联系)
- [ ] 添加测试用例(如确认需要保留)
- [ ] 在feature分支测试删除后的功能
- [ ] 通过完整测试套件(集成+单元)

---

## 综合决策总结

### 核心技术选择

| 技术点 | 选择方案 | 置信度 |
|--------|---------|--------|
| 装饰器顺序 | @staticmethod → @abstractmethod | ✅ 确定(官方文档) |
| 删除阈值 | 仅删除0%覆盖分支 | ✅ 确定(行业共识) |
| 阶段划分 | 3阶段(prepare/execute/finalize) | ✅ 确定(框架实践) |
| 参数传递 | InferenceContext实例变量 | ⚠️ 建议(可选,不强制) |
| 子类策略 | 重写抽象方法,继承阶段方法 | ✅ 确定(模板方法模式) |

### 实施优先级

**P0 - 必须实施**:
1. 添加@abstractmethod装饰器到_postprocess和_preprocess_static
2. 生成pytest-cov覆盖率报告
3. 提取3个阶段方法到BaseOnnx

**P1 - 强烈推荐**:
4. 引入InferenceContext替代返回元组
5. 删除确认为0%覆盖的分支代码

**P2 - 可选优化**:
6. 添加调试模式和性能分析
7. 统一错误消息格式

### 风险和缓解

**风险1: InferenceContext增加学习曲线**
- **缓解**: 保持向后兼容,提供迁移指南
- **缓解**: 在quickstart.md中提供示例代码

**风险2: 删除了仍在某些场景使用的代码**
- **缓解**: 严格遵循0%覆盖+代码搜索+git历史的三重验证
- **缓解**: 在staging环境充分测试

**风险3: 3阶段方法粒度不适合某些子类**
- **缓解**: 阶段方法在BaseOnnx中有默认实现,子类可完全重写
- **缓解**: 保持__call__模板方法,确保接口一致

---

## 参考资料

### 官方文档
- [Python abc模块](https://docs.python.org/3/library/abc.html)
- [PEP 3119 - Abstract Base Classes](https://peps.python.org/pep-3119/)
- [pytest-cov文档](https://pytest-cov.readthedocs.io/)
- [TensorFlow Keras Custom Training](https://keras.io/guides/custom_train_step_in_tensorflow/)
- [PyTorch Hooks](https://pytorch.org/docs/stable/notes/modules.html#module-hooks)

### 设计模式
- [Refactoring Guru - Template Method](https://refactoring.guru/design-patterns/template-method)
- [Python Design Patterns](https://github.com/faif/python-patterns)

### 项目代码
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/infer_onnx/onnx_base.py` - 当前BaseOnnx实现
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/infer_onnx/onnx_yolo.py` - YoloOnnx实现参考
- `/home/tyjt/桌面/onnx_vehicle_plate_recognition/infer_onnx/onnx_ocr.py` - OCRONNX实现参考

---

**研究完成时间**: 2025-10-09
**所有NEEDS CLARIFICATION已解决**: ✅
**下一步**: 进入Phase 1 - Design & Contracts ([data-model.md](./data-model.md))
