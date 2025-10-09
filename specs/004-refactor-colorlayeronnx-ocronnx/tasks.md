# Tasks: 重构ColorLayerONNX和OCRONNX以继承BaseOnnx

**Input**: Design documents from `/specs/004-refactor-colorlayeronnx-ocronnx/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions
- Single Python library project at repository root
- Paths: `infer_onnx/`, `utils/`, `tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 项目初始化和基础测试结构

- [x] T001 创建测试目录结构 (`tests/unit/`, `tests/contract/`)
- [x] T002 [P] 配置pytest fixtures用于OCR和颜色分类模型测试 (`tests/conftest.py`)
- [x] T003 [P] 准备测试数据集: 单层车牌、双层车牌、各种颜色车牌图像 (`tests/fixtures/plates/`)

**Checkpoint**: 测试基础设施就绪

---

## Phase 2: Foundational (Blocking Prerequisites - 必须在重构前完成)

**Purpose**: 创建基准测试和golden数据,锁定当前行为,防止重构回归

**⚠️ CRITICAL**: 必须先完成此阶段才能开始任何重构任务

### 基准单元测试创建

- [x] T004 [P] [FOUNDATIONAL] 为现有OCRONNX创建基准单元测试 (`tests/unit/test_ocr_onnx_baseline.py`)
  - 测试现有`infer()`方法的基本功能
  - 测试单层车牌OCR识别准确性
  - 测试双层车牌OCR识别准确性
  - 记录性能基准(推理时间)

- [x] T005 [P] [FOUNDATIONAL] 为现有ColorLayerONNX创建基准单元测试 (`tests/unit/test_color_layer_onnx_baseline.py`)
  - 测试现有`infer()`方法的基本功能
  - 测试颜色分类准确性(5种颜色)
  - 测试层级分类准确性(单/双层)
  - 记录性能基准(推理时间)

### Golden Test数据集创建

- [x] T006 [FOUNDATIONAL] 创建OCR golden test数据集 (`tests/fixtures/golden_ocr_outputs.json`)
  - 收集10张单层车牌图像
  - 收集10张双层车牌图像
  - 使用现有OCRONNX生成golden输出(文本、置信度、字符置信度)
  - 保存为JSON格式用于回归测试

- [x] T007 [FOUNDATIONAL] 创建颜色分类golden test数据集 (`tests/fixtures/golden_color_layer_outputs.json`)
  - 收集每种颜色(蓝、黄、白、黑、绿)的车牌图像各5张
  - 收集单层和双层车牌各10张
  - 使用现有ColorLayerONNX生成golden输出
  - 保存为JSON格式用于回归测试

### 双层车牌处理逻辑验证

- [x] T008 [FOUNDATIONAL] 为双层车牌处理创建中间状态golden数据 (`tests/fixtures/double_plate_processing_stages/`)
  - 记录`process_plate_image()`的中间状态:
    - 倾斜检测角度
    - 校正后图像
    - 分割线位置
    - 分割后的上下层图像
    - 最终拼接图像
  - 保存为图像文件和JSON元数据

**Checkpoint**: 基准测试全部通过,golden数据集创建完成,可以开始重构

---

## Phase 3: User Story 1 - 统一的模型初始化和管理 (Priority: P1) 🎯 MVP

**Goal**: 使ColorLayerONNX和OCRONNX继承自BaseOnnx,统一初始化模式和会话管理

**Independent Test**: 创建实例后,验证Polygraphy懒加载和provider配置正确,无需实际推理

### US1: ColorLayerONNX继承BaseOnnx

- [x] T009 [P] [US1] 添加类型别名定义到`infer_onnx/ocr_onnx.py`顶部
  - `from typing import List, Tuple, Optional, Dict, TypeAlias`
  - `from numpy.typing import NDArray`
  - `ColorLogits: TypeAlias = Tuple[NDArray[np.float32], float]`
  - `LayerLogits: TypeAlias = Tuple[NDArray[np.float32], float]`

- [x] T010 [US1] 重构ColorLayerONNX类继承BaseOnnx (`infer_onnx/ocr_onnx.py`)
  - 修改`class ColorLayerONNX(BaseOnnx):`
  - 更新`__init__()`方法:
    - 添加`color_map: Dict[int, str]`和`layer_map: Dict[int, str]`参数
    - 调用`super().__init__(onnx_path, input_shape, conf_thres, providers)`
    - 保存color_map和layer_map属性
  - 移除旧的`self.session`创建代码(继承自BaseOnnx)

- [x] T011 [US1] 实现ColorLayerONNX._preprocess()实例方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def _preprocess(self, image: NDArray[np.uint8]) -> PreprocessResult`
  - 调用`_image_pretreatment_static(image, self.input_shape)`
  - 返回(input_tensor, scale, original_shape, ratio_pad)

- [x] T012 [US1] 实现ColorLayerONNX._postprocess()实例方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def _postprocess(self, prediction: NDArray[np.float32], conf_thres: float, **kwargs) -> Dict[str, any]`
  - 分离color_logits和layer_logits
  - 应用softmax
  - 取argmax获取索引
  - 从color_map和layer_map查找名称
  - 返回`{'color': str, 'layer': str, 'color_conf': float, 'layer_conf': float}`

- [x] T013 [US1] 实现ColorLayerONNX.__call__()方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def __call__(self, image: NDArray[np.uint8], conf_thres: Optional[float] = None) -> Tuple[Dict[str, any], Tuple[int, int]]`
  - 调用`super().__call__(image, conf_thres=conf_thres)`
  - 返回分类结果和原始形状

### US1: OCRONNX继承BaseOnnx

- [x] T014 [P] [US1] 添加OCR类型别名定义到`infer_onnx/ocr_onnx.py`
  - `OCRResult: TypeAlias = Tuple[str, float, List[float]]`
  - `PreprocessResult: TypeAlias = Tuple[NDArray[np.float32], float, Tuple[int, int], Optional[Tuple]]`
  - `OCROutput: TypeAlias = Tuple[NDArray[np.int_], Optional[NDArray[np.float32]]]`

- [x] T015 [US1] 重构OCRONNX类继承BaseOnnx (`infer_onnx/ocr_onnx.py`)
  - 修改`class OCRONNX(BaseOnnx):`
  - 更新`__init__()`方法:
    - 添加`character: List[str]`参数(OCR字典)
    - 调用`super().__init__(onnx_path, input_shape, conf_thres, providers)`
    - 保存character属性
  - 移除旧的`self.session`创建代码

- [x] T016 [US1] 实现OCRONNX._preprocess()实例方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def _preprocess(self, image: NDArray[np.uint8], is_double_layer: bool = False) -> PreprocessResult`
  - 调用`_process_plate_image_static(image, is_double_layer)`
  - 调用`_resize_norm_img_static(processed_img, self.input_shape)`
  - 返回(input_tensor, scale, original_shape, ratio_pad)

- [x] T017 [US1] 实现OCRONNX._postprocess()实例方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def _postprocess(self, prediction: NDArray[np.float32], conf_thres: float, **kwargs) -> List[OCRResult]`
  - 从prediction提取text_index和text_prob
  - 调用`_decode_static(self.character, text_index, text_prob)`
  - 过滤低置信度结果
  - 返回OCR结果列表

- [x] T018 [US1] 实现OCRONNX.__call__()方法 (`infer_onnx/ocr_onnx.py`)
  - 签名: `def __call__(self, image: NDArray[np.uint8], conf_thres: Optional[float] = None, is_double_layer: bool = False) -> Tuple[List[OCRResult], Tuple[int, int]]`
  - 需要传递`is_double_layer`参数到预处理
  - 调用父类`super().__call__()`并传递额外参数
  - 返回OCR结果和原始形状

### US1: 合约测试(验证继承正确性)

- [x] T019 [P] [US1] 创建ColorLayerONNX合约测试 (`tests/contract/test_color_layer_onnx_contract.py`)
  - 验证`__init__()`参数符合合约(color_map, layer_map必需)
  - 验证`__call__()`返回格式符合合约(字典键名正确)
  - 验证输入验证逻辑(图像形状、数据类型)
  - 验证异常抛出符合合约(ValueError, RuntimeError)

- [x] T020 [P] [US1] 创建OCRONNX合约测试 (`tests/contract/test_ocr_onnx_contract.py`)
  - 验证`__init__()`参数符合合约(character必需)
  - 验证`__call__()`返回格式符合合约(OCRResult元组)
  - 验证`is_double_layer`参数功能
  - 验证输入验证逻辑
  - 验证异常抛出符合合约

### US1: 单元测试(验证初始化和基本功能)

- [x] T021 [P] [US1] 创建ColorLayerONNX初始化单元测试 (`tests/unit/test_color_layer_onnx.py`)
  - 测试Polygraphy懒加载(session未立即创建)
  - 测试provider自动检测
  - 测试自定义provider参数
  - 测试模型文件不存在时的错误处理
  - 测试color_map/layer_map为空时的错误处理

- [x] T022 [P] [US1] 创建OCRONNX初始化单元测试 (`tests/unit/test_ocr_onnx.py`)
  - 测试Polygraphy懒加载
  - 测试provider配置
  - 测试character参数验证
  - 测试模型文件不存在时的错误处理

**Checkpoint**: User Story 1完成 - 两个类成功继承BaseOnnx,初始化模式统一,合约测试和单元测试通过

---

## Phase 4: User Story 2 - 标准化的推理接口 (Priority: P1)

**Goal**: 将utils中的预处理和后处理函数迁移到类内部,实现完整的推理流程

**Independent Test**: 使用`__call__()`方法进行推理,验证输出格式和准确性与golden数据一致

### US2: 迁移ColorLayerONNX预处理函数

- [x] T023 [US2] 迁移`image_pretreatment`函数为ColorLayerONNX静态方法 (`infer_onnx/ocr_onnx.py`)
  - 从`utils/ocr_image_processing.py`复制`image_pretreatment()`函数
  - 创建`@staticmethod def _image_pretreatment_static(img: NDArray[np.uint8], image_shape: Tuple[int, int]) -> NDArray[np.float32]:`
  - 添加完整类型提示
  - 添加docstring说明:
    - 功能: Resize到目标尺寸、归一化、CHW转换、添加batch维度
    - 参数: img (BGR图像), image_shape (目标尺寸)
    - 返回: [1, 3, H, W] float32张量
  - 验证处理逻辑与原函数完全一致

### US2: 迁移OCRONNX预处理函数

- [x] T024 [US2] 创建双层车牌处理辅助方法1: 倾斜检测 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _detect_skew_angle(image: NDArray[np.uint8]) -> float:`
  - 从`process_plate_image`中提取倾斜检测逻辑
  - 使用Canny边缘检测和霍夫直线变换
  - 返回倾斜角度(度),范围[-45, 45]

- [x] T025 [US2] 创建双层车牌处理辅助方法2: 倾斜校正 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _correct_skew(image: NDArray[np.uint8], angle: float) -> NDArray[np.uint8]:`
  - 从`process_plate_image`中提取校正逻辑
  - 使用cv2.getRotationMatrix2D和cv2.warpAffine
  - 保持图像通道数不变

- [x] T026 [US2] 创建双层车牌处理辅助方法3: 找到分割线 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _find_optimal_split_line(image: NDArray[np.uint8]) -> int:`
  - 从`process_plate_image`中提取水平投影逻辑
  - 计算水平投影直方图
  - 应用高斯平滑
  - 返回最佳分割线y坐标

- [x] T027 [US2] 创建双层车牌处理辅助方法4: 拆分双层 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _split_double_layer(image: NDArray[np.uint8], split_y: int) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:`
  - 根据分割线拆分上下两层
  - 返回(上层图像, 下层图像)

- [x] T028 [US2] 创建双层车牌处理辅助方法5: 拼接层级 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _stitch_layers(top_layer: NDArray[np.uint8], bottom_layer: NDArray[np.uint8]) -> NDArray[np.uint8]:`
  - 对齐两层高度(padding)
  - 水平拼接: np.hstack()
  - 返回拼接后的单行图像

- [x] T029 [US2] 创建OCRONNX主预处理方法 (`infer_onnx/ocr_onnx.py`)
  - 创建`@staticmethod def _process_plate_image_static(img: NDArray[np.uint8], is_double_layer: bool = False) -> NDArray[np.uint8]:`
  - 调用`_detect_skew_angle()`和`_correct_skew()`进行倾斜校正
  - 如果`is_double_layer=True`:
    - 调用`_find_optimal_split_line()`找到分割线
    - 调用`_split_double_layer()`拆分
    - 调用`_stitch_layers()`拼接
  - 返回处理后的单层车牌图像

- [x] T030 [US2] 迁移`resize_norm_img`函数为OCRONNX静态方法 (`infer_onnx/ocr_onnx.py`)
  - 从`utils/ocr_image_processing.py`复制`resize_norm_img()`
  - 创建`@staticmethod def _resize_norm_img_static(img: NDArray[np.uint8], image_shape: Tuple[int, int]) -> NDArray[np.float32]:`
  - 保持宽高比resize到目标高度
  - BGR转RGB、归一化到[-1, 1]、HWC转CHW
  - 右侧padding到目标宽度
  - 返回[1, 3, H, W] float32张量

### US2: 迁移OCRONNX后处理函数

- [x] T031 [US2] 迁移`get_ignored_tokens`函数为OCRONNX静态方法 (`infer_onnx/ocr_onnx.py`)
  - 从`utils/ocr_post_processing.py`复制`get_ignored_tokens()`
  - 创建`@staticmethod def _get_ignored_tokens_static() -> List[int]:`
  - 返回需要忽略的token索引列表(如blank token)

- [x] T032 [US2] 迁移`decode`函数为OCRONNX静态方法 (`infer_onnx/ocr_onnx.py`)
  - 从`utils/ocr_post_processing.py`复制`decode()`
  - 创建`@staticmethod def _decode_static(character: List[str], text_index: NDArray[np.int_], text_prob: Optional[NDArray[np.float32]] = None, is_remove_duplicate: bool = False) -> List[OCRResult]:`
  - 调用`_get_ignored_tokens_static()`获取忽略列表
  - 遍历batch,过滤ignored tokens
  - 可选移除重复字符
  - 拼接字符为文本
  - 计算平均置信度
  - 应用后处理规则(如'苏'->'京')
  - 返回`List[Tuple[str, float, List[float]]]`

### US2: 单元测试(验证迁移的函数正确性)

- [x] T033 [P] [US2] 创建双层车牌辅助方法单元测试 (`tests/unit/test_ocr_onnx.py`)
  - 测试`_detect_skew_angle()`:
    - 使用倾斜图像,验证角度检测准确性
    - 使用水平图像,验证返回0度
  - 测试`_correct_skew()`:
    - 使用已知倾斜角度,验证校正效果
  - 测试`_find_optimal_split_line()`:
    - 使用双层车牌,验证分割线位置合理
  - 测试`_split_double_layer()`和`_stitch_layers()`:
    - 验证拆分和拼接逻辑正确

- [x] T034 [P] [US2] 创建OCRONNX预处理单元测试 (`tests/unit/test_ocr_onnx.py`)
  - 测试`_process_plate_image_static()`:
    - 单层车牌: 验证仅倾斜校正
    - 双层车牌: 验证完整流程(校正+拆分+拼接)
    - 与golden数据对比中间状态
  - 测试`_resize_norm_img_static()`:
    - 验证输出形状[1, 3, 48, 320]
    - 验证归一化范围[-1, 1]
    - 验证padding逻辑

- [x] T035 [P] [US2] 创建OCRONNX后处理单元测试 (`tests/unit/test_ocr_onnx.py`)
  - 测试`_get_ignored_tokens_static()`:
    - 验证返回正确的token索引
  - 测试`_decode_static()`:
    - 使用模拟的text_index和text_prob
    - 验证字符拼接逻辑
    - 验证置信度计算
    - 验证后处理规则('苏'->'京')
    - 验证重复字符移除(如果启用)

- [x] T036 [P] [US2] 创建ColorLayerONNX预处理单元测试 (`tests/unit/test_color_layer_onnx.py`)
  - 测试`_image_pretreatment_static()`:
    - 验证输出形状[1, 3, 224, 224]
    - 验证归一化范围[-1, 1]
    - 验证resize和通道转换正确性

### US2: 集成测试(端到端推理验证)

- [x] T037 [US2] 创建OCRONNX端到端集成测试 (`tests/integration/test_ocr_onnx_inference.py`)
  - 使用真实车牌图像进行完整推理
  - 测试单层车牌OCR:
    - 与golden输出对比(文本、置信度)
    - 允许误差范围±0.02(置信度)
  - 测试双层车牌OCR:
    - 与golden输出对比
    - 验证`is_double_layer=True`参数功能

- [x] T038 [US2] 创建ColorLayerONNX端到端集成测试 (`tests/integration/test_color_layer_onnx_inference.py`)
  - 使用真实车牌图像进行完整推理
  - 测试5种颜色分类:
    - 每种颜色至少5张图像
    - 与golden输出对比
  - 测试单/双层分类:
    - 至少各10张图像
    - 验证分类准确性

**Checkpoint**: User Story 2完成 - 所有预处理和后处理函数成功迁移,单元测试和集成测试通过,输出与golden数据一致

---

## Phase 5: Utils文件删除和调用者修改 (Priority: P1)

**Goal**: 删除utils/ocr_*.py文件,修改所有调用者代码,完成迁移

**Independent Test**: 运行完整的pipeline.py,验证车牌识别流程正常工作

### 调用者修改

- [x] T039 [REFACTOR] 识别utils/ocr_image_processing.py的所有调用者
  - 使用grep搜索: `grep -r "from utils.ocr_image_processing import\|from utils import.*process_plate_image\|from utils import.*resize_norm_img\|from utils import.*image_pretreatment" /home/tyjt/桌面/onnx_vehicle_plate_recognition/`
  - 记录所有调用文件路径和行号
  - 确认调用模式(直接函数调用 vs 作为参数传递)

- [x] T040 [REFACTOR] 识别utils/ocr_post_processing.py的所有调用者
  - 使用grep搜索: `grep -r "from utils.ocr_post_processing import\|from utils import.*decode\|from utils import.*get_ignored_tokens" /home/tyjt/桌面/onnx_vehicle_plate_recognition/`
  - 记录所有调用文件路径和行号

- [x] T041 [REFACTOR] 修改utils/pipeline.py (`utils/pipeline.py`)
  - 移除导入: `from utils.ocr_image_processing import process_plate_image, resize_norm_img, image_pretreatment`
  - 移除导入: `from utils.ocr_post_processing import decode`
  - 添加导入: `from infer_onnx import OCRONNX, ColorLayerONNX` (如果未导入)
  - 修改第224-242行的调用逻辑:
    - **选项A**(推荐): 直接使用`ocr_model(plate_img, is_double_layer=True)`
    - **选项B**(如需独立预处理): 调用`OCRONNX._process_plate_image_static()`等静态方法
  - 验证修改后逻辑与原逻辑等价

- [x] T042 [P] [REFACTOR] 修改MCP模块调用者 (`mcp_vehicle_detection/main.py`或其他)
  - 根据T039/T040的搜索结果
  - 更新导入和调用逻辑
  - 如果MCP模块需要独立预处理:
    - 调用OCRONNX/ColorLayerONNX的静态方法
  - 如果使用完整推理:
    - 使用`__call__()`方法

- [x] T043 [P] [REFACTOR] 修改tools/目录下的调用者(如有)
  - 根据搜索结果更新
  - 验证工具脚本功能不受影响

- [x] T044 [REFACTOR] 更新utils/__init__.py (`utils/__init__.py`)
  - 移除导出: `from .ocr_image_processing import process_plate_image, resize_norm_img, image_pretreatment`
  - 移除导出: `from .ocr_post_processing import decode, get_ignored_tokens`
  - 验证没有其他代码依赖这些导出

### 文件删除

- [x] T045 [REFACTOR] 删除utils/ocr_image_processing.py
  - 确认T041-T044完成,所有调用者已修改
  - 执行: `rm /home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/ocr_image_processing.py`
  - 验证git status显示文件已删除

- [x] T046 [REFACTOR] 删除utils/ocr_post_processing.py
  - 确认所有调用者已修改
  - 执行: `rm /home/tyjt/桌面/onnx_vehicle_plate_recognition/utils/ocr_post_processing.py`
  - 验证git status显示文件已删除

### 回归测试

- [x] T047 [REGRESSION] 运行所有单元测试确认无回归
  - 执行: `pytest tests/unit/ -v`
  - 确认所有测试通过
  - 特别关注OCR和颜色分类相关测试

- [x] T048 [REGRESSION] 运行集成测试确认pipeline功能正常
  - 执行: `pytest tests/integration/test_refactored_pipeline.py -v`
  - 验证完整的车牌识别流程
  - 对比输出与golden数据

- [x] T049 [REGRESSION] 使用真实数据测试pipeline.py
  - 使用10张真实车牌图像
  - 执行完整的检测+OCR+颜色分类流程
  - 验证准确性与重构前一致
  - 记录性能指标(推理时间)

**Checkpoint**: 重构完成 - utils文件已删除,所有调用者修改完成,回归测试通过

---

## Phase 6: User Story 3 - TensorRT引擎比较能力 (Priority: P2)

**Goal**: 添加TensorRT引擎比较功能,支持精度验证

**Independent Test**: 独立测试engine比较功能,无需依赖实际车牌检测流程

### US3: TensorRT数据加载器支持

- [ ] T050 [P] [US3] 为OCRONNX实现create_engine_dataloader()支持
  - 继承自BaseOnnx的`create_engine_dataloader()`方法
  - 确认数据加载器正确使用`_resize_norm_img_static()`预处理
  - 测试加载器生成的数据格式正确

- [ ] T051 [P] [US3] 为ColorLayerONNX实现create_engine_dataloader()支持
  - 继承自BaseOnnx的`create_engine_dataloader()`方法
  - 确认数据加载器使用`_image_pretreatment_static()`预处理
  - 测试加载器生成的数据格式正确

### US3: 引擎比较功能

- [ ] T052 [US3] 验证OCRONNX的compare_engine()功能
  - 准备测试ONNX模型和对应的TensorRT引擎(或现场构建)
  - 调用`ocr_model.create_engine_dataloader(test_images)`
  - 调用`ocr_model.compare_engine(engine_path, tolerance=1e-3)`
  - 验证返回的比较报告包含:
    - max_diff, mean_diff, pass/fail状态
    - 详细的差异统计信息

- [ ] T053 [US3] 验证ColorLayerONNX的compare_engine()功能
  - 准备测试数据
  - 调用`color_model.create_engine_dataloader(test_images)`
  - 调用`color_model.compare_engine(engine_path, tolerance=1e-3)`
  - 验证比较结果准确性

### US3: 单元测试

- [ ] T054 [P] [US3] 创建OCRONNX engine比较单元测试 (`tests/unit/test_ocr_onnx_engine.py`)
  - 测试`create_engine_dataloader()`返回格式正确
  - 模拟engine比较流程
  - 测试容差阈值调整功能(FP16, INT8)

- [ ] T055 [P] [US3] 创建ColorLayerONNX engine比较单元测试 (`tests/unit/test_color_layer_onnx_engine.py`)
  - 测试数据加载器功能
  - 测试engine比较逻辑

### US3: 集成测试

- [ ] T056 [US3] 创建端到端引擎比较集成测试 (`tests/integration/test_engine_comparison.py`)
  - 使用真实ONNX模型和TensorRT引擎
  - 验证OCR模型的ONNX vs TRT精度
  - 验证颜色分类模型的ONNX vs TRT精度
  - 确认精度损失在容差范围内(<1e-3)

**Checkpoint**: User Story 3完成 - TensorRT引擎比较功能可用,精度验证工作流完整

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 文档更新、性能验证、最终清理

### 文档更新

- [x] T057 [P] [DOCS] 更新OCRONNX和ColorLayerONNX的docstring
  - 添加类级别docstring:
    - 继承关系说明
    - 使用示例
    - 参数说明
    - 与BaseOnnx的关系
  - 为每个方法添加详细docstring:
    - 参数类型和说明(使用data-model.md的类型定义)
    - 返回值类型和说明
    - 异常说明
    - 使用示例

- [x] T058 [P] [DOCS] 更新infer_onnx/CLAUDE.md
  - 更新OCR和颜色分类的API说明
  - 添加继承BaseOnnx的说明
  - 更新使用示例(使用`__call__()`而不是`infer()`)
  - 说明utils文件已删除

- [x] T059 [P] [DOCS] 更新utils/CLAUDE.md
  - 移除ocr_image_processing.py的文档
  - 移除ocr_post_processing.py的文档
  - 更新pipeline.py的说明(反映新的调用方式)

- [x] T060 [P] [DOCS] 更新根目录CLAUDE.md
  - 在"变更日志"添加此次重构记录
  - 更新模块关系图(如有)
  - 更新常见问题FAQ(如有OCR相关问题)

### 性能验证

- [ ] T061 [PERF] 验证成功标准SC-003: 首次推理时间<200ms
  - 测试OCRONNX首次推理(含Polygraphy懒加载)
  - 测试ColorLayerONNX首次推理
  - 记录性能数据
  - 与基准对比,确认符合要求

- [ ] T062 [PERF] 验证成功标准SC-006: API响应时间误差±5%
  - 对比重构前后的推理时间
  - OCR推理时间(后续调用,不含懒加载)
  - 颜色分类推理时间
  - 确认误差在±5%范围内

- [ ] T063 [PERF] 验证成功标准SC-002: 代码重复度降低40%
  - 统计删除的重复代码行数:
    - provider选择逻辑
    - 会话管理逻辑
  - 与重构前对比
  - 确认降低至少40%

- [ ] T064 [PERF] 验证成功标准SC-005: 内存占用优化
  - 测试懒加载效果:
    - 创建实例但不推理,GPU内存不增加
    - 首次推理后,GPU内存增加
  - 对比重构前的内存占用
  - 确认优化效果

### 最终验证

- [ ] T065 [VERIFY] 运行完整的test suite
  - `pytest tests/ -v --cov=infer_onnx --cov=utils`
  - 确认覆盖率符合要求
  - 所有测试通过

- [ ] T066 [VERIFY] 使用quickstart.md验证迁移指南
  - 按照quickstart.md的"迁移指南"步骤执行
  - 验证所有代码示例可运行
  - 验证性能基准表数据准确

- [ ] T067 [VERIFY] 验证边界情况处理
  - 测试输入形状不匹配
  - 测试模型输出格式不一致
  - 测试配置文件不存在
  - 测试FP16/INT8引擎的容差调整
  - 确认所有边界情况符合spec.md的Edge Cases

### 代码清理

- [ ] T068 [CLEANUP] 移除弃用的infer()方法(可选)
  - 如果决定完全移除旧接口:
    - 从OCRONNX移除`infer()`方法
    - 从ColorLayerONNX移除`infer()`方法
  - 或者保留并添加弃用警告:
    - `warnings.warn("infer() is deprecated, use __call__() instead", DeprecationWarning)`

- [ ] T069 [CLEANUP] 清理未使用的导入
  - 检查infer_onnx/ocr_onnx.py
  - 移除未使用的import语句
  - 运行linter验证代码风格

- [ ] T070 [CLEANUP] Git提交准备
  - 确认所有测试通过
  - 创建详细的commit message:
    - 标题: "refactor: ColorLayerONNX和OCRONNX继承BaseOnnx"
    - 正文:
      - 主要变更列表
      - 删除的文件
      - 修改的文件
      - 测试覆盖
      - 性能影响
  - 准备PR描述(如需要)

**Checkpoint**: 重构完成 - 所有文档更新,性能验证通过,代码清理完成,准备合并

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖 - 立即开始
- **Foundational (Phase 2)**: 依赖Setup完成 - **阻塞所有用户故事**
- **User Story 1 (Phase 3)**: 依赖Foundational完成
- **User Story 2 (Phase 4)**: 依赖User Story 1完成(需要继承结构)
- **Utils文件删除 (Phase 5)**: 依赖User Story 2完成(函数已迁移)
- **User Story 3 (Phase 6)**: 依赖User Story 1完成(但可与Phase 4/5并行)
- **Polish (Phase 7)**: 依赖所有用户故事完成

### User Story Dependencies

- **US1 (统一初始化)**: 依赖Foundational,无其他依赖 - **核心阻塞**
- **US2 (标准化接口)**: 依赖US1(需要继承结构) - **必须顺序执行**
- **US3 (TensorRT比较)**: 依赖US1(需要继承结构),但可与US2并行 - **可并行**

### Within Each Phase

- **Phase 1**: 所有任务可并行[P]
- **Phase 2**: T004-T005可并行,T006-T007可并行,T008独立
- **Phase 3 (US1)**:
  - T009-T010 → T011-T012 → T013 (ColorLayer顺序)
  - T014-T015 → T016-T017 → T018 (OCR顺序)
  - T019-T020可并行,T021-T022可并行
- **Phase 4 (US2)**:
  - T023独立
  - T024-T028可部分并行(但逻辑上有顺序)
  - T029依赖T024-T028
  - T030-T032独立
  - T033-T036可并行
  - T037-T038可并行
- **Phase 5**:
  - T039-T040可并行
  - T041-T044顺序执行
  - T045-T046顺序执行(在T041-T044之后)
  - T047-T049顺序执行
- **Phase 6 (US3)**:
  - T050-T051可并行
  - T052-T053顺序
  - T054-T055可并行
  - T056独立
- **Phase 7**: 多数任务可并行[P]

### Parallel Opportunities

- **Setup**: T002-T003
- **Foundational**: T004-T005, T006-T007
- **US1 ColorLayer vs OCR**: 两条线可完全并行
- **US1 Tests**: T019-T020, T021-T022
- **US2 Tests**: T033-T036, T037-T038
- **US2 Refactor**: T042-T043
- **US3**: T050-T051, T054-T055
- **Polish**: T057-T060, T061-T064

---

## Parallel Example: User Story 1

```bash
# ColorLayerONNX重构线(独立执行)
Task T009: 添加类型别名
Task T010: 重构继承BaseOnnx
Task T011: 实现_preprocess()
Task T012: 实现_postprocess()
Task T013: 实现__call__()

# OCRONNX重构线(并行执行)
Task T014: 添加类型别名
Task T015: 重构继承BaseOnnx
Task T016: 实现_preprocess()
Task T017: 实现_postprocess()
Task T018: 实现__call__()

# 测试线(前两条线完成后并行)
Task T019: ColorLayer合约测试 [P]
Task T020: OCR合约测试 [P]
Task T021: ColorLayer初始化测试 [P]
Task T022: OCR初始化测试 [P]
```

---

## Implementation Strategy

### MVP First (Minimum Viable Product)

**MVP范围**: Phase 1-3 (Setup + Foundational + US1)

1. 完成Phase 1: Setup (T001-T003)
2. 完成Phase 2: Foundational (T004-T008) - **关键门控**
3. 完成Phase 3: User Story 1 (T009-T022)
4. **STOP and VALIDATE**:
   - 运行T019-T022的合约测试和单元测试
   - 验证两个类成功继承BaseOnnx
   - 验证Polygraphy懒加载工作正常
   - 验证初始化不会崩溃
5. 如果通过,继续Phase 4

### Incremental Delivery

1. **Foundation Ready** (Phase 1-2): 基准测试和golden数据就绪
2. **US1 Complete** (Phase 3): 继承BaseOnnx完成,初始化统一
3. **US2 Complete** (Phase 4-5): 函数迁移完成,utils文件删除,完整推理可用
4. **US3 Complete** (Phase 6): TensorRT比较功能可用
5. **Production Ready** (Phase 7): 文档更新,性能验证通过

### Parallel Team Strategy

如果有2名开发者:

1. **共同完成** Phase 1-2 (Setup + Foundational) - 约1-2天
2. **Phase 3拆分**:
   - Developer A: T009-T013 + T019 + T021 (ColorLayerONNX)
   - Developer B: T014-T018 + T020 + T022 (OCRONNX)
3. **Phase 4协作**:
   - Developer A: T023 + T036 (ColorLayer预处理)
   - Developer B: T024-T032 + T033-T035 + T030 (OCR预处理+后处理)
   - 共同: T037-T038 (集成测试)
4. **Phase 5共同**: 修改调用者和删除文件
5. **Phase 6拆分**:
   - Developer A: T050 + T054 (ColorLayer engine)
   - Developer B: T052 + T052 + T056 (OCR engine)
6. **Phase 7共同**: 文档和最终验证

---

## Critical Path Analysis

### 最长路径 (估计时间)

```
Setup (0.5天)
  → Foundational (2天) 🔴 关键路径
    → US1 OCRONNX (2天) 🔴
      → US2 OCRONNX迁移 (3天) 🔴 最复杂部分
        → Utils删除和调用者修改 (1天) 🔴
          → 回归测试 (0.5天)
            → US3 TensorRT (1天)
              → Polish (1天)
```

**总关键路径**: 约11天 (单人顺序执行)
**并行优化后**: 约6-7天 (2人团队)

### 高风险任务标识

- 🔴 T006-T008: Golden test数据集创建 - **质量关键**
- 🔴 T024-T029: 双层车牌逻辑拆分 - **最复杂部分**
- 🔴 T032: decode函数迁移 - **OCR核心逻辑**
- 🔴 T041: pipeline.py修改 - **影响主流程**
- 🔴 T047-T049: 回归测试 - **验证关键**

---

## Notes

- **[P]**: 不同文件,无依赖,可并行执行
- **[Story]**: 任务所属用户故事(US1/US2/US3)
- **[FOUNDATIONAL]**: 阻塞所有用户故事的前置任务
- **[REFACTOR]**: 重构现有代码
- **[REGRESSION]**: 回归测试任务
- **[PERF]**: 性能验证任务
- **[DOCS]**: 文档更新任务
- **测试优先**: Foundational阶段创建基准测试,锁定行为
- **增量验证**: 每个Phase结束都有Checkpoint
- **独立测试**: 每个User Story都可独立验证
- **避免**: 跨文件依赖、模糊任务、缺少golden数据的回归测试

---

**Total Tasks**: 70
**Setup**: 3 tasks
**Foundational**: 5 tasks
**US1 (P1)**: 14 tasks
**US2 (P1)**: 16 tasks
**Refactor & Deletion**: 11 tasks
**US3 (P2)**: 7 tasks
**Polish**: 14 tasks

**MVP Scope**: Phase 1-3 (T001-T022, 22 tasks)
**Full Feature**: All phases (70 tasks)
