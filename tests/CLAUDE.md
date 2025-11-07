[根目录](../CLAUDE.md) > **tests**

# 测试体系模块 (tests)

## 模块职责

提供完整的测试框架,包括单元测试、集成测试、合约测试和性能测试,确保系统功能正确性、API合约一致性和性能指标达标。采用pytest框架,支持代码覆盖率分析和性能基准测试。

## 入口和启动

- **测试配置**: `conftest.py` - 全局pytest配置和fixtures
- **单元测试**: `unit/` - 组件级功能测试(62+ 测试)
- **集成测试**: `integration/` - 端到端流程验证(30+ 测试)
- **合约测试**: `contract/` - API合约和数据模型验证(15+ 测试)
- **性能测试**: `performance/` - 性能基准和优化验证(2+ 测试)

### 快速开始

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试套件
pytest tests/unit/ -v                    # 单元测试
pytest tests/integration/ -v             # 集成测试
pytest tests/contract/ -v                # 合约测试
pytest tests/performance/ -v --benchmark-only  # 性能测试

# 运行单个测试文件
pytest tests/unit/test_ocr_metrics.py -v

# 生成覆盖率报告
pytest tests/ --cov=onnxtools --cov-report=html
# 查看报告: htmlcov/index.html

# 并行测试(需要pytest-xdist)
pytest tests/ -n 4  # 4个worker并行
```

## 外部接口

### 1. 测试Fixtures使用

```python
import pytest

# 使用项目配置fixtures
def test_with_config(project_root, plate_config, ocr_character):
    """使用配置fixtures的测试"""
    assert project_root.exists()
    assert 'ocr_dict' in plate_config
    assert len(ocr_character) == 85

# 使用模型路径fixtures
def test_with_models(ocr_model_path, color_layer_model_path):
    """使用模型路径fixtures的测试"""
    assert ocr_model_path.exists()
    assert color_layer_model_path.suffix == '.onnx'

# 使用样本图像fixtures
def test_with_samples(sample_single_layer_plate, sample_blue_plate):
    """使用样本图像fixtures的测试"""
    assert sample_single_layer_plate.shape[2] == 3  # BGR
    assert sample_blue_plate is not None

# 使用golden测试数据
def test_with_golden(golden_ocr_outputs, golden_color_layer_outputs):
    """使用golden数据fixtures的测试"""
    assert 'single_layer' in golden_ocr_outputs
    assert len(golden_ocr_outputs['single_layer']) > 0
```

### 2. 自定义Fixtures

```python
# tests/conftest.py中添加新的fixtures
@pytest.fixture
def custom_detector(project_root):
    """创建自定义检测器fixture"""
    from onnxtools import create_detector
    model_path = project_root / "models" / "rtdetr.onnx"
    if not model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    return create_detector('rtdetr', str(model_path))

# 在测试中使用
def test_custom_detector(custom_detector, sample_image):
    result = custom_detector(sample_image)
    assert len(result) >= 0
```

## 模块结构

```
tests/
├── conftest.py                          # pytest配置和全局fixtures
├── __init__.py                          # 测试模块初始化
├── fixtures/                            # 测试数据目录
│   ├── plates/                          # 车牌样本图像
│   │   ├── single_layer_sample.jpg
│   │   ├── double_layer_sample.jpg
│   │   ├── blue_plate.jpg
│   │   └── yellow_plate.jpg
│   └── golden/                          # Golden测试数据
│       ├── golden_ocr_outputs.json
│       └── golden_color_layer_outputs.json
│
├── unit/                                # 单元测试(62+ 测试)
│   ├── __init__.py
│   ├── test_ocr_metrics.py              # OCR指标计算(23个)
│   ├── test_load_label_file.py          # 标签文件加载(12个)
│   ├── test_ocr_onnx_refactored.py      # OCR推理(27个)
│   ├── test_result.py                   # Result类功能测试
│   └── test_result_property.py          # Result属性测试
│
├── integration/                         # 集成测试(30+ 测试)
│   ├── __init__.py
│   ├── test_pipeline_integration.py     # 完整推理管道
│   ├── test_ocr_integration.py          # OCR识别流程
│   ├── test_ocr_evaluation_integration.py  # OCR评估(8个)
│   ├── test_result_integration.py       # Result集成测试
│   ├── test_result_visualization.py     # Result可视化测试
│   ├── test_supervision_only.py         # Supervision库集成
│   ├── test_basic_drawing.py            # 基础绘制功能
│   ├── test_annotator_pipeline.py       # Annotator管道
│   ├── test_preset_scenarios.py         # 预设场景测试
│   ├── test_round_box_integration.py    # 圆角框annotator
│   ├── test_box_corner_integration.py   # 边框角annotator
│   ├── test_dot_annotator.py            # 点标记annotator
│   ├── test_geometric_annotators.py     # 几何annotators
│   ├── test_fill_annotators.py          # 填充annotators
│   ├── test_privacy_annotators.py       # 隐私annotators
│   ├── test_halo_annotator.py           # 光晕annotator
│   └── test_percentage_bar_integration.py  # 置信度条annotator
│
├── contract/                            # 合约测试(15+ 测试)
│   ├── __init__.py
│   ├── test_ocr_evaluator_contract.py   # OCR评估器合约(11个)
│   ├── test_ocr_onnx_refactored_contract.py  # OCR推理合约
│   ├── test_result_contract.py          # Result类合约
│   ├── test_convert_detections_contract.py  # 数据转换合约
│   ├── test_draw_detections_contract.py     # 可视化API合约
│   ├── test_annotator_factory_contract.py   # Annotator工厂合约
│   ├── test_annotator_pipeline_contract.py  # Annotator管道合约
│   └── test_preset_loading_contract.py      # 预设加载合约
│
└── performance/                         # 性能测试(2+ 测试)
    ├── __init__.py
    ├── test_annotator_benchmark.py      # Annotator性能基准
    └── test_result_plot_benchmark.py    # Result.plot()性能基准
```

## 关键依赖和配置

### 测试框架依赖
```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",          # 代码覆盖率
    "pytest-benchmark>=4.0.0",    # 性能基准测试
    "pytest-xdist>=3.0.0",        # 并行测试
    "pytest-timeout>=2.0.0",      # 测试超时控制
]
```

### pytest配置
```ini
# pytest.ini (项目根目录)
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=onnxtools
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
    contract: marks contract tests
    unit: marks unit tests
```

### 测试数据配置
```python
# conftest.py中的fixtures
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
PLATES_DIR = FIXTURES_DIR / "plates"
GOLDEN_DIR = FIXTURES_DIR / "golden"

fixtures = {
    'project_root': '项目根目录路径',
    'plate_config': 'plate.yaml配置字典',
    'ocr_character': 'OCR字符字典列表',
    'color_map': '颜色映射字典',
    'layer_map': '层级映射字典',
    'ocr_model_path': 'OCR模型路径',
    'color_layer_model_path': '颜色/层级模型路径',
    'sample_single_layer_plate': '单层车牌样本图像',
    'sample_double_layer_plate': '双层车牌样本图像',
    'sample_blue_plate': '蓝牌样本图像',
    'sample_yellow_plate': '黄牌样本图像',
    'golden_ocr_outputs': 'Golden OCR测试数据',
    'golden_color_layer_outputs': 'Golden颜色/层级测试数据',
    'enable_gpu': 'GPU可用性检测'
}
```

## 数据模型

### 测试配置模型
```python
benchmark_config = {
    'target_time_ms': 30.0,          # 目标处理时间(ms)
    'iterations': 100,                # 测试迭代次数
    'max_objects': 20,                # 最大目标数量
    'warmup_iterations': 10           # 预热迭代次数
}

sample_detection_format = [
    [x1, y1, x2, y2, confidence, class_id],  # 单个检测结果
    # ... 更多检测
]

sample_plate_result = {
    'plate_text': str,                # 车牌文本
    'color': str,                     # 车牌颜色
    'layer': str,                     # 车牌层数
    'confidence': float,              # 置信度
    'should_display_ocr': bool        # 是否显示OCR结果
}
```

### 合约验证模型
```python
contract_validation = {
    'input_format': {
        'type': 'API输入类型',
        'shape': 'ndarray shape',
        'dtype': 'numpy dtype',
        'range': 'value range'
    },
    'output_format': {
        'type': 'API输出类型',
        'fields': '必需字段列表',
        'constraints': '约束条件'
    },
    'error_handling': {
        'invalid_input': '预期异常类型',
        'edge_cases': '边界情况处理'
    },
    'performance_constraint': {
        'max_time_ms': float,         # 最大执行时间
        'max_memory_mb': float        # 最大内存占用
    }
}
```

### Golden测试数据格式
```json
// golden_ocr_outputs.json
{
    "single_layer": [
        {
            "image": "plate_001.jpg",
            "text": "京A12345",
            "confidence": 0.95,
            "char_scores": [0.98, 0.97, 0.96, 0.94, 0.93, 0.92, 0.91]
        }
    ],
    "double_layer": [
        {
            "image": "plate_002.jpg",
            "text": "京AF1234学",
            "confidence": 0.92,
            "char_scores": [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88]
        }
    ]
}
```

## 测试和质量

### 测试覆盖率统计

**总体统计**:
- 总测试用例: 109+ (unit: 62, integration: 30, contract: 15, performance: 2)
- 测试通过率: 96.6% (105/109)
- 代码覆盖率: ~85% (核心模块)

**模块覆盖率**:
| 模块 | 覆盖率 | 测试数量 | 通过率 |
|------|--------|---------|--------|
| onnxtools.infer_onnx | 88% | 45 | 97.8% |
| onnxtools.utils | 82% | 38 | 94.7% |
| onnxtools.eval | 90% | 19 | 100% |
| onnxtools.pipeline | 75% | 7 | 100% |

### 单元测试覆盖
- [x] **OCR指标计算** (`test_ocr_metrics.py` - 23个测试)
  - 编辑距离边界情况(空字符串、长度差异、插入删除替换)
  - 中文字符处理测试
  - 真实OCR场景(常见混淆、部分识别、双层车牌、噪声)
  - 准确率计算和归一化指标

- [x] **标签文件加载** (`test_load_label_file.py` - 12个测试)
  - 单张图像格式解析
  - JSON数组格式解析
  - 边界情况处理(空行、无效格式、缺失文件)
  - 中文路径支持

- [x] **OCR推理** (`test_ocr_onnx_refactored.py` - 27个测试)
  - 单层/双层车牌识别
  - 置信度计算
  - 字符级置信度
  - 错误处理

- [x] **Result类功能** (`test_result.py`, `test_result_property.py`)
  - 属性访问和索引操作
  - 过滤和统计方法
  - 可视化接口

- [ ] **图像预处理函数** (待补充)
- [ ] **NMS算法** (待补充)
- [ ] **数据转换工具** (待补充)

### 集成测试覆盖
- [x] **完整推理管道** (`test_pipeline_integration.py`)
  - 端到端检测+OCR流程
  - 多模型协同工作
  - 结果聚合和后处理

- [x] **OCR评估集成** (`test_ocr_evaluation_integration.py` - 8个测试)
  - 端到端评估(table和JSON格式)
  - 参数验证(max_images、置信度阈值扫描)
  - 边界情况处理(缺失图像、损坏图像)
  - 性能测试(<1秒处理5张图像)

- [x] **Result集成测试** (`test_result_integration.py`)
  - 与检测器集成
  - 与可视化工具集成

- [x] **Supervision集成** (`test_supervision_only.py`, `test_annotator_pipeline.py`)
  - 13种Annotator类型集成测试
  - Annotator管道组合测试
  - 5种预设场景验证

- [x] **可视化集成** (`test_basic_drawing.py`, `test_result_visualization.py`)
  - 基础绘制功能
  - Result.plot()方法
  - 各类annotator集成

### 合约测试覆盖
- [x] **OCR评估器合约** (`test_ocr_evaluator_contract.py` - 11个测试)
  - 基础评估流程合约(返回格式、数值范围)
  - 编辑距离指标合约(完美匹配、部分匹配、per_sample_results)
  - 置信度过滤合约(阈值行为、样本守恒)
  - JSON导出格式合约(有效性、必需字段)
  - 表格对齐合约(中文列名、数值格式)

- [x] **Result类合约** (`test_result_contract.py`)
  - API接口稳定性验证
  - 数据类型约束
  - 方法签名一致性

- [x] **数据转换合约** (`test_convert_detections_contract.py`)
  - 检测结果到Supervision格式转换
  - 数据完整性保证

- [x] **可视化API合约** (`test_draw_detections_contract.py`)
  - 绘制函数接口稳定性
  - 参数验证

- [x] **Annotator合约** (`test_annotator_factory_contract.py`, `test_annotator_pipeline_contract.py`)
  - 工厂模式接口约束
  - 管道组合行为验证

### 性能测试覆盖
- [x] **Annotator性能基准** (`test_annotator_benchmark.py`)
  - 13种Annotator类型的渲染时间(75μs ~ 1.5ms)
  - 内存占用监控
  - 性能回归检测

- [x] **Result.plot()性能基准** (`test_result_plot_benchmark.py`)
  - 不同对象数量下的渲染性能
  - 预设场景性能对比

- [ ] **推理延迟基准** (待补充)
- [ ] **批处理吞吐量测试** (待补充)

## 常见问题 (FAQ)

### Q: 如何添加新的测试用例?
A:
1. 在对应测试目录(unit/integration/contract/performance)创建`test_*.py`文件
2. 使用`conftest.py`中的fixtures或创建新的fixtures
3. 遵循现有测试命名规范(`test_<function_name>`)
4. 添加必要的文档字符串说明测试目的
5. 运行`pytest tests/ -v`验证测试通过

```python
# tests/unit/test_new_feature.py
import pytest

def test_new_feature_basic(project_root):
    """测试新功能的基础行为"""
    # 测试代码
    assert True

def test_new_feature_edge_cases():
    """测试新功能的边界情况"""
    # 测试代码
    assert True
```

### Q: 性能测试失败怎么调试?
A:
1. 检查GPU资源占用(`nvidia-smi`)
2. 确认测试环境稳定(无其他负载)
3. 增加`iterations`参数获取稳定结果
4. 查看`pytest-benchmark`详细报告(`--benchmark-verbose`)
5. 使用`--benchmark-only`跳过非性能测试

```bash
# 调试性能测试
pytest tests/performance/ -v --benchmark-verbose --benchmark-only

# 保存基准结果
pytest tests/performance/ --benchmark-save=baseline

# 对比基准
pytest tests/performance/ --benchmark-compare=baseline
```

### Q: 如何生成测试覆盖率报告?
A:
```bash
# HTML报告
pytest tests/ --cov=onnxtools --cov-report=html
# 查看: open htmlcov/index.html

# 终端报告
pytest tests/ --cov=onnxtools --cov-report=term-missing

# XML报告(CI集成)
pytest tests/ --cov=onnxtools --cov-report=xml

# 排除特定文件
pytest tests/ --cov=onnxtools --cov-report=html --cov-config=.coveragerc
```

### Q: 合约测试的作用是什么?
A: 合约测试验证API接口的:
- **输入输出格式**: 确保数据类型和结构稳定
- **错误处理**: 验证异常情况下的行为
- **性能约束**: 保证关键路径的性能指标
- **向后兼容**: 支持安全重构,避免破坏现有API

合约测试是模块间接口稳定性的保证,适合在重构或API变更时使用。

### Q: 如何跳过需要GPU的测试?
A:
```python
# 在测试函数中使用fixture
def test_gpu_inference(enable_gpu):
    if not enable_gpu:
        pytest.skip("GPU not available")
    # GPU相关测试代码

# 或使用marker
@pytest.mark.gpu
def test_gpu_only():
    # GPU测试代码
    pass

# 运行时跳过GPU测试
pytest tests/ -m "not gpu"
```

### Q: 如何并行运行测试?
A:
```bash
# 安装pytest-xdist
pip install pytest-xdist

# 并行运行(4个worker)
pytest tests/ -n 4

# 自动检测CPU核心数
pytest tests/ -n auto

# 注意: 并行测试可能导致fixtures冲突,需要注意scope设置
```

### Q: 如何调试失败的测试?
A:
```bash
# 详细输出
pytest tests/unit/test_ocr_metrics.py -vv

# 进入Python调试器
pytest tests/unit/test_ocr_metrics.py --pdb

# 只运行失败的测试
pytest tests/ --lf

# 在第一个失败时停止
pytest tests/ -x

# 显示本地变量
pytest tests/unit/test_ocr_metrics.py -l
```

## 相关文件列表

### 测试配置文件
- `tests/conftest.py` - 全局pytest配置和fixtures定义(261行)
- `tests/__init__.py` - 测试模块初始化
- `pytest.ini` - pytest配置文件(项目根目录)
- `.coveragerc` - 覆盖率配置文件(可选)

### 测试数据目录
- `tests/fixtures/` - 测试数据根目录
- `tests/fixtures/plates/` - 车牌样本图像
- `tests/fixtures/golden/` - Golden测试数据(JSON)

### 单元测试套件(62+ 测试)
- `tests/unit/test_ocr_metrics.py` (23个)
- `tests/unit/test_load_label_file.py` (12个)
- `tests/unit/test_ocr_onnx_refactored.py` (27个)
- `tests/unit/test_result.py`
- `tests/unit/test_result_property.py`

### 集成测试套件(30+ 测试)
- `tests/integration/test_pipeline_integration.py`
- `tests/integration/test_ocr_evaluation_integration.py` (8个)
- `tests/integration/test_result_integration.py`
- `tests/integration/test_result_visualization.py`
- `tests/integration/test_supervision_only.py`
- `tests/integration/test_basic_drawing.py`
- `tests/integration/test_annotator_pipeline.py`
- `tests/integration/test_preset_scenarios.py`
- `tests/integration/test_*_annotator*.py` (13个annotator测试)

### 合约测试套件(15+ 测试)
- `tests/contract/test_ocr_evaluator_contract.py` (11个)
- `tests/contract/test_result_contract.py`
- `tests/contract/test_ocr_onnx_refactored_contract.py`
- `tests/contract/test_convert_detections_contract.py`
- `tests/contract/test_draw_detections_contract.py`
- `tests/contract/test_annotator_factory_contract.py`
- `tests/contract/test_annotator_pipeline_contract.py`
- `tests/contract/test_preset_loading_contract.py`

### 性能测试套件(2+ 测试)
- `tests/performance/test_annotator_benchmark.py`
- `tests/performance/test_result_plot_benchmark.py`

## 变更日志 (Changelog)

**2025-11-07** - 大幅更新测试体系文档
- 补充完整的模块结构和文件清单
- 添加详细的fixtures使用说明和自定义方法
- 补充测试覆盖率统计表和各测试套件详情
- 添加golden测试数据格式说明
- 扩展常见问题(并行测试、调试方法、覆盖率报告)
- 添加pytest配置示例和最佳实践
- 更新变更日志记录

**2025-10-10** - OCR评估测试完成
- 新增`test_ocr_evaluation_integration.py` (8个集成测试)
- 新增`test_ocr_evaluator_contract.py` (11个合约测试)
- 新增`test_ocr_metrics.py` (23个单元测试)
- 测试总数增至109+,通过率96.6%

**2025-09-30** - Supervision集成测试完成
- 13种Annotator类型集成测试
- 5种可视化预设场景验证
- 新增`test_annotator_benchmark.py`性能测试

**2025-09-30 11:05:14 CST** - 初始化测试体系模块文档
- 建立测试框架和合约测试规范
- 初始测试覆盖率约70%

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/`*
*最后更新: 2025-11-07 16:35:25*
