[根目录](../CLAUDE.md) > **tests**

# 测试体系模块 (tests)

## 模块职责

提供完整的测试框架，包括单元测试、集成测试、合约测试和性能测试，确保系统功能正确性、API合约一致性和性能指标达标。

## 入口和启动

- **测试配置**: `conftest.py` - 全局pytest配置和fixtures
- **集成测试**: `integration/` - 端到端流程验证
- **合约测试**: `contract/` - API合约和数据模型验证
- **单元测试**: `unit/` - 组件级功能测试
- **性能测试**: `performance/` - 性能基准和优化验证

## 外部接口

### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试套件
pytest tests/integration/ -v
pytest tests/contract/ -v
pytest tests/unit/ -v
pytest tests/performance/ -v

# 运行性能基准测试
pytest tests/performance/ -v --benchmark-only

# 生成覆盖率报告
pytest tests/ --cov=infer_onnx --cov=utils --cov-report=html
```

### 测试fixtures使用
```python
import pytest
from conftest import sample_image, sample_detections

def test_example(sample_image, sample_detections):
    # 使用共享的测试数据
    assert sample_image.shape == (480, 640, 3)
    assert len(sample_detections[0]) == 2
```

## 关键依赖和配置

### 测试框架依赖
- **pytest**: 测试框架核心
- **pytest-benchmark**: 性能基准测试
- **pytest-cov**: 代码覆盖率分析
- **supervision**: 可视化测试验证

### 测试数据配置
```python
# conftest.py中的fixtures
fixtures = {
    'sample_image': '测试图像数据',
    'sample_detections': '检测结果样例',
    'sample_plate_results': 'OCR结果样例',
    'sample_class_names': '类别名称映射',
    'sample_colors': '可视化颜色配置',
    'font_path': '字体文件路径',
    'benchmark_config': '性能基准配置'
}
```

## 数据模型

### 测试配置模型
```python
benchmark_config = {
    'target_time_ms': 30.0,    # 目标处理时间(ms)
    'iterations': 100,          # 测试迭代次数
    'max_objects': 20           # 最大目标数量
}

sample_detection_format = [
    [x1, y1, x2, y2, confidence, class_id],  # 单个检测结果
    # ... 更多检测
]

sample_plate_result = {
    'plate_text': str,              # 车牌文本
    'color': str,                   # 车牌颜色
    'layer': str,                   # 车牌层数
    'should_display_ocr': bool      # 是否显示OCR结果
}
```

### 合约验证模型
```python
contract_validation = {
    'input_format': 'API输入格式规范',
    'output_format': 'API输出格式规范',
    'error_handling': '错误处理行为验证',
    'performance_constraint': '性能约束条件'
}
```

## 测试和质量

### 集成测试覆盖
- [x] `test_pipeline_integration.py` - 完整推理管道测试
- [x] `test_ocr_integration.py` - OCR识别流程测试
- [x] `test_supervision_only.py` - Supervision库集成测试
- [x] `test_basic_drawing.py` - 基础绘制功能测试
- [x] **`test_ocr_evaluation_integration.py`** - OCR数据集评估集成测试 (8个测试)
  - 端到端评估（table和JSON格式）
  - 参数验证（max_images、置信度阈值扫描）
  - 边界情况处理（缺失图像、损坏图像）
  - 性能测试（<1秒处理5张图像）

### 合约测试覆盖
- [x] `test_convert_detections_contract.py` - 检测数据转换合约
- [x] `test_draw_detections_contract.py` - 可视化API合约
- [x] `test_benchmark_contract.py` - 性能基准合约
- [x] **`test_ocr_evaluator_contract.py`** - OCR评估器API合约测试 (11个测试)
  - 基础评估流程合约（返回格式、数值范围）
  - 编辑距离指标合约（完美匹配、部分匹配、per_sample_results）
  - 置信度过滤合约（阈值行为、样本守恒）
  - JSON导出格式合约（有效性、必需字段）
  - 表格对齐合约（中文列名、数值格式）

### 单元测试覆盖
- [ ] 推理引擎基类测试
- [ ] 图像预处理函数测试
- [x] **OCR指标计算测试** (`test_ocr_metrics.py` - 23个测试用例)
  - 编辑距离边界情况（空字符串、长度差异、插入删除替换）
  - 中文字符处理测试
  - 真实OCR场景（常见混淆、部分识别、双层车牌、噪声）
- [ ] NMS算法测试
- [ ] 数据转换工具测试

### 性能测试覆盖
- [ ] 推理延迟基准测试
- [ ] 内存使用监控测试
- [ ] 可视化渲染性能测试
- [ ] 批处理吞吐量测试

## 常见问题 (FAQ)

### Q: 如何添加新的测试用例？
A: 1) 在对应测试目录创建test_*.py文件; 2) 使用conftest.py中的fixtures; 3) 遵循现有测试命名规范; 4) 添加必要的文档字符串

### Q: 性能测试失败怎么调试？
A: 1) 检查GPU资源占用; 2) 确认测试环境稳定; 3) 增加iterations获取稳定结果; 4) 查看pytest-benchmark详细报告

### Q: 如何生成测试覆盖率报告？
A: 运行 `pytest tests/ --cov=infer_onnx --cov=utils --cov-report=html`，查看htmlcov/index.html

### Q: 合约测试的作用是什么？
A: 合约测试验证API接口的输入输出格式、错误处理和性能约束，确保模块间接口稳定，支持安全重构

## 相关文件列表

### 测试配置文件
- `conftest.py` - 全局pytest配置和fixtures定义
- `__init__.py` - 测试模块初始化

### 集成测试套件
- `integration/test_pipeline_integration.py` - 端到端管道测试
- `integration/test_ocr_integration.py` - OCR流程集成测试
- `integration/test_supervision_only.py` - Supervision库集成
- `integration/test_basic_drawing.py` - 基础绘制测试

### 合约测试套件
- `contract/test_convert_detections_contract.py` - 数据转换合约
- `contract/test_draw_detections_contract.py` - 可视化API合约
- `contract/test_benchmark_contract.py` - 性能基准合约

### 单元测试套件
- `unit/` - 单元测试目录（待扩展）

### 性能测试套件
- `performance/` - 性能测试目录（待扩展）

### 可视化测试
- `visual/` - 可视化输出验证（可选）

## 变更日志 (Changelog)

**2025-09-30 11:05:14 CST** - 初始化测试体系模块文档，建立测试框架和合约测试规范

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/tests/`*