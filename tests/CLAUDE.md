[根目录](../CLAUDE.md) > **tests**

# 测试体系 (tests)

## 模块职责

基于 pytest 的测试套件,分为单元 / 集成 / 合约 / 性能四类,覆盖推理类、Result、可视化 annotator、OCR/分类评估与 2D 跟踪。支持覆盖率统计、性能基准与并行执行。

## 目录结构

```
tests/
├── conftest.py            # 全局 fixtures
├── fixtures/plates/       # 车牌样本(合成生成,见 README)
├── unit/                  # 单组件/函数测试
├── integration/           # 端到端流程 + annotator 集成
├── contract/              # API 合约/数据模型稳定性
└── performance/           # pytest-benchmark 基准
```

| 目录 | 职责 | 主要覆盖 |
|------|------|---------|
| `unit/` | 函数/类隔离测试 | OCR/分类指标、NMS、图像处理、Result、config、tracking(kalman/matching/bytetrack/ocsort/factory)、各 ORT |
| `integration/` | 多组件协作 | 推理管道、OCR 流程、Supervision 可视化、13 类 annotator、预设场景 |
| `contract/` | 接口契约 | Result、OCR/分类 evaluator、检测转换、draw API、annotator 工厂/管道、预设加载 |
| `performance/` | 性能基准 | annotator 渲染、Result.plot()、tracking |

## 运行测试

```bash
pytest tests/ -v                          # 全量
pytest tests/unit/ -v                     # 按目录
pytest tests/performance/ --benchmark-only  # 仅基准
pytest tests/unit/test_nms.py -v          # 单文件
pytest tests/ -m "not slow"               # 按 marker 过滤
pytest tests/ -n auto                     # 并行(pytest-xdist)
pytest tests/ --cov=onnxtools --cov-report=html   # 覆盖率 -> htmlcov/
pytest tests/ --lf -x                     # 只跑上次失败,首错即停
```

## pytest 配置 (pytest.ini)

- `testpaths=tests`,发现规则 `test_*.py` / `Test*` / `test_*`。
- `addopts`: `-v --tb=short --strict-markers`(marker 必须注册)。
- Marker: `unit` / `integration` / `contract` / `performance` / `benchmark` / `visual` / `slow`。
- `filterwarnings` 忽略 `UserWarning` 与 `DeprecationWarning`。

## 关键 fixtures (conftest.py)

| fixture | 含义 |
|---------|------|
| `project_root` | 仓库根目录 Path |
| `plate_config` | plate.yaml 配置字典 |
| `ocr_character` | OCR 字符字典列表 |
| `color_map` / `layer_map` | 颜色 / 层级 ID 映射 |
| `ocr_model_path` / `color_layer_model_path` | 模型路径(缺失时 skip) |
| `sample_{single_layer,double_layer,blue,yellow}_plate` | 合成车牌样本图像(BGR) |
| `golden_ocr_outputs` / `golden_color_layer_outputs` | golden 期望输出 |
| `enable_gpu` | GPU 可用性检测 |

> 样本图像由 `_generate_synthetic_plate` 合成,无需外部数据文件。

## 新增测试约定

1. 按类型放入对应目录,文件名 `test_*.py`,函数 `test_<被测对象>`,加一行 docstring 说明意图。
2. 优先复用 conftest fixtures;新增公共 fixture 写入 `conftest.py`,注意 scope 避免并行冲突。
3. 需要模型/GPU 的测试用 `pytest.skip(...)` 或 `@pytest.mark.gpu`(运行时 `-m "not gpu"` 跳过)。
4. 对应规则:新函数 → unit;新推理类 → integration;API 变更 → contract;性能声明 → performance benchmark。
5. 提交前跑 `pytest tests/ -v` 确认通过。

## FAQ

- **性能测试不稳定?** 用 `--benchmark-verbose` 看详情,`--benchmark-save=baseline` / `--benchmark-compare=baseline` 做回归对比;确认无其他负载。
- **合约测试作用?** 锁定 API 输入输出格式、错误行为与向后兼容,重构 / 改接口时先跑它。
