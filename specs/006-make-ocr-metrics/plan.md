# Implementation Plan: OCR Metrics Evaluation Functions

**Branch**: `006-make-ocr-metrics` | **Date**: 2025-10-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-make-ocr-metrics/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

创建OCR指标评估功能模块,提供与目标检测评估(DatasetEvaluator)一致的架构模式,支持完全准确率、归一化编辑距离和编辑距离相似度计算。评估结果以两行表格格式输出到终端(支持中文对齐)或导出为JSON格式,用于车牌OCR模型性能评估和优化。

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**:
- numpy >=2.2.6 (数值计算)
- Levenshtein >=0.25.0 (编辑距离计算)
- logging (标准库,日志记录)
- pathlib (标准库,路径处理)
- json (标准库,JSON导出)

**Storage**: 文件系统 (tab分隔的label list文件: train.txt, val.txt)
**Testing**: pytest (单元测试) + contract tests (API合约测试)
**Target Platform**: Linux server (支持UTF-8终端和东亚字符宽度显示)
**Project Type**: single (添加到现有`infer_onnx/`模块)
**Performance Goals**:
- 1000张图像评估完成时间 < 5分钟 (GPU)
- 内存占用 < 500MB (beyond model weights)
- 编辑距离计算精度 < 0.001容差

**Constraints**:
- 必须与现有DatasetEvaluator架构模式一致
- 中文输出必须在终端正确对齐(列宽18和12)
- 支持OCRONNX类的__call__()接口
- 向后兼容现有数据集格式

**Scale/Scope**:
- 支持1000+图像的批量评估
- 3个核心指标(完全准确率、归一化编辑距离、编辑距离相似度)
- 2种输出模式(表格对齐、JSON导出)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Research)

| Principle | Compliance | Notes |
|-----------|-----------|-------|
| **I. Modular Architecture** | ✅ PASS | OCRDatasetEvaluator将作为独立类添加到infer_onnx/eval_ocr.py,继承评估模式架构 |
| **II. Configuration-Driven Design** | ✅ PASS | 评估参数(confidence threshold, max_images)通过参数传递,无硬编码配置 |
| **III. Performance First** | ✅ PASS | 批量处理设计,目标1000图像<5分钟,内存高效(仅存储统计数据) |
| **IV. Type Safety** | ✅ PASS | 所有公共API使用类型提示,输入验证(label文件格式、路径存在性) |
| **V. Test-Driven Development** | ✅ PASS | 先编写合约测试(验证输出格式、指标正确性),后实现功能 |
| **VI. Observability** | ✅ PASS | 使用logging模块记录进度(每50张)、警告(跳过样本)、错误(文件不存在) |
| **VII. Simplicity (YAGNI)** | ✅ PASS | 最简实现:仅3个核心指标,无可视化、无per-character分析、无A/B测试 |

**Result**: ✅ All gates PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```
specs/006-make-ocr-metrics/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── ocr_evaluator_api.yaml
├── checklists/          # Quality validation
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
infer_onnx/
├── __init__.py          # 添加OCRDatasetEvaluator导出
├── eval_coco.py         # 现有的目标检测评估器(参考实现)
└── eval_ocr.py          # 新增: OCR评估器模块

utils/
├── ocr_metrics.py       # 新增: OCR指标计算函数
│   ├── calculate_accuracy()
│   ├── calculate_edit_distance()
│   └── print_ocr_metrics()
└── __init__.py          # 导出ocr_metrics函数

tests/
├── contract/
│   └── test_ocr_evaluator_contract.py  # 新增: OCR评估器合约测试
├── integration/
│   └── test_ocr_evaluation_integration.py  # 新增: 端到端评估测试
└── unit/
    └── test_ocr_metrics.py  # 新增: 指标计算单元测试
```

**Structure Decision**: 采用Single Project结构,新功能作为模块添加到现有`infer_onnx/`和`utils/`目录。遵循与`eval_coco.py`一致的评估器架构模式,确保代码风格和接口设计的统一性。

## Complexity Tracking

*无Constitution违规项 - 本节留空*

