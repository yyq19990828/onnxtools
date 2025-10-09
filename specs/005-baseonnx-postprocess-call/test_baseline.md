# 测试基准 (Baseline Test Results)

**日期**: 2025-10-09
**分支**: `005-baseonnx-postprocess-call`
**重构前状态记录**

## 测试环境

- Python版本: 3.10.9
- pytest版本: 8.4.2
- 测试执行时间: ~1.72秒

## 测试统计

### 总体通过率

| 测试类型 | 通过 | 失败 | 总计 | 通过率 |
|---------|------|------|------|--------|
| **单元测试** | 27 | 0 | 27 | 100% ✅ |
| **集成测试** | 116 | 6 | 122 | 95.1% ✅ |
| **总计** | 143 | 6 | 149 | 96.0% ✅ |

### 失败测试详情 (非核心功能)

所有6个失败测试均为**非核心推理功能**,不影响BaseOnnx重构:

1. `test_conflict_warning_generation` - Annotator冲突警告测试,属于可视化模块
2. `test_supervision_vs_pil_output_comparison` - PIL vs Supervision对比,参数问题
3. `test_error_handling_pipeline_compatibility` - Pipeline错误处理,坐标类型问题
4. `test_memory_usage_pipeline_compatibility` - 内存测试,缺少psutil模块
5. `test_preset_rendering[high_contrast]` - 预设场景渲染,Annotator冲突
6. `test_high_contrast_preset_detailed` - 高对比度预设,Annotator冲突

### 核心推理测试状态

| 模块 | 测试数量 | 状态 | 备注 |
|------|---------|------|------|
| OCR单元测试 | 27 | ✅ 100% | ColorLayerONNX/OCRONNX完整测试 |
| OCR集成测试 | 12 | ✅ 100% | OCR文本识别和显示测试 |
| Pipeline集成 | 10/13 | ✅ 76.9% | 3个非核心测试失败 |
| Supervision集成 | 9 | ✅ 100% | Supervision库集成正常 |

## 重构前的BaseOnnx状态

### 当前抽象方法

```python
@abstractmethod
def _postprocess(self, prediction: np.ndarray, conf_thres: float, **kwargs) -> List[np.ndarray]:
    """后处理抽象方法,子类需要实现"""
    pass  # 仅使用pass,无NotImplementedError
```

### 当前__call__方法复杂度

- **总行数**: ~60行 (line 145-204)
- **分支逻辑**:
  - line 162-168: 3元组/4元组兼容性分支
  - line 177-180: batch维度调整
  - line 193-197: RF-DETR特殊处理
  - line 200-202: 多batch结果过滤

### 已知问题

1. `_postprocess`有@abstractmethod但无NotImplementedError实现
2. `_preprocess_static`无@abstractmethod装饰器
3. `__call__`方法代码过长,职责混合
4. 3元组/4元组兼容性分支覆盖率未知 (需pytest-cov确认)

## 预期重构目标

基于spec.md的成功标准:

- **SC-001**: 实例化未实现抽象方法的子类时抛出TypeError ⏳ 待实现
- **SC-002**: 错误消息格式统一 ⏳ 待实现
- **SC-003**: 测试通过率保持100%/96% ✅ 当前基准达标
- **SC-004**: 5个子类实现验证 ⏳ 待验证
- **SC-005**: @abstractmethod装饰器顺序正确 ⏳ 待实现
- **SC-006**: 性能指标<50ms, <2GB ⏳ 待测试
- **SC-007**: 代码质量pylint 8.0+, mypy通过 ⏳ 待测试
- **SC-008**: 仅删除0%覆盖分支 ⏳ 需覆盖率报告
- **SC-009**: 代码行数减少30%+ ⏳ 待实现

## 下一步行动

1. ✅ T001-T003: Setup完成
2. ⏳ T004-T005: 添加@abstractmethod装饰器
3. ⏳ T006-T009: 提取3个阶段方法
4. ⏳ T010-T012: User Story 1验证
5. ⏳ T013-T020: User Story 2验证

---

*基准记录时间: 2025-10-09*
*下一阶段: Phase 2 - Foundational Tasks*
