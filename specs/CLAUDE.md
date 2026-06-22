[根目录](../CLAUDE.md) > **specs**

# 功能规范归档 (specs)

## 模块职责

已完成的 feature spec 归档（spec-kit 流程产出）。每个子目录是一次功能交付的设计快照，包含 `spec.md` / `plan.md` / `tasks.md` 及可选的 `research.md`、`data-model.md`、`contracts/` 等。这里是**只读历史**，用于追溯设计决策，不作为当前需求来源。

## 关键约定

- **不要编辑 `specs/` 下已完成的 spec** —— 它们是历史归档。
- **新需求走 OpenSpec proposal**（见 [openspec/CLAUDE.md](../openspec/CLAUDE.md)），不要在此新增或改写 spec。

## 已有 spec 清单

| 目录 | 说明 |
|------|------|
| `001-baseort-result-third/` | BaseORT 统一结果包装类（Result） |
| `001-supervision-plate-box/` | 用 Supervision 库增强可视化 |
| `002-delete-old-draw/` | 移除旧版 PIL 绘制函数 |
| `003-add-more-annotators/` | 新增更多 Supervision Annotator 类型 |
| `004-refactor-colorlayeronnx-ocronnx/` | ColorLayerONNX / OCRONNX 重构为继承 BaseOnnx |
| `005-baseonnx-postprocess-call/` | BaseOnnx 抽象方法强制实现 |
| `006-make-ocr-metrics/` | OCR 指标评估函数 |
