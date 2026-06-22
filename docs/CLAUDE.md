[根目录](../CLAUDE.md) > **docs**

# 项目文档模块 (docs)

## 模块职责

存放用户指南、API 参考、跟踪算法讲解和 Polygraphy 工具文档，由 MkDocs (material 主题) 构建为站点。`mkdocs.yml` 排除了 `polygraphy使用指南/`、`CLAUDE.md`、`rfdetr_coco_evaluation_report.md`、`_hooks/`，这些不进站点。

## 构建与预览

```bash
mkdocs serve            # 本地预览 (http://127.0.0.1:8000)
mkdocs build --strict   # 严格模式构建，链接/语法错误即失败
```

依赖见 `pyproject.toml` 文档组：`mkdocs-material`、`mkdocstrings[python]`、`mkdocs-gen-files`、`mkdocs-literate-nav`、`mkdocs-mermaid2-plugin`。

## 文档清单

### 顶层

| 文件 | 内容 |
|------|------|
| `index.md` | 站点首页 |
| `getting-started.md` | 安装与快速上手 |

### API 参考 (`api/`)

逐模块的 API 文档（mkdocstrings 自动生成）：`index`、`detectors`、`classifiers`、`ocr`、`pipeline`、`result`、`config`、`eval`、`tracking`、`utils`。

### 使用指南 (`guides/`)

| 文件 | 内容 |
|------|------|
| `annotator_usage.md` | Supervision 标注器用法 |
| `cli-tools.md` | 命令行工具 |
| `evaluation_guide.md` | 模型评估指南 |
| `mcp.md` | MCP 服务接口 |

### 跟踪算法 (`guides/tracking/`)

`index.md` 为入口，按算法/主题分篇：

- **基础**：`math-foundations`、`metrics`、`traditional-methods`
- **SORT 系**：`ocsort`、`deep-ocsort`、`hybrid-sort`、`memosort`、`strongsort`、`botsort`、`bytetrack`
- **JDE/FairMOT 系**：`jde`、`jde-family`、`fairmot`、`centertrack`
- **Transformer 系**：`transformer-mot`、`trackformer`、`motr`、`motrv2`、`matr`、`putr`、`fasttracktr`

### 模型支持 (`models/`)

| 文件 | 内容 |
|------|------|
| `model_support_list.md` | 支持的模型清单 |

### Polygraphy 指南 (`polygraphy使用指南/`，不进站点)

| 路径 | 内容 |
|------|------|
| `README.md` / `目录索引.md` | 概述与目录 |
| `run.md` `convert.md` `inspect.md` `debug.md` `surgeon.md` `check.md` | 各命令详解 |
| `api_guide.md` | API 编程指南 |
| `faq.md` | 常见问题 |
| `how-to/` | 操作指南（调试精度、自定义输入、降精度处理、debug/reduce 子工具） |
| `examples/` | `api/` `cli/` `dev/` 示例代码 |

### 其他

- `rfdetr_coco_evaluation_report.md`：RF-DETR COCO 评估报告（不进站点）
- `_hooks/`、`javascripts/`、`stylesheets/`：MkDocs 构建钩子与静态资源

## 维护约定

- 公共 API、模块结构、CLI、配置或工作流变更时，同步更新对应文档（见根 CLAUDE.md「Keep Docs in Sync」）。
- 重命名/删除符号后，grep 所有 `*.md` 修正引用与链接。
