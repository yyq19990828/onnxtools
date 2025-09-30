[根目录](../CLAUDE.md) > **docs**

# 项目文档模块 (docs)

## 模块职责

存储和管理项目文档、使用指南、工具说明和技术文档，为开发者和用户提供全面的参考资料。

## 入口和启动

- **评估指南**: `evaluation_guide.md` - 模型评估完整指南
- **Polygraphy指南**: `polygraphy使用指南/` - NVIDIA Polygraphy工具详细文档

## 外部接口

### 文档访问
```bash
# 查看评估指南
cat docs/evaluation_guide.md

# 浏览Polygraphy文档
ls docs/polygraphy使用指南/

# 查看特定工具文档
cat docs/polygraphy使用指南/run.md
cat docs/polygraphy使用指南/debug.md
```

### 文档结构
```
docs/
├── evaluation_guide.md          # 模型评估指南
├── polygraphy使用指南/          # Polygraphy完整文档
│   ├── README.md               # Polygraphy概述
│   ├── run.md                  # run命令使用
│   ├── convert.md              # convert命令使用
│   ├── inspect.md              # inspect命令使用
│   ├── debug.md                # debug命令使用
│   ├── surgeon.md              # surgeon命令使用
│   ├── check.md                # check命令使用
│   ├── faq.md                  # 常见问题
│   ├── api_guide.md            # API编程指南
│   ├── how-to/                 # 操作指南
│   └── examples/               # 示例代码
└── ...                         # 其他文档
```

## 关键依赖和配置

### 文档格式
- **Markdown**: 主要文档格式
- **Mermaid**: 流程图和架构图
- **代码示例**: 嵌入式代码块
- **图片**: PNG/SVG图表

### 文档编写规范
```markdown
# 文档标题

## 概述
简要说明文档目的和适用范围

## 前提条件
列出必要的环境和依赖

## 步骤/内容
详细的操作步骤或内容说明

## 示例
实际的代码示例和命令

## 常见问题
FAQ和疑难解答

## 参考资料
相关链接和扩展阅读
```

## 数据模型

### 文档元数据
```yaml
document:
  title: "模型评估指南"
  type: "user_guide"  # user_guide, api_doc, tutorial, reference
  version: "1.0.0"
  last_updated: "2025-09-30"
  author: "Project Team"
  tags: ["evaluation", "coco", "metrics"]
  related_docs: ["polygraphy使用指南/"]
```

### Polygraphy文档分类
```yaml
polygraphy_docs:
  command_reference:
    - run: "模型推理和对比"
    - convert: "模型格式转换"
    - inspect: "模型结构检查"
    - debug: "精度调试工具"
    - surgeon: "ONNX模型手术"
    - check: "模型验证"

  how_to_guides:
    - "使用自定义输入数据"
    - "调试精度问题"
    - "处理降精度"
    - "有效使用debug工具"

  examples:
    - api: "API编程示例"
    - cli: "命令行示例"
    - dev: "开发者示例"
```

## 测试和质量

### 文档完整性检查
- [ ] 所有命令有使用说明
- [ ] 所有示例可运行
- [ ] 链接有效性验证
- [ ] 代码块语法正确

### 文档维护
- [ ] 定期更新版本号
- [ ] 修复过期信息
- [ ] 添加新功能文档
- [ ] 收集用户反馈

## 常见问题 (FAQ)

### Q: 如何为新功能编写文档？
A: 1) 确定文档类型（指南/参考/教程）; 2) 遵循文档模板; 3) 提供实际示例; 4) 添加到文档索引; 5) 请他人审阅

### Q: 文档和代码不一致怎么办？
A: 1) 优先更新文档; 2) 在代码注释中引用文档; 3) CI中添加文档检查; 4) 定期审核文档准确性

### Q: Polygraphy文档太长怎么快速查找？
A: 1) 查看README目录索引; 2) 使用grep搜索关键词; 3) 参考examples/快速开始; 4) 查阅FAQ常见问题

### Q: 如何贡献文档改进？
A: 1) 发现问题记录issue; 2) 提交PR修复; 3) 添加示例代码; 4) 完善中文翻译

## 相关文件列表

### 核心文档
- `evaluation_guide.md` - 模型评估完整指南
- `README.md` - 项目主README（根目录）

### Polygraphy完整文档集
- `polygraphy使用指南/README.md` - Polygraphy概述
- `polygraphy使用指南/run.md` - run命令详解
- `polygraphy使用指南/convert.md` - convert命令详解
- `polygraphy使用指南/inspect.md` - inspect命令详解
- `polygraphy使用指南/debug.md` - debug命令详解
- `polygraphy使用指南/surgeon.md` - surgeon命令详解
- `polygraphy使用指南/check.md` - check命令详解
- `polygraphy使用指南/faq.md` - 常见问题
- `polygraphy使用指南/api_guide.md` - API编程指南
- `polygraphy使用指南/目录索引.md` - 完整目录

### 操作指南
- `polygraphy使用指南/how-to/debug_accuracy.md` - 调试精度问题
- `polygraphy使用指南/how-to/use_custom_input_data.md` - 自定义输入
- `polygraphy使用指南/how-to/work_with_reduced_precision.md` - 降精度处理
- `polygraphy使用指南/how-to/use_debug_subtools_effectively.md` - 调试工具使用
- `polygraphy使用指南/how-to/use_debug_reduce_effectively.md` - reduce工具使用

### 示例代码
- `polygraphy使用指南/examples/api/` - API编程示例
- `polygraphy使用指南/examples/cli/` - 命令行示例
- `polygraphy使用指南/examples/dev/` - 开发者扩展示例

## 变更日志 (Changelog)

**2025-09-30 11:05:14 CST** - 初始化项目文档模块，建立文档管理体系和Polygraphy完整指南

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/docs/`*