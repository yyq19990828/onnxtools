[根目录](../CLAUDE.md) > **openspec**

# OpenSpec规范管理系统 (openspec)

## 模块职责

提供规范驱动开发(Spec-Driven Development)流程支持,管理功能规范(specs)和变更提案(changes),确保新功能开发有据可依、有规可循。OpenSpec是一个轻量级的规范管理工具,帮助团队在编码前明确需求和设计决策。

## 入口和启动

- **AI助手指南**: `openspec/AGENTS.md` - 为AI编码助手提供OpenSpec工作流程说明
- **项目约定**: `openspec/project.md` - 项目特定的约定和上下文(待填充)
- **规范目录**: `specs/` - 已实现的功能规范(当前真实状态)
- **变更目录**: `changes/` - 待实施的变更提案(未来计划)
- **归档目录**: `changes/archive/` - 已完成的变更记录

### 快速开始

```bash
# 查看活跃的变更提案
openspec list

# 查看现有功能规范
openspec list --specs

# 查看特定变更详情
openspec show <change-id>

# 查看特定规范详情
openspec show <spec-id> --type spec

# 验证变更提案
openspec validate <change-id> --strict

# 归档已完成的变更
openspec archive <change-id> --yes
```

## 外部接口

### 1. OpenSpec CLI命令

```bash
# 核心命令
openspec list                    # 列出活跃变更
openspec list --specs             # 列出功能规范
openspec show <item>              # 显示变更或规范详情
openspec validate <item>          # 验证变更或规范
openspec archive <change-id> [--yes|-y]  # 归档变更(--yes跳过确认)

# 项目管理
openspec init [path]              # 初始化OpenSpec系统
openspec update [path]            # 更新指令文件

# 交互模式
openspec show                     # 提示选择要查看的项目
openspec validate                 # 批量验证模式

# 调试命令
openspec show <change> --json --deltas-only  # 查看变更的delta详情
openspec validate <change> --strict           # 严格验证
```

### 2. 命令标志

| 标志 | 作用 |
|------|------|
| `--json` | 机器可读的JSON输出 |
| `--type change\|spec` | 消歧义:指定查看的是变更还是规范 |
| `--strict` | 综合验证(推荐用于提交前检查) |
| `--no-interactive` | 禁用交互提示 |
| `--skip-specs` | 归档时跳过规范更新 |
| `--yes`/`-y` | 跳过确认提示(非交互式归档) |
| `--deltas-only` | 只显示变更的delta部分 |

## 模块结构

```
openspec/
├── AGENTS.md                   # AI助手工作流程指南(457行)
├── project.md                  # 项目约定和上下文(32行,待填充)
│
├── specs/                      # 功能规范(已实现的真实状态)
│   └── <capability>/
│       ├── spec.md             # 需求和场景
│       └── design.md           # 技术模式(可选)
│
└── changes/                    # 变更提案(未来计划)
    ├── <change-id>/            # 单个变更提案
    │   ├── proposal.md         # 变更说明(为什么、改什么、影响)
    │   ├── tasks.md            # 实施清单
    │   ├── design.md           # 技术决策(可选)
    │   └── specs/              # Delta规范变更
    │       └── <capability>/
    │           └── spec.md     # ADDED/MODIFIED/REMOVED需求
    └── archive/                # 已完成的变更
        └── YYYY-MM-DD-<change-id>/  # 归档时间戳+变更ID
```

## 关键依赖和配置

### OpenSpec工具依赖
- **openspec CLI**: 规范管理命令行工具(需要独立安装)
- **ripgrep (rg)**: 全文搜索工具(用于跨规范搜索)

### 规范文件格式约定

**Requirement格式** (必须使用三级标题):
```markdown
### Requirement: Feature Name
The system SHALL provide...

#### Scenario: Success case
- **WHEN** user performs action
- **THEN** expected result
```

**Delta操作标记**:
```markdown
## ADDED Requirements        # 新增功能
## MODIFIED Requirements     # 修改现有功能(需粘贴完整需求)
## REMOVED Requirements      # 删除功能
## RENAMED Requirements      # 重命名(需同时使用MODIFIED更新内容)
```

**重要约束**:
- 每个Requirement必须至少有一个Scenario
- Scenario必须使用`#### Scenario: Name`格式(四级标题)
- 不能使用bullet points或bold标记Scenario
- MODIFIED操作必须粘贴完整的需求内容(包括所有Scenario)

## 数据模型

### 变更提案模型
```markdown
# proposal.md
## Why
[1-2句话说明问题或机会]

## What Changes
- [变更列表]
- [标记破坏性变更: **BREAKING**]

## Impact
- 影响的规范: [capability列表]
- 影响的代码: [关键文件/系统]
```

### 任务清单模型
```markdown
# tasks.md
## 1. Implementation
- [ ] 1.1 Create database schema
- [ ] 1.2 Implement API endpoint
- [ ] 1.3 Add frontend component
- [ ] 1.4 Write tests

## 2. Documentation
- [ ] 2.1 Update API docs
- [ ] 2.2 Add user guide
```

### 设计文档模型(可选)
```markdown
# design.md
## Context
[背景、约束、利益相关者]

## Goals / Non-Goals
- Goals: [...]
- Non-Goals: [...]

## Decisions
- Decision: [决策内容和原因]
- Alternatives considered: [备选方案及取舍]

## Risks / Trade-offs
- [风险] → 缓解措施

## Migration Plan
[迁移步骤、回滚策略]

## Open Questions
- [待解决问题]
```

**何时创建design.md**:
- 跨模块变更或新架构模式
- 新增外部依赖或重大数据模型变更
- 安全、性能或迁移复杂性高
- 需要在编码前明确技术决策

### 规范Delta模型
```markdown
# changes/<change-id>/specs/<capability>/spec.md

## ADDED Requirements
### Requirement: Two-Factor Authentication
Users MUST provide a second factor during login.

#### Scenario: OTP required
- **WHEN** valid credentials are provided
- **THEN** an OTP challenge is required

## MODIFIED Requirements
### Requirement: User Login
[完整的修改后需求,包括所有Scenario]

## REMOVED Requirements
### Requirement: Old Feature
**Reason**: [删除原因]
**Migration**: [迁移方案]

## RENAMED Requirements
- FROM: `### Requirement: Login`
- TO: `### Requirement: User Authentication`
```

## 工作流程

### 阶段1: 创建变更提案

**何时创建提案**:
- 添加新功能或能力
- 破坏性变更(API、数据模式)
- 架构或模式变更
- 性能优化(改变行为)
- 安全模式更新

**何时跳过提案**:
- Bug修复(恢复预期行为)
- 拼写、格式、注释
- 依赖更新(非破坏性)
- 配置变更
- 为现有行为添加测试

**创建步骤**:
1. 搜索现有工作: `openspec list --specs`, `openspec list`
2. 选择唯一的`change-id`(kebab-case,动词开头,如`add-two-factor-auth`)
3. 创建目录结构: `openspec/changes/<change-id>/`
4. 编写`proposal.md`、`tasks.md`、可选的`design.md`
5. 为受影响的capability创建delta规范
6. 验证: `openspec validate <change-id> --strict`
7. 请求审批(在实施前)

**示例**:
```bash
# 1. 探索当前状态
openspec list --specs
openspec list

# 2. 选择change ID并创建结构
CHANGE=add-two-factor-auth
mkdir -p openspec/changes/$CHANGE/{specs/auth}

# 3. 创建提案文件
cat > openspec/changes/$CHANGE/proposal.md << 'EOF'
## Why
增强账户安全性,防止密码泄露导致的未授权访问

## What Changes
- 添加两步验证(2FA)功能
- 支持TOTP和短信OTP
- **BREAKING**: 登录流程需要额外步骤

## Impact
- 影响的规范: auth
- 影响的代码: auth/login.py, auth/models.py
EOF

# 4. 创建任务清单
cat > openspec/changes/$CHANGE/tasks.md << 'EOF'
## 1. Implementation
- [ ] 1.1 添加2FA数据模型
- [ ] 1.2 实现OTP生成和验证
- [ ] 1.3 更新登录API
- [ ] 1.4 编写测试
EOF

# 5. 创建delta规范
cat > openspec/changes/$CHANGE/specs/auth/spec.md << 'EOF'
## ADDED Requirements
### Requirement: Two-Factor Authentication
用户必须在登录时提供第二因子验证。

#### Scenario: TOTP验证
- **WHEN** 用户输入正确密码
- **THEN** 系统要求输入TOTP代码
- **AND** 验证成功后允许登录
EOF

# 6. 验证
openspec validate $CHANGE --strict
```

### 阶段2: 实施变更

**实施顺序** (作为TODO追踪):
1. 阅读`proposal.md` - 理解变更目的
2. 阅读`design.md`(如果存在) - 查看技术决策
3. 阅读`tasks.md` - 获取实施清单
4. 按顺序实施任务 - 逐项完成
5. 确认完成 - 确保`tasks.md`中的每个任务都已完成
6. 更新清单 - 所有任务完成后,将每项标记为`- [x]`
7. 审批门 - 在开始实施前,提案必须经过审查和批准

### 阶段3: 归档变更

**归档时机**: 变更部署到生产环境后

**归档步骤**:
```bash
# 方式1: 使用CLI工具(推荐)
openspec archive <change-id> --yes

# 方式2: 手动归档
# 1. 移动变更目录到archive/
mkdir -p openspec/changes/archive/
mv openspec/changes/<change-id> openspec/changes/archive/$(date +%Y-%m-%d)-<change-id>

# 2. 更新specs/(如果capability变更)
# 将delta合并到对应的specs/<capability>/spec.md

# 3. 验证归档后的状态
openspec validate --strict
```

**归档后文件结构**:
```
openspec/changes/archive/2025-11-07-add-two-factor-auth/
├── proposal.md
├── tasks.md
├── design.md
└── specs/
    └── auth/
        └── spec.md  # delta规范(保留用于历史追溯)
```

## 项目中的OpenSpec使用

### 现有规范 (specs/)

项目当前有6个已完成的规范(位于`specs/`目录):

1. **001-supervision-plate-box/** - Supervision库可视化集成
   - 状态: In Progress (Phase 2)
   - 集成13种Annotator类型和5种预设场景

2. **002-delete-old-draw/** - 旧版绘制代码重构
   - 状态: Completed
   - 移除PIL绘图实现,完全迁移到Supervision

3. **003-add-more-annotators/** - Annotators扩展集成
   - 状态: Completed
   - 添加几何、填充、隐私、特效类annotators

4. **004-refactor-colorlayeronnx-ocronnx/** - ColorLayerONNX和OCRONNX重构
   - 状态: Completed
   - 统一推理接口,继承BaseORT

5. **005-baseonnx-postprocess-call/** - BaseOnnx抽象方法强制实现
   - 状态: Completed
   - 重构`__call__`方法,代码减少83.3%

6. **006-make-ocr-metrics/** - OCR指标评估功能
   - 状态: Completed (Phase 4/7)
   - 完全匹配率、编辑距离、相似度指标

### 规范使用示例

查看已完成规范的实施历史:
```bash
# 列出所有规范
openspec list --specs

# 查看特定规范
openspec show 001-supervision-plate-box --type spec

# 查看归档的变更
ls openspec/changes/archive/
```

## 常见问题 (FAQ)

### Q: 何时使用OpenSpec创建提案?
A: 创建提案适用于:
- 新功能或能力开发
- 破坏性变更(API、数据模式)
- 架构调整或性能优化
- 需要跨团队协调的变更
- 不明确的需求(通过提案澄清)

跳过提案适用于:
- Bug修复、拼写错误、格式调整
- 非破坏性依赖更新
- 配置变更

### Q: 如何搜索现有规范?
A:
```bash
# 列出所有规范和变更
openspec list --specs
openspec list

# 显示特定规范详情
openspec show <spec-id> --type spec

# 全文搜索(使用ripgrep)
rg -n "Requirement:|Scenario:" openspec/specs
rg -n "^#|Requirement:" openspec/changes
```

### Q: 为什么Scenario解析失败?
A: 常见错误:
- **错误**: 使用`- **Scenario: Name**`(bullet point + bold)
- **错误**: 使用`### Scenario: Name`(三级标题)
- **正确**: 使用`#### Scenario: Name`(四级标题)

调试方法:
```bash
# 查看delta解析结果
openspec show <change-id> --json --deltas-only

# 严格验证
openspec validate <change-id> --strict
```

### Q: MODIFIED操作如何正确使用?
A: MODIFIED操作会**完全替换**现有需求,因此必须粘贴完整内容:

**错误做法** (会丢失原有详情):
```markdown
## MODIFIED Requirements
### Requirement: User Login
添加两步验证支持
```

**正确做法** (保留原有内容+新增内容):
```markdown
## MODIFIED Requirements
### Requirement: User Login
用户必须提供凭证进行登录。**新增**: 支持两步验证。

#### Scenario: 成功登录
- **WHEN** 用户输入正确凭证
- **THEN** 系统验证并授予访问权限

#### Scenario: 两步验证流程
- **WHEN** 用户启用2FA
- **THEN** 系统要求提供第二因子
```

### Q: 如何处理跨多个capability的变更?
A: 为每个受影响的capability创建独立的delta文件:

```
openspec/changes/add-2fa-notify/
├── proposal.md
├── tasks.md
└── specs/
    ├── auth/
    │   └── spec.md   # ADDED: Two-Factor Authentication
    └── notifications/
        └── spec.md   # ADDED: OTP email notification
```

### Q: design.md何时必须创建?
A: 以下情况建议创建`design.md`:
- 跨服务/模块变更或新架构模式
- 新增外部依赖或重大数据模型变更
- 安全、性能或迁移复杂性高
- 技术决策不明确,需要在编码前讨论

简单功能或单模块变更可以跳过`design.md`。

### Q: 归档后的变更可以恢复吗?
A: 可以。归档的变更保留在`changes/archive/`目录,包含完整的提案和delta信息,可用于:
- 历史追溯和审计
- 回滚参考
- 类似功能的实施参考

### Q: OpenSpec与Git工作流如何集成?
A: 推荐工作流:
1. 创建功能分支(如`feature/add-2fa`)
2. 在分支中创建OpenSpec提案并提交
3. 提案通过审查后,在同一分支实施
4. 部署后,在单独的PR中归档变更
5. 归档PR合并到主分支

## 相关文件列表

### OpenSpec系统文件
- `openspec/AGENTS.md` - AI助手工作流程指南(457行,完整)
- `openspec/project.md` - 项目约定和上下文(32行,待填充)

### 项目规范
- `specs/001-supervision-plate-box/` - Supervision可视化集成(进行中)
- `specs/002-delete-old-draw/` - 旧版绘制代码重构(已完成)
- `specs/003-add-more-annotators/` - Annotators扩展(已完成)
- `specs/004-refactor-colorlayeronnx-ocronnx/` - OCR重构(已完成)
- `specs/005-baseonnx-postprocess-call/` - BaseORT优化(已完成)
- `specs/006-make-ocr-metrics/` - OCR评估功能(已完成)

### 归档变更
- `openspec/changes/archive/` - 已完成变更的历史记录

### 根级集成
- `CLAUDE.md` - 根级文档,包含OpenSpec使用说明(<!-- OPENSPEC:START -->块)

## 最佳实践

### 简单优先
- 默认<100行新代码的变更
- 单文件实施,直到证明需要更复杂结构
- 避免引入框架,除非有明确理由
- 选择已验证的、稳定的模式

### 复杂性触发条件
仅在以下情况添加复杂性:
- 性能数据显示当前方案过慢
- 具体的规模需求(>1000用户、>100MB数据)
- 多个已验证用例需要抽象

### 清晰引用
- 使用`file.ts:42`格式引用代码位置
- 使用`specs/auth/spec.md`格式引用规范
- 链接相关变更和PR

### Capability命名
- 使用动词-名词格式: `user-auth`, `payment-capture`
- 单一职责per capability
- 10分钟可理解性规则
- 如果描述需要"AND",考虑拆分

### Change ID命名
- 使用kebab-case,简短且描述性: `add-two-factor-auth`
- 优先使用动词开头: `add-`, `update-`, `remove-`, `refactor-`
- 确保唯一性;如果重复,追加`-2`、`-3`等

## 变更日志 (Changelog)

**2025-11-07** - 创建OpenSpec模块文档
- 初始化完整的OpenSpec系统文档
- 记录CLI命令、工作流程和文件格式
- 补充项目中现有的6个规范清单
- 添加常见问题和最佳实践
- 提供创建提案、实施和归档的完整示例

**2025-09-30** - OpenSpec系统集成
- 项目采用规范驱动开发流程
- 完成5个规范实施(002-006)
- 1个规范进行中(001 Phase 2)

**2025-09-15** - OpenSpec初始化
- 建立openspec目录结构
- 创建AGENTS.md指南

---

*模块路径: `/home/tyjt/桌面/onnx_vehicle_plate_recognition/openspec/`*
*最后更新: 2025-11-07 16:35:25*
