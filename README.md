# 大白（dabai）

> 一个自进化 AI 智能体运行时：3D 虚拟形象 + LLM 决策 + 技能/插件扩展框架 + 任务系统。

![Python](https://img.shields.io/badge/Python-3.13-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-blue) ![JavaScript](https://img.shields.io/badge/JavaScript-blue) ![HTML](https://img.shields.io/badge/HTML-blue)

## ✨ 项目简介

「大白」是一个完整的 **AI 智能体运行时**：以 LLM 为核心决策者，驱动 3D 虚拟形象
（VRM 模型）实时互动，并通过 **harness 监督运行时 + 技能（Skill）/ 插件（Plugin）
扩展框架**持续获得新能力——无需改动核心代码，放一份文件进 `skills/` 即获得新工具。

**规模**：约 1,400+ 代码文件 / 30 万行（Python 7.5 万 + TypeScript 5.7 万 + Markdown 10.9 万 + JSON 4 万）。

## 🧠 核心架构

```
大白 主 Agent（agent.py AIAgent）—— 唯一智能体底座
 ├── LLM 决策（chat/game/decision/character_line/memory/plan 六渠道）
 │     └── 全部经 harness.runtime.supervise_llm（熔断 + 重试 + 超时 + token 计量）
 ├── 工具执行（技能/插件/内置兜底，已全面 skill 化）
 │     └── 全部经 harness.runtime.supervise_tool；同一轮多个工具自动并行
 ├── 3D 形象（VRM 模型 + 动作 + 表情 + TTS 语音）
 ├── 分层记忆（短期窗口 / 长期摘要 / 常驻 top-k / 按需召回，实测省 token 70%+）
 └── harness/                    稳定扩展与控制层
       ├── core.py     Harness 门面：工具收集 / 稳定路由 / 健康状态 / 热重载
       ├── runtime.py  Agent 监督运行时：熔断器 / 重试 / 超时 / 计量 / RunSpan
       ├── tasks.py    任务系统：队列调度 / DAG 流程 / 批量 / 持久化断点续跑
       ├── skills.py   技能注册表（skills/<名称>/）
       ├── plugins.py  插件管理器（plugins/<名称>/）
       └── state.py    启停状态持久化
```

## 🧩 技能体系（skills/）

| 技能 | 能力 |
|------|------|
| `code_ops` | 代码工程一体化：检索/分析/修改/验证（AST 感知）、git 全流程、隔离工作树、GitHub 协作 |
| `github` | GitHub 智能体：gh CLI 全套（仓库/PR/issue/搜索/CI/release）+ PR 深度审查工作流 |
| `agent_ops` | 委派外部智能体（codex/opencode/DSH）、子智能体、技能开发/拉取 |
| `search` | 四引擎合一搜索（anysearch/tavily/exa/web）+ 翻墙代理 |
| `appearance` | 3D 形象管理：VRM/PMX 模型、Mixamo 动作、换装 |
| `media` | 媒体能力：音乐/图片/视频 |
| `tasks` | 任务中心：长任务/批量任务/队列统摄 |
| `smell-check-main` | 代码坏味道检查 |

> 渐进式披露：`on_demand` 技能按需注入（一句话摘要 + `skill_help` 拉取完整说明书），
> 技能再多也不撑爆上下文。

## 🚀 快速开始

```bash
# 安装依赖
npm install

# 启动（含 harness 运行时 + 3D 前端 + 管理台）
npm start
```

## 🖥 管理台

浏览器打开 `http://<大白地址>/harness`：

- 技能/插件热启停、热重载
- 任务与队列控制（暂停/恢复/取消/重试）
- 运行时健康状态、熔断器复位、token 用量

REST API：`/api/harness/status`、`/api/harness/skills`、`/api/harness/plugins`、`/api/harness/reload`、`/api/harness/runtime`

## 📁 项目结构

```text
dabai/
    ├── agent.py            主 Agent（LLM 决策 + 工具路由 + 记忆）
    ├── server.py           服务端（WebSocket + REST API）
    ├── harness/            监督运行时：core / runtime / tasks / skills / plugins
    ├── skills/             技能体系（code_ops / github / search / appearance / media ...）
    ├── web/                3D 前端（VRM 渲染 / 交互 / 大屏）
    ├── models/             VRM 形象模型库
    ├── tools/              工具链（gh CLI / mixamo / VRM 转换 / gpt_sovits）
    ├── rl_coordinator.py   RL 双轨协调（AgentBandit / PushPull / 校准器）
    ├── memory.py           分层记忆
    └── settings.json       配置
```

## 🧠 关键机制

- **监督运行时**：每次 LLM 调用与工具执行都经 harness——瞬时错误自动退避重试、
  整体超时保护、连续失败自动熔断（冷却后半开探测恢复）、全量计量与 token 记账；
- **任务系统**：多步骤 DAG 流程与批量 map 后台执行，步骤级重试、流程级看门狗、
  失败策略（abort/continue）、**服务重启断点续跑**；
- **并行工具**：同一轮多个独立工具调用自动并行执行；
- **分层记忆**：短期窗口 + 长期摘要 + 常驻 top-k + 按需召回，实测省 token 70%+；
- **工具参数严格校验**：执行前按定义校验 required/类型/enum/边界，错误中文回填给模型自修正；
- **工具心跳**：长任务实时推送进度事件，不再"静默无输出"。

详细设计见 [HARNESS.md](HARNESS.md)、[MEMORY_HIERARCHY.md](MEMORY_HIERARCHY.md)。

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request。

## 📄 许可证

[MIT](LICENSE)