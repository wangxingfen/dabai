---
name: mixamo-anim
description: 操作大白自己的 Mixamo 动作库（D:\AI\dabai\web\anim）：下载 3D 表情动作（自动 Without Skin）、管理动作文件与 animation-library.json、优化情绪映射。大白需要下载新动作、整理动作库、检查缺失文件、调整情绪映射时使用本 skill。
---

# Mixamo 动作库操作手册

项目根目录：`D:\AI\dabai`。本技能直接复用 `mixamo_download_service.py`
（Playwright + fq 代理，与 server.py 共用同一单例），管理/优化直接操作
`web/anim/` 目录和 `web/anim/animation-library.json` 配置，无需启动额外服务。

## 心智模型

```
anim_start(起浏览器) → 登录(唯一需用户参与) → anim_download/anim_batch(下载)
    → anim_library(管理/归类) → anim_optimize(优化情绪映射)
```

- 下载服务是**单例**，与网页控制台（`/anim-library`）共享状态，两边不要同时操作；
- 下载自动选 **Without Skin**（纯动作 FBX，约 560KB，无网格），这是硬性要求；
- 登录是**唯一需要用户参与的环节**：Adobe 会话会过期，每次会话需在弹出窗口登录一次；
- 批量下载是后台异步任务，启动后立即返回，用 `anim_status` 轮询进度。

## 标准工作流

### 1. 查看状态

```text
anim_status
```

返回：浏览器是否运行、登录状态、代理、正在下载/队列、动作库各分类在盘数、
缺失动作、未注册文件、最近日志。**任何操作前先看这个。**

### 2. 启动浏览器并登录

```text
anim_start            # 自动起 clash 代理 + Playwright 浏览器（弹出窗口）
anim_check_login      # 检测是否已登录
```

若未登录：**让用户在弹出的浏览器窗口完成 Mixamo/Adobe 登录**（这是唯一手动步骤），
登录完成后：

```text
anim_check_login      # 确认已登录
anim_save_cookies     # 保存 cookies（下次重启浏览器可免登录）
```

### 3. 下载动作

单个：

```text
anim_download { "name": "idle_normal" }
```

批量（后台执行，立即返回）：

```text
anim_batch { "names": ["idle_normal", "wave", "laugh"] }
```

- 已存在的动作会自动跳过（`status: skipped`）；
- 批量未登录会拒绝启动——先完成登录再重试；
- 批量进行中用 `anim_status` 轮询，`anim_stop` 可中止。

### 4. 管理动作库

```text
anim_library { "action": "verify" }        # 配置 vs 磁盘：缺失/未注册文件
anim_library { "action": "scan" }          # 扫描磁盘 .fbx 并匹配配置
anim_library { "action": "list", "status": "missing" }   # 列缺失动作
anim_library { "action": "categorize", "dry_run": true } # 预览归类（放错位置的文件）
anim_library { "action": "categorize", "dry_run": false }# 实际移动文件到分类目录
```

### 5. 优化情绪映射

```text
anim_optimize { "action": "validate" }     # 校验配置结构
anim_optimize { "action": "emotions" }     # 查看 emotionMap 及在盘情况
anim_optimize { "action": "fix" }          # 自动修复（写回配置）
```

`fix` 会：清理 emotionMap 中引用不存在动作的条目、把未进入任何情绪池的动作按
自身 `emotion` 字段加入对应情绪池。

## 工具速查

| 工具 | 作用 |
| --- | --- |
| `anim_status` | 综合状态：浏览器/登录/代理 + 动作库统计 + 日志 |
| `anim_start [proto]` | 启动代理浏览器（自动起 clash） |
| `anim_stop` | 关闭浏览器（中止批量） |
| `anim_check_login` | 检测登录（ims_sid/idg_token + 页面无 Log In 按钮） |
| `anim_save_cookies` | 保存 cookies 到 data/mixamo_cookies.json |
| `anim_download name` | 下载单个动作（自动 Without Skin） |
| `anim_batch names[]` | 批量下载（后台，需已登录） |
| `anim_library action` | 管理：stats/list/scan/verify/categorize |
| `anim_optimize action` | 优化：validate/emotions/fix |

## 动作库结构

```
web/anim/
├── animation-library.json    # 配置：categories（分类+动作）+ emotionMap（情绪映射）
├── idle/  gesture/  emotion/  walk/  dance/  pose/   # 分类目录，.fbx 文件
└── index.html                # 网页控制台（/anim-library，与技能共享状态）
```

配置条目字段：`name`（动作标识）、`file`（分类目录/文件名.fbx）、`emotion`（情绪标签）、
`loop`、`description`，可选 `search`（Mixamo 搜索词）。

支持的情绪标签：`happy, excited, shy, sad, pout, angry, surprised, thoughtful,
calm, proud, tired, playful, love, neutral`。

## 排障

- **批量下载没反应**：先 `anim_status` 看 `is_running` 和 `is_logged_in`；
  未登录会被拒绝启动，先登录 + `anim_save_cookies`。
- **下载失败 `card timeout` / `card not found`**：多为未登录或代理不通。
  检查登录状态；代理问题用 fq-proxy 技能（`fq_ctl`）换协议/换 IP。
- **下载到带皮肤的 9MB 文件**：说明 Without Skin 选择失败（正常应为 ~560KB 纯动作）。
  重新下载该动作，若持续失败需检查 Mixamo 弹窗结构（服务端已内置多种兜底策略）。
- **动作库缺文件**：`anim_library verify` 看缺失清单，`anim_batch` 补齐。
- **文件放错位置**：`anim_library categorize` 自动移到配置指定分类目录。

## 安全红线

- 批量下载前必须确认已登录（`anim_check_login`），不要盲目启动；
- 登录凭据只存本机 `data/mixamo_cookies.json`，不外发；
- `categorize` 默认 dry_run 预览，确认无误再实际移动；
- 不要与网页控制台（/anim-library）同时操作下载服务。
