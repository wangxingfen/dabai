# skill_pull 安装与运维说明

`skill_pull`（全网拉取技能）让大白能按任务需求自动从 GitHub 检索、筛选、下载、
校验并安装成熟的 Agent Skill，形成可复用的技能库。

## 一、安装（3 步，约 1 分钟）

1. **放置目录**：把本目录（`skills/skill_pull/`，含 skill.json / SKILL.md / skill.py / INSTALL.md）
   完整复制到大白根目录的 `skills/` 下。若本文件已位于 `D:\AI\dabai\skills\skill_pull\`，已算安装完成。
2. **生效**：hot_reload 守护会在 1 秒内自动加载（无需重启服务）；
   若未生效，在 /harness 管理页点击「重载」，或请求 `POST http://127.0.0.1:8000/api/harness/reload`。
3. **验证**：对大白说「拉取一个网页抓取 skill」，或直接调用 `skill_help("skill_pull")` 查看本说明书。

环境要求：Python 3.10+（与大白一致），**仅使用标准库**，无需 pip 安装任何依赖。

## 二、调用方式

工具（OpenAI function-calling 定义，已注册进大白工具列表）：

| 工具 | 必填参数 | 可选参数 |
| --- | --- | --- |
| `skill_pull_search` | query | min_stars(默认50)、max_results(默认8) |
| `skill_pull_inspect` | repo | — |
| `skill_pull_install` | repo | select（合集内子路径）、force(默认false) |

返回：可读中文文本（候选清单 / 结构结论 / 审查报告 + 安装结果 + 使用说明）。

## 三、工作流程（安装一个技能的完整链路）

`任务描述 → 关键词提取（中文自动增强英文词）→ GitHub Search API 检索 →
star/活跃度/结构特征筛选 → 下载官方 tarball → 安全解压（路径穿越防护）→
结构校验（必须有 SKILL.md 或 skill.json）→ 静态安全审查（BLOCK 级规则命中即拒）→
注册 skills/<name>/（无 skill.json 时自动生成最小清单）→ POST /api/harness/reload 生效 →
输出 skill_help 使用说明`

## 四、安全边界（务必阅读）

| 边界 | 说明 |
| --- | --- |
| 来源白名单 | 仅 api.github.com / codeload.github.com；不接受任意 URL 直装 |
| 不执行代码 | 下载/解压/审查/复制全部纯静态；安装完成也不自动运行新技能代码 |
| BLOCK 级审查 | 系统 shell 执行、eval/exec、pickle、>1KB base64 混淆、下载并执行、进程注入、路径穿越 → 拒绝安装 |
| 体积限制 | 单文件 ≤ 1MB，解压总量 ≤ 16MB，超限中止 |
| 文件类型 | .exe/.dll/.bat/.ps1 等可疑后缀列出提示 |
| 幂等与覆盖 | 同名技能默认跳过；force=true 才覆盖（覆盖前先 rmtree 旧目录） |
| 名称白名单 | 技能目录名 ^[A-Za-z0-9_-]{1,64}$，拒绝非法字符 |
| 残留说明 | 静态审查通过 ≠ 绝对安全；异常时 /harness 管理页可禁用；缓存位于 data/skill_pull_cache/（不入库） |

## 五、配置项

- **GITHUB_TOKEN**：可选。写入环境变量 GITHUB_TOKEN 或 `data/github_token.txt`，
  将 GitHub API 配额从 60 次/小时提升到 5000 次/小时（用 unauthenticated 配额也能工作）。
- **RELOAD_PORTS**：skill.py 顶部元组 (8000, 8900, 7860)，按需调整本地服务端口。
- **成熟度门槛**：search 的 min_stars 参数（默认 50）。

## 六、卸载

删除 `skills/skill_pull/` 目录即可（hot_reload 自动移除）。若只禁用：/harness 管理页关闭该技能。

## 七、局限与后续扩展点

- 搜索源目前为 GitHub（覆盖绝大多数 Agent Skill 生态）；如需要 Bing/DuckDuckGo 补充检索、
  awesome-list 扫描、OTA 更新（git pull 式 diff 升级）、推荐安装（根据任务自动按相关性打分）等，
  可在 skill.py 中扩展 `_search_github_request` / `skill_pull_install` 相应函数。
- 合集仓库（多技能）默认全装；按需用 select 参数精确控制。
