# 全网拉取技能（skill_pull）

根据当前任务需求，自动从全网检索、拉取、校验并安装成熟的 Agent Skill（技能工具库），
装好后立即注册进大白技能体系，并输出每个新技能的使用说明。**本技能自身就是范例**：
通过 `skill_help("skill_pull")` 可以随时查看本说明书。

## 适用场景
- 用户说「拉一个 XX 的 skill」「找个会 XX 的技能」「帮我装个 XX 工具」
- 当前任务缺少合适工具，需要引入现成的开源能力（网页抓取、数据分析、翻译、写作、记忆、代码等）
- 想把社区成熟的技能库（如 anthropics/skills、obra/superpowers 风格的仓库）批量接入大白

## 工具与调用方式

| 工具 | 参数 | 作用 |
| --- | --- | --- |
| `skill_pull_search` | query（任务描述/关键词，必填）；min_stars=50；max_results=8 | 关键词提取 + GitHub API 检索，返回按 star 降序的候选清单（含成熟度标记） |
| `skill_pull_inspect` | repo（owner/name，必填） | 安装前深检：结构、许可证、文件树、可疑文件，给出「是否标准 Skill 仓库」结论 |
| `skill_pull_install` | repo（必填）；select（合集内子路径，可选）；force=false | 下载→安全解压→结构校验→静态安全审查→注册安装→热重载→输出使用说明 |

返回格式：均为可读文本；候选清单/审查报告/安装结果直接可转述给用户。

## 标准三步用法（示例）

### 示例 1：拉取一个网页抓取 skill
1. `skill_pull_search(query: "拉一个网页抓取 skill")` → 自动增强为 "网页抓取 web scraping agent skill"，
   返回候选（如 `browser-use/browser-use`、`Doriandarko/claude-web-scraper` 等，含 ⭐ 与更新时间）。
2. （可选）`skill_pull_inspect(repo: "目标仓库")` 核实结构。
3. `skill_pull_install(repo: "目标仓库")` → 审查通过后装入 `skills/<name>/`，
   返回新技能名称、工具列表与 SKILL.md 摘要。

### 示例 2：拉取一个数据分析 skill
1. `skill_pull_search(query: "数据分析")` → 自动增强为 "数据分析 data analysis agent skill"。
2. `skill_pull_install(repo: "选中的仓库", select: "skills/data-analysis")`（合集仓库指定子目录）。

### 示例 3：整车拉取官方 skills 合集
`skill_pull_install(repo: "anthropics/skills")` → 发现多个技能子目录，全部安装；
或 `select: "artifacts-builder"` 只装其中一个。安装输出含每个技能的使用说明，
后续用 `skill_help("技能名")` 随时查阅。

## 检索与筛选逻辑
- 关键词提取：中文任务描述自动映射英文增强词（抓取→web scraping、数据分析→data analysis 等）；
  否则自动追加 "agent skill" 提高命中率；使用 GitHub Search API（sort=stars）。
- 成熟度筛选：star 数门槛（默认 50）、最近推送时间、许可证、是否已归档（archived）均为参考维度。
- 结构筛选：描述含 "skill" 且含 agent/llm/claude 等词的仓库标记为「✓ 像 skill 仓库」。

## 安全边界（重要）
- **只从 GitHub 官方域下载**（api.github.com / codeload.github.com），不接受其他来源。
- **绝不执行下载的代码**：下载、解压、审查、复制全部是纯静态操作。
- 静态安全审查规则（命中即拒绝安装，BLOCK 级）：
  系统 shell 命令执行（os.system / subprocess shell=True）、eval/exec 动态执行、
  pickle 反序列化、>1KB 的 base64 混淆块、下载并执行模式、进程注入/底层系统调用、路径穿越写入。
- 体积与类型限制：单文件 ≤ 1MB、解压总量 ≤ 16MB；.exe/.dll/.bat 等可疑文件列出并提示。
- 技能名白名单（^[A-Za-z0-9_-]{1,64}$）；**幂等**：同一来源（仓库+仓库内路径）重装会自动跳过
  （force=true 才强制覆盖）；不同仓库的同名技能会自动加后缀保留两个版本。
- 静态审查通过 ≠ 绝对安全：安装输出会提示人工留意；运行异常可在 /harness 管理页禁用该技能。
- GitHub API 未配置 token 时约 60 次/小时配额；建议将 token 写入环境变量 GITHUB_TOKEN
  或 data/github_token.txt（提额到 5000 次/小时）。

## 安装后生效机制
- 安装即写入 `skills/<name>/`（skill.json + 原仓库文件），并尝试 POST 本地
  `/api/harness/reload` 触发热重载；即使失败，hot_reload 守护也会在 1 秒内自动加载。
- 被安装的技能在 system prompt 中以摘要形式注入（渐进式披露），详见 `skill_help("技能名")`。

## 常见问题
- 搜不到：改用英文关键词、降低 min_stars（如 10）、换更具体的描述。
- "未发现标准 Skill 结构"：该仓库不是 skill 仓库（没有 SKILL.md/skill.json），拒绝安装是符合预期的。
- 审查被拒：查看返回的命中规则与文件行号，人工核实 data/skill_pull_cache/ 中解压内容。
- 卸载：删除 `skills/<名字>/` 目录即可（管理页可禁用）。

## 文件
- skill.json —— 清单与三个工具定义（disclosure: on_demand，渐进披露）
- skill.py —— 实现（搜索/检查/下载/安全解压/静态审查/安装/热重载，全部标准库）
- INSTALL.md —— 安装与运维说明
