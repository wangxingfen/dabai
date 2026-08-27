# 技能工坊（skill_dev）—— 大白自建自改技能

本技能让大白具备**自我开发能力**：自己创建全新技能（Skill）、修改既有技能、
静态校验、热重载生效——全程无需人工写文件。用它创建的技能与手写技能完全等价，
因此大白可以持续给自己迭代新能力（自举）。**本技能自身也是用它这套规范写出来的范例**。

## 适用场景
- 用户说「给你加个 XX 能力」「帮我做个 XX 技能」「你能不能自己学会 XX」
- 当前任务缺少合适工具，需要把一段可复用的能力封装成技能
- 修改既有技能：改引导语/提示词、加或改工具、调整行为边界、改名、启停
- 校验某个技能是不是合格、为什么某个技能工具报错/不生效

## 技能体系速览（大白侧机制）

每个技能 = `skills/<技能名>/` 目录下的三件套：
| 文件 | 作用 |
| --- | --- |
| `skill.json` | 清单：name/title/version/description/author/enabled/disclosure/prompt/tools（OpenAI function-call 格式） |
| `skill.py` | 实现：TOOLS + PROMPT + HANDLERS（工具名→函数映射），或 execute(name,args) 分发器；也可只写 PROMPT |
| `SKILL.md` | 中文说明书：`skill_help("技能名")` 的数据源（渐进披露时按需拉取） |

生效机制：`hot_reload` 守护每秒扫描 skills/ 下 .py/.json/.md 变化 → 自动热重载（约 1 秒）；
本技能的工具也会主动 POST `/api/harness/reload` 即时生效。
enabled/disclosure 说明：enabled 默认 true（harness_state.json 记录覆盖）；
disclosure=on_demand（默认推荐）：对话里只注入一句话摘要（description 前 88 字），需要时用 `skill_help` 拉全文；
disclosure=full：完整 prompt 常驻系统提示词。

## 工具一览

| 工具 | 参数 | 作用 |
| --- | --- | --- |
| `skill_dev_list` | （无） | 列出全部技能及状态（启用/披露/工具数/损坏） |
| `skill_dev_read` | skill_name, file | 读取 skill.json / skill.py / SKILL.md（改前必读、可留备份） |
| `skill_dev_create` | name, title, description, prompt?, disclosure?, tools_spec?, overwrite? | 脚手架新技能三件套（含工具桩），重名默认拒绝 |
| `skill_dev_edit` | skill_name, field, value | 改清单字段；tools 整体替换；enabled 启停；name 自动改名目录 |
| `skill_dev_write_file` | skill_name, filename, content | 整体写入 skill.py / SKILL.md |
| `skill_dev_validate` | skill_name | 静态校验：JSON/命名/工具格式/HANDLERS 一致性/编译/SKILL.md |
| `skill_dev_reload` | （无） | 强制热重载并汇报运行状态与损坏清单 |
| `skill_dev_remove` | skill_name, confirm=true | 删除技能（无回收站） |

## 标准流程 A：创建一个全新技能

1. **skill_dev_list** —— 看现状（避免重名、工具名冲突）。
2. **skill_dev_create** —— 传 name/title/description/prompt/disclosure，有工具就传 tools_spec（JSON 数组字符串）。
   自动生成三件套骨架 + 工具桩，并热重载注册。
3. **skill_dev_write_file** 补全 skill.py（实现每个工具）与 SKILL.md（说明书）。
4. **skill_dev_validate** 校验，把所有 ✘ 清零；实测调用每个新工具一次验收（改动自动热重载，无需等）。

### skill.json 模板
```json
{
  "name": "weather",
  "title": "天气查询",
  "version": "1.0.0",
  "description": "查询城市实时天气与穿衣建议。",
  "author": "dabai",
  "enabled": true,
  "disclosure": "on_demand",
  "prompt": "【天气查询】用户问天气/气温/穿衣建议时用 weather_query 查询并给出建议。",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "weather_query",
        "description": "查询指定城市的当前天气，返回温度、天气现象与穿衣建议。",
        "parameters": {
          "type": "object",
          "properties": {
            "city": { "type": "string", "description": "城市名，如 上海" }
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

### skill.py 模板（HANDLERS 风格，推荐）
```python
"""天气查询（weather）—— 使用说明见 SKILL.md。"""
from __future__ import annotations

TOOLS = [...]                    # 与 skill.json 的 tools 保持一致（可省略，清单为准）
PROMPT = "【天气查询】……"        # 可选：会覆盖清单 prompt

def weather_query(args):
    city = args.get("city") or ""
    # ……真实实现（调用 API/本地数据），返回可读文本
    return f"上海 当前 26℃ 多云，建议薄外套。"

HANDLERS = {"weather_query": weather_query}
```
写法约定：handler 接收参数字典、返回 **可读文本**（会原样转述给用户）；
需要异步就 `async def`（await 自动支持）；多个工具就多注册几条。

### SKILL.md 模板
```markdown
# 天气查询（weather）

## 适用场景
- 用户问某地天气/气温/穿什么衣服

## 工具
| 工具 | 参数 | 作用 |
| --- | --- | --- |
| weather_query | city（必填） | 查询城市当前天气与穿衣建议 |

## 边界
- 只支持国内城市；查不到就如实说明，不要编造数据
```

## 标准流程 B：修改既有技能

| 要改什么 | 做法 |
| --- | --- |
| 引导语/描述/标题/启停/披露模式 | `skill_dev_edit`（field=prompt/description/title/enabled/disclosure） |
| 工具列表（加/删/改参数） | `skill_dev_read` 读清单 → `skill_dev_edit`(field=tools, value=新 JSON 数组) → 同步 skill.py |
| 工具实现/行为 | `skill_dev_read` 读 skill.py → `skill_dev_write_file` 整体写新版本 → validate |
| 说明书 | `skill_dev_write_file`(filename=SKILL.md) |
| 改名 | `skill_dev_edit`(field=name, value=新名)（自动重命名目录） |
| 复制出一个变体 | read 三件套 → create（新名）+ write_file 回填 → validate |

改完一律 **skill_dev_validate** 收尾：✘ 清零才合格，⚠ 建议处理。

## 完整实战示例（掷骰子 dice 技能）

1）创建脚手架：
```
skill_dev_create(
  name="dice", title="掷骰子",
  description="掷一个或多个骰子，返回每次点数与合计。",
  prompt="【掷骰子】用户要掷骰子/随机数/抽签时用 dice_roll。",
  tools_spec='[{"type":"function","function":{"name":"dice_roll","description":"掷 n 个 m 面骰子，返回每次点数与合计。","parameters":{"type":"object","properties":{"count":{"type":"integer","description":"骰子个数，默认 1"},"sides":{"type":"integer","description":"面数，默认 6"}},"required":[]}}}]')
```

2）补全实现 skill.py：
```python
"""掷骰子（dice）"""
from __future__ import annotations
import random

def dice_roll(args):
    count = max(1, int(args.get("count") or 1))
    sides = max(2, int(args.get("sides") or 6))
    results = [random.randint(1, sides) for _ in range(count)]
    return "🎲 掷出 %d 个 %d 面骰子：%s（合计 %d）" % (
        count, sides, "、".join(map(str, results)), sum(results))

HANDLERS = {"dice_roll": dice_roll}
```
（用 `skill_dev_write_file(skill_name="dice", filename="skill.py", content=...)` 写入）

3）补全 SKILL.md、4）`skill_dev_validate("dice")` 清零 ✘ → 完成，直接可用。

## 校验结果解读
- ✘ = 致命问题：不改技能无法注册/工具无法执行（必须清零）
- ⚠ = 建议问题：缺 description/prompt/SKILL.md、description 超 88 字、代码与清单不一致等
- ✔ = 通过项：JSON 合法、命名合规、py_compile 编译通过、HANDLERS 完整、SKILL.md 存在
- 「清单工具没有实现」= manifest 里声明了工具但 skill.py 没注册对应 handler（或没有 execute 分发器）
- 「代码含未实现桩」= 脚手架生成的桩还没被替换，工具调用只会提示未实现

## 命名与规范（务必遵守）
- 技能名：`^[A-Za-z0-9_-]{1,64}$`（仅字母/数字/下划线/连字符）；目录名 = skill.json 的 name
- 工具名：字母/数字/下划线/连字符；**全局唯一**（先 skill_dev_list 核对，别与现有工具撞名）
- description ≤ 88 字（on_demand 摘要截断线）；disclosure：新技能默认 on_demand
- prompt 风格：以【技能名】开头，写触发条件 + 规则 + 边界，中文、简洁、可执行
- SKILL.md 风格：仿本文件（适用场景/工具表/用法示例/边界），一句话能看懂怎么用
- 实现准则：handler 返回可读文本；异步用 async def；不执行外部代码、不引入危险操作

## 安全边界与自愈
- **只操作 skills/ 目录**；绝不修改 harness/*.py 与根目录 *.py（会触发整进程自动重启）
- 校验是纯静态的（py_compile + AST），不会执行被校验技能的代码；也不要自己运行陌生技能代码
- 把本技能（skill_dev）自己改坏怎么办：技能变 broken、工具全部失效。
  用其它可用途径（如 shell 技能执行命令）直接修复 `skills/skill_dev/skill.py`，
  hot_reload 1 秒内自动重新加载恢复；或将目录改名备份后重新 create。
- 删除（skill_dev_remove）必须 confirm=true 且无回收站；改前先 skill_dev_read 留备份
- 改完立刻 validate；validate 通过 ≠ 行为正确，新工具要**实测调用一次**验收

## 常见问题
- 「工具调用报错：尚未实现」→ 脚手架桩未替换，补全 skill.py 后重新 validate
- 「清单工具没有实现」→ 在 HANDLERS 里注册同名 handler，或提供 execute 分发器
- 改了没生效 → 调 skill_dev_reload 强制刷新；确认文件确实写入了（权限问题少见）
- 想删掉不用的技能 → skill_dev_remove 传 confirm=true；也可 /harness 管理页禁用
- validate 报损坏/加载失败 → 多半是 skill.py 语法错误或 skill.json 解析失败，读文件修复即可

## 文件
- skill.json —— 清单与 8 个工具定义（disclosure: on_demand，渐进披露）
- skill.py —— 实现（创建/读取/编辑/写文件/校验/重载/列表/删除，纯标准库）
- SKILL.md —— 本说明书（skill_help("skill_dev") 查看）
