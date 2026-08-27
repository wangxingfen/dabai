# 代码工程（code_ops）—— 大白自带的顶级编程能力

让大白像资深程序员一样工作：批量检索代码 → 定位符号 → 读懂上下文 →
分析结构与依赖 → 精准修改 → 验证改动 → 交付前自审。全程自己做，不需要委派给其它智能体。

## 适用场景
- 用户要求检索/查找代码：搜报错关键字、找某个函数被谁调用、查配置项、魔法数字
- 用户要求分析代码：看懂陌生项目、评估改动影响面、找重构切入点、查依赖关系
- 用户要求修改代码：改 bug、重构、批量替换、新增文件/函数/配置
- 用户要求排错：定位问题 → 读取上下文 → 修改 → 验证 → 自审
- 用户要求审查改动/看提交历史/查代码来历：git 感知工具直接上

## 核心工作流（按顺序走，缺一环就补）
1. **摸清结构**：`code_list_files` 看项目文件清单，
   `code_git_status`/`code_git_log` 看当前改动与历史
2. **定位**：`code_search` 检索关键词/正则，`code_locate` 找函数/类/变量的定义与引用
3. **读懂**：`code_read` 读文件（可指定 路径:起-止 行区间），
   `code_git_blame` 查某行来历
4. **分析**：`code_analyze` 分析文件/目录，`code_deps` 看依赖图（改前必查谁在引用）
5. **修改**：`code_edit` 精准替换/插入（唯一锚点，可 preview 先预览），
   `code_patch` 补丁式批量改动，`code_create_file` 新建文件
6. **验证**：`code_verify` 语法检查，`code_test` 跑完整测试套件
7. **自审**：`code_git_diff` 看自己改了什么，`code_review` 打包审查
   （变更清单 + diff 统计 + 语法检查），确认无误再向用户汇报

## 工具一览

| 工具 | 参数 | 作用 |
| --- | --- | --- |
| `code_search` | query（必填）, root?, exts?, paths?, context?, case_sensitive?, regex?, limit?, include_noise? | 正则/关键词批量检索代码，带上下文行 |
| `code_list_files` | root?, exts?, dirs?, max_depth?, limit?, summary_only?, include_noise? | 列出代码文件或按扩展名统计 |
| `code_read` | files（必填）, root?, max_lines? | 批量读文件，支持 `路径:行号` / `路径:起-止` |
| `code_locate` | symbol（必填）, root?, kind?, exts?, paths?, limit? | 定位函数/类/变量的定义与引用位置 |
| `code_analyze` | files?, root?, dir? | 逐文件结构分析或目录级统计 |
| `code_deps` | files?, root?, limit? | 文件间依赖关系、孤立文件、循环依赖 |
| `code_edit` | file, mode, new（必填）, old?/anchor?, position?, replace_all?, root?, allow_core?, preview? | 唯一锚点精准替换/插入，自动备份 + diff 预览 |
| `code_create_file` | path, content（必填）, root?, overwrite?, check_syntax? | 新建文件，自动建目录、防覆盖、语法预检 |
| `code_verify` | files（必填）, root?, mode?, timeout? | 语法检查（py/json/js）、运行 .py 测试文件 |
| `code_git_status` | root?, short? | git 工作区状态：改动清单、分支、统计 |
| `code_git_diff` | root?, staged?, ref?, files?, stat?, max_lines? | 查看未提交/指定范围的差异，可带统计 |
| `code_git_log` | root?, limit?, file? | 提交历史（hash/日期/作者/说明） |
| `code_git_blame` | file, lines（必填）, root? | 某文件指定行的归属与责任提交 |
| `code_patch` | patch（必填）, root?, preview?, allow_core? | 应用 unified diff 补丁（多文件、严格上下文校验） |
| `code_test` | root?, files?, pattern?, timeout?, verbose? | 跑完整 pytest 套件，返回失败用例清单 |
| `code_review` | root?, ref? | 审查自己的改动：变更清单 + diff 统计 + 语法检查 |

## 用法示例

### 检索
```
code_search(query="TODO", exts="py,js,ts", context=2)
code_search(query="def (fetch|save)\w+", root="D:\\AI\\other-project")
code_search(query="timeout=120", regex=false, context=1)
```

### 定位与读取
```
code_locate(symbol="fetch_data")
code_locate(symbol="UserModel", kind="ref")
code_read(files="src/a.py:1-50, src/b.py")
```

### 分析与依赖
```
code_analyze(files="src/a.py, src/b.py")
code_analyze(root="D:\\AI\\other-project")        # 整个目录统计
code_deps(root="D:\\AI\\other-project")
```

### 修改（先读 → 再改 → 后验证 → 自审）
```
code_git_status(root="D:\\AI\\other-project")
code_edit(file="src/a.py", mode="replace",
          old="timeout = 120", new="timeout = 300")
code_edit(file="src/a.py", mode="replace",
          old="timeout = 120", new="timeout = 300", preview=true)   # 先预览
code_edit(file="src/a.py", mode="insert",
          anchor="def main():", position="after",
          new="    setup_logger()")
code_verify(files="src/a.py")
code_test(root="D:\\AI\\other-project")        # 跑全量测试
code_git_diff()                                # 看自己改了什么
code_review()                                  # 交付前自审
```

### git 感知与补丁
```
code_git_status()                     # 改前看工作区
code_git_log(file="src/a.py")         # 查这个文件最近的提交
code_git_blame(file="src/a.py", lines="10-20")   # 查某段代码是谁写的
code_patch(patch="diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,4 @@\n x=1\n+print(1)\n", preview=true)
```

## 边界与安全（务必遵守）
1. **不臆造路径**：引用文件前先 `code_list_files`/`code_search` 确认真实路径；
   越界路径（不在 root 内）一律拒绝。
2. **唯一锚点**：`code_edit` 的 old/anchor 必须唯一；出现 0 次或多于 1 次时
   只报告、不擅改。批量替换用 `replace_all=true` 前先确认影响范围。
3. **大白核心保护**：默认禁止修改 `harness/*.py` 与项目根目录 `*.py`
   （会触发整进程自动重启）；确需修改须 `allow_core=true` 并先告知用户。
4. **噪音目录**：检索/分析默认跳过 node_modules/.git/__pycache__/dist 等，
   确需搜索时传 `include_noise=true`。
5. **改完必验**：每次 `code_edit`/`code_patch`/`code_create_file` 后都要
   `code_verify` 确认语法；改了模块接口后跑 `code_test` 相关测试。
6. **git 只读**：`code_git_*` 只读不改仓库，绝不自动提交/回退/删分支；
   提交等写操作由用户手动执行。
7. **输出截断**：结果过长会截断显示，必要时缩小 limit / 分批查询。

## 与其它技能的分工
- 检索/分析/修改是**大白自己的能力**，直接调用 code_* 工具，不需要委派；
- 要委派给 DSH/Codex/OpenCode 做大型独立任务时用 agent_ops；
- 修改大白自身技能体系（skill.json/skill.py/SKILL.md）优先用 skill_dev
  （本技能也能改，但注意热重载与校验规范）。

## 常见问题
- 「锚点出现 N 次」→ 补上更多上下文（行首缩进、相邻行）让锚点唯一
- 「未找到要替换的原文」→ 先 code_read 核对原文（注意引号/缩进/换行）
- 中文文件乱码 → 本技能自动识别 UTF-8/GBK 编码，正常无需处理
- 文件没生效 → 确认写入了目标路径，并 code_verify 语法 + code_read 复核
- pytest 未安装 → code_test 会提示，改用 code_verify(mode=test) 逐文件跑，
  或先安装 pytest
- 补丁应用失败 → 多半是上下文行与文件不符，先 code_read 核对再生成补丁
