# 本机命令行技能（shell）

让大白直接在用户的 Windows 电脑上执行命令行操作——取代旧"分流 LLM + 批量 steps"链路：
主 Agent 通过 function calling 逐条调用、逐条看结果，天然具备反思能力。

## 工具

| 工具 | 用途 |
|------|------|
| shell_run(command, timeout?) | 执行一条 Windows 命令并返回输出（打开程序/文件、dir 查目录、tasklist 看进程等） |
| find_file(name) | 按名字关键词搜索真实文件的完整路径（桌面/下载/视频/音乐/文档/项目目录/盘符浅层） |
| search_text(query, root?, glob?, max_results?) | 按关键词搜索**文件内容**（类 grep，优先 ripgrep，缺失回退 findstr）：查函数/变量/配置定义与引用、统计日志报错、跨文件定位代码；只读，返回 文件:行号:内容 |
| list_files(root?, depth?, glob?, max_entries?) | 列出目录结构与文件清单（只读，优先 rg 尊重 .gitignore）：动手前先看项目长什么样、文件在哪 |
| read_lines(path, start?, max_lines?) | 按行区间读取文件（只读，自动识别 UTF-8/GBK）：search_text 定位到行号后，读附近上下文，禁止整读大文件 |
| git_status(root?, max_lines?) | 查看 git 工作区状态（只读）：分支 + 改动/新增/删除清单，续接任务前确认改动面 |
| git_diff(root?, path?, staged?, max_lines?) | 查看 git 改动差异（只读）：默认未暂存，staged=true 看暂存区，可指定单文件；出问题定位回滚点 |
| system_check(what?, keyword?, port?, max_lines?) | 系统只读体检：进程（按关键词过滤）/ 监听端口（按端口过滤）/ 磁盘剩余；排查服务、推流、端口占用 |
| symbols(path?, max_results?) | 列出代码文件的类/函数/方法符号表（只读）：Python 用标准库 ast；JS/TS/C/C++/C#/Java/Go/Rust/Bash/Lua/PHP/Ruby/Kotlin/Swift 用 tree-sitter 真实语法树（已内置语法库），缺失自动回退正则 |
| read_json(path?, key?, max_chars?) | 读取并校验 JSON 配置（只读）：报错行列、美化打印、点路径取子字段；查配置损坏/字段用 |

## 使用规则

1. **绝不臆造文件路径**：引用用户文件前必须先 `find_file` 拿到真实完整路径。
2. **危险命令会被拦截**（复用与 /cmd 手动命令相同的守卫），拦截时如实告知用户。
3. **删除白名单**：删除/清理类命令只允许操作白名单临时文件（*.pyc-check、*.tmp、*.bak-*、
   _check*.txt、codex_logs/ 运行日志、%TEMP% 临时文件），其余删除一律被拦截；
   git 已跟踪的项目文件（源码/素材/网页/音频/游戏等）禁止删除。
4. **边界**：编写完整项目/游戏/应用等大型任务 → 用 delegate_agent_task 委派给
   codex/opencode/DSH；本技能只做快速本机操作。
5. 输出超长会截断；命令超时默认 60s（最大 300s）。
6. **查内容优先 search_text**：需要定位某段代码/配置/日志内容时用 search_text（rg 按关键词定位），
   禁止整读超大文件；search_text 是只读操作，不会修改任何文件。
7. **动手前先 list_files，定位后 read_lines**：先看项目结构，搜索定位到 `文件:行号` 后用 read_lines
   只读附近几十行，不要整读文件。
8. **改自己前先 git_status/git_diff**：确认改动面与差异（只读），续接任务先看已落盘部分，
   出问题用 git 回滚，不要盲目重做。
9. **排查用 system_check**：服务没起/端口占用/推流进程卡住时，先查进程与端口，不要靠猜。
10. **快速了解代码文件用 symbols，查配置用 read_json**：symbols 秒出类/函数清单（支持十余种语言，不用整读）；read_json
    能直接告诉你 JSON 哪一行坏了（codex_runtime.json 被写坏这类问题一眼定位），支持 agent.tools.cx 式点路径。
