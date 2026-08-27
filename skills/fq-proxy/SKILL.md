# 翻墙与代理（fq-proxy）

由原 app-proxy 与 fq-launcher 两个技能合并而成。两部分职责：

- **操作 fq 翻墙启动器**：启动/停止/切换代理核心、换 IP、测连通、开关系统代理（工具 `fq_ctl`）。
- **给任意应用配代理**：让 git/pip/npm/curl/node/浏览器等走本地代理端口，并验证、清理。

> 推荐流程：先 `proxy_test`（不传参）摸清本机哪些代理端口活着 → 不够就 `fq_ctl` 起一个 →
> 按下面「给应用加代理」的方式 A–D 配置 → 再 `proxy_test` 验证生效 → 用完按清理清单收尾。

---

## 一、本机两套代理源

| 来源 | 地址 | 说明 |
|---|---|---|
| dev-sidecar（常驻） | `http://127.0.0.1:31181` | 系统平时就在用它，勿动它的注册表设置 |
| fq 多协议翻墙 | 见下表 | 需先用 `fq_ctl` 启动对应协议 |

### fq 协议与端口速查

| 编号 | 名称 | 别名 | 本地代理 |
|---|---|---|---|
| 1 | clash | clash.meta, meta | http://127.0.0.1:7890 |
| 2 | xray | — | socks5://127.0.0.1:1080 |
| 3 | hysteria | hy, hy1 | socks5://127.0.0.1:1080 |
| 4 | singbox | sing-box, sb | socks5://127.0.0.1:1080 |
| 5 | naive | naiveproxy | socks5://127.0.0.1:1080 |
| 6 | hysteria2 | hy2 | socks5://127.0.0.1:1080 |
| 7 | juicity | — | socks5://127.0.0.1:1080 |
| 8 | mieru | — | socks5://127.0.0.1:3080 |
| 9 | shadowquic | sq | socks5://127.0.0.1:4080 |

项目路径固定为 `D:\AI\Chrome141_AllNew_2025.10.3`，CLI 入口 `fq.cmd`（内部调 `fq.ps1`）。
各协议可用 IP 源数量不同（clash 6 个，xray/hysteria/hysteria2 各 4 个，其余 2 个），`fq_ctl list` 可查。

## 二、fq_ctl 命令对照

```bash
# 以下命令行形式供人工参考；大白一律用 fq_ctl 工具调用
./fq.cmd list                    # 全部协议的状态、端口、IP源数量、文件完整性
./fq.cmd status [协议]           # 运行进程 PID + 端口监听详情
./fq.cmd start <协议> [选项]     # 启动核心，默认同时启动便携 Chrome
./fq.cmd stop [协议|all]         # 停止；-KeepChrome 保留浏览器
./fq.cmd update <协议> [n]       # 云端拉新 IP 配置（有副作用！）
./fq.cmd chrome [协议]           # 只开便携 Chrome（自动挂当前监听端口）
./fq.cmd test [协议|all]         # 走代理访问 gstatic generate_204，204=通
./fq.cmd sysproxy on [协议]|off|status   # Windows 系统代理
```

典型任务：
- **一键翻墙**：`fq_ctl(start, target="clash")`
- **被封换 IP**：`fq_ctl(update, target="xray", ip_source=2, confirm_update=true)` 后重启该协议；或逐个 `-Ip 1..N` 试
- **换协议**：`fq_ctl(stop, target="all")` → `fq_ctl(start, target="hysteria2")`
- **诊断**：`fq_ctl(test)` / `fq_ctl(status)`
- **关闭**：`fq_ctl(stop, target="all")`

### 重要注意事项

1. **UAC 提权**：agent 点不了 UAC 弹窗。start 默认带 `-NoElevate -NoChrome`（纯 socks/http 协议够用）；确需提权时提醒用户手动点『是』。
2. **代理不通 ≠ 启动失败**：端口在监听但远端 IP 被墙（test 超时）→ 换 `ip_source`，全不行换协议，还不行可能是本地网络封 UDP/443。
3. **端口可能被外部程序占用**（如 3080 是 dev-sidecar 的 node）：LISTEN 但核心没跑就是占用，test 结果不可信。必须真实探测，别只看端口。
4. **update 有副作用**：会备份旧配置（*_backup）再替换，仅在用户明确要换 IP 时执行。
5. Python 里调 `.cmd` 要显式 `cmd /c`；Git Bash 的 `./fq.cmd` 在 cmd.exe 下不识别。

## 三、给应用加代理（方式按覆盖面从大到小）

**第 0 步永远是确认代理端口活着**：`proxy_test()` 或 `fq_ctl(status)`。

### A. 系统代理 —— 覆盖所有"守规矩"的桌面应用

```bash
./fq.cmd sysproxy on        # 开：浏览器/pip/requests/大部分桌面软件生效
./fq.cmd sysproxy off       # 用完必须关！
```

⚠️ **dev-sidecar 冲突**：本机系统代理平时由 dev-sidecar 托管（指向 127.0.0.1:31181）。
`sysproxy on` 会覆盖它，dev-sidecar 检测到后可能自动关掉系统代理开关。
用完恢复原状：设回 `ProxyServer=https=http://127.0.0.1:31181`、`ProxyEnable=1` 并广播 WinINET 刷新；
若开关又被关掉是它的自我保护，重设一次或在它界面里重新打开。

### B. 终端环境变量 —— 只影响当前会话，最安全，适合 git/pip/curl/python/node

```bash
# Git Bash / Linux 风格
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 ALL_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1
# 取消: unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"; $env:HTTPS_PROXY="http://127.0.0.1:7890"
# 取消: Remove-Item Env:HTTP_PROXY
```

注意：
- 走 socks 端口(1080 等)时写 `socks5h://`（h=域名解析也走代理），部分工具不认 `socks5://`。
- 只对当前终端有效；**不要**持久化到系统环境变量，容易忘清理。

### C. 单个浏览器实例带参启动 —— 不影响其他应用

```bash
"D:\AI\Chrome141_AllNew_2025.10.3\App\chrome.exe" --proxy-server=http://127.0.0.1:7890 --user-data-dir="%TEMP%\fq_chrome_profile"
```

⚠️ 必须带独立 `--user-data-dir`！否则参数被已运行的 Chrome 吞掉、新窗口照样直连——最常见的坑。

### D. 常用工具的持久化配置

```bash
git config --global http.proxy http://127.0.0.1:7890    # 取消: --unset
npm config set proxy http://127.0.0.1:7890              # 取消: npm config delete
pip install --proxy http://127.0.0.1:7890 包名          # pip 建议单次
```

## 四、加完必验证 & 收尾清理

验证：

```bash
curl -sS --ssl-no-revoke -x http://127.0.0.1:7890 https://httpbin.org/ip
# 出口 IP 变成代理节点 IP = 生效；浏览器则开 ip138.com 看（也可直接用 proxy_test 工具）
```

每次操作结束逐项核对：

- [ ] 系统代理：`sysproxy off` 或恢复 dev-sidecar 原值
- [ ] 终端环境变量：unset / 删除
- [ ] 工具配置：git/npm config 取消
- [ ] 残留进程：杀掉所有带 proxy-server 参数的自动化浏览器

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'chrome|chromium|msedge' -and $_.CommandLine -match 'proxy-server' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 五、常见故障速查

| 报错/现象 | 原因 | 处理 |
|---|---|---|
| 浏览器 `ERR_SOCKS_CONNECTION_FAILED` | 有浏览器挂着失效的 socks 参数（多为自动化残留） | 杀掉带 proxy-server 的 chrome 进程，换正常入口重开 |
| `ERR_PROXY_CONNECTION_FAILED` | 系统代理指向的端口没监听 | `fq_ctl(status)` 查端口；起对应协议或关系统代理 |
| 开了 sysproxy 反而全断网 | 系统代理指向了已死的端口 | 立即 `sysproxy off` |
| sysproxy on 后一会儿又失效 | dev-sidecar 自我保护关掉了开关 | 见方式 A 冲突说明 |
| 端口在监听但连不通 | 端口被无关程序占用，或远端 IP 被墙 | 用 `proxy_test` 实测；换 ip_source 或换协议 |

## 六、浏览器自动化与外部项目接入

项目根目录的 `fq_browser.py` 封装 Playwright/Selenium，自动检测监听中的代理端口。
思路：**fq 只负责开代理端口；自动化用 Playwright 自带 Chromium 走这个端口出去**。
不要用 CDP 接管项目自带便携 Chrome（141 旧版连不上）；socks5 端口都支持，clash 的 HTTP 端口(7890) 兼容性最好。

```bash
cd "D:\AI\Chrome141_AllNew_2025.10.3"
./fq.cmd start clash -NoChrome -NoElevate        # 1. 先起代理，不带浏览器
python fq_browser.py playwright https://example.com --demo   # 2. 自动化访问
./fq.cmd stop all                                # 3. 用完关闭
# 更多: --headless 无头 / selenium 后端 / -p xray 指定协议 / repl-pw 交互式
```

其他项目的代码若硬编码了代理地址，把端口换成 fq 本地端口即可（参考项目内 `yt_fq.py`）：
1. 最简：先 `start clash -NoChrome`，代码写 `proxy={"server": "http://127.0.0.1:7890"}`
2. 自动检测：扫端口表并**真实探测**（不能只看 LISTEN，曾有外部程序占 3080 导致误判）
3. 全自动：检测不到就用 `subprocess.run(["cmd","/c","fq.cmd","start","clash","-NoChrome","-NoElevate"], cwd=FQ_ROOT)` 拉起，轮询就绪后再用

依赖状态：playwright 1.58 与 selenium 4.47 已装好（含自带 Chromium）；DrissionPage 与本项目 Chrome 141 不兼容，勿用。
