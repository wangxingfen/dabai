---
name: yt-hanhua-pipeline
description: 操作「油管视频汉化流水线」项目（D:\AI\油管视频汉化）的 CLI：搜索下载 YouTube 视频、Whisper 转录、智能翻译、字幕烧录、AI 配音、B 站发布、背景音乐混音。用户提到汉化视频、翻译配音视频、下载油管/YouTube 视频、发布B站、烧字幕、垫背景音乐、查看流水线任务时使用本 skill。项目同时有 Web 控制台和 CLI，两者共享任务状态，优先用 CLI。
---

# 油管视频汉化流水线 CLI 操作手册

项目根目录：`D:\AI\油管视频汉化`。所有命令先 `cd` 到该目录再执行，
统一用项目自带的 Python：`.venv\Scripts\python.exe cli.py <命令>`（下文简写 `cli.py`；
双击 `cli.bat` 亦可）。

## 心智模型

流水线四个阶段自动衔接，任务状态持久化在 `pipeline/data/state.json`，
CLI 与 Web 控制台（`python app.py`）看到的是同一份状态，**两边不能同时跑调度器**：

```
download(搜索下载) → process(汉化处理) → divide(分区归类) → publish(发布B站)
```

文件流转（video_key = 视频文件名去扩展名）：

```
temp_videos/<视频>.mp4
  → subtitles/<video_key>_with_subtitles_final.mp4 + _zh.srt + _zh.txt
  → subtitles/<合集名>/…           （AI 判断归属，失败则跳过、发布时补）
  → 已发布视频/<合集名>/…
```

- 重活（Whisper/翻译/TTS/浏览器）由 worker 子进程完成，用哪个 Python 由
  `cli.py config get python` 指定（必须是装了 whisper/moviepy/playwright 的环境）；
- CLI 本身零依赖，只做调度/查看/配置；
- 新下载的文件有 30 秒静默期（`process.max_file_age_seconds`）才会被认领处理；
- 失败/取消的任务会把认领的文件自动还原，`retry` 即可重新排队。

## 标准工作流

端到端跑一批视频：

```bash
cd /d/AI/油管视频汉化
.venv/Scripts/python.exe cli.py doctor            # 1. 先体检（GPT-SoVITS 未启动会影响配音）
.venv/Scripts/python.exe cli.py download "ESP32 教程"   # 2. 搜索关键词入队（或 add 本地文件）
.venv/Scripts/python.exe cli.py run               # 3. 前台跑整条流水线，实时看板
```

`run` 的看板每级一段显示各阶段排队/运行数；任务完成/失败会即时打印事件行。
退出方式：Ctrl+C 一次＝停止派发、等运行中任务跑完；再按一次＝强制终止并还原文件。
`run -w`＝队列清空后自动退出（批处理场景）。`run --stages process,divide`＝只跑部分阶段。

跟踪某个任务：

```bash
cli.py status                    # 总览：阶段统计、文件流转、进行中任务
cli.py jobs --all                # 全部任务；--stage process --status failed 过滤
cli.py log <任务ID> -f           # 实时跟踪该任务日志直到结束
cli.py retry <任务ID>            # 重试失败/取消/中断的任务
cli.py cancel <任务ID>           # 取消（文件自动还原）
cli.py delete <任务ID>           # 删除任务记录（连同日志）
```

处理本地视频（不经过下载阶段）：

```bash
cli.py add D:\videos\a.mp4 D:\videos\b.mkv   # 也可以传目录
cli.py run
```

## 命令速查

| 命令 | 作用 |
| --- | --- |
| `doctor [--deep]` | 环境体检：worker Python、ffmpeg、目录、AI key、GPT-SoVITS、调度冲突；`--deep` 检查 worker 依赖 |
| `download <关键词>` / `dl` | 创建搜索下载任务（Playwright 打开浏览器搜并批量下载） |
| `add <文件/目录…>` | 本地视频入汉化队列（文件会被暂存到 temp/work/ 认领） |
| `run [--stages…] [-w] [--force]` | 前台调度器；`-w` 空闲退出；Web 控制台在跑会拒绝启动（`--force` 强制） |
| `status [--json]` | 总览；`--json` 给脚本/Agent 用 |
| `jobs [--stage] [--status] [--all] [--json]` | 任务列表（默认不含已完成） |
| `log <任务ID> [-f] [-n 行数]` | 看日志（logs/<任务ID>.log） |
| `retry / cancel / delete <任务ID>…` | 任务操作，可一次多个 |
| `pause` / `resume` | 暂停/恢复派发（运行中的任务会跑完） |
| `config show/get/set/edit/path` | 读写 pipeline/config.json，点号路径，如 `config set process.model_size=large` |
| `bgm show/files/global/set/clear` | 背景音乐：全局默认 + 单视频专属 |
| `remix <成品.mp4> [--bgm p] [--volume v] [--out o]` | 给已汉化成品直接垫背景乐（不重新汉化，默认原地覆盖） |

## 常用配置键（`config set 键=值`）

| 键 | 说明 |
| --- | --- |
| `python` | worker 解释器路径（whisper 等重依赖所在环境） |
| `dry_run` | `true`＝整条流水线只模拟，验证链路用，改完务必改回 `false` |
| `process.model_size` | Whisper 模型（medium/large…） |
| `process.translation_mode` | `smart`＝智能连贯翻译（默认）；`literal`＝逐条直译 |
| `process.speaking_style` | 固定说话风格/人设（打造个人 IP 的核心参数） |
| `process.timing_mode` | `audio_fit`＝语音变速凑视频（默认）；`video_fit`＝视频变速凑语音 |
| `process.bgm_path` / `process.bgm_volume` | 全局背景乐（音量建议 0.1~0.2） |
| `stages.<阶段>.enabled` | 启停某阶段——**任务一直排队不动时先查这个** |
| `stages.<阶段>.concurrency` | 各阶段并发（1–8） |
| `publish.space_name` | B 站投稿空间名 |

## 背景音乐两板斧

```bash
cli.py bgm files                                   # 看可用音频（bgm/ 目录）
cli.py bgm global --path bgm/卡农.mp3 --volume 0.15   # 之后所有汉化任务默认垫这首
cli.py bgm set <video_key> --path bgm/xx.mp3          # 单视频专属（优先于全局）
cli.py remix subtitles/<合集>/xxx_with_subtitles_final.mp4 --bgm bgm/xx.mp3 --volume 0.2
# remix 直接对成品混音（几十秒），复用 <video_key>_voice_only.m4a 人声缓存，不会越叠越厚
```

## 排障

- **任务一直「排队等待」**：`cli.py status` 看该阶段是否显示停用（`(停)`/表格「启用=否」），
  `config set stages.<阶段>.enabled=true` 打开；或整条流水线被 `pause` 了（`resume` 恢复）。
- **`run` 拒绝启动**：Web 控制台（app.py）正在跑，二者共用调度会冲突。关掉 Web 再 run；
  或直接用 Web 界面操作。另查 `pipeline/data/scheduler.lock` 是否有残留死锁（进程不在了可删）。
- **汉化失败**：`cli.py log <任务ID>` 看末尾 traceback。常见：GPT-SoVITS（127.0.0.1:7860）
  没启动→先启服务再 `retry`；AI key 失效→改 `settings.json`。
- **发布失败**：B 站 cookie 过期（`all_cookies.json`），或合集目录名与 B 站不一致。
  修好后 `cli.py retry pub_xxx`。
- **怀疑流水线坏了**：`config set dry_run=true` → `download 测试` → `run -w` 走一遍全模拟，
  验证完 `config set dry_run=false`。dry-run 会产生 `dry_*.mp4` 假文件，测完删掉。
- **状态残留**：`run` 启动时会自动清理上次异常中断的任务（kill 残留进程并还原文件）。

## 安全红线

- 改动任何流程参数后，先用 `dry_run=true` 冒烟，再切回真实模式；
- 不要在 `run`/Web 同时调度时手工移动 `temp_videos`、`subtitles` 里的业务文件；
- `settings.json` 含 API key，不要外发；`retry` 前先看日志定位原因，避免反复撞同一错误。
