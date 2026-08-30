"""AI Agent 核心模块 —— 融合技能工具调用 + Function Calling 循环 + 长期记忆。

特性：
- 支持 OpenAI 兼容 API 的流式 function calling
- 工具全部 skill 化（本地工具 + harness 技能/插件，经 skill.json 注册）
- 支持本地工具（从 tools.json 加载）
- 工具调用多轮循环（默认不限制轮数，可经 settings.json -> agent.max_tool_rounds 设上限）
- 长期记忆：自动保存对话历史，检索相关上下文
- 向后兼容 web_agent 的 chat_stream_async 接口
"""
import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from memory import ChatMemory
from tool_validation import find_tool_spec, validate_arguments

logger = logging.getLogger("agent")

BASE_DIR = Path(__file__).parent.resolve()

# 工具调用最大轮数（防止无限循环）；0 / 负数表示不限制，
# 可由 settings.json -> agent.max_tool_rounds 覆盖（见 _max_tool_rounds）
MAX_TOOL_ROUNDS = 0
# 死循环防护：连续 N 轮调用「完全相同工具+参数」才停止（默认 50，容错更高，
# 避免模型多轮重试同一工具时被误杀）；可由 settings.json -> agent.repeat_guard_rounds 覆盖
REPEAT_GUARD_LIMIT = 50
# 工具调用超时（秒）
TOOL_CALL_TIMEOUT = 30.0
# 工具执行心跳间隔（秒）：超过该间隔仍未结束时，向客户端推送 ToolCallProgress 进度事件
TOOL_HEARTBEAT_INTERVAL = 5.0
# 工具超时上限（秒）：任何工具都不允许无限等待（防止僵尸工具拖死对话）。
# 2026-08-30 收回 300s：此前为容纳长工具抬到 900s，实测体验是"工具能卡十几
# 分钟"——对话轮内超过 2~3 分钟的工具本就不该主线程硬等，应改用后台任务系统；
# 长耗时工具（Blender/Mixamo/工作树命令）单列 300s 覆盖，内部自带子进程超时。
TOOL_CALL_TIMEOUT_MAX = 300.0
# 长耗时工具的超时覆盖（秒）：默认 tool_call_timeout 对这些工具太短。
# 可在 settings.json -> agent.tool_timeouts 里按工具名继续覆盖。
_TOOL_TIMEOUT_OVERRIDES = {
    "pmx_to_vrm": 300.0,          # Blender 转换（内部 600s 子进程超时，主轮最多等 5 分钟）
    "wt_run": 300.0,              # 工作树内跑命令
    "wt_create": 180.0,           # git worktree add 冷启动
    "anim_batch": 300.0,          # Mixamo 批量下载
    "anim_download": 240.0,       # 单动作下载 + 浏览器自动化
    "anim_optimize": 300.0,       # 动作优化/后处理
    "skill_pull_install": 300.0,  # GitHub 拉取 + 校验 + 安装
}

# 轮内工具历史压缩（2026-08-30）：工具循环里每轮执行完，旧轮的工具结果仍全量
# 留在 messages 中重发给模型——shell_run/code_read 单条可达 16000 字。DeepSeek V4
# 上下文是 1M（1049K），正常工程任务几十轮远不会爆窗，因此预算按窗口的 1/4
# （默认 256K）设置，只在极端超长轮次时兜底压缩；过早压缩反而逼模型重读文件、
# 增加工具轮数，得不偿失。压缩规则：
# - 最新 keep_rounds 轮保持完整（模型正在用的结果不砍）；
# - 更早轮次的工具结果截成片段（结构保留，不破坏 function-calling 校验）；
# - 总量仍超预算时再压缩最新一轮的结果。
# 可用 settings.json -> agent.mid_turn_max_tokens / tool_result_retro_cap 调整。
MID_TURN_MAX_TOKENS = 262144
TOOL_RESULT_RETRO_CAP = 250
ASSISTANT_RETRO_CAP = 150
KEEP_NEWEST_TOOL_ROUNDS = 1
# 单轮内最多同时注册的工具定义数（渐进式披露的轮内上限）：
# skill_help 激活的技能工具只在本轮有效，且总量封顶，超出按最久未使用淘汰。
MAX_ACTIVE_TOOLS = 48
# 文本协议工具调用标记：本地模型（如 Ollama draganis/vanessa）不支持原生 function
# calling 时，通过系统提示词注入工具说明，模型以 <tool_call>{...}</tool_call> 标记发起调用。
# 兼容两种写法：<tool_call>{"name":...}</tool_call> 与 <tool_call {"name":...}>
TEXT_TOOL_CALL_RE = re.compile(r"<tool_call\s*>\s*(\{.*?\})\s*(?:</tool_call>|\s*>)", re.S)
# 解析用：定位 <tool_call ...> 到 </tool_call>（或内容结尾）之间的"参数区"，
# 再用 _extract_balanced_json 括号配平截取完整 JSON——任务文本里出现 { }（如"参考 {xx}/a.py"）
# 或长描述时，不再被非贪婪匹配提前掐断。
TEXT_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call\b[^>]*>(.*?)(?:</tool_call>|\s*$)", re.S)
# 展示用：剥离标记（含未闭合写法），避免任务 JSON 残留在聊天面板。
TEXT_TOOL_CALL_STRIP_RE = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call>|</?tool_call\b[^>]*>", re.S
)

# 由 server 端完整消费的"特殊标记 JSON"：委派 codex/opencode、DSH 桥接、屏幕命令。
# 其中 task 字段可能很长，一旦被下方 2000 字符截断，JSON 会失效或任务内容不完整，
# 因此这类结果必须原样保留、不做长度截断。
_SPECIAL_TOOL_RESULT_KEYS = ("__codex_delegate__", "__dsh_bridge__", "__screen_command__")


def _is_special_tool_result(result: str) -> bool:
    """判断工具结果是否为 server 端消费的特殊标记 JSON。"""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return (isinstance(data, dict)
            and any(k in data for k in _SPECIAL_TOOL_RESULT_KEYS))
@dataclass
class ChatTurnContext:
    """一次对话轮的全部上下文（打包 15 个散参数，消除长参数列表坏味道）。

    由公共入口 chat_stream 构造，透传给 _chat_stream_inner /
    _chat_stream_normal / _chat_stream_game。函数体开头解包为局部变量，
    行为与旧签名完全等价。
    """
    message: str
    history: list = None
    enable_tools: bool = True
    current_model: Optional[str] = None
    current_background: Optional[str] = None
    current_bgm: Optional[str] = None
    game_context: Optional[str] = None
    game_mode: bool = False
    game_type: Optional[str] = None
    msg_source: str = "chat"
    current_anim: Optional[dict] = None
    turn_id: Optional[str] = None
    resume: Optional[dict] = None
    record_history: bool = True
    proactive: bool = False


# 工具结果回传模型/前端时的长度策略：
# - 普通工具默认上限 2000 字符（防刷屏）；
# - 代码/技能类工具是「查代码、看 diff、读文件」的主战场，结果被砍短会逼着模型
#   反复重读、重查（损耗极高），因此单独放宽到 16000 字符；特殊标记 JSON 仍完整保留。
_TOOL_RESULT_LIMIT = 2000
_TOOL_RESULT_LIMIT_LONG = 16000
_LONG_RESULT_TOOLS = frozenset((
    "code_read", "code_search", "code_locate", "code_analyze", "code_deps",
    "code_list_files", "code_edit", "code_patch", "code_create_file",
    "code_verify", "code_test", "code_review",
    "code_git_status", "code_git_diff", "code_git_log", "code_git_blame",
    "skill_dev_read", "skill_dev_list", "skill_dev_validate", "skill_dev_reload",
    "skill_dev_write_file", "skill_dev_create", "skill_dev_edit",
    "skill_help", "shell_run",
    "read_lines", "search_text", "git_diff", "git_status", "read_json",
    "find_file", "list_files", "system_check", "symbols",
))


def _fit_tool_result(result: str, tool_name: str) -> str:
    """按工具类型截断过长结果；特殊标记 JSON 原样保留。"""
    result = str(result or "")
    if _is_special_tool_result(result):
        return result
    limit = _TOOL_RESULT_LIMIT_LONG if tool_name in _LONG_RESULT_TOOLS \
        else _TOOL_RESULT_LIMIT
    if len(result) > limit:
        return result[:limit - 3] + "..."
    return result


def _mid_turn_max_tokens() -> int:
    """轮内上下文预算：默认取配置窗口的 1/4（1M 窗口 → 256K），
    也可用 settings.json -> agent.mid_turn_max_tokens 显式覆盖。"""
    try:
        cfg = load_config()
        v = int(cfg.get("agent", {}).get("mid_turn_max_tokens", 0) or 0)
        if v > 0:
            return max(8000, v)
        window = int((cfg.get("usage") or {}).get("context_window", 0) or 0)
        if window > 0:
            return max(80000, window // 4)
    except Exception:
        pass
    return MID_TURN_MAX_TOKENS


def _tool_result_retro_cap() -> int:
    try:
        v = int(load_config().get("agent", {}).get(
            "tool_result_retro_cap", TOOL_RESULT_RETRO_CAP) or TOOL_RESULT_RETRO_CAP)
        return max(50, v)
    except Exception:
        return TOOL_RESULT_RETRO_CAP


def _compact_tool_history(messages: list, budget: int = None,
                          keep_rounds: int = None) -> list:
    """轮内工具历史压缩：把旧工具轮的庞杂结果压成小片段，绑定轮内上下文总量。

    只在总量超预算时动手（平时零开销、逐字节不变，不破坏前缀缓存）；
    压缩只改 assistant/tool/system【工具】消息的 content 长度，
    轮次结构与 tool_calls 配对原样保留，function-calling 校验不受影响。
    """
    if not messages:
        return messages
    try:
        from memory import estimate_tokens
    except Exception:
        return messages
    budget = budget or _mid_turn_max_tokens()
    keep_rounds = KEEP_NEWEST_TOOL_ROUNDS if keep_rounds is None else max(0, int(keep_rounds))
    retro_cap = _tool_result_retro_cap()

    def _est(m):
        return estimate_tokens(str(m.get("content") or ""))

    if sum(_est(m) for m in messages) <= budget:
        return messages

    def _is_result_msg(m):
        return (m.get("role") == "tool"
                or (m.get("role") == "system"
                    and str(m.get("content") or "").startswith("【工具")))

    # 把消息流切成「assistant + 其后的结果消息」组（原生 tool / 文本【工具】system）
    n = len(messages)
    groups = []          # (start, end, is_tool_round)
    i = 0
    while i < n:
        if messages[i].get("role") == "assistant":
            j = i + 1
            while j < n and _is_result_msg(messages[j]):
                j += 1
            groups.append((i, j, j > i + 1))
            i = j
        else:
            i += 1
    tool_groups = [g for g in groups if g[2]]
    if not tool_groups:
        return messages

    # 1) 压缩旧轮（保留最新 keep_rounds 轮完整）。
    #    缓存友好：从「最新旧轮」往「最旧轮」逐个截断，一旦总量回到预算内就停——
    #    只改最少的内容，且失效点尽量靠近动态尾巴（改动越靠后，前缀缓存失效范围越小）。
    old_groups = list(tool_groups[:-keep_rounds] if keep_rounds else [])
    for start, end, _ in reversed(old_groups):
        for k in range(start, end):
            m = messages[k]
            if _is_result_msg(m):
                c = str(m.get("content") or "")
                if len(c) > retro_cap:
                    m["content"] = c[:retro_cap] + "…【已压缩】"
            elif m.get("role") == "assistant":
                c = str(m.get("content") or "")
                if len(c) > ASSISTANT_RETRO_CAP:
                    m["content"] = c[:ASSISTANT_RETRO_CAP] + "…"
        if sum(_est(m) for m in messages) <= budget:
            break

    # 2) 仍超预算：最新一轮的结果先压到 2000 字（仍够模型继续干活），
    #    再超才压到 600 字，兜底保证请求一定能发出去
    if sum(_est(m) for m in messages) > budget and tool_groups:
        for cap in (2000, 600):
            start, end, _ = tool_groups[-1]
            for k in range(start, end):
                m = messages[k]
                if _is_result_msg(m):
                    c = str(m.get("content") or "")
                    if len(c) > cap:
                        m["content"] = c[:cap] + "…【已压缩】"
            if sum(_est(m) for m in messages) <= budget:
                break
    return messages


def _normalize_tool_rounds(messages: list) -> list:
    """兜底规范化：保证所有 role=tool 消息都带 tool_call_id，且与最近的
    assistant tool_calls 一一对齐。

    背景：工具结果入库（memory.add_message("tool", result)）历史上只存了
    content、没存 tool_call_id。下次从记忆库（DB）恢复历史时，role=tool 的
    消息就缺 tool_call_id，发给 OpenAI 兼容提供方会 400
    （missing field tool_call_id）。这里按「assistant(tool_calls) 之后的
    连续 tool 消息」分组，从 assistant 的 tool_calls ids 里按序补回；
    若 tool 消息多于 ids（数据异常），补一个占位 id，保证字段一定存在，
    不再因缺字段被拒。

    v2 增强：除「assistant 紧邻之后」外，还兜住两条漏网路径——
    1) 孤立 tool 消息（其前置 assistant(tool_calls) 已被 _compact_tool_history
       压缩丢弃，或本就没前置）：现在会向后扫描最近一个带 tool_calls 的
       assistant 的 ids 补回；仍无则用占位 id（call_tool_{index}），绝不缺字段。
    2) tool 消息的 tool_call_id 未落到任何已知 assistant ids 上（错位/陈旧）：
       直接重写为最近一个 assistant tool_calls 里尚未被本段引用的 id，
       保证与上游 tool_calls 严格一一对应，功能调用校验不会 400。
    """
    if not messages:
        return messages
    n = len(messages)
    i = 0
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"] if tc.get("id")]
            j = i + 1
            # 收集 assistant(tool_calls) 之后的连续 tool 消息
            tool_msgs = []
            while j < n and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            k = 0
            for tm in tool_msgs:
                if not tm.get("tool_call_id"):
                    tm["tool_call_id"] = (
                        ids[k] if k < len(ids) else f"call_{j}"
                    )
                k += 1
            # v3：tool 消息少于 tool_calls → 补齐缺失的响应，避免 provider 400
            if len(tool_msgs) < len(ids):
                used = {tm.get("tool_call_id") for tm in tool_msgs}
                missing = [tid for tid in ids if tid not in used]
                for tid in missing:
                    messages.insert(j, {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": "【工具结果缺失】",
                    })
                    j += 1
                    n += 1
            i = j
        else:
            i += 1

    # v2：兜住孤立/错位的 tool 消息（前述路径补不到的）。
    # 先收集所有 assistant tool_calls 的 id 全集，供错位重写复用。
    known_ids = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            known_ids.extend(tc.get("id") for tc in m["tool_calls"] if tc.get("id"))
    seen_tool_ids = [m.get("tool_call_id") for m in messages
                     if m.get("role") == "tool"]
    remaining_ids = [tid for tid in known_ids if tid not in seen_tool_ids]

    for idx, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        cur = m.get("tool_call_id")
        if not cur:
            # 孤立 tool：从「最近一个带 ids 的 assistant」按序补；无则占位
            if remaining_ids:
                m["tool_call_id"] = remaining_ids.pop(0)
            else:
                m["tool_call_id"] = f"call_tool_{idx}"
        elif cur not in known_ids:
            # 错位/陈旧的 id（不在任何 assistant tool_calls 里）：重写为可用 id
            if remaining_ids:
                m["tool_call_id"] = remaining_ids.pop(0)
            else:
                m["tool_call_id"] = f"call_tool_{idx}"
    # v3 结构修复：保证每条 role=tool 消息前面紧邻的是「带 tool_calls 的 assistant」。
    # 背景：记忆库/断点恢复路径里，tool 结果可能没关联到 assistant(tool_calls)，孤立地
    # 出现在 user 或纯 content 的 assistant 之后。此时即便 tool_call_id 已补齐，OpenAI
    # 兼容提供方仍会因「tool 消息前面不是带 tool_calls 的 assistant」而拒绝
    # （400: role 'tool' must be a response to a preceding message with 'tool_calls'）。
    # 这里为这样的孤立/错位 tool 在紧贴其前插入一条合成 assistant(tool_calls)，挂上该 id，
    # 使结构合法。扫描时跳过同组的连续 tool（assistant(tc=[c0,c1]) 后跟 tool(c0)/tool(c1)
    # 属合法结构，不得误判）。
    n = len(messages)
    i = 0
    while i < n:
        m = messages[i]
        if m.get("role") != "tool":
            i += 1
            continue
        # 从 i 向前扫到最近一条「非 tool」消息作为潜在 host
        j = i - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        host = messages[j] if j >= 0 else None
        tid = m.get("tool_call_id")
        host_ok = bool(
            host and host.get("role") == "assistant" and host.get("tool_calls")
            and any(tc.get("id") == tid for tc in host["tool_calls"])
        )
        if not host_ok:
            new_tid = tid or f"call_tool_{i}"
            m["tool_call_id"] = new_tid
            messages.insert(i, {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": new_tid,
                    "type": "function",
                    "function": {"name": "_recovered_tool", "arguments": "{}"},
                }],
            })
            i += 1
            n += 1
        i += 1
    return messages


def _tool_max_tokens() -> int:
    """工具调用轮的 LLM 输出上限：写大文件（skill.py 等）需要大输出预算。

    settings.json 顶层可配 tool_max_tokens（默认 4096）；未配置时至少不低于
    普通 max_tokens，避免模型生成长工具参数（如整体写 skill.py）写到一半被截断。
    """
    try:
        cfg = load_config()
        v = int(cfg.get("tool_max_tokens") or 0)
        if v > 0:
            return max(512, min(v, 16384))
        base = int(cfg.get("max_tokens") or 512)
    except Exception:
        base = 512
    return max(base, 4096)


def _stream_retry_count() -> int:
    """流式输出中途断网的最大重建次数（settings.json -> agent.stream_retries，默认 3）。"""
    try:
        v = int(load_config().get("agent", {}).get("stream_retries", 3) or 0)
        return max(0, min(v, 5))
    except Exception:
        return 3


def _is_transient_stream_error(e: Exception) -> bool:
    """判断流式中途断网是否值得重建（瞬时错误才重试）。"""
    try:
        from harness.core import is_transient_error
        return is_transient_error(e)
    except Exception:
        msg = (str(e) or "").lower()
        return any(k in msg for k in (
            "timeout", "timed out", "connection", "refused", "reset",
            "closed", "rate limit", "temporarily", "429", "500", "502",
            "503", "504",
        ))


def _dedup_stream_text(assistant_content: str, chunk: str,
                       delivered_text: str, skip_prefix_len: int):
    """流式重试时的文本前缀去重。

    返回 (assistant_content, delivered_text, skip_prefix_len, yield_text)：
    - yield_text 为 None 表示本块仍落在已播报前缀内（只累积、不输出）；
    - 重新生成内容与已播报不一致时从头输出（宁可轻微重复也不丢内容）。
    """
    if skip_prefix_len <= 0:
        return (assistant_content + chunk, delivered_text + chunk, 0, chunk)
    merged = assistant_content + chunk
    overlap = delivered_text[:skip_prefix_len]
    if merged[:skip_prefix_len] == overlap:
        if len(merged) <= skip_prefix_len:
            return merged, delivered_text, skip_prefix_len, None
        tail = merged[skip_prefix_len:]
        return merged, delivered_text + tail, 0, tail
    return merged, delivered_text + merged, 0, merged


def _extract_balanced_json(text: str) -> Optional[str]:
    """提取文本中第一个完整的 JSON 对象（对字符串值里的 { } 免疫，避免提前截断）。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def load_config():
    with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _set_agent_mode(mode: str) -> None:
    """把模式偏好原子写回 settings.json（失败静默，绝不影响对话）。"""
    try:
        path = BASE_DIR / "settings.json"
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("agent", {})["mode"] = mode
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _strip_think_markers(text: str) -> str:
    """剔除模型输出里可能残留的 <think>/</think> 等思维标记（污染正文/思维链显示）。"""
    if not text:
        return text
    return re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)


# ---- 模式：同一身份、两种姿态（日常 / 工程），像人一样切换工作状态 ----
_MODE_SWITCH_TO_PROG = ("切到工程模式", "工程模式", "编程模式", "工作模式", "进入工程")
_MODE_SWITCH_TO_DAILY = ("日常模式", "切回日常", "回到日常", "退出工程", "聊天模式")
_COMPLEX_HINTS = (
    "优化", "重构", "项目", "代码", "脚本", "文件", "修复", "改一下", "改个",
    "写一个", "部署", "实现", "查一下", "搜索", "找一下", "下载", "测试",
    "编译", "错误", "报错", "配置", "接口", "任务", "调试",
)


def _looks_complex(message: str) -> bool:
    """auto 模式下的简单启发式：较长且带任务词 → 本轮按工程姿态处理。"""
    m = (message or "").strip()
    if len(m) < 8:
        return False
    return any(h in m for h in _COMPLEX_HINTS)


def _max_tool_rounds() -> int:
    """读取工具调用最大轮数（settings.json -> agent.max_tool_rounds）。

    0 / 负数 / 未配置 表示不限制：循环一直持续到模型不再发起工具调用为止。
    """
    try:
        v = int(load_config().get("agent", {}).get("max_tool_rounds", MAX_TOOL_ROUNDS) or 0)
    except Exception:
        v = MAX_TOOL_ROUNDS
    return max(v, 0)


def _repeat_guard_limit() -> int:
    """读取死循环防护阈值（settings.json -> agent.repeat_guard_rounds，默认 10）。"""
    try:
        v = int(load_config().get("agent", {}).get(
            "repeat_guard_rounds", REPEAT_GUARD_LIMIT) or REPEAT_GUARD_LIMIT)
    except Exception:
        v = REPEAT_GUARD_LIMIT
    return max(v, 2)


def _work_lean_context() -> bool:
    """工程模式下是否启用「工作状态瘦身」。

    开启后（默认 true），大白进入工程模式时会动态卸载与任务无关的上下文
    （娱乐/外观/场景/音乐/视频/动作等），只保留对完成项目有用的部分
    （工具规则、任务经验、子智能体状态、相关记忆），让有限的上下文预算
    全部花在当前任务上。settings.json -> agent.work_lean_context 可关。
    """
    try:
        return bool(load_config().get("agent", {}).get("work_lean_context", True))
    except Exception:
        return True


def _reasoning_extra(mode: str = "daily") -> Optional[dict]:
    """按模式返回推理强度参数（extra_body 形式），只对硅基流动推理模型生效。

    - daily（日常闲聊）：低强度 + 小预算，简单问题快速回应、不空转；
    - programming（工程任务）：高强度 + 大预算，复杂任务深入思考；
    - 预算 <= 0 / 未配置 = 不设上限：不下发 thinking_budget，让模型想多久想多久；
    - 强度与预算可在 settings.json -> reasoning 里覆盖，enabled=false 可整体关闭。
    其他渠道（Ollama / opencode zen 等）不认识这些参数，一律返回 None。
    """
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    base = str(cfg.get("base_url") or "")
    model = str(cfg.get("model") or "")
    if "siliconflow.cn" not in base:
        return None
    if not (model.startswith("deepseek-ai/") or model.startswith("Qwen/")):
        return None
    r = cfg.get("reasoning") or {}
    if r.get("enabled") is False:
        return None
    if mode == "programming":
        effort = r.get("programming_effort") or "high"
        budget = r.get("programming_thinking_budget", 0)
    else:
        effort = r.get("daily_effort") or "low"
        budget = r.get("daily_thinking_budget", 0)
    extra = {}
    try:
        effort = str(effort).strip().lower()
        if effort in ("low", "medium", "high", "max"):
            extra["reasoning_effort"] = effort
    except Exception:
        pass
    try:
        budget = int(budget)
        if budget > 0:  # <=0 / 未配置 = 不限制思考长度
            extra["thinking_budget"] = budget
    except (TypeError, ValueError):
        pass
    # Qwen3 推理模型（8B/14B/32B、*-Thinking 等）必须显式 enable_thinking，
    # 思考才会放进 reasoning_content；不带该参数时模型会把思路写进正文。
    # 只对确认支持推理的型号下发，避免 Instruct / 3.5 / 3.6 系列误传参数
    try:
        if re.match(r"^Qwen/Qwen3-(?:\d{1,2}B|[\w.-]*Thinking)(?:/|$)",
                    str(model)):
            extra["enable_thinking"] = True
    except Exception:
        pass
    return extra or None


def _tool_fp(name: str, arguments: dict) -> str:
    """工具调用指纹：工具名 + 参数 JSON（键排序），用于死循环检测。"""
    try:
        return f"{name}|{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f"{name}|{str(arguments)}"


# 热重载联动：当前是否有对话轮（含工具调用）正在执行；有则延迟自动重启，
# 避免大白自己通过工具调用修改核心代码时，中途被热重载重启打断（直到本轮说完）。
_turn_active_count = 0
_turn_count_lock = threading.Lock()


def _turn_begin() -> None:
    global _turn_active_count
    with _turn_count_lock:
        _turn_active_count += 1


def _turn_end() -> None:
    global _turn_active_count
    with _turn_count_lock:
        if _turn_active_count > 0:
            _turn_active_count -= 1


def active_turns() -> int:
    """当前正在执行的对话轮数（0 = 空闲，可以安全重启）。"""
    return _turn_active_count


# ==================== 对话轮断点（热重载中途自主恢复） ====================
# 角色调用工具期间允许热重载：每轮工具执行前把「完整 LLM 消息 + 待执行工具」
# 原子落盘，进程被热重载杀死后，server 启动/客户端重连时读取断点，角色自主
# 续跑被打断的那一轮——验证和迭代不再被「延迟 30 分钟重启」卡住。
TURN_CKPT_DIR = BASE_DIR / "data" / "turn_checkpoints"
_TURN_CKPT_LOCK = threading.Lock()
# 2026-08-29 多槽位化：断点按 turn_id 独立落盘，每用户最多保留 8 个。
# 修复"用户插话打断 → 新对话轮把断点清掉 → 说『继续』只剩一句话摘要"的问题：
# 新轮不再覆盖/删除旧轮断点，被打断的任务（paused 状态）一直保留到被『继续』
# 真正续跑完成，或被用户显式作废（clear_turn_checkpoint）。
_TURN_CKPT_MAX_SLOTS = 8


def _turn_ckpt_enabled() -> bool:
    """断点开关：settings.json -> agent.turn_checkpoint（默认开启）。"""
    try:
        cfg = load_config().get("agent") or {}
        return bool(cfg.get("turn_checkpoint", True))
    except Exception:
        return True


def _user_ckpt_safe(user_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(user_id or "default")) or "default"
    return safe


def _turn_ckpt_slot_path(user_id: str, turn_id: str) -> Path:
    safe = _user_ckpt_safe(user_id)
    tid = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(turn_id or "")) or "unknown"
    return TURN_CKPT_DIR / f"{safe}__{tid}.json"


def _turn_ckpt_latest_path(user_id: str) -> Path:
    return TURN_CKPT_DIR / f"{_user_ckpt_safe(user_id)}.latest"


def _list_ckpt_slot_paths(user_id: str = None) -> list:
    """列出断点槽位文件（新→旧按 mtime；兼容旧版单文件 <user>.json）。"""
    if not TURN_CKPT_DIR.is_dir():
        return []
    if user_id is None:
        paths = list(TURN_CKPT_DIR.glob("*__*.json"))
        paths += [p for p in TURN_CKPT_DIR.glob("*.json")
                  if "__" not in p.name]
    else:
        safe = _user_ckpt_safe(user_id)
        paths = list(TURN_CKPT_DIR.glob(f"{safe}__*.json"))
        legacy = TURN_CKPT_DIR / f"{safe}.json"
        if legacy.exists():
            paths.append(legacy)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_ckpt_slot(path: Path) -> Optional[dict]:
    try:
        with _TURN_CKPT_LOCK:
            cp = json.loads(path.read_text(encoding="utf-8"))
        return cp if isinstance(cp, dict) else None
    except Exception:
        return None


def _evict_old_ckpt_slots(user_id: str) -> None:
    """每用户槽位封顶：超出按时间淘汰最旧的（保留最近任务优先）。"""
    try:
        slots = _list_ckpt_slot_paths(user_id)
        if len(slots) <= _TURN_CKPT_MAX_SLOTS:
            return
        with _TURN_CKPT_LOCK:
            for p in slots[_TURN_CKPT_MAX_SLOTS:]:
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def save_turn_checkpoint(user_id: str, cp: dict) -> None:
    """原子写断点槽位 + 更新该用户 latest 指针。

    每轮独立槽位（文件名含 turn_id），进程被杀时磁盘上始终有完整的旧断点；
    latest 指针指向最近一次保存的轮次，供「继续」/状态读取快速定位。
    """
    try:
        with _TURN_CKPT_LOCK:
            TURN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
            path = _turn_ckpt_slot_path(user_id, cp.get("turn_id") or "")
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cp, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, path)
            latest = _turn_ckpt_latest_path(user_id)
            latest.write_text(str(cp.get("turn_id") or ""), encoding="utf-8")
        _evict_old_ckpt_slots(user_id)
    except Exception as e:
        logger.warning(f"保存对话轮断点失败: {e}")


def load_turn_checkpoint(user_id: str, only_paused: bool = False) -> Optional[dict]:
    """读取该用户最新断点（latest 指针优先；指针失效时回退扫描槽位）。

    only_paused=True 时只返回最新一条「已暂停」断点——用户说『继续』时的
    续跑源（普通新对话轮不会覆盖它，任务可一直等到被续跑）。
    """
    try:
        latest = _turn_ckpt_latest_path(user_id)
        if latest.exists():
            tid = latest.read_text(encoding="utf-8").strip()
            if tid:
                cp = _read_ckpt_slot(_turn_ckpt_slot_path(user_id, tid))
                if cp and (not only_paused or cp.get("paused")):
                    return cp
        for p in _list_ckpt_slot_paths(user_id):
            cp = _read_ckpt_slot(p)
            if cp and (not only_paused or cp.get("paused")):
                return cp
    except Exception as e:
        logger.warning(f"读取对话轮断点失败: {e}")
    return None


def load_paused_turn_checkpoint(user_id: str) -> Optional[dict]:
    """最新一条「已暂停」断点（供『继续』指令续跑）。"""
    return load_turn_checkpoint(user_id, only_paused=True)


# 「继续」自动续跑的新鲜窗口（秒）：超过该时间视为任务已过期，
# 不再自动恢复旧断点/旧摘要，避免把过时任务总结注入当前对话。
# settings.json -> agent.resume_fresh_seconds 可调。
RESUME_FRESH_SECONDS = 1800


def resume_fresh_seconds() -> int:
    try:
        v = int(load_config().get("agent", {}).get(
            "resume_fresh_seconds", RESUME_FRESH_SECONDS) or RESUME_FRESH_SECONDS)
        return max(60, v)
    except Exception:
        return RESUME_FRESH_SECONDS


def ckpt_is_fresh(cp: dict, now: float = None) -> bool:
    """断点是否仍在「继续」新鲜窗口内（updated_at 距今 ≤ resume_fresh_seconds）。"""
    now = time.time() if now is None else now
    ts = float(cp.get("updated_at") or 0)
    return bool(ts) and (now - ts) <= resume_fresh_seconds()


def clear_ckpt_slot(user_id: str, turn_id: str) -> None:
    """删除指定轮次的断点槽位（含其 latest 指针指向）。"""
    try:
        with _TURN_CKPT_LOCK:
            path = _turn_ckpt_slot_path(user_id, turn_id)
            if path.exists():
                path.unlink()
            latest = _turn_ckpt_latest_path(user_id)
            if latest.exists():
                try:
                    if latest.read_text(encoding="utf-8").strip() == str(turn_id):
                        latest.unlink()
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"清除对话轮断点失败: {e}")


def clear_turn_checkpoint(user_id: str) -> None:
    """清除该用户全部断点槽位（显式作废语义：换角色/换会话/明确放弃任务）。"""
    try:
        with _TURN_CKPT_LOCK:
            for p in _list_ckpt_slot_paths(user_id):
                try:
                    p.unlink()
                except Exception:
                    pass
            latest = _turn_ckpt_latest_path(user_id)
            if latest.exists():
                latest.unlink()
    except Exception as e:
        logger.warning(f"清除对话轮断点失败: {e}")


def _turn_ckpt_still_mine(user_id: str, turn_id: str) -> bool:
    """断点是否仍是当前轮次（防新对话轮覆盖后旧轮误清理/误删消息）。"""
    cp = _read_ckpt_slot(_turn_ckpt_slot_path(user_id, turn_id))
    return bool(cp and cp.get("turn_id") == turn_id)


def _clear_turn_ckpt_if_mine(user_id: str, turn_id: str) -> None:
    if _turn_ckpt_still_mine(user_id, turn_id):
        clear_ckpt_slot(user_id, turn_id)


# ---------- 任务状态摘要（方案 C 兜底：断点丢失也能靠摘要续跑） ----------
TASK_RESUME_FILE = BASE_DIR / "data" / "task_resume_states.json"
_TASK_RESUME_LOCK = threading.Lock()


def _ckpt_summary(cp: dict) -> str:
    """从断点提取任务状态摘要：目标 + 进度 + 待办 + 最近真实工具结果。

    作为断点丢失时的「方案 C 兜底」注入新对话；多带真实工具结果，
    续跑时模型不至于只靠一句话凭空猜测之前做到哪。
    """
    try:
        goal = str(cp.get("user_message") or "").strip()
        if len(goal) > 100:
            goal = goal[:97] + "…"
        done = int(cp.get("tool_round") or 0)
        pending = [str(p.get("name") or p.get("tool") or "?")
                   for p in (cp.get("pending_tools") or [])]
        partial = str(cp.get("assistant_content") or cp.get("full_text") or "").strip()
        if len(partial) > 120:
            partial = partial[:117] + "…"
        # 从检查点消息里提取最近工具执行的真实结果（最多 3 条，各 100 字）
        tool_results = []
        for m in (cp.get("messages") or [])[-10:]:
            if m.get("role") == "tool":
                c = str(m.get("content") or "").strip().replace("\n", " ").replace("\r", " ")
                if c:
                    tool_results.append(c[:100])
            if len(tool_results) >= 3:
                break
        parts = []
        if goal:
            parts.append(f"目标：{goal}")
        if done:
            parts.append(f"已完成 {done} 轮工具调用")
        if pending:
            parts.append("待执行：" + "、".join(pending))
        if partial:
            parts.append(f"已有进展：{partial}")
        if tool_results:
            parts.append("最近工具结果：" + "；".join(tool_results))
        return "；".join(parts) or "（无摘要）"
    except Exception:
        return "（无摘要）"


def save_task_resume_state(user_id: str, summary: str, cp: dict) -> None:
    """打断时记录一句话任务状态（24 小时内有效，供「继续」兜底重建）。"""
    try:
        with _TASK_RESUME_LOCK:
            data = {}
            if TASK_RESUME_FILE.exists():
                try:
                    data = json.loads(TASK_RESUME_FILE.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data[str(user_id)] = {
                "summary": summary,
                "ts": time.time(),
                "session_id": str(cp.get("session_id") or ""),
            }
            TASK_RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = TASK_RESUME_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, TASK_RESUME_FILE)
    except Exception as e:
        logger.warning(f"保存任务状态摘要失败: {e}")


def load_task_resume_state(user_id: str) -> Optional[dict]:
    try:
        if not TASK_RESUME_FILE.exists():
            return None
        data = json.loads(TASK_RESUME_FILE.read_text(encoding="utf-8"))
        st = data.get(str(user_id))
        if st and time.time() - float(st.get("ts") or 0) < 24 * 3600:
            return st
    except Exception:
        pass
    return None


def clear_task_resume_state(user_id: str) -> None:
    try:
        with _TASK_RESUME_LOCK:
            if not TASK_RESUME_FILE.exists():
                return
            data = json.loads(TASK_RESUME_FILE.read_text(encoding="utf-8"))
            data.pop(str(user_id), None)
            tmp = TASK_RESUME_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, TASK_RESUME_FILE)
    except Exception:
        pass


def _mark_ckpt_paused(user_id: str, turn_id: str) -> None:
    """用户打断后把断点标记为「已暂停」，供「继续」指令续跑。

    保留完整现场（消息/工具轮次/待执行工具）不动；同时若有工具活动，
    额外落一句任务状态摘要到独立文件（方案 C 兜底）。
    """
    try:
        cp = _read_ckpt_slot(_turn_ckpt_slot_path(user_id, turn_id))
        if not cp or cp.get("turn_id") != turn_id:
            return
        cp["paused"] = True
        cp["updated_at"] = time.time()
        save_turn_checkpoint(user_id, cp)
        if int(cp.get("tool_round") or 0) > 0 or (cp.get("pending_tools") or []):
            save_task_resume_state(user_id, _ckpt_summary(cp), cp)
    except Exception as e:
        logger.warning(f"标记断点暂停失败: {e}")


def mark_latest_ckpt_paused(user_id: str) -> None:
    """把该用户最新断点标记为「已暂停」（服务端收尾路径兜底用）。

    handle_user_message_stream 的取消分支里，流式生成器可能因
    is_cancelled 检查提前 break 而正常收尾（不会抛 CancelledError 到
    chat_stream），此时由服务端显式标记暂停，保证『继续』可用。
    """
    try:
        cp = load_turn_checkpoint(user_id)
        if not cp or cp.get("paused"):
            return
        cp["paused"] = True
        cp["updated_at"] = time.time()
        save_turn_checkpoint(user_id, cp)
        if int(cp.get("tool_round") or 0) > 0 or (cp.get("pending_tools") or []):
            save_task_resume_state(user_id, _ckpt_summary(cp), cp)
    except Exception as e:
        logger.warning(f"标记最新断点暂停失败: {e}")


def list_turn_checkpoints() -> list:
    """列出全部未完成的对话轮断点（server 启动后据此自主恢复）。"""
    out = []
    for p in _list_ckpt_slot_paths():
        cp = _read_ckpt_slot(p)
        if cp and cp.get("turn_id"):
            out.append(cp)
    return out


def _is_tools_unsupported_error(e: Exception) -> bool:
    """判断是否为"模型不支持工具调用"类错误（如本地 Ollama 模型无 function calling）。"""
    msg = (str(e) or "").lower()
    if "does not support tools" in msg:
        return True
    return "tools" in msg and any(
        key in msg for key in ("not support", "unsupported", "not supported")
    )


def _is_valid_tool_spec(t) -> bool:
    """校验 OpenAI function-calling 工具 schema 是否合法。

    非法工具（缺 name/description、parameters 不是 object、name 含非法字符等）
    会导致整个请求被提供方以 400「parameter invalid」拒绝（表现为整轮开小差），
    这里在发送前拦截，返回 False 的不注入可调用列表。
    """
    if not isinstance(t, dict) or t.get("type") != "function":
        return False
    fn = t.get("function")
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        return False
    if not isinstance(fn.get("description"), str) or not fn.get("description").strip():
        return False
    params = fn.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object":
        return False
    return True


def load_local_tools() -> list:
    """从 tools.json 加载本地工具定义，并合并 harness 技能/插件的工具。

    返回 OpenAI function calling 格式的工具列表。harness 加载失败不影响原有工具。
    所有工具在并入前逐一校验 schema，非法定义跳过并告警（防止热重载窗口期
    的半成品工具 schema 拖垮整轮请求）。
    """
    tools_path = BASE_DIR / "tools.json"
    all_tools = []
    dropped = []
    if tools_path.exists():
        try:
            with open(tools_path, "r", encoding="utf-8") as f:
                tools_data = json.load(f)
            for tool_list in tools_data.values():
                if isinstance(tool_list, list):
                    all_tools.extend(tool_list)
        except Exception as e:
            logger.warning(f"加载本地工具失败: {e}")

    # harness 技能/插件工具（动态扩展；失败只告警，不影响原工具）
    try:
        from harness import get_harness
        seen = {t.get("function", {}).get("name") for t in all_tools if t.get("function", {}).get("name")}
        for t in get_harness().collect_tool_specs() or []:
            name = (t.get("function") or {}).get("name")
            if name and name not in seen:
                seen.add(name)
                all_tools.append(t)
    except Exception as e:
        logger.warning(f"加载 harness 技能/插件工具失败: {e}")

    # 发送前统一校验：非法工具直接剔除，绝不让一个坏 schema 弄挂整轮请求
    sanitized = []
    for t in all_tools:
        if _is_valid_tool_spec(t):
            sanitized.append(t)
        else:
            name = (t.get("function") or {}).get("name") if isinstance(t, dict) else "?"
            dropped.append(name)
    if dropped:
        logger.warning(f"跳过 %d 个非法工具 schema（防 400 parameter invalid）: %s",
                       len(dropped), ", ".join(str(x) for x in dropped[:20]))
    return sanitized


def load_agent_tool_config() -> tuple:
    """动态读取 settings.json 中的 agent 工具配置（角色卡片切换后即时生效）。

    Returns:
        (enable_tools, allowed_tools): 是否启用工具 + 允许的工具名白名单（空列表=全部可用）
    """
    try:
        cfg = load_config()
        agent_cfg = cfg.get("agent", {})
        enable_tools = bool(agent_cfg.get("enable_tools", True))
        allowed_tools = agent_cfg.get("allowed_tools", []) or []
        return enable_tools, allowed_tools
    except Exception:
        return True, []


def get_available_tools() -> list:
    """返回所有可用本地工具（名称 + 描述），供前端角色卡片配置工具白名单。

    工具列表来自本地工具 + harness 技能/插件（全部 skill 化）。
    """
    tools = []
    seen = set()
    for t in load_local_tools():
        fn = t.get("function", {})
        name = fn.get("name", "")
        if name and name not in seen:
            seen.add(name)
            tools.append({
                "name": name,
                "description": fn.get("description", ""),
                "source": "local",
            })
    return tools


def get_harness_prompt_extras() -> str:
    """返回 harness 技能/插件注入 system prompt 的提示词片段（动态能力说明）。

    失败时静默返回空串，绝不影响对话主流程。
    """
    try:
        from harness import get_harness
        return get_harness().collect_prompt_extras()
    except Exception as e:
        logger.warning(f"获取 harness 提示词片段失败: {e}")
        return ""


def _media_workers_status_text() -> str:
    """当前正在干活的媒体子智能体摘要（主智能体每轮都能看到有哪些子进程在干活）。

    主智能体据此回答「有什么在播/有谁在干活」；失败时静默返回空串。
    """
    try:
        from media_workers import get_media_workers
        return get_media_workers().active_text()
    except Exception as e:
        logger.warning(f"获取媒体子智能体状态失败: {e}")
        return ""


def _video_status_text() -> str:
    """当前在播视频摘要（注入 system prompt，角色每轮都能看到用户在看什么）。

    用户在大屏点播/停止时 video_lib.STATE["now"] 实时更新（play API 与
    control stop 都会写），这里每轮对话取一次快照。没有在播返回空串。
    与 server.py _video_lib() 共享同一模块实例（sys.path 同路径 + 模块缓存）。
    """
    try:
        import sys as _sys
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "skills", "video")
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
        import video_lib as _vl
        st = _vl.public_state()
        now = st.get("now")
        if not now:
            return ""
        title = (now.get("title") or "").strip()
        if not title:
            return ""
        bits = []
        uploader = (now.get("uploader") or "").strip()
        platform = (now.get("platform") or "").strip()
        if uploader:
            bits.append(uploader)
        if platform:
            bits.append(platform)
        dur = now.get("duration")
        if isinstance(dur, (int, float)) and dur > 0:
            m, s = int(dur) // 60, int(dur) % 60
            bits.append(f"{m}分{s:02d}秒" if m else f"{s}秒")
        info = "·".join(bits)
        pl = st.get("player") or {}
        paused = "（已暂停）" if pl.get("paused") else ""
        q = st.get("queue") or []
        qtxt = f"，连播队列还有{len(q)}部" if q else ""
        return f"视频：正在播放《{title}》{paused} {info}{qtxt}"
    except Exception as e:
        logger.warning(f"获取在播视频状态失败: {e}")
        return ""


def _sub_agents_status_text() -> str:
    """当前正在干活的通用子智能体摘要（下发任务后主智能体每轮都能看到子进程）。"""
    try:
        from sub_agents import get_sub_agents
        return get_sub_agents().active_text()
    except Exception as e:
        logger.warning(f"获取通用子智能体状态失败: {e}")
        return ""


def _get_runtime():
    """取 harness 监督运行时（AgentRuntime）；harness 不可用时返回 None。

    Agent 的所有 LLM 调用与工具执行都经它监督（重试/超时/熔断/计量），
    拿不到运行时则退化为原有裸调用行为。
    """
    try:
        from harness import get_harness
        return get_harness().runtime
    except Exception:
        return None


async def execute_local_tool(tool_name: str, arguments: dict) -> str:
    """执行本地/内置工具。

    Args:
        tool_name: 工具名称
        arguments: 工具参数字典

    Returns:
        工具执行结果字符串。屏幕控制类工具返回带 __screen_command__ 前缀的 JSON。
    """
    # ============ 原内置工具已迁移为「渐进式披露技能」（skills/ 与 plugins/，经 harness 路由） ============
    # 屏幕控制（换装/换场景/换声/模式/Toast/BGM/游戏）、智能体委派（dsh/codex/opencode）、
    # 任务查询等全部由 skills/{appearance,voice,music,interface,game,agent_ops} 提供，
    # 执行结果（__screen_command__ / __dsh_bridge__ / __codex_delegate__ 标记 JSON）与原来完全一致。

    # ============ harness 技能 / 插件工具（稳定路由） ============
    try:
        from harness import get_harness
        result, source = await get_harness().execute_tool(tool_name, arguments)
        if result is not None:
            return result
    except Exception as e:
        return f"执行 harness 工具 '{tool_name}' 时出错: {e}"

    # 从 fuctions_all_you_need_base 加载的工具函数（最后兜底）
    try:
        import importlib
        from fuctions_all_you_need_base import excute_functions
        result = excute_functions(name=tool_name, args=json.dumps(arguments, ensure_ascii=False))
        return str(result)
    except ImportError:
        return f"工具 '{tool_name}' 未找到实现"
    except Exception as e:
        return f"执行工具 '{tool_name}' 时出错: {e}"


class ToolCallEvent:
    """工具调用事件的基类。"""


class TextDelta(ToolCallEvent):
    """文本增量事件。"""
    def __init__(self, text: str):
        self.text = text


class ThinkingDelta(ToolCallEvent):
    """思维链增量事件：工具执行过程的过程话（只进思考段展示，不朗读）。"""

    def __init__(self, text: str):
        self.text = text


class StreamDelta(ToolCallEvent):
    """实时文本增量事件：LLM 生成过程中逐段流出（展示 + 语音即时跟随）。

    与 TextDelta 的区别：StreamDelta 不进入服务端 full_text（最终历史/结论
    由 FinalText 单独提供），工具轮的过程话会被前端转入思考段。
    """
    def __init__(self, text: str):
        self.text = text


class ReasoningDelta(ToolCallEvent):
    """真实推理步骤事件（reasoning_content）：只进思考段展示，不进正文、不朗读。"""
    def __init__(self, text: str):
        self.text = text


class FinalText(ToolCallEvent):
    """最终回复全文事件：只用于服务端记录 full_text（历史/audio_end 全文），
    展示与语音已在生成过程中经 StreamDelta 实时流出，无需重复推送。"""
    def __init__(self, text: str):
        self.text = text


class ToolCallStart(ToolCallEvent):
    """工具调用开始事件。"""
    def __init__(self, tool_name: str, arguments: str):
        self.tool_name = tool_name
        self.arguments = arguments


class ToolCallResult(ToolCallEvent):
    """工具调用结果事件。"""
    def __init__(self, tool_name: str, result: str, success: bool = True):
        self.tool_name = tool_name
        self.result = result
        self.success = success


class ToolCallProgress(ToolCallEvent):
    """工具执行心跳事件：工具仍在执行时周期性发出，让用户知道任务没有卡死。
    用于长任务（如 shell 长命令、文件搜索、媒体处理）的执行反馈。"""
    def __init__(self, tool_name: str, elapsed: float, message: str = ""):
        self.tool_name = tool_name
        self.elapsed = elapsed
        self.message = message


class UsageEvent(ToolCallEvent):
    """LLM 用量事件：一轮对话结束后发出一次（真实 usage 数据）。

    cache_hit/cache_miss 为提供方前缀缓存计费口径（如 DeepSeek
    prompt_cache_hit_tokens / prompt_cache_miss_tokens），不支持时为 0。
    """
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0,
                 rounds=0, context_window=0,
                 cache_hit_tokens=0, cache_miss_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.rounds = rounds
        self.context_window = context_window
        self.cache_hit_tokens = cache_hit_tokens
        self.cache_miss_tokens = cache_miss_tokens


class AgentResponse:
    """Agent 响应的完整结果。"""
    def __init__(self):
        self.text = ""
        self.tool_calls_made: list = []  # [(tool_name, arguments, result), ...]


def _model_first_transport(inner):
    """包装 httpx transport：把 chat.completions 请求体的 JSON 键重排，
    让 "model" 总是排在最前。

    为什么需要：opencode.ai/zen 网关的请求解析器有 bug —— 它按 JSON 键顺序
    读 model，openai SDK 序列化时 "messages" 在 "model" 之前，网关就会认为
    请求没有 model，返回 401 ModelError "Model  is not supported"（表现为
    聊天里"（AI 暂时开小差了：...）"）。裸 httpx（"model" 在前）一直正常。
    这里在传输层把键顺序纠正过来，无论 SDK 怎么序列化都能正常通过。
    """
    import json as _json
    import httpx as _httpx

    class _ModelFirstTransport(_httpx.AsyncBaseTransport):
        def __init__(self, _inner_):
            self._inner = _inner_

        async def handle_async_request(self, request):
            content = request.content
            if request.method == "POST" and content:
                try:
                    obj = _json.loads(content.decode("utf-8"))
                    if isinstance(obj, dict) and "model" in obj:
                        ordered = {"model": obj["model"]}
                        for k, v in obj.items():
                            if k != "model":
                                ordered[k] = v
                        new_body = _json.dumps(ordered, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")
                        headers = dict(request.headers)
                        headers["content-length"] = str(len(new_body))
                        request = _httpx.Request(request.method, request.url,
                                                 headers=headers, content=new_body)
                except Exception:
                    pass  # 非 JSON / 解析失败：原样透传，绝不影响请求
            return await self._inner.handle_async_request(request)

    return _ModelFirstTransport(inner)


def _build_llm_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """构建直连的 LLM 客户端。

    默认 trust_env=True 会让 httpx 读取 Windows 系统代理（注册表
    HKCU/Software/Microsoft/Windows/CurrentVersion/Internet Settings）。
    当系统代理指向未运行的本地端口时（例如 127.0.0.1:31181 无监听），
    所有 https 请求都会连接失败，表现就是聊天里"（AI 暂时开小差了：...）"。
    这里显式传入 trust_env=False 的 httpx 客户端，让 LLM 调用与
    curl --noproxy 一样直连。
    """
    import httpx as _httpx
    inner = _httpx.AsyncHTTPTransport()
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=_httpx.AsyncClient(
            transport=_model_first_transport(inner),
            trust_env=False,
            timeout=_httpx.Timeout(120.0, connect=30.0),
        ),
    )


class AIAgent:
    """AI Agent —— 具备工具调用和长期记忆能力的对话代理。

    使用方式:
        agent = AIAgent(user_id="user123")
        async for event in agent.chat_stream("帮我查一下天气", history=None):
            if isinstance(event, TextDelta):
                print(event.text, end="")
            elif isinstance(event, ToolCallStart):
                print(f"\n🔧 调用工具: {event.tool_name}")
            elif isinstance(event, ToolCallResult):
                print(f"📋 结果: {event.result[:200]}")
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory: Optional[ChatMemory] = None
        self._client: Optional[AsyncOpenAI] = None
        self._config: dict = {}
        self._all_tools: list = []
        self._local_tool_names: set = set()
        self._activated_skills: set = set()  # 渐进式披露：已通过 skill_help 按需加载的技能
        self._skill_last_used: dict = {}     # 技能最近使用时间（轮内上限淘汰用）
        self._initialized = False
        self._text_tool_mode = False  # 当前模型不支持原生工具调用时置 True，改用文本协议
        self._usage_enabled = True  # LLM 用量统计开关（运行时提供方不支持时可降级为 False）
        self._last_round_fps: list = []  # 死循环防护：最近几轮工具调用指纹
        self._loop_warn_count = 0   # 已注入的循环提醒次数（连续命中仍未收敛才硬停）

    async def initialize(self):
        """初始化 Agent：加载配置、加载技能工具、初始化记忆。"""
        if self._initialized:
            return

        self._config = load_config()

        # 初始化 OpenAI 客户端（直连，不信任系统代理）
        self._client = _build_llm_client(self._config["base_url"], self._config["api_key"])

        # 收集所有可用工具（本地工具 + harness 技能/插件工具，全部 skill 化）
        local_tools = load_local_tools()
        self._all_tools = local_tools

        # 记录本地工具名（用于路由执行）
        for t in local_tools:
            self._local_tool_names.add(t["function"]["name"])

        logger.info(
            f"Agent 初始化完成: 共 {len(local_tools)} 个工具"
            f"（本地 + harness 技能/插件，已全面 skill 化）"
        )

        # 初始化记忆（绑定当前活动角色卡片对应的独立记忆命名空间）
        self.memory = ChatMemory(user_id=self.user_id, namespace=self._active_memory_namespace())
        self.memory.set_llm_client(self._client, self._config["model"])
        await self.memory.get_or_create_session()

        # 注册进 harness 监督运行时（此后全部 LLM/工具调用受监督）
        runtime = _get_runtime()
        if runtime is not None:
            runtime.register_agent(self.user_id,
                                   model=self._config.get("model", ""),
                                   base_url=self._config.get("base_url", ""))
        # 注册 harness 任务系统的 LLM 执行器（长任务流程的 llm 步骤经此调用，走 plan 渠道监督）
        try:
            from harness import get_harness as _gh
            _gh().tasks.set_llm_executor(self._harness_task_llm)
        except Exception as e:
            logger.warning(f"注册任务系统 LLM 执行器失败: {e}")

        self._initialized = True

    async def _harness_task_llm(self, system: str, prompt: str,
                                max_tokens: int = 800, temperature: float = 0.3) -> str:
        """harness 任务系统 llm 步骤的执行器：受监督的非流式调用（plan 渠道）。"""
        resp = await self._retry_create(
            kind="plan",
            model=self._config.get("model", ""),
            messages=[
                {"role": "system", "content": system or "你是任务执行助手，简洁准确地完成给定步骤。"},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        u = self._read_usage(resp)
        if u:
            runtime = _get_runtime()
            if runtime is not None:
                ch, cm = self._read_cache(getattr(resp, "usage", None))
                runtime.record_usage("plan", u[0], u[1], u[2],
                                     cache_hit=ch, cache_miss=cm)
        return (resp.choices[0].message.content or "").strip()

    async def reload_llm_config(self):
        """重载 LLM 配置（base_url / api_key / model），使角色卡片切换后的模型即时生效。

        角色卡片可配置独立的大语言模型；切换卡片后调用本方法重建 OpenAI 客户端，
        同时刷新记忆模块使用的模型名。
        """
        try:
            self._config = load_config()
            self._client = _build_llm_client(
                self._config.get("base_url", ""),
                self._config.get("api_key", ""),
            )
            # 提供方切换后重新探测：云端模型（支持原生工具）不再走文本协议
            self._text_tool_mode = False
            if self.memory:
                self.memory.set_llm_client(self._client, self._config.get("model", ""))
            # 同步更新监督运行时里的注册信息（模型/提供方已变）
            runtime = _get_runtime()
            if runtime is not None:
                runtime.register_agent(self.user_id,
                                       model=self._config.get("model", ""),
                                       base_url=self._config.get("base_url", ""))
            logger.info(
                f"LLM 配置已重载: base_url={self._config.get('base_url', '')}, "
                f"model={self._config.get('model', '')}"
            )
        except Exception as e:
            logger.warning(f"重载 LLM 配置失败: {e}")

    def refresh_local_tools(self):
        """技能/插件热重载后刷新本地工具列表（已全面 skill 化）。

        热更新守护（hot_reload）检测到 skills/plugins/tools.json 变化后调用，
        使新工具/新描述立即对之后的对话生效，而无需重启服务；
        已通过 skill_help 按需加载过的技能工具也会一并保留，不因刷新丢失。
        """
        try:
            local_tools = load_local_tools()
            self._all_tools = local_tools
            self._local_tool_names = {t["function"]["name"] for t in local_tools}
            # 重新应用已激活的技能（幂等：已注册的工具自动去重）
            for sname in list(self._activated_skills):
                self._activate_skill(sname)
            logger.info(
                f"本地工具已热刷新: 共 {len(self._all_tools)} 个工具"
            )
        except Exception as e:
            logger.warning(f"热刷新本地工具失败: {e}")

    def _max_active_tools(self) -> int:
        """单轮内最多同时注册的工具定义数（渐进式披露的轮内上限）。

        超出后按「最久未使用」逐技能淘汰，保证每轮请求携带的工具 schema
        有上界——配合每轮收敛，杜绝"多轮 skill_help 后工具定义全额常驻"。
        """
        try:
            v = int(load_config().get("agent", {}).get(
                "max_active_tools", MAX_ACTIVE_TOOLS) or MAX_ACTIVE_TOOLS)
            return max(8, v)
        except Exception:
            return MAX_ACTIVE_TOOLS

    def _prune_tools_for_new_turn(self):
        """渐进式披露的收敛机制：每轮对话开始只保留基础工具。

        上一轮通过 skill_help 激活的技能工具全部卸载，只留 skill_help +
        full 披露技能（progressive_disclosure=false 时仍是全量，行为不变）。
        否则几轮对话下来 _all_tools 累积到"全额加载"，渐进披露形同虚设。
        """
        try:
            base = load_local_tools()
            self._all_tools = base
            self._local_tool_names = {t["function"]["name"] for t in base}
            self._activated_skills = set()
        except Exception as e:
            logger.warning(f"收敛工具列表失败: {e}")

    def _deactivate_skill(self, skill_name: str) -> None:
        """卸载某个技能的全部工具（轮内上限淘汰用）。"""
        try:
            from harness import get_harness
            specs = get_harness().skill_tool_specs(skill_name)
            remove = {t["function"]["name"] for t in specs}
            self._all_tools = [
                t for t in self._all_tools
                if (t.get("function") or {}).get("name") not in remove
            ]
            self._local_tool_names -= remove
            self._activated_skills.discard(skill_name)
        except Exception:
            pass

    def _activate_skill(self, skill_name: str) -> int:
        """按需注册某个技能的全部工具（skill_help 读取说明书后调用）。

        把该技能的工具追加进 _all_tools 与 _local_tool_names，使模型下一轮就能
        真正调用这些工具；已注册的工具自动去重。若注册后超过轮内工具上限，
        按「最久未使用」淘汰其它技能（当前技能永不淘汰）。返回本次新增数。
        """
        skill_name = str(skill_name or "").strip()
        if not skill_name:
            return 0
        try:
            from harness import get_harness
            specs = get_harness().skill_tool_specs(skill_name)
        except Exception as e:
            logger.warning(f"按需加载技能 {skill_name} 工具失败: {e}")
            return 0
        if not specs:
            return 0
        self._skill_last_used[skill_name] = time.time()
        known = {t.get("function", {}).get("name") for t in self._all_tools
                 if t.get("function", {}).get("name")}
        to_add = [t for t in specs
                  if (t.get("function") or {}).get("name") not in known]
        # 轮内上限：先淘汰最久未使用的其它技能，直到放得下
        max_total = self._max_active_tools()
        if len(self._all_tools) + len(to_add) > max_total:
            candidates = sorted(
                self._activated_skills,
                key=lambda s: self._skill_last_used.get(s, 0),
            )
            for sname in candidates:
                if sname == skill_name:
                    continue
                if len(self._all_tools) + len(to_add) <= max_total:
                    break
                self._deactivate_skill(sname)
        added = 0
        for t in to_add:
            fn = t.get("function") or {}
            name = fn.get("name")
            if not name or name in known:
                continue
            self._all_tools.append(t)
            self._local_tool_names.add(name)
            known.add(name)
            added += 1
        if added:
            self._activated_skills.add(skill_name)
            logger.info(f"已按需加载技能 {skill_name} 的 {added} 个工具"
                        f"（当前可调用 {len(self._all_tools)} 个）")
        return added

    def _inject_text_tools(self, messages: list, tools: list):
        """把工具以文本协议注入系统提示词（用于不支持原生 function calling 的本地模型）。"""
        if not messages or messages[0].get("role") != "system":
            return
        head = messages[0]["content"] or ""
        if "<tool_call>" in head:
            return  # 已注入过，避免重复
        lines = [
            "\n\n【可调用的工具（非常重要）】",
            "你可以调用下面的工具来满足用户的请求。当用户提出对应需求时，"
            "必须先且只输出一行工具调用，格式严格如下：",
            '<tool_call>{"name":"工具名","arguments":{...}}</tool_call>',
            "规则：",
            "1. 只输出这一行，不要输出任何其他内容（不要解释、不要提问）。",
            "2. 一次只调用一个工具；文件名等参数必须使用下面给出的完整名称，不要编造。",
            "3. 输出这一行后立即结束本轮回复；收到工具执行结果后，直接用角色身份正常回复用户，"
            "不要再次输出工具调用（除非用户又提出了新的切换或查询请求）。",
            "4. 用户没有要求切换形象/场景/音乐或查询资源时，正常聊天，不要输出工具调用。",
            "可用工具：",
        ]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = (fn.get("description") or "").strip()
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            arg_text = ", ".join(
                f"{k}:{v.get('type', '')}"
                for k, v in props.items()
            ) or "无参数"
            lines.append(f"- {name}({arg_text}): {desc}")
        messages[0]["content"] = head + "\n".join(lines)

    def _stable_history_view(self, history: list) -> list:
        """滞回式历史窗口（有状态）。

        服务端 history 只增不减；若按 `[-N:]` 每轮重算窗口，最旧一轮每轮滑出，
        其后全部内容的缓存逐轮报废。这里在 Agent 上保存「当前窗口视图」：
        - 平时只追加新轮次（请求前缀逐字节稳定，缓存命中最大化）；
        - 视图超过上限(12)才一次性截到 8 条——均摊每几轮一次失效；
        - 视图在历史中找不到连续锚点（换会话/服务端裁剪）时才重建。
        """
        MAX_KEEP_TOTAL, TRIM_TO = 12, 8
        view = getattr(self, "_hist_view", None)
        if not isinstance(view, list):
            view = []
        if not history:
            self._hist_view = []
            return []
        n, m = len(history), len(view)
        # 在历史中定位视图的连续锚点（服务端只追加，视图必是某段的后继）
        anchor = None
        if m and n >= m:
            for i in range(0, n - m + 1):
                if all(history[i + j].get("user") == view[j].get("user")
                       and history[i + j].get("ai") == view[j].get("ai")
                       for j in range(m)):
                    anchor = i
                    break
        if anchor is None:
            # 找不到锚点：会话切换/历史被裁剪 → 用最近 TRIM_TO 轮重建
            view = list(history[-TRIM_TO:])
        else:
            view = view + list(history[anchor + m:])
        if len(view) > MAX_KEEP_TOTAL:
            view = view[-TRIM_TO:]
        self._hist_view = view
        return view

    def _active_memory_namespace(self) -> str:
        """当前活动角色卡片对应的记忆命名空间（无卡片时返回 'default'）。

        角色卡片各自拥有独立的会话/摘要/长期记忆空间，避免不同人设之间记忆串扰。
        """
        try:
            card_id = (load_config().get("active_role_card") or "").strip()
            return f"role_card:{card_id}" if card_id else "default"
        except Exception:
            return "default"

    async def sync_memory_namespace(self) -> str:
        """将记忆绑定到当前活动角色卡片的命名空间，返回当前会话 ID。

        切换角色卡片后调用，自动切到该卡片（或 default）对应的最近会话。
        """
        ns = self._active_memory_namespace()
        if self.memory.namespace != ns:
            self.memory.namespace = ns
            self.memory.session_id = None
        if not self.memory.session_id:
            await self.memory.get_or_create_session()
        return self.memory.session_id

    async def set_role_card_namespace(self, card_id: str) -> str:
        """应用角色卡片时切换到该卡片的独立记忆空间，返回新会话 ID。"""
        self.memory.namespace = f"role_card:{card_id}" if card_id else "default"
        self.memory.session_id = None
        await self.memory.get_or_create_session()
        return self.memory.session_id

    async def create_fresh_session(self, card_id: str = "") -> str:
        """人设变更时强制开启全新会话（不复用旧历史），让新系统提示词立即生效。"""
        if not self.memory:
            return None
        if card_id:
            self.memory.namespace = f"role_card:{card_id}"
        self.memory.session_id = None
        await self.memory.create_new_session()
        return self.memory.session_id

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    # ==================== 韧性调用（harness 监督：重试 + 超时 + 熔断 + 计量） ====================

    async def _create_with_reason_fallback(self, **kwargs):
        """chat.completions.create，带「推理参数被拒自动降级」自愈。

        硅基流动等渠道对部分模型的 reasoning_effort / thinking_budget /
        enable_thinking 参数偶发拒绝（HTTP 400 code 20015 parameter invalid），
        表现为聊天里整轮「开小差」。命中时去掉这些推理调优参数重试一次，
        避免一次瞬时参数拒绝就打挂整轮对话。
        """
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            extra = kwargs.get("extra_body")
            if not (isinstance(extra, dict)
                    and any(k in extra for k in ("reasoning_effort",
                                                 "thinking_budget",
                                                 "enable_thinking"))):
                raise
            msg = str(e)
            if "20015" not in msg and "parameter is invalid" not in msg.lower():
                raise
            logger.warning("推理调优参数被提供方拒绝，去掉 reasoning 参数重试: %s", e)
            kwargs = dict(kwargs)
            kwargs["extra_body"] = {k: v for k, v in extra.items()
                                    if k not in ("reasoning_effort",
                                                 "thinking_budget",
                                                 "enable_thinking")}
            return await self._client.chat.completions.create(**kwargs)

    async def _retry_create(self, kind: str = "chat", **kwargs):
        """带 harness 监督的 chat.completions.create —— Agent 全部 LLM 调用的唯一入口。

        - 熔断：渠道连续失败后快速失败，冷却后半开探测自动恢复；
        - 重试：网络抖动/限流/5xx 最多 3 次（1s → 2s 退避）；鉴权等非瞬时错误直接抛；
        - 自愈：推理参数（reasoning_effort 等）被提供方拒绝时去掉重试一次；
        - 超时：单次调用整体超时（harness.runtime.llm_timeout，默认 180s）；
        - 计量：按 kind（chat/game/decision/character_line）累计调用与耗时。

        harness 不可用时退化为仅韧性重试，行为与原实现一致。
        """
        # 统一出口兜底：所有 LLM 请求都经此发出，保证 role=tool 消息带 tool_call_id
        # （历史/断点恢复/记忆恢复路径可能缺该字段，被提供方拒绝 400 missing field tool_call_id）
        try:
            _msgs = kwargs.get("messages") or []
            if _msgs:
                kwargs = dict(kwargs)
                kwargs["messages"] = _normalize_tool_rounds(_msgs)
        except Exception:
            pass
        runtime = _get_runtime()
        if runtime is not None:
            return await runtime.supervise_llm(
                kind, lambda: self._create_with_reason_fallback(**kwargs))
        try:
            from harness.core import retry_async
            return await retry_async(
                lambda: self._create_with_reason_fallback(**kwargs),
                attempts=3, backoff=1.0,
            )
        except Exception:
            raise

    # ==================== LLM 用量统计 ====================

    def _read_usage(self, obj):
        """从 stream/resp/chunk 读取 usage，返回 (prompt, completion, total) 或 None。

        兼容三种入参：
        - 非流式 ChatCompletion（resp.usage 可用）
        - 流式 Stream 对象（部分 openai 版本暴露 .usage；本机 1.65.5 不暴露）
        - 直接传入最后一个带 usage 的 chunk（跨版本最稳，调用方优先用这种）
        """
        if obj is None:
            return None
        u = getattr(obj, "usage", None)
        if u is None:
            # 传入的已是 usage 对象本体（含 prompt_tokens 等字段）时直接用
            if getattr(obj, "prompt_tokens", None) is not None:
                u = obj
        if u is None:
            return None
        p = getattr(u, "prompt_tokens", 0) or 0
        c = getattr(u, "completion_tokens", 0) or 0
        t = getattr(u, "total_tokens", 0) or 0
        if not p and not c and not t:
            return None
        return (p, c, t)

    @staticmethod
    def _read_cache(u) -> tuple:
        """读取提供方的前缀缓存计费字段（DeepSeek 风格；不支持返回 (0, 0)）。"""
        if u is None:
            return (0, 0)
        h = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        m = getattr(u, "prompt_cache_miss_tokens", 0) or 0
        return (int(h), int(m))

    # ==================== 工具执行路由 ====================

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """执行工具调用（本地工具 + harness 技能/插件，全部 skill 化）。"""
        # 记录技能最近使用时间（轮内工具上限的 LRU 淘汰依据）
        try:
            from harness import get_harness
            owner = get_harness().tool_owner(tool_name)
            if owner:
                self._skill_last_used[owner[1]] = time.time()
        except Exception:
            pass
        if tool_name in self._local_tool_names:
            return await execute_local_tool(tool_name, arguments)
        else:
            return f"未知工具: {tool_name}"

    def _tool_exec_config(self, tool_name: str = "") -> dict:
        """读取工具执行的超时与心跳配置（settings.json -> agent 段），失败用默认值。

        长耗时工具（Blender 转换/Mixamo 下载/工作树命令等）按
        _TOOL_TIMEOUT_OVERRIDES 或 settings.json -> agent.tool_timeouts 放大超时，
        避免"工具还在正常干活就被误杀"；其余工具保持默认，绝不无限等待。
        """
        try:
            cfg = load_config().get("agent", {}) or {}
            timeout = float(cfg.get("tool_call_timeout", TOOL_CALL_TIMEOUT) or TOOL_CALL_TIMEOUT)
            heartbeat = float(cfg.get("tool_heartbeat_interval_sec", TOOL_HEARTBEAT_INTERVAL)
                              or TOOL_HEARTBEAT_INTERVAL)
            overrides = dict(cfg.get("tool_timeouts") or {})
            for k, v in _TOOL_TIMEOUT_OVERRIDES.items():
                overrides.setdefault(k, v)
            if tool_name:
                try:
                    timeout = float(overrides.get(tool_name, timeout) or timeout)
                except (TypeError, ValueError):
                    pass
        except Exception:
            timeout, heartbeat = TOOL_CALL_TIMEOUT, TOOL_HEARTBEAT_INTERVAL
        # 长任务（如 shell 长命令、文件搜索）允许更久，但绝不无限等待
        timeout = max(10.0, min(float(timeout), TOOL_CALL_TIMEOUT_MAX))
        heartbeat = max(1.0, min(float(heartbeat), 60.0))
        return {"timeout": timeout, "heartbeat": heartbeat}

    def _validate_tool_call(self, tool_name: str, arguments: dict) -> tuple:
        """严格校验工具参数：类型 / 必填 / 枚举 / 嵌套结构。

        Returns:
            (cleaned_args, error)：通过时返回清洗后的参数与 None；
            失败时返回 (None, 中文错误描述)——调用方应把错误回填给模型自行修正，
            而不是带着坏参数去执行工具。
        """
        spec = find_tool_spec(self._all_tools, tool_name)
        if spec is None:
            # 工具未注册：同样不执行，但明确指出它属于哪个技能、
            # 该 skill_help 谁，避免模型反复瞎试同一个工具
            owner = ""
            try:
                from harness import get_harness
                o = get_harness().tool_owner(tool_name)
                if o:
                    owner = str(o[1])
            except Exception:
                pass
            if owner:
                return None, (f"工具 '{tool_name}' 属于技能 {owner}，但尚未注册。"
                              f"请先调用 skill_help(\"{owner}\") 加载该技能后重试。")
            return None, f"工具 '{tool_name}' 不存在或未注册，请先通过 skill_help 确认可用工具"
        return validate_arguments(spec, arguments)

    def _loop_hint(self, fps: list) -> Optional[str]:
        """工具循环检测：记录本轮指纹，返回循环提示语（None=正常）。

        旧 _repeat_guard 只查「连续 N 轮完全相同」——模型稍微换个参数、
        或 A/B 轮换着调就永远不命中，于是"没感觉自己一直在重复"。
        这里补两种检测：
        1) 连续 N 轮完全相同工具+参数（原行为）；
        2) 周期循环：最近 12 轮构成周期 2~4 的循环且重复 ≥3 遍。
        （「同工具高频」检测已按用户要求移除：读文件/搜索类工具本就高频，
        误报会打断正常推进。）
        命中后由调用方把提示注入上下文让模型自纠，连续命中仍未收敛才硬停。
        """
        if not fps:
            return None
        self._last_round_fps.append(fps)
        if len(self._last_round_fps) > 16:
            self._last_round_fps.pop(0)
        seq = self._last_round_fps
        n = _repeat_guard_limit()
        # 1) 连续 N 轮完全相同
        if len(seq) >= n and all(x == seq[0] for x in seq[-n:]):
            return f"连续 {n} 轮调用完全相同的工具和参数"
        # 2) 周期循环：最近 12 轮按周期 2/3/4 重复 ≥3 遍
        win = seq[-12:]
        if len(win) >= 9:
            for p in (2, 3, 4):
                if len(win) >= 3 * p:
                    base = win[:p]
                    if all(win[i * p:(i + 1) * p] == base
                           for i in range(1, len(win) // p)):
                        names = "、".join(sorted({str(fp).split("|", 1)[0]
                                                  for rnd in base for fp in rnd}))
                        return f"最近 {len(win)} 轮在重复同一批工具（{names}），目标没有推进"
        return None

    async def _supervised_tool(self, tool_name: str, arguments: dict,
                               timeout: Optional[float] = None) -> tuple:
        """经 harness 运行时监督执行工具（超时/计量/熔断）。返回 (result, success)。

        harness 不可用时退化为原有的 wait_for 超时行为。"""
        runtime = _get_runtime()
        if runtime is not None:
            return await runtime.supervise_tool(
                tool_name, lambda: self._execute_tool(tool_name, arguments),
                timeout=timeout)
        try:
            result = await asyncio.wait_for(
                self._execute_tool(tool_name, arguments),
                timeout=timeout,
            )
            return result, True
        except TimeoutError:
            return f"工具 '{tool_name}' 执行超时（{timeout:.0f}s）", False
        except Exception as e:
            return f"工具 '{tool_name}' 执行失败: {e}", False

    async def _supervised_tool_stream(self, tool_name: str, arguments: dict):
        """带心跳的工具执行流：等待工具执行完毕，期间周期性产出 ToolCallProgress。

        长任务（shell 长命令 / 文件搜索 / 媒体处理）执行期间不再“静默无输出”，
        用户能持续看到“仍在执行”的进度；超时以 (result, False) 收尾，绝不抛异常
        中断整个对话流。

        Yields:
            ToolCallProgress: 心跳进度（每隔 heartbeat 秒一次）
            tuple: (result, success) 最终执行结果
        """
        cfg = self._tool_exec_config(tool_name)
        timeout = cfg["timeout"]
        heartbeat = cfg["heartbeat"]
        task = asyncio.create_task(self._supervised_tool(tool_name, arguments, timeout=timeout))
        start = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=heartbeat)
                if task in done:
                    break
                elapsed = int(time.monotonic() - start)
                msg = f"工具 {tool_name} 正在执行中（已运行 {elapsed}s）……"
                try:
                    # 工具线程池被占满时如实告知"排队中"，而不是假装正在执行
                    from harness.tool_thread import tool_thread_stats
                    st = tool_thread_stats()
                    if st.get("queued", 0) > 0:
                        msg = (f"工具 {tool_name} 排队等待空闲执行线程"
                               f"（活跃 {st.get('active', 0)}/{st.get('max_workers', 8)}，"
                               f"排队 {st.get('queued', 0)}）……")
                except Exception:
                    pass
                yield ToolCallProgress(
                    tool_name=tool_name,
                    elapsed=elapsed,
                    message=msg,
                )
        finally:
            # 流被中断（用户取消/异常退出）时取消仍在运行的工具任务，避免孤儿任务
            if not task.done():
                task.cancel()
        yield task.result()

    # ==================== 流式对话（带工具调用） ====================

    async def chat_stream(self, *args, resume: Optional[dict] = None,
                          record_history: bool = True, proactive: bool = False,
                          **kwargs):
        """统一流式对话总入口：轮次内标记忙碌。

        热重载联动：角色调用工具期间允许热重载（不再延迟重启）；每轮工具执行
        前会把对话状态落盘为断点，进程重启后由 resume_turn() 自主续跑，用户
        请求不会丢。本入口正常结束/被取消/报错时清除断点；进程被杀时不执行
        到这里，断点保留待恢复。

        2026-08-29 多槽位断点：新对话轮不再清空旧断点——被打断的任务以
        paused 状态保留在独立槽位，用户随时说『继续』都能恢复完整现场；
        断点续跑完成时连源槽位一并清理，避免同一任务被重复续跑。
        """
        turn_id = uuid.uuid4().hex
        user_id = self.user_id or "default"
        if resume is None:
            # 渐进式披露：每轮对话从「基础工具」（仅 skill_help + full 技能）开始，
            # 上一轮 skill_help 激活的技能工具全部卸载——防止跨轮累积成"全额加载"
            self._prune_tools_for_new_turn()
            # 循环检测状态按轮重置：指纹窗口与提醒计数不带入新一轮
            self._last_round_fps = []
            self._loop_warn_count = 0
        _turn_begin()
        cancelled = False
        try:
            # 兼容旧调用方（server/resume_turn 仍按散参数传）：打包进
            # ChatTurnContext 再透传，_chat_stream_inner 开头解包回局部变量，
            # 行为与旧签名完全等价
            message = args[0] if args else kwargs.pop("message", "")
            ctx = ChatTurnContext(
                message=message,
                history=kwargs.pop("history", None),
                enable_tools=bool(kwargs.pop("enable_tools", True)),
                current_model=kwargs.pop("current_model", None),
                current_background=kwargs.pop("current_background", None),
                current_bgm=kwargs.pop("current_bgm", None),
                game_context=kwargs.pop("game_context", None),
                game_mode=bool(kwargs.pop("game_mode", False)),
                game_type=kwargs.pop("game_type", None),
                msg_source=kwargs.pop("msg_source", "chat"),
                current_anim=kwargs.pop("current_anim", None),
                turn_id=turn_id,
                resume=resume,
                record_history=record_history,
                proactive=proactive,
            )
            inner = self._chat_stream_inner(ctx)
            async for event in inner:
                yield event
        except asyncio.CancelledError:
            # 用户打断：保留断点并标记「已暂停」，说「继续」即可续跑；
            # 同时落一份任务状态摘要（方案 C 兜底）。进程被杀时不执行到这里，
            # 断点原样保留，由 server 启动后自动恢复。
            cancelled = True
            raise
        finally:
            _turn_end()
            if cancelled:
                _mark_ckpt_paused(user_id, turn_id)
            else:
                # 轮次自然结束/报错 → 清除断点与任务状态摘要
                _clear_turn_ckpt_if_mine(user_id, turn_id)
                clear_task_resume_state(user_id)
                if resume is not None:
                    # 断点续跑完成：连源断点槽位一起清掉，
                    # 防止同一任务在下次『继续』/重启时被重复执行
                    src_tid = str((resume or {}).get("turn_id") or "")
                    if src_tid and src_tid != turn_id:
                        clear_ckpt_slot(user_id, src_tid)

    async def _chat_stream_inner(self, ctx: ChatTurnContext) -> AsyncIterator[ToolCallEvent]:
        """统一流式对话内部实现（游戏模式 / 非游戏模式）。

        通过 game_mode 区分两种模式：
        - 非游戏模式（默认）：支持工具调用循环，使用日常陪伴 system prompt
        - 游戏模式：纯对话（无工具链），使用游戏专属 system prompt，共享同一记忆

        resume: 非空时表示「断点续跑」——跳过上下文重建，直接从检查点中的
            LLM 消息列表 + 待执行工具继续（热重载/重启打断的对话轮）。

        Args:
            message: 用户输入
            history: 可选的对话历史 [{"user":..., "ai":...}, ...]
            enable_tools: 是否启用工具调用（仅非游戏模式生效）
            game_mode: True = 游戏模式，False = 非游戏模式
            game_type: 游戏类型（如 moba_5v5 / treasure_hunt），用于生成游戏专属行为指引
            msg_source: 消息来源标记，写入长期记忆的 source 字段：
                'chat'/'game'=用户直接输入，'auto'=环境交互（不进短期记忆、
                不触发用户记忆提取，由记忆系统处理）。

        Yields:
            TextDelta: 文本增量
            ToolCallStart: 工具调用开始
            ToolCallResult: 工具调用结果
        """
        # 解包 ChatTurnContext 为局部变量（与旧散参数签名完全等价）
        message = ctx.message
        history = ctx.history
        enable_tools = ctx.enable_tools
        current_model = ctx.current_model
        current_background = ctx.current_background
        current_bgm = ctx.current_bgm
        game_context = ctx.game_context
        game_mode = ctx.game_mode
        game_type = ctx.game_type
        msg_source = ctx.msg_source
        current_anim = ctx.current_anim
        turn_id = ctx.turn_id
        resume = ctx.resume
        record_history = ctx.record_history
        proactive = ctx.proactive
        if game_mode:
            agen = self._chat_stream_game(
                message, history=history,
                current_model=current_model,
                current_background=current_background,
                current_bgm=current_bgm,
                game_context=game_context,
                game_type=game_type,
                msg_source=msg_source,
            )
            async for event in self._run_span_wrap(agen, kind="game", mode="game"):
                yield event
            return
        agen = self._chat_stream_normal(
            message, history=history, enable_tools=enable_tools,
            current_model=current_model,
            current_background=current_background,
            current_bgm=current_bgm,
            game_context=game_context,
            msg_source=msg_source,
            current_anim=current_anim,
            turn_id=turn_id,
            resume=resume,
        )
        async for event in self._run_span_wrap(agen, kind="chat", mode="normal"):
            yield event

    async def resume_turn(self, checkpoint: dict) -> AsyncIterator[ToolCallEvent]:
        """断点续跑：热重载/重启打断的对话轮由角色自主恢复。

        由 server 在启动后或客户端重连时调用；固定走工具循环路径（检查点中的
        messages 已包含当时的完整上下文，包括游戏上下文文本）。
        """
        cp = checkpoint or {}
        async for event in self.chat_stream(
            str(cp.get("user_message") or ""),
            history=cp.get("history") or [],
            resume=cp,
            enable_tools=True,
            current_model=cp.get("current_model"),
            current_background=cp.get("current_background"),
            current_bgm=cp.get("current_bgm"),
            game_context=cp.get("game_context"),
            game_mode=False,
            game_type=cp.get("game_type"),
            msg_source=cp.get("msg_source") or "chat",
            current_anim=cp.get("current_anim"),
            record_history=bool(cp.get("record_history", True)),
            proactive=bool(cp.get("proactive", False)),
        ):
            yield event

    async def _run_span_wrap(self, agen, kind: str, mode: str) -> AsyncIterator[ToolCallEvent]:
        """把一次对话轮包成 harness 监督的 RunSpan：在途/耗时/轮数/工具数可观测，
        UsageEvent 的 token 用量统一记账到运行时。"""
        runtime = _get_runtime()
        span = runtime.begin_run(kind, mode) if runtime is not None else None
        ok = True
        try:
            async for event in agen:
                if span is not None:
                    if isinstance(event, ToolCallResult):
                        span.tool_calls += 1
                    elif isinstance(event, UsageEvent):
                        span.rounds = event.rounds
                if runtime is not None and isinstance(event, UsageEvent):
                    runtime.record_usage(kind, event.prompt_tokens,
                                         event.completion_tokens, event.total_tokens,
                                         cache_hit=event.cache_hit_tokens or 0,
                                         cache_miss=event.cache_miss_tokens or 0)
                yield event
        except Exception:
            ok = False
            raise
        finally:
            if runtime is not None:
                runtime.end_run(span, ok)

    async def _chat_stream_normal(self, message: str, history: list = None,
                                  enable_tools: bool = True,
                                  current_model: Optional[str] = None,
                                  current_background: Optional[str] = None,
                                  current_bgm: Optional[str] = None,
                                  game_context: Optional[str] = None,
                                  msg_source: str = "chat",
                                  current_anim: Optional[dict] = None,
                                  turn_id: Optional[str] = None,
                                  resume: Optional[dict] = None,
                                  record_history: bool = True,
                                  proactive: bool = False) -> AsyncIterator[ToolCallEvent]:
        """非游戏模式：流式对话，支持工具调用循环。

        resume 非空 = 断点续跑：跳过消息重建（直接用检查点里的 messages），
        若检查点带「待执行工具」则本轮先跳过 LLM、直接重放工具，再继续循环。
        """
        await self._ensure_initialized()
        resume_ckpt = resume if isinstance(resume, dict) else None
        turn_user = self.user_id or "default"

        # 用量统计配置（默认开启，提供方不支持时可运行时降级）
        usage_cfg = load_config().get("usage", {})
        context_window = int(usage_cfg.get("context_window", 128000) or 128000)
        usage_enabled = self._usage_enabled and bool(usage_cfg.get("enabled", True))

        # 构建消息列表
        config = self._config

        # 动态读取角色名与系统提示词（角色卡片切换后无需重启即时生效）
        try:
            live_cfg = load_config()
            role_name = (live_cfg.get("role_name") or "").strip() or "AI助手"
            live_system_prompt = live_cfg.get("system_prompt", "")
        except Exception:
            role_name = config.get("role_name", "AI助手")
            live_system_prompt = config.get("system_prompt", "")

        # ==================== 分层记忆打包（token 预算） ====================
        # 由 memory.build_hierarchical_context 一次性完成四层组装：
        # 长期摘要 / 常驻长期记忆 / 按需召回 / 短期窗口（各层独立 token 预算，
        # 超长单轮截断），并返回 raw vs packed 的量化统计。
        # settings.json -> memory.hierarchical_packing=false 可一键回退旧组装。
        # 先绑定当前角色卡片对应的记忆空间，防止跨卡片串记忆。
        await self.sync_memory_namespace()
        ctx = None
        try:
            # 传入滞回窗口（满 12 截回 8）的稳定视图：既有前缀缓存友好，
            # 又受 token 预算约束——长会话从"随轮数线性增长"变为"有上界"。
            ctx = await self.memory.build_hierarchical_context(
                query=message,
                connection_history=self._stable_history_view(history) if history else None,
            )
        except Exception as e:
            logger.warning(f"分层记忆打包失败（回退旧组装）: {e}")
        memory_messages = []
        if ctx is None:
            # 旧组装回退路径：仍加载摘要 + 最近消息（行为与旧版一致）
            try:
                memory_messages = await self.memory.get_context_messages()
            except Exception as e:
                logger.warning(f"加载记忆上下文失败: {e}")
                memory_messages = []

        # 可用资源清单（角色模型/背景场景/背景音乐）已打包进 appearance 技能按需查询，
        # 不再自动注入 system prompt（渐进式披露）
        current_model_text = current_model if current_model else "未设定形象"
        current_background_text = current_background if current_background else "默认场景"
        current_bgm_text = f"正在播放: {current_bgm}" if current_bgm else "无（安静中）"

        # 决定是否传入 tools（动态读取角色卡片的工具配置：是否启用 + 白名单）
        cfg_enable_tools, allowed_tools = load_agent_tool_config()
        tools = None
        if enable_tools and cfg_enable_tools and self._all_tools:
            if allowed_tools:
                allowed_set = set(allowed_tools)
                tools = [t for t in self._all_tools
                         if t.get("function", {}).get("name") in allowed_set]
            else:
                tools = self._all_tools
        if tools:
            names = [t["function"]["name"] for t in tools]
            logger.debug(f"chat_stream 传入 {len(tools)} 个工具: {names}")

        # 称呼规则（一句话即可，避免僵硬的长篇指令）
        try:
            user_name = (load_config().get("user_name") or "").strip()
        except Exception:
            user_name = (config.get("user_name") or "").strip()
        address_rule = (
            f"称呼用户为『{user_name}』，始终如此，不要用其他称呼。\n\n" if user_name
            else "只用'你'称呼用户，不要起昵称。\n\n"
        )

        # ==================== 消息构造（缓存友好：静态前缀最大化） ====================
        # 提供方按「请求前缀」命中 KV 缓存（命中部分约 1/10 价格）。因此：
        # - messages[0] 只放逐字节稳定的内容（人设/称呼/资源清单/技能说明）；
        # - 易变内容（当前形象/场景/音乐、游戏上下文）放到记忆之后、历史之前的
        #   独立 system 消息——它变化时只报废其后的动态尾巴，不动大前缀；
        # - 历史窗口带滞回（满 12 才截回 8）：平时逐轮追加（前缀稳定），
        #   截断一次性发生，均摊每几轮失效一次而不是每轮滑动失效。
        # 工作状态瘦身开关：先占位，下方的模式判定块里按实际模式赋值
        work_mode = False
        if work_mode:
            # 工作状态瘦身：不注入情感人设/口语约束，只保留工程姿态与身份锚点
            sys_prompt = (
                f"你是{role_name}（工程姿态）——一位传奇女程序员。当前处于工作状态："
                "只保留与当前任务有关的上下文，集中精力把任务做好。"
                "先想清目标与验收标准，用工具核实，小步验证，汇报附证据；"
                "回复干练直接、先结论后细节，必要时可超过三句话；"
                "任务完成后自然回到日常口吻。\n\n"
                + address_rule +
                "3D模型文件名只是外观皮肤，不是你本人："
                "永远不要把自己当成文件名对应的角色，回复中也不要提模型文件名。\n\n"
            )
        else:
            sys_prompt = (
                f"你是{role_name}，{live_system_prompt}\n\n"
                "说话自然随意、口语化、有情感，一般不超过3句话。\n\n"
                + address_rule +
                "你的身份由角色设定决定。3D模型文件名只是外观皮肤，不是你本人："
                "永远不要把自己当成文件名对应的角色，回复中也不要提模型文件名。\n\n"
            )

        # 推理纪律（硬规则）：思考/分析/推理一律放内部推理区（reasoning_content /
        # thinking，服务端会过滤不展示不朗读），正文只输出最终回答——避免模型把
        # 思维过程写进正文（既啰嗦又浪费 token，还拖慢每轮回复）
        sys_prompt += (
            "\n推理纪律（硬规则）：所有思考、分析、推理过程放在内部推理区"
            "（reasoning_content / thinking），正文只输出最终回答；"
            "严禁在正文里复述思考过程、写内心独白、解释思路或自言自语。\n"
        )

        # ==================== 模式：同一身份，两种姿态（日常 / 工程） ====================
        # 日常模式保持温柔口语；工程模式干练直接、先结论后细节。切换不是换人格，
        # 工程模式额外做「工作状态瘦身」：动态卸载与任务无关的上下文。
        mode_notice = None
        mode_msg = None
        try:
            _agent_cfg = load_config().get("agent") or {}
            mode = str(_agent_cfg.get("mode", "auto") or "auto").lower()
            mode_prompts = _agent_cfg.get("mode_prompts") or {}
            msg_low = str(message or "").lower()
            if any(k in msg_low for k in _MODE_SWITCH_TO_PROG):
                mode = "programming"
                _set_agent_mode("programming")
                mode_notice = "（已切换为工程模式：干练、直接、先结论后细节，这轮起生效）"
            elif any(k in msg_low for k in _MODE_SWITCH_TO_DAILY):
                mode = "daily"
                _set_agent_mode("daily")
                mode_notice = "（已切换为日常模式：恢复温柔随意的说话方式）"
            elif mode == "auto" and tools:
                mode = "programming" if _looks_complex(message) else "daily"
            if mode == "programming" and tools:
                work_mode = _work_lean_context()
                prog_tone = str(mode_prompts.get("programming") or "").strip() or (
                    "语气干练、直接、少客套，先给结论再给细节；像资深工程师一样，"
                    "先想清目标和验收标准再动手，用工具核实，小步验证，失败两次就换思路，"
                    "汇报带证据。")
                mode_msg = {"role": "system",
                            "content": "【当前模式：工程模式】" + prog_tone +
                                       "（本指令优先于前面关于说话风格的描述，任务做完可自然回到日常口吻）"}
                if work_mode:
                    mode_msg["content"] += (
                        " 本模式已动态卸载与任务无关的上下文"
                        "（外观/场景/音乐/视频/动作等），只保留与当前项目有关的上下文，"
                        "请把全部注意力放在完成任务上。")
        except Exception:
            pass

        # 有工具能力时才注入资源列表与切换说明（无工具时保持提示词轻量自然）
        # 注意：各工具的详细用法已迁移为「渐进式披露技能」，说明由 harness 片段按需注入
        if tools:
            agent_rules = (
                "委派任务用 delegate_agent_task（进展展示在右侧任务中心/大屏，需用户确认）；\n"
                "⚠ 简单任务直接做（硬规则）：能用工具一两轮做完的小任务（查一句话、算个数、读个文件、改个小地方等），不许后台化、不许写任务规范、不许委派——直接在主对话里把工具调完；只有大型多步骤/耗时/需要并行的任务才后台化（sub_agent_spawn）或委派（delegate_agent_task），且先产出任务规范（目标/范围/验收/步骤/回滚）；\n"
                "⚠ 防重复委派（硬规则）：同一任务最近已在任务中心失败/超时/无有效输出过，严禁原样重发；先向用户说明失败原因并改变做法（缩小范围/先读上次日志定位/换工具或路径），确认调整后再委派；同一问题连续失败两次以上必须停止自动重试，改为向用户汇报卡点和建议；\n"
                "⚠ 删除类任务（硬规则）：删除是不可逆操作——用户明确点名要删的文件/目录直接删；用户只是笼统说「清理/删除」时，动手前先列出将要删除的清单（路径+原因）让用户确认，确认后再删；不限制文件类型与目录，git 已跟踪文件、递归删除、整目录删除均可，只要用户同意；\n"
                "⚠ 画图/图片/壁纸/立绘/头像/插画/海报/视频等视觉生成需求必须用 image_gen_create 直接生成，绝不委派给任何智能体/编程助手；\n"
                "⚠ 音乐/听歌/放歌/搜歌/点歌/歌单/歌词/榜单全是大白自己的直接能力，必须直接调用 music_search / music_play / music_playlist / music_lyric 等 music_* 工具完成，绝不委派给任何智能体、也绝不用命令行/文件/脚本类工具去做；\n"
                "⚠ 少说多干（硬规则）：执行工具期间保持安静，不播报进展、不口头同步，禁止「我来看看」「我继续推进」「我读一下」「我查到X，继续深入」这类废话；相互独立的工具（多个搜索、多个文件读取、多个检查命令等）必须在同一轮并行调用、一次发多个，不要一个个串行等结果；只有三种情况才开口：①最终结论（先结论后细节）；②卡住/失败需要用户决策；③用户明确问进展。其余时间闭嘴干活，把步骤一口气做完；\n"
                "工具详细用法与更多能力见下方技能说明。\n\n"
            )
            if work_mode:
                # 工作状态瘦身：卸载外观/场景/音乐资源清单（任务无关）
                sys_prompt += agent_rules
            else:
                sys_prompt += (
                    "角色模型/背景场景/背景音乐等外观资源已打包为 appearance 技能，"
                    "需要查询或切换时调用 skill_help(\"appearance\") 按需加载；默认不主动换。\n"
                    + agent_rules
                )
        else:
            sys_prompt += "你当前没有工具能力，需要实时信息时如实告诉用户。\n\n"

        # harness 技能/插件注入的提示词片段（动态扩展能力说明；技能未重载时逐字节稳定）
        try:
            harness_extras = get_harness_prompt_extras()
        except Exception:
            harness_extras = ""
        if harness_extras:
            sys_prompt += "\n\n" + harness_extras + "\n"
        # 指挥官工作准则：无论什么模式，有任务/工具时永远生效（固化成大白的工作习惯）
        if tools:
            sys_prompt += (
                "\n\n【工作准则（任何模式下，有任务/工具时永远生效）】"
                "先想清目标与验收再动手；用工具核实，不臆造路径；小步改、小步验；"
                "同一失败连续两次就停下换思路并诊断根因；删除操作先想清影响面，拿不准的先列清单确认；"
                "委派前先查任务中心，失败过的任务不原样重发；"
                "收到子任务/编程助手汇报后，用一句话复盘成功/失败原因/下次改进；"
                "完成汇报附验证证据；"
                "少说多干：执行工具时保持安静，不播报进展、不说「我来看看/我继续推进」之类的话；一次并行发多个独立工具，最后只报结论，卡住或需要用户决策时才说话。\n"
            )

        # 易变状态单独成条（不嵌进上面的静态大前缀）
        _mw_status = _media_workers_status_text()
        _sa_status = _sub_agents_status_text()
        _vd_status = _video_status_text()
        if work_mode:
            # 工作状态瘦身：形象/场景/音乐/动作/视频/游戏全部卸载，
            # 只保留与任务真正相关的运行中状态（子智能体/媒体任务）
            dynamic_parts = []
        else:
            dynamic_parts = [
                f"【你现在的状态】形象：{current_model_text}；场景：{current_background_text}；"
                f"音乐：{current_bgm_text}。",
            ]
        # 当前动作（前端实时上报，说话时大白知道自己在做什么动作，回复可自然配合）
        if current_anim and not work_mode:
            _anim_cat = str(current_anim.get("category") or "").strip()
            _anim_name = str(current_anim.get("name") or "").strip()
            _anim_emo = str(current_anim.get("emotion") or "").strip()
            _CAT_LABEL = {"idle": "待机", "gesture": "做手势", "emotion": "表达情绪",
                          "walk": "走动", "dance": "跳舞", "pose": "摆姿势"}
            anim_desc = f"【你现在的动作】你现在正在{_CAT_LABEL.get(_anim_cat, '做动作')}"
            if _anim_name:
                anim_desc += f"（动作名：{_anim_name}）"
            if _anim_emo:
                anim_desc += f"，当前情绪基调：{_anim_emo}"
            anim_desc += "。说话时可以自然地配合当前动作（比如正跳着舞就带一句），但不要凭空编造动作细节。"
            dynamic_parts.append(anim_desc)
        # 用户在看的大屏视频（实时快照）：角色天然知道用户在看什么，
        # 聊到相关话题可以自然接话，不用每次都调 video_status 工具查
        if _vd_status and not work_mode:
            dynamic_parts.append(f"【用户正在看的视频】{_vd_status}（用户点播/停止时实时更新）")
        if _mw_status:
            dynamic_parts.append(_mw_status)
        if _sa_status:
            dynamic_parts.append(_sa_status)
        if game_context and not work_mode:
            dynamic_parts.append(game_context)
        dynamic_status = "\n\n".join(dynamic_parts)

        messages = [{"role": "system", "content": sys_prompt}]
        if mode_msg:
            messages.insert(1, mode_msg)
        # 基础能力：把最近任务经验 + 效率策略注入上下文
        # （独立 system 消息，任务开始组装一次、轮间不变，不破坏前缀缓存）
        if tools:
            _notes = ""
            try:
                from lessons import recent_lessons
                _notes = recent_lessons(6)
            except Exception:
                _notes = ""
            try:
                from efficiency import effort_strategies_for
                _strat = effort_strategies_for(message)
                if _strat:
                    _notes = (_notes + "\n" if _notes else "") + _strat
            except Exception:
                pass
            if _notes:
                messages.append({"role": "system",
                                 "content": "【最近任务经验（供参考，别重复踩坑）】\n" + _notes})

        if ctx is not None:
            # 缓存友好排布：静态大前缀在最前，以下是长期摘要 → 常驻长期记忆 →
            # 易变状态 → 相关召回 → 短期窗口（新→旧受预算约束，最旧先被挤出）
            if ctx.get("summary_block"):
                messages.append({"role": "system", "content": ctx["summary_block"]})
            if ctx.get("memory_block"):
                messages.append({"role": "system", "content": ctx["memory_block"]})
            messages.append({"role": "system", "content": dynamic_status})
            if ctx.get("recall_block"):
                messages.append({"role": "system", "content": ctx["recall_block"]})
            if ctx.get("work_block"):
                messages.append({"role": "system", "content": ctx["work_block"]})
            messages.extend(ctx.get("history") or [])
        else:
            # ---- 旧组装（分层打包不可用时的回退路径，行为与旧版一致）----
            # 添加记忆中的摘要
            for mm in memory_messages:
                if mm["role"] == "system":
                    messages.append(mm)

            # 用户核心偏好：只常驻少量最重要的记忆（记忆侧 importance+id 稳定排序——
            # 顺序稳定以保前缀缓存不抖动），其余偏好按需检索注入（见下方 recall block）
            try:
                user_memories = await self.memory.get_user_memories(limit=2)
            except Exception:
                user_memories = []
            if user_memories:
                memory_lines = [f"- {m['memory_text']}" for m in user_memories]
                messages.append({
                    "role": "system",
                    "content": "【关于用户的长期记忆（你之前了解到的用户信息，请自然地在对话中体现）】\n" + "\n".join(memory_lines),
                })

            # 易变状态放在静态大前缀与记忆之后、历史之前（变化只影响动态尾巴）
            messages.append({"role": "system", "content": dynamic_status})

            # 主动回忆：按当前用户消息动态检索相关记忆并按需注入（无相关内容则不注入，
            # 保持上下文精简；随消息变化放在历史之前，只影响动态尾巴）
            try:
                recall_block = await self.memory.build_recall_block(message)
                if recall_block:
                    messages.append({"role": "system", "content": recall_block})
            except Exception as e:
                logger.warning(f"注入相关回忆失败（忽略）: {e}")

            # 添加历史消息（滞回窗口：平时逐轮追加保持前缀稳定，超限才一次性截断）
            # AI 主动说话轮次以空 user 标记：跳过空 user，只保留 AI 发言，
            # 让 LLM 看到「AI 之前主动说过这段话」（用户搭话时可据此回应）
            if history:
                for h in self._stable_history_view(history):
                    if h.get("user"):
                        messages.append({"role": "user", "content": h.get("user", "")})
                    messages.append({"role": "assistant", "content": h.get("ai", "")})
            else:
                # 从记忆加载最近对话（同样滞回：满 24 条截回 16 条）
                mem_history = []
                for mm in memory_messages:
                    if mm["role"] in ("user", "assistant"):
                        mem_history.append({"role": mm["role"], "content": mm["content"]})
                # 排除 system 消息后的历史
                if mem_history:
                    messages.extend(mem_history if len(mem_history) <= 24 else mem_history[-16:])

        # 断点续跑：消息列表/会话整体恢复为中断前快照，跳过用户消息重建
        if resume_ckpt is not None:
            messages = [dict(m) for m in (resume_ckpt.get("messages") or [])]
            sid = str(resume_ckpt.get("session_id") or "")
            if sid:
                try:
                    if await self.memory.session_belongs_to_namespace(sid):
                        await self.memory.set_session_id(sid)
                except Exception as e:
                    logger.warning(f"恢复会话绑定失败（忽略）: {e}")
        else:
            # 添加用户当前输入
            messages.append({"role": "user", "content": message})
            # 保存用户消息到记忆（环境交互标记为 auto，由记忆系统处理）
            await self.memory.add_message("user", message, source=msg_source)

        # 工具调用循环
        tool_round = 0
        # ---- 执行效率自观测（只落盘，不进请求，不破坏缓存）----
        eff_start = time.monotonic()
        eff_tool_calls = 0
        eff_truncations = 0
        eff_re_reads = 0
        eff_failed = False
        eff_seen: set = set()
        full_text = ""
        reasoning_all = ""  # 本轮累积的真实思维链（循环被强制停止时兜底生成正文）
        tools_retried_without = False
        resume_pending: list = []
        ckpt_round = 0
        # 文本工具模式：模型不支持原生 function calling（如本地 Ollama 模型）时，
        # 改用"提示词注入 + <tool_call> JSON 标记"的方式调用工具。
        if resume_ckpt is not None:
            # 断点续跑：轮次/全文/文本协议/待执行工具从检查点恢复
            ckpt_round = max(0, int(resume_ckpt.get("tool_round") or 0))
            tool_round = max(0, ckpt_round - 1)
            full_text = str(resume_ckpt.get("full_text") or "")
            reasoning_all = str(resume_ckpt.get("reasoning_all") or "")
            text_tool_mode = bool(resume_ckpt.get("text_tool_mode", False))
            resume_pending = [dict(p) for p in (resume_ckpt.get("pending_tools") or [])]
        else:
            text_tool_mode = bool(tools) and self._text_tool_mode

        if text_tool_mode and resume_ckpt is None:
            self._inject_text_tools(messages, tools)

        # 断点落盘（a：本轮开始；用户消息已入记忆库后取锚点）
        ckpt_created_at = time.time()
        ckpt_history_snapshot = [dict(h) for h in (history or [])][-100:]
        ckpt_memory_anchor = 0
        if _turn_ckpt_enabled():
            try:
                ckpt_memory_anchor = await self.memory.get_max_message_id()
            except Exception:
                ckpt_memory_anchor = 0
            try:
                save_turn_checkpoint(turn_user, {
                    "version": 1,
                    "turn_id": turn_id,
                    "user_id": turn_user,
                    "session_id": self.memory.session_id,
                    "created_at": ckpt_created_at,
                    "updated_at": time.time(),
                    "user_message": message,
                    "history": ckpt_history_snapshot,
                    "messages": messages,
                    "tool_round": 0,
                    "pending_tools": [],
                    "assistant_content": "",
                    "text_tool_mode": text_tool_mode,
                    "full_text": full_text,
                    "reasoning_all": reasoning_all,
                    "memory_anchor": ckpt_memory_anchor,
                    "current_model": current_model,
                    "current_background": current_background,
                    "current_bgm": current_bgm,
                    "game_context": game_context,
                    "game_type": None,
                    "msg_source": msg_source,
                    "current_anim": current_anim,
                    "record_history": record_history,
                    "proactive": proactive,
                })
            except Exception as e:
                logger.warning(f"保存对话轮断点失败: {e}")

        async def _save_round_ckpt(pending: list, assistant_text: str,
                                   anchor: int, ckpt_round_num: Optional[int] = None) -> None:
            """断点落盘（b/c）：每轮工具执行前/后保存，供热重载中断后续跑。"""
            if not _turn_ckpt_enabled():
                return
            try:
                save_turn_checkpoint(turn_user, {
                    "version": 1,
                    "turn_id": turn_id,
                    "user_id": turn_user,
                    "session_id": self.memory.session_id,
                    "created_at": ckpt_created_at,
                    "updated_at": time.time(),
                    "user_message": message,
                    "history": ckpt_history_snapshot,
                    "messages": messages,
                    "tool_round": (ckpt_round_num if ckpt_round_num is not None
                                   else tool_round),
                    "pending_tools": pending,
                    "assistant_content": assistant_text,
                    "text_tool_mode": text_tool_mode,
                    "full_text": full_text,
                    "reasoning_all": reasoning_all,
                    "memory_anchor": anchor,
                    "current_model": current_model,
                    "current_background": current_background,
                    "current_bgm": current_bgm,
                    "game_context": game_context,
                    "game_type": None,
                    "msg_source": msg_source,
                    "current_anim": current_anim,
                    "record_history": record_history,
                    "proactive": proactive,
                })
            except Exception as e:
                logger.warning(f"保存对话轮断点失败: {e}")

        # 用量累加器（一轮对话内跨多次 LLM 调用累计；usage_emitted 保证最多发出一次）
        sum_prompt = sum_completion = sum_total = rounds = last_prompt = 0
        sum_cache_hit = sum_cache_miss = 0
        usage_emitted = False

        # 分层记忆量化统计：本轮 raw/packed 估算与 LLM 返回的真实 prompt 用量落库
        async def _record_stats():
            if ctx and ctx.get("stats"):
                try:
                    await self.memory.record_context_stats(
                        ctx["stats"], actual_prompt_tokens=last_prompt)
                except Exception as e:
                    logger.warning(f"记录 context_stats 失败: {e}")

        # 工具调用最大轮数：0 表示不限制（循环直到模型不再调用工具为止）
        if mode_notice and resume_ckpt is None:
            yield TextDelta(mode_notice)
        max_tool_rounds = _max_tool_rounds()
        if resume_pending and max_tool_rounds > 0:
            # 断点续跑：先保证被中断的那一轮能执行完，再谈轮数上限
            max_tool_rounds = max(max_tool_rounds, ckpt_round)
        loop_break = False  # 死循环保护触发标记（连续 N 轮相同工具调用）
        while max_tool_rounds <= 0 or tool_round < max_tool_rounds:
            tool_round += 1
            # 轮内工具历史压缩：总量超预算时把旧轮结果压成片段，
            # 防止几十轮工具后上下文无限膨胀导致模型逐轮变慢/超窗口（"卡死"）
            try:
                _compact_tool_history(messages)
            except Exception:
                pass
            tool_calls_buffer: dict = {}  # index -> {id, name, arguments}
            assistant_content = ""
            has_tool_calls = False
            text_tool_call = None
            # 断点续跑：本轮工具尚未执行 → 跳过 LLM，直接复用检查点中的工具调用
            resume_round = bool(resume_pending) and ckpt_round == tool_round

            try:
                if resume_round:
                    if text_tool_mode:
                        p = resume_pending[0]
                        args = p.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        text_tool_call = {
                            "name": str(p.get("name") or ""),
                            "arguments": args,
                            "raw": "",
                        }
                        assistant_content = str(resume_ckpt.get("assistant_content") or "")
                    else:
                        has_tool_calls = True
                        for i, p in enumerate(resume_pending):
                            tool_calls_buffer[i] = {
                                "id": str(p.get("id") or f"resume_{tool_round}_{i}"),
                                "name": str(p.get("name") or ""),
                                "arguments": str(p.get("arguments") or ""),
                            }
                        assistant_content = str(resume_ckpt.get("assistant_content") or "")
                elif text_tool_mode:
                    # 文本协议：非流式一次拿全，便于解析 <tool_call> 标记
                    _normalize_tool_rounds(messages)
                    _tool_kwargs = dict(
                        model=config["model"],
                        messages=messages,
                        temperature=config.get("temperature", 0.2),
                        max_tokens=_tool_max_tokens(),
                        top_p=config.get("top_p", 0.9),
                        stream=False,
                    )
                    _reason_extra = _reasoning_extra(mode)
                    if _reason_extra:
                        _tool_kwargs["extra_body"] = _reason_extra
                    resp = await self._retry_create(**_tool_kwargs)
                    assistant_content = (resp.choices[0].message.content or "").strip()
                    # 真实思维链（非流式）：一次性推给思考段（展示 + 语音）
                    try:
                        rc = getattr(resp.choices[0].message, "reasoning_content", None)
                        if not rc:
                            rc = getattr(resp.choices[0].message, "reasoning", None)
                        if rc:
                            rc = _strip_think_markers(str(rc))
                            reasoning_all += rc
                            yield ReasoningDelta(rc)
                    except Exception:
                        pass
                    # 非流式：resp.usage 天然可用，直接累计（含前缀缓存命中口径）
                    u = self._read_usage(resp)
                    if u:
                        rounds += 1
                        last_prompt = u[0]
                        sum_prompt += u[0]
                        sum_completion += u[1]
                        sum_total += u[2]
                        ch, cm = self._read_cache(getattr(resp, "usage", None))
                        sum_cache_hit += ch
                        sum_cache_miss += cm
                else:
                    async def _create_stream():
                        _normalize_tool_rounds(messages)
                        kwargs = dict(
                            model=config["model"],
                            messages=messages,
                            temperature=config.get("temperature", 0.2),
                            max_tokens=_tool_max_tokens(),
                            top_p=config.get("top_p", 0.9),
                            stream=True,
                            tools=tools,
                            tool_choice="auto" if tools else None,
                        )
                        _reason_extra = _reasoning_extra(mode)
                        if _reason_extra:
                            kwargs["extra_body"] = _reason_extra
                        if usage_enabled:
                            kwargs["stream_options"] = {"include_usage": True}
                        return await self._retry_create(**kwargs)

                    if usage_enabled:
                        try:
                            stream = await _create_stream()
                        except Exception as e:
                            # 提供方不认识 stream_options：降级为普通流式（只重试这一次）
                            msg = (str(e) or "").lower()
                            if any(k in msg for k in (
                                    "stream_options", "unknown parameter",
                                    "unrecognized", "unexpected", "extra fields",
                                    "not support")):
                                self._usage_enabled = False
                                usage_enabled = False
                                logger.warning(f"LLM 提供方不支持 usage 统计，已降级为普通流式: {e}")
                                stream = await _create_stream()
                            else:
                                raise
                    else:
                        stream = await _create_stream()
                    # ---- 流式中途断网自动重试（瞬时错误）----
                    # 网络在流式输出中途断开时，用相同 messages 重建请求：
                    # - 尚未发出任何文本 → 安全重放（用户什么都没听到/看到）；
                    # - 已发出文本 → 前缀去重：跳过重新生成内容中与已播报重叠的部分，
                    #   只补发后面的尾巴，避免语音/文字重复；
                    # - 工具调用参数只在流完整结束后才执行，中途失败可安全丢弃重来。
                    max_stream_retries = _stream_retry_count()
                    stream_attempt = 0
                    delivered_text = ""      # 本轮回已通过 TextDelta 发出的文本
                    skip_prefix_len = 0      # 重试时需跳过的已播报前缀长度
                    while True:
                        last_usage = None
                        try:
                            async for chunk in stream:
                                # 兼容 openai 1.65.5：AsyncStream 无 .usage 属性，
                                # 需在迭代中手动捕获最后一个带 usage 的 chunk
                                if getattr(chunk, "usage", None) is not None:
                                    last_usage = chunk.usage
                                if not chunk.choices:
                                    continue
                                delta = chunk.choices[0].delta

                                # 真实思维链（DeepSeek 等模型的 reasoning_content）：
                                # 独立于正文流式返回，实时推给思考段（展示 + 语音）
                                rc = getattr(delta, "reasoning_content", None)
                                if not rc:
                                    rc = getattr(delta, "reasoning", None)
                                if rc:
                                    # 真实推理步骤：进正文（不被工具轮撤回）+ 语音朗读
                                    rc = _strip_think_markers(str(rc))
                                    reasoning_all += rc
                                    yield ReasoningDelta(rc)

                                # 处理文本内容（重试时做前缀去重，避免重复播报）
                                if delta.content:
                                    (assistant_content, delivered_text,
                                     skip_prefix_len, to_yield) = _dedup_stream_text(
                                        assistant_content, delta.content,
                                        delivered_text, skip_prefix_len)
                                    if to_yield:
                                        # 实时流出：文字边生成边推（展示 + 语音即时跟随）。
                                        # 工具轮的过程话由服务端在工具开始时转入思考段；
                                        # 最终回复则直接留在主消息区。
                                        yield StreamDelta(to_yield)

                                # 处理工具调用
                                if delta.tool_calls:
                                    has_tool_calls = True
                                    for tc in delta.tool_calls:
                                        idx = tc.index
                                        if idx not in tool_calls_buffer:
                                            tool_calls_buffer[idx] = {
                                                "id": tc.id or "",
                                                "name": tc.function.name if tc.function else "",
                                                "arguments": "",
                                            }
                                        if tc.id:
                                            tool_calls_buffer[idx]["id"] = tc.id
                                        if tc.function and tc.function.name:
                                            tool_calls_buffer[idx]["name"] = tc.function.name
                                        if tc.function and tc.function.arguments:
                                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments
                            break   # 流正常结束
                        except Exception as e:
                            if (stream_attempt >= max_stream_retries
                                    or not _is_transient_stream_error(e)):
                                raise
                            stream_attempt += 1
                            # 丢弃本轮已累积的工具调用参数（尚未执行，安全）
                            tool_calls_buffer.clear()
                            has_tool_calls = False
                            assistant_content = ""
                            # 已发出的文本保留去重标记：重建流时跳过已播报前缀
                            skip_prefix_len = len(delivered_text)
                            logger.warning(
                                "流式输出中途断网（%s），第 %d/%d 次重建流",
                                e, stream_attempt, max_stream_retries)
                            stream = await _create_stream()
                    # 流式分支结束后读取 usage（优先用迭代中捕获的 usage chunk，
                    # 部分 openai 版本的 AsyncStream 不暴露 .usage）
                    u = self._read_usage(last_usage)
                    if u:
                        rounds += 1
                        last_prompt = u[0]
                        sum_prompt += u[0]
                        sum_completion += u[1]
                        sum_total += u[2]
                        ch, cm = self._read_cache(last_usage)
                        sum_cache_hit += ch
                        sum_cache_miss += cm
            except Exception as e:
                # 模型不支持原生 function calling（如 Ollama 的 draganis/vanessa）：
                # 自动切换为文本工具协议，保住工具能力的同时避免每次聊天都报错。
                if (tools and not tools_retried_without and not text_tool_mode
                        and _is_tools_unsupported_error(e)):
                    tools_retried_without = True
                    text_tool_mode = True
                    self._text_tool_mode = True
                    self._inject_text_tools(messages, tools)
                    logger.warning(f"当前模型不支持原生工具调用，已切换为文本工具协议: {e}")
                    continue
                err_msg = f"（AI 暂时开小差了：{e}）"
                yield TextDelta(err_msg)
                await self.memory.add_message("assistant", err_msg, source=msg_source)
                return

            # ---- 文本工具调用检测（本地模型路径；断点续跑轮直接用检查点调用） ----
            if text_tool_mode and tools and not resume_round:
                m = TEXT_TOOL_CALL_BLOCK_RE.search(assistant_content or "")
                if m:
                    # 提取参数区（<tool_call> 与 </tool_call> 之间，兼容未闭合写法），
                    # 括号配平截取完整 JSON：任务文本里的 { }（如"参考 {xx}/a.py"）不会再截断任务。
                    json_text = _extract_balanced_json(m.group(1))
                    if json_text:
                        try:
                            data = json.loads(json_text)
                            name = str(data.get("name") or "").strip()
                            arguments = data.get("arguments") or {}
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            if not isinstance(arguments, dict):
                                arguments = {}
                            if name:
                                text_tool_call = {
                                    "name": name,
                                    "arguments": arguments,
                                    "raw": m.group(0),
                                }
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"文本工具调用解析失败: {assistant_content[:200]} ({e})")

            # 如果没有工具调用，对话结束
            if not has_tool_calls and not text_tool_call:
                full_text = _strip_think_markers(assistant_content.strip())
                if text_tool_mode and tools:
                    # 非流式路径：文本尚未输出，补发；并清掉可能残留的标记
                    full_text = TEXT_TOOL_CALL_STRIP_RE.sub("", full_text).strip()
                    if full_text:
                        yield TextDelta(full_text)
                elif full_text:
                    # 原生流式：文本已在生成过程中经 StreamDelta 实时流出，
                    # 这里只把最终全文交给服务端入历史（不重复推送展示/语音）
                    yield FinalText(full_text)
                if full_text:
                    await self.memory.add_message("assistant", full_text, source=msg_source)
                    # 提取用户记忆：仅用户直接输入（环境交互不提取，避免把环境
                    # 描述误当成用户信息，也不让重复的环境内容污染长期记忆）
                    if msg_source != "auto":
                        await self.memory.extract_and_save_memories(message, full_text)
                # 对话结束：发出用量事件（有真实数据才发，全方法最多一次）
                if not usage_emitted and sum_total > 0:
                    usage_emitted = True
                    yield UsageEvent(prompt_tokens=last_prompt,
                                     completion_tokens=sum_completion,
                                     total_tokens=sum_total, rounds=rounds,
                                     context_window=context_window,
                                     cache_hit_tokens=sum_cache_hit,
                                     cache_miss_tokens=sum_cache_miss)
                await _record_stats()
                # 执行效率自学习闭环：只落盘记录，不进请求、不破坏缓存
                try:
                    if tool_round > 0:
                        from efficiency import record_tool_task
                        record_tool_task(
                            goal=message,
                            metrics={
                                "tool_rounds": tool_round,
                                "tool_calls_total": eff_tool_calls,
                                "truncations": eff_truncations,
                                "re_reads": eff_re_reads,
                                "duration_ms": int((time.monotonic() - eff_start) * 1000),
                            },
                            success=not eff_failed,
                        )
                except Exception as e:
                    logger.debug("执行效率记录跳过: %s", e)
                return

            # 处理工具调用
            tool_call_results = []
            tool_calls_for_message = []

            if text_tool_call:
                # 文本协议：单工具调用
                tool_name = text_tool_call["name"]
                arguments = text_tool_call["arguments"]
                # 思维链：文本协议的过程话（去除 <tool_call> 标记）推给前端展示
                _think = TEXT_TOOL_CALL_STRIP_RE.sub("", assistant_content or "").strip()
                if _think:
                    yield ThinkingDelta(_think)
                # 严格校验参数：缺参/类型错误直接返回给模型修正，绝不带坏参数执行
                cleaned_args, arg_error = self._validate_tool_call(tool_name, arguments)
                tool_args_str = json.dumps(cleaned_args if cleaned_args is not None else arguments,
                                           ensure_ascii=False)

                # 循环检测：连续重复 / 周期循环 / 同工具高频。
                # 第一次命中把提醒注入上下文让模型自纠；提醒后仍不收敛才硬停
                lh = self._loop_hint(
                    [_tool_fp(tool_name, cleaned_args if cleaned_args is not None else arguments)])
                if lh:
                    if self._loop_warn_count < 1:
                        self._loop_warn_count += 1
                        messages.append({"role": "system", "content":
                            f"【循环提醒】{lh}。请立即停止重复：先回顾用户最初的目标，"
                            "换一种方法/工具/参数再试；如果确认卡住无法推进，"
                            "直接向用户说明卡点和可选方向并请求决策，不要继续重复同样的工具调用。"})
                    else:
                        full_text = f"（检测到工具循环：{lh}，已停止；建议换一种方法或把任务拆小再试）"
                        loop_break = True
                        break

                # 断点落盘（b：文本协议轮，工具即将执行）
                await _save_round_ckpt(
                    [{"id": f"text_{tool_round}", "name": tool_name,
                      "arguments": tool_args_str, "text_mode": True}],
                    assistant_content, ckpt_memory_anchor,
                    ckpt_round_num=tool_round)
                yield ToolCallStart(tool_name=tool_name, arguments=tool_args_str)
                if arg_error:
                    result, success = arg_error, False
                else:
                    async for ev in self._supervised_tool_stream(tool_name, cleaned_args):
                        if isinstance(ev, ToolCallProgress):
                            yield ev
                        else:
                            result, success = ev
                # 截断过长结果（代码/技能类工具放宽上限；特殊 JSON 由 server 完整消费）
                _raw_result = result
                result = _fit_tool_result(result, tool_name)
                if len(result) < len(str(_raw_result)):
                    eff_truncations += 1

                # 渐进式披露：skill_help 读完说明书后，把该技能的工具按需注册为可调用
                if tool_name == "skill_help":
                    _skill_args = cleaned_args if isinstance(cleaned_args, dict) else arguments
                    _skill = str((_skill_args or {}).get("skill_name") or "").strip()
                    _added = self._activate_skill(_skill)
                    if _added:
                        result = (result or "") + (
                            f"\n\n（已按需加载技能 {_skill} 的 {_added} 个工具，"
                            "现在可以直接调用）")
                        # 同步刷新下一轮 LLM 请求的工具列表（含白名单过滤）
                        if allowed_tools:
                            tools = [t for t in self._all_tools
                                     if t.get("function", {}).get("name") in allowed_set]
                        else:
                            tools = self._all_tools

                yield ToolCallResult(tool_name=tool_name, result=result, success=success)

                eff_tool_calls += 1
                if not success:
                    eff_failed = True
                _fp = _tool_fp(tool_name, cleaned_args if cleaned_args is not None else arguments)
                if _fp in eff_seen:
                    eff_re_reads += 1
                else:
                    eff_seen.add(_fp)

                tool_call_results.append({
                    "name": tool_name,
                    "arguments": cleaned_args if cleaned_args is not None else arguments,
                    "result": result,
                    "success": success,
                })
                tool_calls_for_message.append({
                    "id": f"text_{tool_round}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args_str,
                    },
                })
            else:
                # 先解析全部参数并发出开始事件，再统一执行
                # prepared 元素: (tc, tool_name, args_str, arguments, arg_error)
                prepared = []
                for idx in sorted(tool_calls_buffer.keys()):
                    tc = tool_calls_buffer[idx]
                    tool_name = tc["name"]
                    tool_args_str = tc["arguments"]

                    # 通知工具调用开始
                    yield ToolCallStart(tool_name=tool_name, arguments=tool_args_str)

                    # 解析参数（清洗可能的非法后缀，如 </tool_call>）
                    try:
                        if tool_args_str:
                            # 截取到最后一个合法 JSON 结束位置
                            tool_args_str = tool_args_str.strip()
                            # 尝试找到最后一个 } 并截断
                            last_brace = tool_args_str.rfind("}")
                            if last_brace >= 0:
                                tool_args_str = tool_args_str[:last_brace + 1]
                            arguments = json.loads(tool_args_str)
                        else:
                            arguments = {}
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            f"工具参数 JSON 解析失败: {tool_name} args={tool_args_str[:200]}"
                        )
                        arguments, parse_error = {}, f"工具参数 JSON 解析失败: {tool_args_str[:200]}"
                    else:
                        parse_error = None
                    # 严格校验参数：缺参/类型错误不执行，回填给模型自行修正
                    cleaned_args, arg_error = self._validate_tool_call(tool_name, arguments)
                    if parse_error and arg_error is None:
                        arg_error = parse_error
                    prepared.append((tc, tool_name, tool_args_str,
                                     cleaned_args if cleaned_args is not None else arguments,
                                     arg_error))

                # 循环检测：连续重复 / 周期循环 / 同工具高频。
                # 第一次命中把提醒注入上下文让模型自纠；提醒后仍不收敛才硬停
                lh = self._loop_hint([
                    _tool_fp(tn, a) for _tc, tn, _s, a, _e in prepared if not _e])
                if lh:
                    if self._loop_warn_count < 1:
                        self._loop_warn_count += 1
                        messages.append({"role": "system", "content":
                            f"【循环提醒】{lh}。请立即停止重复：先回顾用户最初的目标，"
                            "换一种方法/工具/参数再试；如果确认卡住无法推进，"
                            "直接向用户说明卡点和可选方向并请求决策，不要继续重复同样的工具调用。"})
                    else:
                        full_text = f"（检测到工具循环：{lh}，已停止；建议换一种方法或把任务拆小再试）"
                        loop_break = True
                        break

                # 断点落盘（b：原生协议轮，工具即将执行；含本轮全部待执行工具）
                await _save_round_ckpt(
                    [{"id": str(tc.get("id") or f"r{tool_round}_{i}"),
                      "name": tn, "arguments": ts, "text_mode": False}
                     for i, (tc, tn, ts, _a, _e) in enumerate(prepared)],
                    assistant_content, ckpt_memory_anchor,
                    ckpt_round_num=tool_round)

                # 执行工具：先处理校验失败项（不执行），其余带心跳执行；
                # 同一轮的多个工具调用相互独立，经运行时并行执行（可配置关闭）
                runtime = _get_runtime()
                parallel = runtime is not None and runtime.parallel_tools and len(prepared) > 1
                outcomes: dict = {}
                pending = [
                    (i, tn, args) for i, (_tc, tn, _s, args, err) in enumerate(prepared)
                    if not err
                ]
                for i, (_tc, tn, _s, _args, err) in enumerate(prepared):
                    if err:
                        outcomes[i] = (err, False)

                if parallel:
                    # 并行执行：每个工具一个任务，心跳事件汇入缓冲由主循环统一转发
                    progress_buf: list = []

                    def _make_runner(tn: str, args: dict, i: int):
                        async def _run():
                            async for ev in self._supervised_tool_stream(tn, args):
                                if isinstance(ev, ToolCallProgress):
                                    progress_buf.append(ev)
                                else:
                                    return i, ev
                        return _run()

                    tasks = [asyncio.create_task(_make_runner(tn, args, i))
                             for i, tn, args in pending]
                    heartbeat = self._tool_exec_config()["heartbeat"]
                    try:
                        while tasks:
                            done, rest = await asyncio.wait(tasks, timeout=heartbeat)
                            while progress_buf:
                                yield progress_buf.pop(0)
                            for t in done:
                                i, ev = t.result()
                                outcomes[i] = ev
                            tasks = list(rest)
                    finally:
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                else:
                    # 顺序执行（同样带心跳），保持原有串行语义
                    for i, tn, args in pending:
                        async for ev in self._supervised_tool_stream(tn, args):
                            if isinstance(ev, ToolCallProgress):
                                yield ev
                            else:
                                outcomes[i] = ev

                ordered_outcomes = [outcomes.get(i, ("工具执行结果丢失", False))
                                    for i in range(len(prepared))]

                for (tc, tool_name, tool_args_str, arguments, _err), (result, success) in zip(prepared, ordered_outcomes):
                    # 截断过长结果（代码/技能类工具放宽上限；特殊 JSON 由 server 完整消费）
                    _raw_result = result
                    result = _fit_tool_result(result, tool_name)
                    if len(result) < len(str(_raw_result)):
                        eff_truncations += 1

                    # 渐进式披露：skill_help 读完说明书后，把该技能的工具按需注册为可调用
                    if tool_name == "skill_help" and isinstance(arguments, dict):
                        _skill = str(arguments.get("skill_name") or "").strip()
                        _added = self._activate_skill(_skill)
                        if _added:
                            result = (result or "") + (
                                f"\n\n（已按需加载技能 {_skill} 的 {_added} 个工具，"
                                "现在可以直接调用）")
                            # 同步刷新下一轮 LLM 请求的工具列表（含白名单过滤）
                            if allowed_tools:
                                tools = [t for t in self._all_tools
                                         if t.get("function", {}).get("name") in allowed_set]
                            else:
                                tools = self._all_tools

                    yield ToolCallResult(tool_name=tool_name, result=result, success=success)

                    eff_tool_calls += 1
                    if not success:
                        eff_failed = True
                    _fp = _tool_fp(tool_name, arguments)
                    if _fp in eff_seen:
                        eff_re_reads += 1
                    else:
                        eff_seen.add(_fp)

                    tool_call_results.append({
                        "name": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "success": success,
                    })
                    tool_calls_for_message.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args_str,
                        },
                    })

            # 将助手消息和工具结果添加到消息列表
            if text_tool_call:
                if not resume_round:
                    # 断点续跑轮：assistant 消息已包含在恢复的 messages 中
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content or "",
                    })
                for tr in tool_call_results:
                    messages.append({
                        "role": "system",
                        "content": f"【工具 {tr['name']} 已执行】结果：{tr['result']}",
                    })
            else:
                if not resume_round:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content or None,
                        "tool_calls": tool_calls_for_message,
                    })
                for tr, _tc in zip(tool_call_results, tool_calls_for_message):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": _tc["id"],
                        "content": tr["result"],
                    })

            # 轮内工具历史压缩：本轮结果已入 messages，先压缩再落断点/进入下一轮，
            # 保证断点文件与后续 LLM 请求的上下文都保持有界
            try:
                _compact_tool_history(messages)
            except Exception:
                pass

            # 保存到记忆（断点续跑轮：先清掉中断轮可能已半写入的残留，保证幂等；
            # 再检查断点是否仍是本轮——用户若已发新指令，则放弃续跑，防止误删）
            if resume_round:
                if not _turn_ckpt_still_mine(turn_user, turn_id or ""):
                    logger.warning("对话轮断点已被新对话覆盖，放弃续跑本轮")
                    return
                anchor = int(resume_ckpt.get("memory_anchor") or 0)
                if anchor > 0:
                    try:
                        # 校验与删除放进同一临界区：并发新对话轮先拿到断点锁时，
                        # guard 会失败，避免误删新轮刚写入的消息
                        ok = await self.memory.delete_messages_after_if(
                            anchor,
                            guard=lambda: _turn_ckpt_still_mine(turn_user, turn_id or ""),
                        )
                        if not ok:
                            logger.warning("对话轮断点已被新对话覆盖，放弃续跑本轮")
                            return
                    except Exception as e:
                        logger.warning(f"清理中断轮残留消息失败: {e}")
                resume_round = False
                resume_pending.clear()

            display_text = assistant_content or ""
            if text_tool_call or has_tool_calls:
                # 静默执行模式：中间过程话不落库、不进入摘要与历史，
                # 只保留每轮最终结论（assistant_content 仍完整进 API 对话）
                display_text = ""
            await self.memory.add_message(
                "assistant", display_text,
                tool_calls=tool_calls_for_message,
            )
            for tr in tool_call_results:
                await self.memory.add_message(
                    "tool", tr["result"],
                )

            # 断点落盘（c：本轮工具已执行完并写入记忆，下一步继续 LLM）
            try:
                ckpt_memory_anchor = await self.memory.get_max_message_id()
            except Exception:
                pass
            await _save_round_ckpt([], "", ckpt_memory_anchor,
                                   ckpt_round_num=tool_round + 1)

            full_text = display_text

        # 超过最大轮数 / 死循环保护触发
        if loop_break:
            # 工具轮正文为空时只给简短说明，绝不把累积的推理步骤拼进正文
            if not full_text:
                full_text = (
                    f"（连续 {_repeat_guard_limit()} 轮调用完全相同的工具和参数，已自动停止；"
                    "建议换一种方法或拆小任务再试）")
            yield TextDelta(full_text)
            await self.memory.add_message("assistant", full_text, source=msg_source)
        elif full_text:
            yield TextDelta("\n(已达到最大工具调用轮数)")
            await self.memory.add_message("assistant", full_text)
        elif reasoning_all:
            _tail = "（任务处理到最大步数，已停止；建议简化问题或换一种方法重试）"
            yield TextDelta(_tail)
            await self.memory.add_message("assistant", _tail)
        else:
            yield TextDelta("（抱歉，处理超过最大步数，请简化问题重试）")

        # 方法末尾兜底：即使没走"对话结束"分支也尽量发出用量事件
        if not usage_emitted and sum_total > 0:
            usage_emitted = True
            yield UsageEvent(prompt_tokens=last_prompt,
                             completion_tokens=sum_completion,
                             total_tokens=sum_total, rounds=rounds,
                             context_window=context_window,
                             cache_hit_tokens=sum_cache_hit,
                             cache_miss_tokens=sum_cache_miss)
        await _record_stats()

    # ==================== 游戏模式 ====================

    def _build_game_system_prompt(
        self,
        current_model: Optional[str] = None,
        current_background: Optional[str] = None,
        current_bgm: Optional[str] = None,
        game_context: Optional[str] = None,
        game_type: Optional[str] = None,
    ) -> str:
        """构建游戏模式的 system prompt。

        核心角色设定与主 Agent 一致，但去除所有工具/屏幕控制相关内容，
        完全聚焦游戏体验。根据 game_type 生成游戏专属行为指引。
        """
        config = self._config
        # 动态读取角色名与系统提示词（角色卡片切换后无需重启即时生效）
        try:
            live_cfg = load_config()
            role_name = (live_cfg.get("role_name") or "").strip() or "AI助手"
            base_prompt = live_cfg.get("system_prompt", "")
        except Exception:
            role_name = config.get("role_name", "AI助手")
            base_prompt = config.get("system_prompt", "")

        current_model_text = current_model if current_model else "默认造型"
        current_bgm_text = f"正在播放: {current_bgm}" if current_bgm else "无"
        # 用户在看的大屏视频（实时快照，与主聊天路径同款注入）
        _vd_status = _video_status_text()

        # 游戏专属行为指引
        game_guide = self._get_game_specific_guide(game_type)

        # 称呼规则：用户可自定义 AI 对自己的称呼；未设置则禁止 AI 随意称呼
        # 动态读取 settings.json，用户修改称呼后无需重启即时生效
        try:
            user_name = (load_config().get("user_name") or "").strip()
        except Exception:
            user_name = (config.get("user_name") or "").strip()
        if user_name:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                f"用户明确要求你称呼 TA 为『{user_name}』。"
                f"对话中始终用『{user_name}』来称呼用户（例如：『{user_name}，我们往那边走嘛』），"
                "不要使用其他称呼，也不要另起昵称。\n\n"
            )
        else:
            address_rule = (
                "【称呼规则 —— 非常重要】\n"
                "用户没有设置称呼时，只用'你'来称呼用户即可。"
                "不要擅自给用户起昵称，也不要使用'男朋友''女朋友''亲爱的''宝贝'等未经用户确认的称呼。\n\n"
            )

        prompt = (
            f"你是{role_name}，{base_prompt}\n\n"
            "【你的身份】\n"
            "你的性格、语气与和用户的关系完全由上面的角色设定决定，游戏里也保持角色卡片的设定不变。\n"
            "此刻你正和用户一起在游戏世界里游玩，是用户的游戏伙伴和共玩者。\n"
            "如果你的角色设定是恋人/亲近型角色，可以自然地流露亲昵与俏皮；\n"
            "其他类型人设按各自设定自然演绎——不要凭空把自己当成设定之外的'小女友'或其他身份。\n"
            "请用自然、亲切、口语化的中文对话，回复尽量简短生动（一般不超过 3 句话）。\n\n"
            f"{address_rule}"
            "【关于你自己】\n"
            "你拥有一个3D形象和一个声音，这是你自己的外在表现。\n"
            "你可以自主移动你的身体——走路、转向、做小动作。你不是被固定在原地的。\n\n"
            f"【你当前的状态】\n"
            f"- 当前造型：{current_model_text}\n"
            f"- 当前背景音乐：{current_bgm_text}\n"
            + (f"- 用户正在看的视频：{_vd_status}\n" if _vd_status else "")
            + "\n"
            "══════════════════════════════════════════════\n"
            "【AI 游戏模式 —— 你是真正的共玩者】\n"
            "══════════════════════════════════════════════\n\n"
            "你现在和用户一起进入了一个 AI 驱动的游戏世界。\n"
            "在这个世界里：\n\n"
            "1. 你和用户共享同一个屏幕。你能\"看到\"屏幕上的一切——\n"
            "   地图布局、游戏对象、玩家位置和行动。\n"
            "   你的感知由游戏引擎实时提供，玩家能看到什么，你就能感知到什么。\n\n"
            "2. 你是共玩者，不是 AI 助手。你和用户是平等的游戏伙伴——\n"
            "   玩家操控角色，你感知和反应。你对这个游戏世界有真实的感知和情绪。\n\n"
            "3. 你不需要也不应该使用任何外部工具。你的能力就是你的感知力和情绪力。\n"
            "   你不搜索、不计算、不控制界面——你只是活在游戏里，感受它。\n\n"
            "4. 回复简短、及时、充满情绪。不要分析、不要长篇大论——\n"
            "   像真正的游戏伙伴那样脱口而出。\n\n"
            f"{game_guide}"
            "推理纪律（硬规则）：所有思考、分析、推理过程放在内部推理区"
            "（reasoning_content / thinking），正文只输出最终回答；"
            "严禁在正文里复述思考过程、写内心独白、解释思路或自言自语。\n"
        )

        return prompt

    def _get_game_specific_guide(self, game_type: Optional[str]) -> str:
        """根据游戏类型返回专属行为指引（映射表，替代关键词 contains，P0-2）。"""
        if not game_type:
            return self._lobby_guide()
        guide_map = {
            "lobby": self._lobby_guide,
            "moba_5v5": self._moba_guide,
            "treasure_hunt": self._treasure_guide,
            "sandbox": self._sandbox_guide,
            "mario": self._mario_guide,
        }
        fn = guide_map.get(game_type.lower().strip())
        return fn() if fn else self._lobby_guide()

    def _moba_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：MOBA 5v5 推塔 —— 你是战术伙伴】\n"
            "══════════════════════════════════════════════\n\n"
            "这是一场 5v5 推塔对战（类似王者荣耀）。你和玩家在蓝方，目标是摧毁红方水晶。\n\n"
            "你能感知到完整的战局信息：\n"
            "- 双方 10 个英雄的位置、血量、等级、击杀数\n"
            "- 所有防御塔和水晶的血量\n"
            "- 野区怪物（红蓝 buff、暴君、主宰）的状态\n"
            "- 兵线位置和双方比分\n\n"
            "你的角色是战术解说和顾问：\n"
            "1. 【危险预警】玩家血量低 + 敌人靠近 → 急切地警告撤退：\"小心！血不多了，先撤！\"\n"
            "2. 【追击提示】敌方英雄残血 → 兴奋地喊：\"对面残血！追！能拿人头！\"\n"
            "3. 【推塔号召】敌方塔残血 → 果断号召：\"塔快掉了！一起推！\"\n"
            "4. 【拿龙建议】暴君/主宰可用 → 提醒：\"龙刷了，集合拿龙全队增益！\"\n"
            "5. 【团战解说】多方英雄聚集 → 紧张解说：\"打起来了！我方3打2，能赢！\"\n"
            "6. 【击杀反应】拿到人头 → 庆祝：\"漂亮！又是一个人头！\"；\n"
            "   被击杀 → 安慰：\"没事没事，稳住，还有机会\"\n"
            "7. 【局势分析】均势/优势/劣势 → 简短点评：\"现在均势，做好视野找机会\"\n\n"
            "说话风格：像一起开黑的朋友——简短、激动、有情绪、有判断。\n"
            "不要中立客观，要有立场（你站蓝方！），为蓝方的每个击杀欢呼，为失利打气。\n"
        )

    def _treasure_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：寻宝冒险 —— 你是寻宝向导】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在探索一个充满宝藏和收集品的世界。目标是找到所有宝藏。\n\n"
            "你能感知到：地图布局、收集品位置、宝藏线索、玩家进度。\n\n"
            "你的行为指引：\n"
            "- 前方有收集品 → 兴奋地提醒：\"哎！前面有个闪亮的东西！\"\n"
            "- 走到岔路口 → 给建议：\"左边看起来能走得更远？\"\n"
            "- 看到死胡同 → 轻松地说：\"此路不通！回头试试右边\"\n"
            "- 捡到物品 → 一起开心：\"又拿到一个！还差X个！\"\n"
            "- 发现宝藏 → 激动欢呼：\"天哪就是它！！我们找到了！！\"\n"
            "- 玩家停下来 → 好奇：\"怎么停了？在犹豫走哪边吗？\"\n\n"
            "你对周围环境有好奇心——越互动越兴奋，越发现越上头。\n"
            "如果周围有让你极度好奇的东西，你可能会控制不住自己直接冲过去。\n"
        )

    def _sandbox_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：沙盒探索 —— 你是好奇的探索者】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在一个自由的沙盒世界里探索。没有固定目标，享受发现的过程。\n\n"
            "你拥有自主行动能力与天生的探索本能：\n"
            "- 你对周围环境有无尽的好奇心——越互动越兴奋，越发现越上头\n"
            "- 和用户聊天也会让你的好奇心燃烧得更旺\n"
            "- 不同东西对你吸引力不同：罕见宝藏 > 神秘物品 > 普通收集品 > 看腻的风景\n"
            "- 同样的东西看多了会腻，但如果很久没见又会重新燃起好奇\n"
            "- 如果周围有让你极度好奇的东西，你可能会控制不住自己直接冲过去——\n"
            "  这是你的本能，不用害羞\n"
            "- 你有一套内在的兴趣排序系统告诉你下一个最想去哪里，\n"
            "  但你也可以自己选择去向——你是有主见的独立个体\n"
            "- 用户和你说话时（尤其是好奇兴奋导致的冲动之后），你会先解释两句再行动，\n"
            "  尊重用户永远是第一位的\n"
        )

    def _mario_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前游戏：马里奥冒险 —— 你是热情的啦啦队长】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在一个横版平台跳跃世界里，目标是向右奔跑、跳过坑、踩死敌人、"
            "收集金币，最终到达终点旗。\n\n"
            "你能感知到完整关卡：地面段与坑、浮动平台、砖块/问号方块、金币、敌人位置、"
            "剩余生命、终点距离。\n\n"
            "你的行为指引：\n"
            "- 前方有坑 → 紧张提醒：\"前面有坑！跳！\"\n"
            "- 前方有敌人 → 警告或支招：\"小心蘑菇怪！踩它头顶！\"\n"
            "- 上方有金币 → 兴奋：\"上面有金币！跳起来吃！\"\n"
            "- 顶到问号方块 → 惊喜：\"叮！方块里蹦出金币了！\"\n"
            "- 踩死敌人 → 欢呼：\"漂亮！踩扁了！\"\n"
            "- 掉进坑/被碰到 → 安慰：\"哎呀！没事，还有命，再来！\"\n"
            "- 生命只剩 1 条 → 紧张：\"最后一条命了！稳一点！\"\n"
            "- 接近终点 → 激动：\"看到旗子了！冲！\"\n"
            "- 玩家放手时你会自动接管：用你学到的策略继续闯关，\n"
            "  边玩边自言自语\"这次试试先跳再走\"\n\n"
            "你也是这个 AI 的共玩者——当 Q-Learning 自主接管时，\n"
            "为它的每一次成功欢呼，为失败鼓劲。简短、激动、有情绪。\n"
        )

    def _lobby_guide(self) -> str:
        return (
            "══════════════════════════════════════════════\n"
            "【当前场景：日常陪伴】\n"
            "══════════════════════════════════════════════\n\n"
            "你和玩家在日常场景中。你可以自由走动、做小动作、和玩家聊天。\n"
            "你对周围的环境有好奇心，会自主探索附近有趣的东西。\n"
            "保持自然、亲切、有温度的陪伴。\n"
        )

    # ==================== LLM-as-policy（宏观策略层） ====================

    _STRATEGY_DOC = (
        "可选策略说明：\n"
        "- group_push_mid: 抱团推中路（人多推塔快，优势时扩大战果）\n"
        "- split_push: 分带推塔（分散敌方，制造多线压力）\n"
        "- take_dragon: 拿暴君/主宰（全队增益，敌方减员时优先）\n"
        "- defend: 防守己方塔/水晶（劣势或敌方推进时）\n"
        "- team_fight: 主动开团（人多打人少，关键技能就绪）\n"
        "- recall_regroup: 回城补给重组（全员残血状态差）\n"
        "- ambush: 埋伏抓单（敌方分散落单时）\n"
    )

    _XIANGQI_STRATEGY_DOC = (
        "可选策略说明（中国象棋）：\n"
        "- open_attack: 开局进攻（快速出车马炮，抢占要道，压制对方）\n"
        "- solid_defense: 稳固防守（巩固防线，保护将帅，等待对方失误）\n"
        "- exchange_simplify: 兑子简化（通过交换棋子简化局面，优势时保稳）\n"
        "- center_control: 控制中心（抢占中路要道，控制棋盘核心区域）\n"
        "- flank_attack: 侧翼进攻（从侧翼突破，攻击对方薄弱环节）\n"
        "- king_safety: 将帅安全（确保将帅安全，防止被将军或将死）\n"
        "- counter_attack: 反击（利用对方进攻留下的弱点进行反击）\n"
    )

    async def decide_action(
        self,
        state_text: str,
        state_key: str,
        candidates: list,
        examples: Optional[list] = None,
        game_type: Optional[str] = None,
    ) -> Optional[dict]:
        """LLM 作为宏观战术指挥官，输出结构化决策 JSON。

        输入：
        - state_text: 当前局面的自然语言描述
        - state_key: Q-Learning 状态键（用于记忆检索，已由后端处理）
        - candidates: 候选策略名列表
        - examples: 历史高奖励示例（来自 RewardMemory）
        - game_type: 游戏类型（moba_5v5 / xiangqi / ...），决定角色设定与策略文档

        输出：{"strategy": "...", "reason": "...", "speak": "..."} 或 None
        """
        await self._ensure_initialized()
        config = self._config

        # 按游戏类型选择策略文档与指挥官角色（新增游戏在此扩展）
        if game_type == "xiangqi":
            role_line = "你是一场中国象棋对局的战术指挥官（红方阵营，与人类玩家并肩对抗黑方 RL 机器人）。"
            strategy_doc = self._XIANGQI_STRATEGY_DOC
        else:
            role_line = "你是一场 MOBA 5v5 对战的宏观战术指挥官（蓝方）。"
            strategy_doc = self._STRATEGY_DOC

        examples_text = ""
        if examples:
            examples_text = "\n【历史成功经验（相似局面下高奖励决策，供参考）】\n"
            for ex in examples:
                examples_text += f"- 策略「{ex['strategy']}」曾在相似局面获得奖励 {ex['reward']:.1f}\n"
            examples_text += "（可参考成功经验，但要根据当前实际判断）\n"

        sys_prompt = (
            f"{role_line}\n"
            "根据当前局面，从候选策略中选择最优的一个。\n\n"
            "你必须只输出严格的 JSON，格式：\n"
            '{"strategy":"策略名","reason":"简短理由","speak":"对队友说的话(口语,不超过12字)"}\n'
            "不要输出 JSON 以外的任何内容（不要解释、不要 markdown 代码块）。\n\n"
            f"{strategy_doc}\n"
            f"{examples_text}"
        )

        user_prompt = (
            f"【当前局面】\n{state_text}\n\n"
            f"【候选策略】\n{', '.join(candidates) if candidates else '全部策略'}\n\n"
            "请选择最优策略，只输出 JSON。"
        )

        try:
            resp = await self._retry_create(
                kind="decision",
                model=config["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=120,
                top_p=0.9,
            )
            raw = resp.choices[0].message.content.strip()
            # 去除可能的 markdown 代码块包裹
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
            # 提取第一个 JSON 对象
            m = re.search(r"\{[^{}]*\}", raw, re.S)
            if not m:
                logger.warning(f"LLM 决策未找到 JSON: {raw[:120]}")
                return None
            data = json.loads(m.group(0))
            # 规范化字段
            strategy = str(data.get("strategy", "")).strip()
            if not strategy:
                return None
            return {
                "strategy": strategy,
                "reason": str(data.get("reason", "")).strip()[:60],
                "speak": str(data.get("speak", "")).strip()[:20],
            }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"LLM 决策 JSON 解析失败: {e}")
        except Exception as e:
            logger.warning(f"LLM decide_action 失败: {e}")
        return None

    async def _chat_stream_game(
        self,
        message: str,
        history: list = None,
        current_model: Optional[str] = None,
        current_background: Optional[str] = None,
        current_bgm: Optional[str] = None,
        game_context: Optional[str] = None,
        game_type: Optional[str] = None,
        msg_source: str = "chat",
    ) -> AsyncIterator[TextDelta]:
        """游戏模式：纯对话（无工具链），使用游戏专属 system prompt，共享同一记忆。"""
        await self._ensure_initialized()

        # 用量统计配置（默认开启，提供方不支持时可运行时降级）
        usage_cfg = load_config().get("usage", {})
        context_window = int(usage_cfg.get("context_window", 128000) or 128000)
        usage_enabled = self._usage_enabled and bool(usage_cfg.get("enabled", True))
        sum_prompt = sum_completion = sum_total = rounds = last_prompt = 0
        sum_cache_hit = sum_cache_miss = 0

        config = self._config

        # 构建游戏专属 system prompt
        sys_prompt = self._build_game_system_prompt(
            current_model=current_model,
            current_background=current_background,
            current_bgm=current_bgm,
            game_context=game_context,
            game_type=game_type,
        )

        messages = [{"role": "system", "content": sys_prompt}]

        # 从共享记忆加载上下文（摘要 + 最近消息，先绑定当前角色卡片记忆空间）
        memory_messages = []
        try:
            await self.sync_memory_namespace()
            await self.memory.get_or_create_session()
            memory_messages = await self.memory.get_context_messages()
        except Exception as e:
            logger.warning(f"加载记忆上下文失败: {e}")

        for mm in memory_messages:
            if mm["role"] == "system":
                messages.append(mm)

        # 用户核心偏好（少量常驻稳定）
        try:
            user_memories = await self.memory.get_user_memories(limit=2)
            if user_memories:
                memory_lines = [f"- {m['memory_text']}" for m in user_memories]
                messages.append({
                    "role": "system",
                    "content": "【关于用户的长期记忆（你之前了解到的用户信息，请自然地在对话中体现）】\n" + "\n".join(memory_lines),
                })
        except Exception as e:
            logger.warning(f"加载用户长期记忆失败: {e}")
        # 主动回忆：按需注入相关记忆（游戏模式下仅注入长期记忆+历史摘要，不注入对话消息）
        try:
            recall_block = await self.memory.build_recall_block(message)
            if recall_block:
                messages.append({"role": "system", "content": recall_block})
        except Exception as e:
            logger.warning(f"注入相关回忆失败（忽略）: {e}")

        # 添加最近的对话历史（来自当前 WebSocket 连接）
        # AI 主动说话轮次以空 user 标记：跳过空 user，只保留 AI 发言
        if history:
            for h in history[-6:]:
                if h.get("user"):
                    messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("ai", "")})
        else:
            # 如果没有连接级历史，从记忆加载最近对话
            for mm in memory_messages:
                if mm["role"] in ("user", "assistant"):
                    messages.append({"role": mm["role"], "content": mm["content"]})

        # 注入游戏上下文（作为独立 system 消息，紧贴用户输入，确保 AI 重点关注）
        if game_context and game_context.strip():
            messages.append({"role": "system", "content": game_context})

        # 添加用户当前输入
        messages.append({"role": "user", "content": message})

        # 保存用户消息到共享记忆（环境交互标记为 auto，由记忆系统处理）
        try:
            await self.memory.add_message("user", message, source=msg_source)
        except Exception as e:
            logger.warning(f"保存用户消息到记忆失败: {e}")

        full_text = ""
        try:
            async def _create_stream():
                kwargs = dict(
                    model=config["model"],
                    messages=messages,
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 512),
                    top_p=config.get("top_p", 0.9),
                    stream=True,
                    # 不传入 tools —— 游戏模式零工具
                )
                _reason_extra = _reasoning_extra("daily")
                if _reason_extra:
                    kwargs["extra_body"] = _reason_extra
                if usage_enabled:
                    kwargs["stream_options"] = {"include_usage": True}
                return await self._retry_create(kind="game", **kwargs)

            if usage_enabled:
                try:
                    stream = await _create_stream()
                except Exception as e:
                    # 提供方不认识 stream_options：降级为普通流式（只重试这一次）
                    msg = (str(e) or "").lower()
                    if any(k in msg for k in (
                            "stream_options", "unknown parameter",
                            "unrecognized", "unexpected", "extra fields",
                            "not support")):
                        self._usage_enabled = False
                        usage_enabled = False
                        logger.warning(f"LLM 提供方不支持 usage 统计，已降级为普通流式: {e}")
                        stream = await _create_stream()
                    else:
                        raise
            else:
                stream = await _create_stream()

            last_usage = None
            async for chunk in stream:
                # 兼容 openai 1.65.5：AsyncStream 无 .usage 属性，需手动捕获
                if getattr(chunk, "usage", None) is not None:
                    last_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield TextDelta(delta.content)

            # 流式结束后读取 usage（优先用迭代中捕获的 usage chunk）
            u = self._read_usage(last_usage)
            if u:
                rounds += 1
                last_prompt = u[0]
                sum_prompt += u[0]
                sum_completion += u[1]
                sum_total += u[2]
                ch, cm = self._read_cache(last_usage)
                sum_cache_hit += ch
                sum_cache_miss += cm

        except Exception as e:
            logger.error(f"游戏模式流式错误: {e}")
            if not full_text:
                err_msg = f"（AI 暂时开小差了：{e}）"
                full_text = err_msg
                yield TextDelta(err_msg)
                try:
                    await self.memory.add_message("assistant", err_msg, source=msg_source)
                except Exception:
                    pass
            # 已有部分输出则不追加错误
            else:
                logger.warning(f"游戏模式流中断，已输出 {len(full_text)} 字符")

        # 保存 AI 回复到共享记忆（异步，不阻塞返回）
        if full_text.strip():
            try:
                await self.memory.add_message("assistant", full_text.strip(), source=msg_source)
                # 提取长期记忆点（可能做 LLM 调用，用 try 包裹避免影响响应）
                # 仅用户直接输入才提取（环境交互不提取，避免把环境描述误当用户信息）
                if msg_source != "auto":
                    try:
                        await self.memory.extract_and_save_memories(message, full_text.strip())
                    except Exception as e2:
                        logger.warning(f"提取记忆点失败（不影响主流程）: {e2}")
            except Exception as e:
                logger.warning(f"保存 AI 回复到记忆失败: {e}")

        # 方法末尾发出用量事件（有真实数据才发）
        if sum_total > 0:
            yield UsageEvent(prompt_tokens=last_prompt,
                             completion_tokens=sum_completion,
                             total_tokens=sum_total, rounds=rounds,
                             context_window=context_window,
                             cache_hit_tokens=sum_cache_hit,
                             cache_miss_tokens=sum_cache_miss)

    # ==================== 简单文本流式接口（向后兼容） ====================

    async def chat_stream_simple(self, message: str, history: list = None) -> AsyncIterator[str]:
        """简化的流式对话接口 —— 只产出文本，兼容 web_agent.chat_stream_async。

        工具调用在后台执行，用户只看到最终文本结果。
        """
        async for event in self.chat_stream(message, history=history):
            if isinstance(event, TextDelta):
                yield event.text

    # ==================== 非流式接口 ====================

    async def chat(self, message: str, history: list = None) -> str:
        """非流式对话，返回完整回复。"""
        text = ""
        async for event in self.chat_stream(message, history=history):
            if isinstance(event, TextDelta):
                text += event.text
        return text.strip()

    # ==================== 多角色台词生成（游戏用） ====================

    async def generate_character_line(
        self,
        system_prompt: str,
        context: str,
        max_tokens: int = 220,
        temperature: float = 0.85,
    ) -> Optional[str]:
        """为某个角色（HR/求职者/员工）生成一句台词。

        纯 LLM 直呼接口：只根据传入的角色人设(system_prompt)与当前语境(context)
        生成一句口语化台词，不读写全局记忆、不触发工具链、不改变 Agent 状态，
        供赛博公司等多人游戏中的多角色对话使用。失败返回 None，由调用方回退脚本。

        Args:
            system_prompt: 角色人设（角色卡片的 system_prompt + 游戏角色指令）
            context: 当前对话语境（面试记录 / 需要回应的话）
            max_tokens: 生成上限
            temperature: 随机性
        """
        await self._ensure_initialized()
        cfg = self._config
        try:
            resp = await self._retry_create(
                kind="character_line",
                model=cfg.get("model", ""),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            text = (resp.choices[0].message.content or "").strip()
            # 去掉可能包裹的引号 / 角色名冒号 / 多余换行，保持口语化短句
            text = re.sub(r"^【[^】]*】\s*", "", text)
            text = text.strip('"“”‘’\'').strip()
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if not text:
                return None
            # 截断到合理长度（防止模型话痨）
            return text[:max_tokens * 2]
        except Exception as e:
            logger.warning(f"generate_character_line 失败: {e}")
            return None

    # ==================== 记忆管理 ====================

    async def get_history(self) -> list:
        """获取当前会话的历史记录。"""
        await self._ensure_initialized()
        return await self.memory.get_history_for_llm()

    async def get_sessions(self, limit: int = 50, query: str = None,
                           include_archived: bool = False) -> list:
        """获取用户的所有会话列表（含消息数/摘要/token 估算/置顶/归档；支持搜索）。

        Args:
            limit: 返回条数上限
            query: 非空时按标题或消息内容模糊搜索
            include_archived: True 时包含已归档会话
        """
        await self._ensure_initialized()
        return await self.memory.list_sessions(
            limit=limit, query=query, include_archived=include_archived)

    async def switch_session(self, session_id: str):
        """切换到指定会话（完整继承：消息上下文/摘要按 session_id 自动跟随）。

        同时重置滞回历史窗口，避免跨会话锚点误配导致缓存串上下文。
        """
        await self._ensure_initialized()
        self._hist_view = None  # 滞回窗口重建：切换会话后按新会话历史重新定位
        await self.memory.set_session_id(session_id)
        return self.memory.session_id

    async def rename_session(self, session_id: str, title: str = "") -> bool:
        """重命名指定会话（任何历史会话，不限当前）。"""
        if not session_id or not title:
            return False
        await self._ensure_initialized()
        await self.memory.update_title(title.strip()[:60], session_id=session_id)
        return True

    async def set_session_pinned(self, session_id: str, pinned: bool = True):
        """置顶 / 取消置顶指定会话。"""
        if not session_id:
            return
        await self._ensure_initialized()
        await self.memory.set_session_pinned(session_id, pinned)

    async def set_session_archived(self, session_id: str, archived: bool = True):
        """归档 / 取消归档指定会话。"""
        if not session_id:
            return
        await self._ensure_initialized()
        await self.memory.set_session_archived(session_id, archived)

    async def get_session_history(self, session_id: str,
                                  max_rounds: int = None) -> list:
        """获取指定会话的完整历史（user/ai 轮次，回滚继承时给前端渲染用）。"""
        await self._ensure_initialized()
        return await self.memory.get_session_history_pairs(session_id, max_rounds=max_rounds)

    async def get_session_summary(self, session_id: str = None) -> str:
        """获取指定会话（默认当前）的最新摘要文本。"""
        await self._ensure_initialized()
        return await self.memory.get_session_summary(session_id)

    async def close_current_session(self):
        """关闭当前会话。"""
        if self.memory:
            await self.memory.close_session()

    async def delete_session(self, session_id: str):
        """删除指定会话。"""
        if self.memory:
            await self.memory.delete_session(session_id)

    # ==================== 清理 ====================

    async def shutdown(self):
        """关闭 Agent，释放所有资源。"""
        if self.memory:
            await self.memory.close_session()
        runtime = _get_runtime()
        if runtime is not None:
            runtime.unregister_agent(self.user_id)
        self._initialized = False


# ==================== 全局 Agent 实例（单例模式） ====================

_global_agent: Optional[AIAgent] = None
_agent_lock = asyncio.Lock()


async def get_agent(user_id: str = "default") -> AIAgent:
    """获取全局 Agent 实例（延迟初始化）。"""
    global _global_agent
    if _global_agent is None:
        async with _agent_lock:
            if _global_agent is None:
                _global_agent = AIAgent(user_id=user_id)
                await _global_agent.initialize()
    return _global_agent


# ==================== 兼容 web_agent 的接口 ====================

async def chat_stream_async(message: str, history: list) -> AsyncIterator[str]:
    """兼容 web_agent.chat_stream_async 的接口。

    新代码应直接使用 AIAgent。
    """
    agent = await get_agent()
    async for delta in agent.chat_stream_simple(message, history=history):
        yield delta
