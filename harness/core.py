"""Harness 运行时核心 —— 「大白」的稳定智能体扩展层。

职责：
1. 统一收集技能（Skill）与插件（Plugin）的工具定义，交给 agent 的 function calling；
2. 统一收集技能/插件注入 system prompt 的提示词片段；
3. 提供稳定工具路由：builtin → 技能 → 插件（未命中即放行给下一层）；
4. 提供健康状态（uptime、工具统计、最近事件）与热重载；
5. 提供通用韧性原语 retry_async（对瞬时错误退避重试）。

技能/插件出现任何加载错误都不会拖垮「大白」：坏条目被标记、
正常条目照常工作，管理页 /api/harness/* 可查看与修复。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .skills import SkillRegistry
from .plugins import PluginManager
from .state import StateStore
from .runtime import AgentRuntime
from .tasks import TaskSystem

logger = logging.getLogger("harness.core")

# 韧性重试的默认参数
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF = 1.0

# 可重试的瞬时错误特征（小写子串匹配；鉴权类错误绝不重试）
_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "refused", "reset", "closed",
    "unavailable", "server error", "internal server", "bad gateway",
    "rate limit", "too many requests", "temporarily", "try again",
    "429", "500", "502", "503", "504",
)
_NON_TRANSIENT_HINTS = (
    "authentication", "api key", "invalid_api_key", "unauthorized",
    "forbidden", "not found", "401", "403", "404", "does not support",
    "unsupported", "doesn't support",
)


def is_transient_error(e: BaseException) -> bool:
    """判断异常是否属于「瞬时错误」（网络抖动/限流/服务端 5xx）。"""
    if isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    msg = (str(e) or "").lower()
    if any(k in msg for k in _NON_TRANSIENT_HINTS):
        return False
    return any(k in msg for k in _TRANSIENT_HINTS)


async def retry_async(coro_factory: Callable[[], Awaitable[Any]],
                      attempts: int = DEFAULT_MAX_ATTEMPTS,
                      backoff: float = DEFAULT_BACKOFF,
                      on_retry: Optional[Callable[[int, BaseException], None]] = None) -> Any:
    """对可等待的调用做退避重试（仅瞬时错误）。

    coro_factory 必须返回新的 coroutine（每次重试都重新创建请求，
    避免复用已被消费过的 stream/response）。

    用法：
        result = await retry_async(lambda: client.chat.completions.create(...))
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 —— 需要捕捉一切以便分类
            last_exc = e
            if attempt >= attempts or not is_transient_error(e):
                raise
            delay = backoff * (2 ** (attempt - 1))
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception:
                    pass
            logger.warning("瞬时错误，%.1fs 后第 %d/%d 次重试: %s",
                           delay, attempt + 1, attempts, e)
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover


# 内置 skill_help 工具：渐进式披露的「按需拉取说明书」入口。
# 渐进披露开启时，on_demand 技能只注入一句话摘要，模型需要时调用本工具
# 获取该技能的完整使用说明（SKILL.md 或 prompt + 工具参数）。
SKILL_HELP_TOOL = {
    "type": "function",
    "function": {
        "name": "skill_help",
        "description": ("获取某个技能（Skill）的完整使用说明书。当用户需求涉及某技能、"
                         "或你想调用它的工具但不确定用法、参数或边界时，先调用本工具读取说明书，再执行对应工具。"),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "技能名，如 filesys、weather、image_gen（以 system prompt 中的技能摘要为准）。",
                }
            },
            "required": ["skill_name"]
        }
    }
}

# 渐进披露模式下注入 system prompt 的引导语
PROGRESSIVE_HINT = (
    "【渐进式技能说明】部分技能采用按需加载：摘要已在上面列出。当你想使用某技能的工具时，"
    "先调用 skill_help(\"技能名\") 读取它的完整使用说明，再执行对应工具；不要猜测参数。"
)


class Harness:
    """稳定运行时的门面：技能 + 插件 + 健康状态 + 热重载。"""

    def __init__(self, base_dir: Path):
        import threading
        self.base_dir = Path(base_dir)
        self.state = StateStore(self.base_dir / "harness_state.json")
        self.runtime = AgentRuntime(self)   # Agent 监督运行时（LLM/工具/对话轮）
        self.tasks = TaskSystem(self)       # 长任务/批量任务/DAG 流程系统
        self.skills = SkillRegistry(self, self.base_dir,
                                    self._opt_dir("skills_dir", "skills"))
        self.plugins = PluginManager(self, self.base_dir,
                                     self._opt_dir("plugins_dir", "plugins"))
        self.started_at = time.time()
        self._events: deque = deque(maxlen=60)   # 最近事件（健康环形缓冲）
        self._tool_index: dict = {}              # tool_name -> (kind, owner_name)
        self._index_lock = threading.Lock()
        self._server_app = None                  # server.py startup 注入的 FastAPI app
        self._loaded = False

    # ---------- 配置 ----------

    def _opt_dir(self, key: str, default: str) -> Path:
        """从 settings.json 的 harness 段读取目录配置（缺省用默认值）。"""
        try:
            path = self.base_dir / "settings.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                h = (cfg or {}).get("harness", {}) or {}
                val = str(h.get(key, "") or "").strip()
                if val:
                    p = Path(val)
                    if not p.is_absolute():
                        p = self.base_dir / p
                    return p
        except Exception:
            pass
        return self.base_dir / default

    # ---------- 加载 / 重载 ----------

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.skills.ensure_loaded()
        self.plugins.ensure_loaded()
        self._rebuild_index()
        self._offer_server_app()
        self._loaded = True
        self.record("harness", f"harness 就绪：技能 {len(self.skills.list_info())} 个，"
                               f"插件 {len(self.plugins.list_info())} 个")

    # ---------- 服务端生命周期（REST 挂载等） ----------

    def on_server_start(self, app) -> None:
        """server.py startup 事件调用：把 app 注入 harness，并通知各插件挂载资源。"""
        self.ensure_loaded()
        self._server_app = app
        self._offer_server_app()
        try:
            self.tasks.ensure_started()
        except Exception as e:
            logger.warning("任务系统启动失败: %s", e)

    def on_server_stop(self) -> None:
        """server.py shutdown 事件调用：停任务系统、通知插件清理已挂载的 REST 路由等。"""
        try:
            self.tasks.stop()
        except Exception as e:
            logger.warning("任务系统停止钩子失败: %s", e)
        for entry in self.plugins._loaded.values():
            inst = entry.get("instance")
            if inst is None:
                continue
            try:
                inst.on_server_stop()
            except Exception as e:
                logger.warning("插件 on_server_stop 钩子失败: %s", e)
        self._server_app = None

    def _offer_server_app(self) -> None:
        """把已注入的 app 提供给新加载的插件实例（幂等：每个实例只 offer 一次）。"""
        if self._server_app is None:
            return
        for entry in self.plugins._loaded.values():
            if entry.get("broken") or not self.plugins.is_enabled(entry["info"]["name"]):
                continue
            inst = entry.get("instance")
            if inst is None or entry.get("server_started"):
                continue
            try:
                inst.on_server_start(self._server_app)
            except Exception as e:
                logger.warning("插件 %s on_server_start 钩子失败: %s",
                               entry["info"].get("name"), e)
            entry["server_started"] = True

    def reload_all(self) -> dict:
        """热重载全部技能与插件（管理页/API 调用）。返回概况。"""
        n_skill = len(self.skills.discover())
        n_plugin = len(self.plugins.discover())
        for name in list(self.skills._loaded.keys()):
            self.skills.reload(name)
        for m in self.plugins.discover():
            self.plugins.reload(m["name"])
        self._rebuild_index()
        self._offer_server_app()
        try:
            self.runtime.reload_cfg()
        except Exception:
            pass
        try:
            self.tasks.reload_cfg()
        except Exception:
            pass
        self._loaded = True
        self.record("harness", f"已热重载：技能 {n_skill} 个、插件 {n_plugin} 个")
        return {"skills": n_skill, "plugins": n_plugin}

    # ---------- 工具索引 ----------

    def _rebuild_index(self) -> None:
        index: dict = {}
        for name, entry in self.skills._loaded.items():
            if entry.get("broken") or not self.skills.is_enabled(name):
                continue
            for tname in entry.get("tools", {}):
                index[tname] = ("skill", name)
        for name, entry in self.plugins._loaded.items():
            if entry.get("broken") or not self.plugins.is_enabled(name):
                continue
            for tname in entry.get("tools", {}):
                index[tname] = ("plugin", name)
        with self._index_lock:
            self._tool_index = index

    def tool_owner(self, tool_name: str) -> Optional[tuple]:
        """返回工具的属主 (kind, owner_name)，未命中返回 None。"""
        self.ensure_loaded()
        with self._index_lock:
            return self._tool_index.get(tool_name)

    # ---------- 供 agent 使用的导出 ----------

    def _progressive_enabled(self) -> bool:
        """渐进式披露全局开关：settings.json 的 harness.progressive_disclosure。默认关闭。"""
        try:
            path = self.base_dir / "settings.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                h = (cfg or {}).get("harness", {}) or {}
                return bool(h.get("progressive_disclosure", False))
        except Exception:
            pass
        return False

    def collect_tool_specs(self) -> list:
        """返回全部启用技能/插件的 OpenAI 工具定义（agent 合并进 _all_tools）。

        始终附带内置 skill_help 工具（唯一例外：已被某技能/插件占用同名工具时）。"""
        self.ensure_loaded()
        tools = self.skills.tool_specs() + self.plugins.tool_specs()
        names = {(t.get("function") or {}).get("name") for t in tools}
        if "skill_help" not in names:
            tools.append(SKILL_HELP_TOOL)
        return tools

    def collect_prompt_extras(self) -> str:
        """返回全部启用技能/插件注入 system prompt 的片段。

        渐进披露开启时：on_demand 技能只注入一句话摘要（skill_help 按需拉详情），
        full 技能仍注入完整 prompt；关闭时保持原有全量行为。"""
        self.ensure_loaded()
        parts = []
        if self._progressive_enabled():
            sbrief = self.skills.brief_extras()
            if sbrief:
                parts.append(sbrief)
            parts.append(PROGRESSIVE_HINT)
        else:
            sp = self.skills.prompt_extras()
            if sp:
                parts.append(sp)
        pp = self.plugins.prompt_extras()
        if pp:
            parts.append(pp)
        return "\n\n".join(parts)

    # ---------- 工具执行（稳定路由） ----------

    async def execute_tool(self, tool_name: str, arguments: dict) -> tuple:
        """执行 harness 工具（内置 skill_help 优先，其次技能/插件）。

        返回 (result_text, source)：
        - 命中内置/技能/插件工具 → (结果文本, 'harness'|'skill'|'plugin')
        - 未命中任何 harness 工具 → (None, '')（调用方继续走本地工具路由）
        """
        if tool_name == "skill_help":
            skill_name = str(arguments.get("skill_name") or arguments.get("name") or "").strip()
            text = self.skills.help_text(skill_name)
            if text is None:
                return f"技能 {skill_name} 不存在或未加载（可用技能见 system prompt 的技能摘要）", "harness"
            return text, "harness"
        owner = self.tool_owner(tool_name)
        if owner is None:
            return None, ""
        kind, owner_name = owner
        try:
            if kind == "skill":
                return await self.skills.execute_tool(owner_name, tool_name, arguments)
            return await self.plugins.execute_tool(owner_name, tool_name, arguments)
        except Exception as e:
            self.record("error", f"harness 工具 {tool_name} 执行异常: {e}")
            return f"harness 工具 {tool_name} 执行异常: {e}", kind

    # ---------- 健康状态 ----------

    def record(self, kind: str, message: str) -> None:
        """记录一条健康事件（环形缓冲，最多 60 条）。"""
        try:
            self._events.append({"at": time.time(), "kind": str(kind), "message": str(message)[:300]})
        except Exception:
            pass

    def status(self) -> dict:
        """运行时健康状态（供 /api/harness/status 与管理页）。"""
        self.ensure_loaded()
        skill_infos = self.skills.list_info()
        plugin_infos = self.plugins.list_info()
        tool_count = {"skill": 0, "plugin": 0}
        with self._index_lock:
            for _, (kind, _owner) in self._tool_index.items():
                tool_count[kind] = tool_count.get(kind, 0) + 1
        broken = [s["name"] for s in skill_infos if s.get("broken")]             + [p["name"] for p in plugin_infos if p.get("broken")]
        try:
            runtime = self.runtime.snapshot()
        except Exception as e:
            runtime = {"error": f"运行时快照失败: {e}"}
        try:
            tasks = self.tasks.summary()
        except Exception as e:
            tasks = {"error": f"任务系统快照失败: {e}"}
        return {
            "ok": not broken,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "skills": skill_infos,
            "plugins": plugin_infos,
            "tool_count": tool_count,
            "broken": broken,
            "state_file": str(self.state.path),
            "skills_dir": str(self.skills.directory),
            "plugins_dir": str(self.plugins.directory),
            "recent_events": list(self._events)[-20:],
            "runtime": runtime,
            "tasks": tasks,
        }
