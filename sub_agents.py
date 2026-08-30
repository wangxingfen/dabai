# -*- coding: utf-8 -*-
"""通用子智能体（General Sub-Agents）—— 主智能体把任意复杂任务下发给独立子智能体，
与 DSH 的 subagent 分工模型对应，是对 media_workers 的一般化：

- 主智能体调用 sub_agent_spawn(task=...) → 这里登记一个子智能体（worker）并立即
  返回任务中心条目；worker 在后台跑自己的「LLM 思考 + 工具执行」循环，不阻塞主对话；
- 可同时大量分派：每个 worker 任务不同、互不影响；全局并发有上限（默认 8），
  超出并发自动排队执行；
- worker 的进度/日志/结果实时写入任务中心（channel=sub），完成后通过 report 回调
  反馈给主智能体（由 server 转成「子智能体汇报」，主智能体得知后向用户转述）；
- 工具：与主智能体共用 harness 技能/插件路由 + 本地工具兜底；
  子智能体工具列表里排除 sub_agent_* 自身，防止无限递归下发。

与媒体子智能体的区别：media_workers 靠前端播放事件驱动完成（等待类任务），
这里是自驱动的通用执行类任务（LLM+工具直到得出结果）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("sub_agents")

# 状态机
ST_QUEUED = "queued"
ST_RUNNING = "running"
ST_DONE = "done"
ST_ERROR = "error"
ST_CANCELLED = "cancelled"

_STATUS_LABEL = {
    ST_QUEUED: "排队中",
    ST_RUNNING: "执行中",
    ST_DONE: "已完成",
    ST_ERROR: "出错",
    ST_CANCELLED: "已取消",
}

MAX_CONCURRENT = 4          # 全局并发上限（超出排队），可经 settings.json -> agent.sub_max_concurrent 覆盖
MAX_ROUNDS = 0             # 单 worker 最大「思考+工具」轮次（0=不限制，与主智能体一致），可经 settings.json -> agent.sub_max_tool_rounds 覆盖
MAX_RUNTIME = 2 * 60 * 60       # 单 worker 最长运行时间（秒），防悬挂（默认 2 小时）
SUB_MAX_EST_TOKENS = 600000  # 单 worker 预估 token 预算（防烧钱），可经 settings.json -> agent.sub_max_est_tokens 覆盖
_MAX_WORKERS = 120          # 注册表上限


def _max_rounds() -> int:
    """读取子智能体最大轮次（settings.json -> agent.sub_max_tool_rounds）。

    0 / 负数表示不限制（与主智能体 max_tool_rounds 语义一致），默认 16。
    """
    try:
        from agent import load_config
        v = int(load_config().get("agent", {}).get("sub_max_tool_rounds", MAX_ROUNDS) or MAX_ROUNDS)
        return v if v > 0 else 0
    except Exception:
        return MAX_ROUNDS


def _max_concurrent() -> int:
    """全局并发上限（settings.json -> agent.sub_max_concurrent），默认 4。"""
    try:
        from agent import load_config
        v = int(load_config().get("agent", {}).get("sub_max_concurrent", MAX_CONCURRENT) or MAX_CONCURRENT)
        return v if v > 0 else MAX_CONCURRENT
    except Exception:
        return MAX_CONCURRENT


def _max_est_tokens() -> int:
    """单 worker 预估 token 预算（settings.json -> agent.sub_max_est_tokens），默认 12 万。"""
    try:
        from agent import load_config
        v = int(load_config().get("agent", {}).get("sub_max_est_tokens", SUB_MAX_EST_TOKENS) or SUB_MAX_EST_TOKENS)
        return v if v > 0 else SUB_MAX_EST_TOKENS
    except Exception:
        return SUB_MAX_EST_TOKENS

SUB_SYSTEM = (
    "你是一个被主智能体派出的子智能体，负责独立完成一项具体任务。\n"
    "规则：\n"
    "1. 需要实时信息、查文件/网页、计算或执行操作时，调用可用工具完成，不要凭空编造；\n"
    "2. 每一步只调用一个工具，拿到结果后再决定下一步；\n"
    "3. 任务完成或无法继续时，停止调用工具，直接用简洁的中文输出最终结果/结论；\n"
    "4. 输出要可直接使用：结论、关键数字或文件路径，不要客套话；\n"
    "5. 你有与主智能体相同的全部工具能力（含技能说明书），长任务可连续多轮调用工具，"
    "没有轮数限制；遇到反爬/失败自动换思路重试，不要轻易放弃。"
)


class SubAgent:
    """一个通用子智能体（一次独立的「LLM+工具」自主执行委托）。"""

    __slots__ = (
        "id", "kind", "title", "task", "status", "logs", "result", "error",
        "ws", "owner", "extra", "created_at", "updated_at",
        "task_ref", "loop_task", "watchdog", "cancelled",
    )

    def __init__(self, task: str, title: str, ws, owner=None, extra: Optional[dict] = None):
        self.id = "agent-" + uuid.uuid4().hex[:10]
        self.kind = "general"
        self.title = (title or task)[:120]
        self.task = task[:4000]
        self.status = ST_QUEUED
        self.logs: list[str] = []
        self.result = ""
        self.error = ""
        self.ws = ws
        self.owner = owner
        self.extra = extra or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        self.task_ref: Any = None        # 任务中心镜像 Task
        self.loop_task: Optional[asyncio.Task] = None
        self.watchdog: Optional[asyncio.Task] = None
        self.cancelled: bool = False

    @property
    def status_label(self) -> str:
        return _STATUS_LABEL.get(self.status, self.status)

    def snapshot(self, full: bool = False) -> dict:
        base = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "task": self.task,
            "status": self.status,
            "status_label": self.status_label,
            "created_at": int(self.created_at * 1000),
            "updated_at": int(self.updated_at * 1000),
        }
        if full:
            base.update({
                "logs": list(self.logs[-20:]),
                "result": self.result,
                "error": self.error,
                "extra": self.extra,
                "task_ref_id": self.task_ref.id if self.task_ref else "",
            })
        return base


class SubAgentManager:
    """通用子智能体注册表：spawn / 并发池 / 自主执行循环 / 汇报回调 + 任务中心镜像。"""

    def __init__(self) -> None:
        self._workers: dict[str, SubAgent] = {}
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(_max_concurrent())
        self._report_handler: Optional[Callable[[SubAgent, str], Awaitable[Any]]] = None
        self._event_handler: Optional[Callable[[str, SubAgent], Awaitable[Any]]] = None
        self._client = None
        self._client_model = ""
        self._tools: Optional[list] = None

    def set_report_handler(self, handler) -> None:
        """设置「worker 完成/出错 → 反馈主智能体」的汇报回调（server 注入）。"""
        self._report_handler = handler

    def set_event_handler(self, handler) -> None:
        """设置注册表事件回调（spawn/running/done/error/cancelled），供 UI 推送。"""
        self._event_handler = handler

    # ---------- 注册 / 查询 ----------

    async def spawn(self, ws, owner, task: str, title: str = "",
                    extra: Optional[dict] = None) -> SubAgent:
        """派出一个通用子智能体（立即返回，后台自主执行，可大量并行分派）。"""
        task = str(task or "").strip()
        if not task:
            raise ValueError("子智能体任务内容不能为空")
        worker = SubAgent(task, title, ws, owner, extra)
        async with self._lock:
            self._workers[worker.id] = worker
            if len(self._workers) > _MAX_WORKERS:
                asyncio.ensure_future(self._sweep())
        await self._mirror_create(worker)
        await self._emit("spawn", worker)
        # 总体看护：超时自动取消，防悬挂
        worker.watchdog = asyncio.ensure_future(self._watchdog(worker.id))
        worker.loop_task = asyncio.ensure_future(self._run(worker))
        logger.info("[SubAgent] 派出《%s》→ [%s]", worker.title, worker.id)
        return worker

    def get(self, worker_id: str) -> Optional[SubAgent]:
        return self._workers.get(worker_id)

    def active(self) -> list[SubAgent]:
        return [w for w in self._workers.values()
                if w.status in (ST_QUEUED, ST_RUNNING)]

    def list(self, limit: int = 80) -> list[dict]:
        ordered = sorted(self._workers.values(),
                         key=lambda w: w.created_at, reverse=True)
        return [w.snapshot(full=False) for w in ordered[:limit]]

    def active_text(self, limit: int = 8) -> str:
        """给主智能体看的「正在干活的子进程」摘要（注入每轮对话动态状态）。"""
        run = [w for w in self._workers.values() if w.status == ST_RUNNING]
        queued = [w for w in self._workers.values() if w.status == ST_QUEUED]
        if not run and not queued:
            return ""
        lines = [f"【子智能体运行中（{len(run)} 个执行，{len(queued)} 个排队）】"]
        for w in run[:limit]:
            lines.append(f"🧠 任务「{w.title}」[{w.id}] 执行中")
        for w in queued[:3]:
            lines.append(f"   ⏳（排队中）「{w.title}」[{w.id}]")
        return "\n".join(lines)

    # ---------- 异步并发池执行 ----------

    async def _run(self, worker: SubAgent) -> None:
        try:
            async with self._sem:            # 并发池：超出上限排队等待
                await self._set_status(worker, ST_RUNNING)
                result = await self._run_loop(worker)
                await self._finish(worker, ST_DONE, result=result)
        except asyncio.CancelledError:
            worker.cancelled = True
            await self._finish(worker, ST_CANCELLED, error="已被取消")
        except Exception as e:
            await self._finish(worker, ST_ERROR,
                               error=f"{e.__class__.__name__}: {e}")

    async def _run_loop(self, worker: SubAgent) -> str:
        client, model = self._get_client()
        tools = self._tool_defs()
        # 与主智能体对齐：注入 harness 技能/插件说明书，让子智能体知道全部能力怎么用
        sys_content = SUB_SYSTEM
        try:
            from agent import get_harness_prompt_extras
            extras = get_harness_prompt_extras()
            if extras:
                sys_content += "\n\n【技能说明书（与主智能体相同）】\n" + extras
        except Exception:
            pass
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": worker.task},
        ]
        rounds = 0
        max_rounds = _max_rounds()          # 0 = 不限制（与主智能体一致）
        max_est_tokens = _max_est_tokens()
        est_tokens_used = 0
        # 死循环防护：记录每轮工具调用指纹（工具名+参数），连续 3 轮完全相同即停止
        recent_fps: list = []
        while max_rounds <= 0 or rounds < max_rounds:
            rounds += 1
            await self._mirror_log(worker, f"第 {rounds} 轮思考…")
            # 上下文压缩：只把最近 60 条消息（保留 system）交给 LLM，防止工具历史无限膨胀拖慢每轮
            llm_messages = [messages[0]] + messages[-120:] if len(messages) > 121 else messages
            # 预估 token 预算（约 3 字符 ≈ 1 token，中英混合粗略估算），防止长任务烧太多 token
            # 注意：这里是「当前上下文」的估算，不是累计消耗——累计会导致长任务几轮就误判超预算提前停止
            try:
                est_tokens_used = sum(
                    len(json.dumps(m, ensure_ascii=False)) for m in llm_messages) // 3
            except Exception:
                pass
            if est_tokens_used > max_est_tokens:
                await self._mirror_log(
                    worker, f"已超出预估 token 预算（约 {est_tokens_used}），提前停止")
                return (f"（子智能体已超出预估 token 预算（约 {est_tokens_used}），"
                        "为控制成本提前停止；请缩小任务范围或分批执行）")
            msg = await self._llm_call(client, model, llm_messages, tools)
            choice = msg.choices[0].message
            content = (choice.content or "").strip()
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                return content or "（子智能体没有给出结论）"

            # 记录本轮工具调用（OpenAI 格式：assistant 消息必须带 tool_calls 原样回传）
            serialized = []
            fps = []
            for tc in tool_calls:
                serialized.append({
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name or "",
                        "arguments": tc.function.arguments or "{}",
                    },
                })
                fps.append((tc.function.name or "",
                            re.sub(r"\s+", "", tc.function.arguments or "{}")))
            recent_fps = (recent_fps + [fps])[-3:]
            if len(recent_fps) >= 3 and recent_fps[-1] == recent_fps[-2] == recent_fps[-3]:
                await self._mirror_log(worker, "连续 3 轮调用相同工具且参数不变，疑似死循环，已自动停止")
                return ("（子智能体连续 3 轮调用相同工具且参数不变，疑似死循环，已自动停止；"
                        "请检查任务描述是否自相矛盾，或相关工具是否异常）")
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": serialized})

            for tc in tool_calls:
                name = tc.function.name or ""
                raw = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except Exception:
                    args = {}
                await self._mirror_log(worker, f"调用工具 {name} {str(args)[:60]}")
                result_text = str(await self._execute_tool(name, args))[:8000]
                await self._mirror_log(worker, f"  {name} → {result_text[:120]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id or "",
                    "content": result_text,
                })
        return f"（子智能体达到最大轮次 {max_rounds} 轮，未能完成全部意图）"

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        # harness 技能/插件路由（与主智能体一致）
        try:
            from harness import get_harness
            result, source = await get_harness().execute_tool(name, arguments)
            if result is not None:
                return result
        except Exception as e:
            return f"工具执行失败：{e.__class__.__name__}: {e}"
        # 本地工具兜底
        try:
            from agent import execute_local_tool
            return await execute_local_tool(name, arguments)
        except Exception as e:
            return f"工具执行失败：{e.__class__.__name__}: {e}"

    async def _llm_call(self, client, model, messages, tools):
        """带瞬态重试的 LLM 调用（429/5xx/网络抖动退避重试，最多 3 次）。"""
        # 兜底：保证 role=tool 消息带 tool_call_id（与主智能体同一规范函数），
        # 否则发给 OpenAI 兼容提供方会被 400（missing field tool_call_id）
        try:
            from agent import _normalize_tool_rounds
            messages = _normalize_tool_rounds(messages)
        except Exception:
            pass
        try:
            from harness.core import retry_async
            return await retry_async(
                lambda: client.chat.completions.create(
                    model=model, messages=messages, tools=tools, tool_choice="auto",
                    max_tokens=4096, temperature=0.3, stream=False),
                attempts=3, backoff=2.0,
            )
        except Exception:
            return await client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
                max_tokens=1024, temperature=0.3, stream=False,
            )

    def _get_client(self):
        from agent import _build_llm_client, load_config
        cfg = load_config()
        model = str(cfg.get("model") or "").strip()
        if self._client is None or self._client_model != model:
            self._client = _build_llm_client(
                str(cfg.get("base_url") or ""), str(cfg.get("api_key") or ""))
            self._client_model = model or "x-preview-f-free"
        return self._client, self._client_model

    def _tool_defs(self) -> list:
        if self._tools is None:
            try:
                from agent import load_local_tools
                self._tools = [
                    t for t in load_local_tools()
                    if not (t.get("function") or {}).get("name", "").startswith("sub_agent_")
                ]
            except Exception as e:
                logger.warning("[SubAgent] 工具列表加载失败: %s", e)
                self._tools = []
        return self._tools

    # ---------- 取消 / 终态 ----------

    async def cancel(self, worker_id: str, reason: str = "") -> Optional[SubAgent]:
        """取消一个子智能体（排队中/执行中均可；静默，不汇报）。"""
        async with self._lock:
            worker = self.get(worker_id)
            if worker is None or worker.status in (ST_DONE, ST_ERROR, ST_CANCELLED):
                return worker
            if worker.loop_task is not None and not worker.loop_task.done():
                worker.loop_task.cancel()
            if worker.watchdog is not None:
                worker.watchdog.cancel()
            worker.cancelled = True
            return worker

    async def _finish(self, worker: SubAgent, status: str, *,
                      result: str = "", error: str = "") -> None:
        worker.status = status
        worker.updated_at = time.time()
        if result:
            worker.result = result[:4000]
        if error:
            worker.error = error[:2000]
            worker.logs.append(f"✗ {error[:200]}")
        if worker.watchdog is not None:
            worker.watchdog.cancel()
            worker.watchdog = None
        await self._mirror_finish(worker, status, result or error)
        await self._emit(status, worker)
        logger.info("[SubAgent] 「%s」→ %s", worker.title, worker.status_label)
        if status == ST_DONE:
            summary = result.strip().split("\n")[0][:160]
            report = f"任务《{worker.title}》已完成：{summary}"
        elif status == ST_ERROR:
            report = f"任务《{worker.title}》出错了：{error[:160]}"
        else:
            report = ""
        if report and self._report_handler is not None and worker.ws is not None:
            try:
                await self._report_handler(worker, report)
            except Exception as e:
                logger.warning("[SubAgent] 汇报回调出错: %s", e)

    async def _set_status(self, worker: SubAgent, status: str) -> None:
        worker.status = status
        worker.updated_at = time.time()
        await self._mirror_set_status(worker, status)
        await self._emit(status, worker)

    async def _emit(self, event: str, worker: SubAgent) -> None:
        if self._event_handler is None or worker.ws is None:
            return
        try:
            await self._event_handler(event, worker)
        except Exception as e:
            logger.warning("[SubAgent] 事件推送失败: %s", e)

    # ---------- 超时看护 ----------

    async def _watchdog(self, worker_id: str) -> None:
        try:
            await asyncio.sleep(MAX_RUNTIME)
            worker = self.get(worker_id)
            if worker is not None and worker.status in (ST_QUEUED, ST_RUNNING):
                if worker.loop_task is not None:
                    worker.loop_task.cancel()
                logger.warning("[SubAgent] 「%s」运行超时，自动取消", worker.title)
        except asyncio.CancelledError:
            pass

    # ---------- 任务中心镜像 ----------

    async def _mirror_create(self, worker: SubAgent) -> None:
        try:
            from task_orchestrator import get_orchestrator
            title = f"子智能体：「{worker.title}」"
            worker.task_ref = await get_orchestrator().create(
                kind="sub-agent",
                title=title,
                ws=worker.ws,
                brief=worker.task,
                confirm=False,
                channel="sub",
                extra={"worker_id": worker.id},
            )
        except Exception as e:
            worker.task_ref = None
            logger.warning("[SubAgent] 任务中心镜像创建失败: %s", e)

    async def _mirror_log(self, worker: SubAgent, text: str) -> None:
        if worker.task_ref is None:
            return
        try:
            from task_orchestrator import get_orchestrator
            worker.logs.append(text[:500])
            await get_orchestrator().add_log(worker.task_ref, text[:500])
        except Exception:
            pass

    async def _mirror_set_status(self, worker: SubAgent, status: str) -> None:
        if worker.task_ref is None:
            return
        try:
            from task_orchestrator import (
                get_orchestrator, STATUS_QUEUED, STATUS_RUNNING,
            )
            await get_orchestrator().set_status(
                worker.task_ref,
                STATUS_RUNNING if status == ST_RUNNING else STATUS_QUEUED,
            )
        except Exception:
            pass

    async def _mirror_finish(self, worker: SubAgent, status: str, tail: str) -> None:
        if worker.task_ref is None:
            return
        try:
            from task_orchestrator import (
                get_orchestrator, STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED,
            )
            orch = get_orchestrator()
            if status == ST_DONE:
                await orch.set_result(worker.task_ref, tail or "子智能体已完成", STATUS_DONE)
            elif status == ST_ERROR:
                await orch.set_error(worker.task_ref, tail or "子智能体执行出错")
            else:
                await orch.set_status(worker.task_ref, STATUS_CANCELLED)
                if tail:
                    await orch.add_step(worker.task_ref, f"（{tail[:200]}）")
        except Exception as e:
            logger.warning("[SubAgent] 任务中心镜像更新失败: %s", e)

    async def _sweep(self) -> None:
        stale = [w for w in self._workers.values()
                 if w.status not in (ST_QUEUED, ST_RUNNING)]
        stale.sort(key=lambda w: w.updated_at, reverse=True)
        for w in stale[_MAX_WORKERS // 2:]:
            self._workers.pop(w.id, None)


_sub_agents: Optional[SubAgentManager] = None


def get_sub_agents() -> SubAgentManager:
    global _sub_agents
    if _sub_agents is None:
        _sub_agents = SubAgentManager()
    return _sub_agents
