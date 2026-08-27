"""与「大白」运行时接线的桥（hooks）。

闭环本身完全独立；这里提供三个场景的接线入口：

1. 对话内工具（推荐，零侵入）：
   skills/execution_loop 技能把 execution_record / execution_review / strategy_lookup
   注册成 harness 工具，HANDLERS 直接调用本模块的三个函数 —— 「大白」在对话里就能
   复盘、查策略、登记任务，无需改动 agent.py / harness 任何核心代码。

2. 任务完成自动记录（可选）：
   在「大白」任务执行点（如 harness 任务的终态回调 / agent 的工具调用包装）追加一行：
       from execution_loop.hooks import record_dabai_task
       record_dabai_task(task_type=..., goal=..., outcome=..., blockers=...)
   即可让每次任务完成自动落执行日志（闭环第 1 环）。

3. 执行前自动检索（可选）：
   在「大白」的决策循环里，对已知任务类型先取
       strategy_notes_for(task_type, goal)
   把结果拼进 LLM 上下文 / 任务描述，即为闭环第 4 环在运行时生效。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .agent import ExecutionLoop, default_loop

logger = logging.getLogger("execution_loop.hooks")


def strategy_notes_for(task_type: str, goal: str = "", top_k: int = 3,
                       loop: Optional[ExecutionLoop] = None) -> str:
    """执行前自动检索：返回可直接拼进上下文的策略提示文本。"""
    return (loop or default_loop()).retriever.build_notes(task_type, goal, top_k=top_k)


def record_dabai_task(*, task_type: str, goal: str, outcome: str,
                      blockers: Optional[list] = None, actions: Optional[list] = None,
                      result: Any = None, meta: Optional[dict] = None,
                      loop: Optional[ExecutionLoop] = None) -> Optional[dict]:
    """任务完成自动记录（闭环第 1 环的运行时接线点）。失败静默。"""
    try:
        return (loop or default_loop()).logger.record(
            task_type=task_type, goal=goal, outcome=outcome,
            blockers=blockers or [], actions=actions or [],
            result=result, meta=meta)
    except Exception as e:
        logger.warning("记录任务执行日志失败: %s", e)
        return None


# ---------------- 对话内工具函数（给 skills/execution_loop 的 HANDLERS 用） ----------------

def execution_record(args: dict) -> str:
    """登记一次任务执行（返回日志 id；不抛异常）。"""
    try:
        entry = record_dabai_task(
            task_type=str(args.get("task_type") or "general"),
            goal=str(args.get("goal") or ""),
            outcome=str(args.get("outcome") or "partial"),
            blockers=args.get("blockers") or [],
            actions=args.get("actions") or [],
            result=args.get("result"))
        if not entry:
            return "执行日志写入失败（已静默降级）"
        return f"已记录执行日志 {entry['id']}：目标「{entry['goal'][:40]}」结果={entry['outcome']}，卡点 {len(entry['blockers'])} 个"
    except Exception as e:
        return f"记录失败：{e}"


def execution_review(args: Optional[dict] = None) -> str:
    """执行一次复盘：把日志卡点提炼为策略（返回报告文本）。"""
    try:
        report = default_loop().review()
        lines = [f"复盘完成：本次梳理日志 {report['reviewed_logs']} 条，"
                 f"识别卡点聚类 {len(report['blocker_clusters'])} 类，"
                 f"沉淀/合并策略 {len(report['strategies_added'])} 条。"]
        for c in report.get("blocker_clusters", [])[:8]:
            lines.append(f"· [{c['scene']} ×{c['count']}] {c['symptom']}")
        for s in report.get("strategies_added", [])[:5]:
            lines.append(f"★ 策略 {s['id']}（{s['scene']}）：{s['title']}")
        return "\n".join(lines)
    except Exception as e:
        return f"复盘失败（不影响主流程）：{e}"


def strategy_lookup(args: dict) -> str:
    """按任务类型 + 目标检索策略库（返回策略要点，供执行时遵循）。"""
    try:
        notes = strategy_notes_for(str(args.get("task_type") or "general"),
                                   str(args.get("goal") or ""))
        if not notes:
            return "策略库暂无该场景策略（首次执行，完成后建议复盘沉淀）"
        return notes
    except Exception as e:
        return f"策略检索失败：{e}"


def strategy_feedback(args: dict) -> str:
    """给策略打效果分：good=true 有效 / false 失效。"""
    try:
        ok = default_loop().feedback(str(args.get("strategy_id") or ""),
                                     bool(args.get("good")))
        return f"策略 {args.get('strategy_id')} 反馈已记录（有效={args.get('good')}）" if ok \
            else "未找到该策略"
    except Exception as e:
        return f"反馈失败：{e}"
