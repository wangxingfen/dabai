# -*- coding: utf-8 -*-
"""通用子智能体技能 —— 主智能体把任意复杂任务下发给独立子智能体并行处理。

- sub_agent_spawn 返回 __sub_agent_spawn__ 标记 → server 用 ws/state 登记
  子智能体并立即开始后台自主执行（LLM 思考 + 工具调用），完成后汇报主智能体；
- sub_agents_list / sub_agent_status / sub_agent_cancel 直接操作子智能体注册表。
"""
from __future__ import annotations

import json


def _sa():
    from sub_agents import get_sub_agents
    return get_sub_agents()


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


def sub_agent_spawn(args: dict) -> str:
    task = str(args.get("task") or "").strip()
    if not task:
        return _err("需要 task 参数：要交给子智能体完成的具体任务（越具体越好）。")
    title = str(args.get("title") or "").strip()[:60]
    note = str(args.get("note") or "").strip()[:200]
    return json.dumps({
        "__sub_agent_spawn__": True,
        "task": task,
        "title": title or task[:40],
        "note": note,
        "message": f"已下发子智能体：「{title or task[:40]}」。子智能体在后台自主执行，"
                   "做完会自动向你汇报；可用 sub_agents_list 随时查看有哪些子进程在干活。",
    }, ensure_ascii=False)


def sub_agents_list(args: dict) -> str:
    try:
        items = _sa().list(80)
    except Exception as e:
        return _err(f"子智能体注册表不可用: {e}")
    active = [w for w in items if w["status"] in ("queued", "running")]
    if not active:
        return "当前没有正在运行的通用子智能体（可以随时用 sub_agent_spawn 下发新任务）。"
    lines = [f"当前有 {len(active)} 个子智能体在忙（并发上限 8，超出自动排队）："]
    for w in active:
        lines.append(f"- [{w['status_label']}] 任务「{w['title']}」[{w['id']}]")
    if len(active) < len(items):
        lines.append(f"（另有 {len(items) - len(active)} 个已结束/已取消的任务在工作中心可查）")
    return "\n".join(lines)


def sub_agent_status(args: dict) -> str:
    wid = str(args.get("worker_id") or "").strip()
    if not wid:
        return _err("需要 worker_id（sub_agents_list 可查）。")
    try:
        w = _sa().get(wid)
    except Exception as e:
        return _err(f"查询失败: {e}")
    if w is None:
        return _err(f"子智能体 {wid} 不存在或已被清理。")
    s = w.snapshot(full=True)
    lines = [f"🧠 子智能体 [{s['id']}]：「{s['title']}」—— {s['status_label']}",
             f"任务：{s['task'][:200]}"]
    logs = s.get("logs") or []
    if logs:
        lines.append("进度：")
        lines.extend(f"  · {x[:120]}" for x in logs[-6:])
    if s.get("result"):
        lines.append(f"结果：{s['result'][:300]}")
    if s.get("error"):
        lines.append(f"错误：{s['error'][:300]}")
    if s.get("task_ref_id"):
        lines.append(f"任务中心条目：{s['task_ref_id']}")
    return "\n".join(lines)


def sub_agent_cancel(args: dict) -> str:
    wid = str(args.get("worker_id") or "").strip()
    if not wid:
        return _err("需要 worker_id（sub_agents_list 可查）。")
    reason = str(args.get("reason") or "").strip() or "用户要求收回"
    try:
        w = _sa().get(wid)
        if w is None:
            return _err(f"子智能体 {wid} 不存在或已结束。")
        await_me = _sa().cancel(wid, reason)
        if await_me is None:
            return _err(f"子智能体 {wid} 已处于终态。")
        return f"已收回子智能体 [{wid}]（「{w.title}」，{reason}）。"
    except Exception as e:
        return _err(f"取消失败: {e}")


HANDLERS = {
    "sub_agent_spawn": sub_agent_spawn,
    "sub_agents_list": sub_agents_list,
    "sub_agent_status": sub_agent_status,
    "sub_agent_cancel": sub_agent_cancel,
}
