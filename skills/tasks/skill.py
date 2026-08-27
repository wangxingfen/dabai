"""长任务与批量任务技能 —— 大白把多步长任务/批量任务交给 harness 任务系统。

工具全部是 harness.tasks（TaskSystem）的薄封装：
- harness_flow_submit / harness_batch_submit  提交（后台执行，立即返回任务ID）
- harness_task_status / harness_task_list     查询（进度到步骤/条目粒度）
- harness_task_cancel / harness_task_retry    控制（取消 / 保留成果重试）

执行特性（由任务系统保证，本技能只做参数透传与结果裁剪）：
- 断点续跑：服务重启自动恢复，已成功步骤不重算；
- 步骤级超时+重试退避，流程级整体看门狗；失败策略 abort / continue；
- 同轮多步骤并行、批量并发受控；工具步骤经 harness 稳定路由 + runtime 监督。
"""
from __future__ import annotations

import json


def _ts():
    from harness import get_harness
    ts = get_harness().tasks
    ts.ensure_started()
    return ts


def _ok(data, extra: str = "") -> str:
    return json.dumps({"ok": True, **data}, ensure_ascii=False, default=str) + extra


def _err(e) -> str:
    return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


async def harness_flow_plan(args: dict) -> str:
    """规划长任务：三言两语 → 理解复述 + 步骤方案（校验/纠错后返回，可选直接开跑）。"""
    try:
        ts = _ts()
        autostart = bool(args.get("autostart"))
        plan = await ts.plan_flow(str(args.get("goal") or ""),
                                  hints=str(args.get("hints") or ""),
                                  critique=True if autostart else None)  # 直跑时强制自评兜底
        if plan.get("needs_clarification"):
            # 信息不足：把问题带回给模型 → 问用户；绝不含糊开跑
            return _ok({"needs_clarification": True,
                        "questions": plan.get("questions") or [],
                        "understanding": plan.get("understanding") or "",
                        "hint": "关键信息缺失——先向用户提出这些问题，答案放入 hints 后重新规划"})
        if plan.get("ok") and autostart:
            tid = ts.submit_flow(
                name=(plan.get("understanding") or "")[:40] or "规划任务",
                goal=plan.get("goal") or "",
                steps=plan.get("steps") or [],
                policy=str(args.get("policy") or "abort"),
            )
            return _ok({"planned": True, "submitted": True, "task_id": tid,
                        "understanding": plan.get("understanding"),
                        "autofixed": plan.get("autofixed"),
                        "hint": "已按规划开始后台执行，稍后用 harness_task_status 查进度。"})
        return json.dumps(plan, ensure_ascii=False, default=str)
    except Exception as e:
        return _err(e)


async def harness_flow_submit(args: dict) -> str:
    try:
        tid = _ts().submit_flow(
            name=str(args.get("name") or "未命名流程"),
            goal=str(args.get("goal") or ""),
            steps=args.get("steps") or [],
            policy=str(args.get("policy") or "abort"),
            timeout=float(args["timeout"]) if args.get("timeout") else None,
        )
        return _ok({"task_id": tid,
                    "hint": "已后台执行；用 harness_task_status 查进度，不要反复重复提交。"})
    except Exception as e:
        return _err(e)


async def harness_batch_submit(args: dict) -> str:
    try:
        tid = _ts().submit_batch(
            name=str(args.get("name") or "批量任务"),
            tool=str(args.get("tool") or ""),
            items=args.get("items") or [],
            concurrency=int(args["concurrency"]) if args.get("concurrency") else None,
        )
        return _ok({"task_id": tid,
                    "hint": "已后台并行执行；用 harness_task_status 查进度。"})
    except Exception as e:
        return _err(e)


def _compact_status(st: dict) -> dict:
    """status 瘦身版：轮询场景给模型看的最小信息量（省 token）。

    步骤只留 状态/错误/结果截断120字；流程终态时附最终摘要（400字）。
    detail=true 时返回全量。"""
    out = {k: st.get(k) for k in
           ("id", "name", "kind", "state", "progress", "status", "goal",
            "waiting_confirm", "error")}
    if st.get("kind") == "flow" and isinstance((st.get("result") or {}).get("steps"), dict):
        steps = {}
        for sid, s in st["result"]["steps"].items():
            r = s.get("result")
            steps[sid] = {"state": s.get("state"),
                          "error": (s.get("error") or "")[:80],
                          "result": (r[:120] if isinstance(r, str)
                                     else str(r)[:120]) if s.get("state") == "succeeded" else None}
        out["steps"] = steps
        if st.get("state") == "succeeded" and steps:
            last = sorted(st["result"]["steps"].items())[-1][1].get("result")
            if isinstance(last, str) and last:
                out["final"] = last[:400]
    elif st.get("kind") == "batch":
        r = st.get("result") or {}
        out["batch"] = {"count": r.get("count"), "ok": r.get("ok"),
                        "failed": r.get("failed")}
        if st.get("state") in ("succeeded", "failed") and r.get("failures"):
            out["batch"]["failures"] = {k: str(v)[:100] for k, v in
                                        list(r["failures"].items())[:5]}
    return out


async def harness_task_status(args: dict) -> str:
    try:
        st = _ts().status(str(args.get("task_id") or ""))
        if st is None:
            return _err(f"任务不存在: {args.get('task_id')}")
        if args.get("detail"):
            return _ok(st)
        return _ok(_compact_status(st))
    except Exception as e:
        return _err(e)


async def harness_task_list(args: dict) -> str:
    try:
        ts = _ts()
        jobs = ts.list_tasks(state=args.get("state") or None,
                             kind=None, limit=int(args.get("limit") or 10))
        flows = [j for j in jobs if j["kind"] in ("flow", "batch")]
        out = {"tasks": flows or jobs}
        reports = ts.drain_reports()
        if reports:
            out["completed_since_last_check"] = reports  # 完成汇报：自动带回，免去反复轮询
        return _ok(out)
    except Exception as e:
        return _err(e)


async def harness_task_cancel(args: dict) -> str:
    try:
        ok = _ts().cancel(str(args.get("task_id") or ""))
        return _ok({"cancelled": ok}) if ok else _err("任务不存在或已结束")
    except Exception as e:
        return _err(e)


async def harness_task_confirm(args: dict) -> str:
    """批准/拒绝流程中等待确认的危险步骤（approve=true 批准，false 拒绝）。"""
    try:
        ts = _ts()
        tid = str(args.get("task_id") or "")
        approve = bool(args.get("approve", True))
        ok, msg = (ts.approve_step if approve else ts.reject_step)(
            tid, str(args.get("note") or ""))
        if not ok:
            return _err(msg)
        return _ok({"result": msg,
                    "hint": ("已批准，流程继续执行" if approve else
                             "已拒绝该步骤，流程将按失败策略处理")})
    except Exception as e:
        return _err(e)


async def harness_task_retry(args: dict) -> str:
    try:
        ok = _ts().retry(str(args.get("task_id") or ""))
        return _ok({"retried": ok}) if ok else _err("任务不存在或仍在运行")
    except Exception as e:
        return _err(e)


HANDLERS = {
    "harness_flow_plan": harness_flow_plan,
    "harness_flow_submit": harness_flow_submit,
    "harness_batch_submit": harness_batch_submit,
    "harness_task_status": harness_task_status,
    "harness_task_list": harness_task_list,
    "harness_task_cancel": harness_task_cancel,
    "harness_task_retry": harness_task_retry,
    "harness_task_confirm": harness_task_confirm,
}
