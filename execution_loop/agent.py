"""ExecutionLoop ——「执行日志 → 复盘 → 策略沉淀 → 自动调用」闭环总入口。

一次 execute() 完成整条自动化链路：
  1. 自动检索：按 task_type + goal 从策略库检索相关策略，注入 run 的上下文
     （ctx["strategy_notes"]，命中策略同时记一次 hit）；
  2. 执行任务：调用你提供的 run(ctx)，由你的执行逻辑决定是否采纳策略；
  3. 自动记录：把目标/动作/结果/卡点写入执行日志（第 1 环）；
  4. 失败闭环：本次若失败/卡壳，复盘时会被提炼成新策略（第 2 环）。

run(ctx) 契约：ctx 含 task_type/goal/task_id/strategy_notes/strategies。
返回 dict：
    {"outcome": "ok"|"partial"|"fail", "result": ..., "actions": [...],
     "blockers": [{"stage","symptom","cause"} 或字符串]}
run 抛异常时按 fail 记录（异常信息进卡点），不会向上抛。

用法示例见 examples/demo_iteration.py 与 README.md。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Optional

from .logger import ExecutionLogger, OUTCOME_FAIL, OUTCOME_OK
from .retriever import StrategyRetriever
from .reviewer import Reviewer
from .store import StrategyStore

logger = logging.getLogger("execution_loop.agent")


class ExecutionLoop:
    def __init__(self, base_dir: Optional[Any] = None,
                 llm_distill: Optional[Callable[[dict], str]] = None,
                 top_k: int = 3):
        self.logger = ExecutionLogger(base_dir=base_dir)
        self.store = StrategyStore(base_dir=base_dir)
        self.retriever = StrategyRetriever(self.store)
        self.reviewer = Reviewer(logger_=self.logger, store=self.store,
                                 base_dir=self.logger.base_dir,
                                 llm_distill=llm_distill)
        self.top_k = max(1, top_k)

    # ==================== 自动调用闭环 ====================
    def execute(self, task_type: str, goal: str, run: Callable[[dict], dict],
                *, task_id: str = "", record: bool = True) -> dict:
        """自动检索 → 注入 → 执行 → 记录。返回 {"log", "strategies_used", "report"}。"""
        t0 = time.time()
        task_id = task_id or uuid.uuid4().hex[:12]
        hits = self.retriever.retrieve(task_type, goal, top_k=self.top_k)
        for h in hits:
            self.store.hit(h["strategy"]["id"])
        ctx: dict = {
            "task_id": task_id,
            "task_type": task_type,
            "goal": goal,
            "strategy_notes": self.retriever.build_notes(task_type, goal, top_k=self.top_k),
            "strategies": [h["strategy"]["id"] for h in hits],
            "strategy_hits": hits,
        }
        # ---- 执行（异常 → 视为失败并进卡点） ----
        report: dict
        try:
            report = dict(run(ctx) or {})
        except Exception as e:
            logger.warning("任务执行异常（记录为失败）: %s", e)
            report = {"outcome": OUTCOME_FAIL, "blockers": [
                {"stage": "执行", "symptom": f"执行抛异常：{e}", "cause": "未捕获异常"}]}
        outcome = report.get("outcome") or (OUTCOME_OK if not report.get("blockers") else "partial")
        if outcome not in ("ok", "partial", "fail"):
            outcome = "partial"
        report["outcome"] = outcome
        # ---- 记录日志 ----
        log: Optional[dict] = None
        if record:
            log = self.logger.record(
                task_id=task_id, task_type=task_type, goal=goal,
                actions=report.get("actions") or [],
                result=report.get("result"),
                blockers=report.get("blockers") or [],
                outcome=outcome,
                duration_ms=int((time.time() - t0) * 1000),
                meta={"strategies_used": [h["strategy"]["id"] for h in hits]})
        return {"log": log, "strategies_used": hits, "report": report,
                "task_id": task_id, "duration_ms": int((time.time() - t0) * 1000)}

    # ==================== 其他环节透传 ====================
    def review(self, since: Optional[float] = None) -> dict:
        """定期复盘：把日志卡点提炼成策略入库。返回复盘报告。"""
        return self.reviewer.review(since=since)

    def lookup(self, scene: str, goal_text: str = "", top_k: Optional[int] = None) -> list[dict]:
        """主动查策略（不执行、不计 hit）。"""
        return self.retriever.retrieve(scene, goal_text, top_k=top_k or self.top_k)

    def feedback(self, strategy_id: str, good: bool) -> bool:
        """人工/环境反馈策略效果：good=True 有效，False 失效降权。"""
        return self.store.feedback(strategy_id, good)

    def snapshot(self) -> dict:
        """运行状态快照：日志统计 + 策略库概览。"""
        logs = self.logger.load_recent(10 ** 9)
        stats = {"total": len(logs),
                 "ok": sum(1 for e in logs if e.get("outcome") == "ok"),
                 "partial": sum(1 for e in logs if e.get("outcome") == "partial"),
                 "fail": sum(1 for e in logs if e.get("outcome") == "fail")}
        return {"logs": stats, "strategies": self.store.snapshot()}


def default_loop() -> ExecutionLoop:
    """进程级默认实例（hooks 与技能共用）。"""
    if not hasattr(default_loop, "_inst"):
        default_loop._inst = ExecutionLoop()
    return default_loop._inst
