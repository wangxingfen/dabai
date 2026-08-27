"""执行日志（Execution Log）——“执行力自我迭代”闭环的第 1 环。

每次任务执行完自动记录四要素：目标(goal)、动作(actions)、结果(result)、
卡点(blockers，哪里失败/卡壳)，外加任务类型(task_type)与耗时，供复盘提炼。

存储：data/execution_logs.jsonl —— JSON Lines 追加写，一条日志一行 JSON。
写入失败静默降级：日志绝不能拖垮任务执行（与 harness 经验记忆同一原则）。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("execution_loop.logger")

# 结果判定口径
OUTCOME_OK = "ok"          # 目标达成
OUTCOME_PARTIAL = "partial"  # 部分达成/有绕行
OUTCOME_FAIL = "fail"      # 失败/卡壳

_LOG_FILENAME = "execution_logs.jsonl"


def _norm_blocker(b: Any) -> dict:
    """卡点归一化：字符串自动转成结构化 {stage, symptom, cause}。"""
    if isinstance(b, dict):
        return {
            "stage": str(b.get("stage") or "执行"),
            "symptom": str(b.get("symptom") or "").strip(),
            "cause": str(b.get("cause") or "").strip(),
        }
    return {"stage": "执行", "symptom": str(b).strip(), "cause": ""}


class ExecutionLogger:
    """结构化执行日志：JSONL 追加写，支持按复盘游标读取未复盘日志。"""

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else (
            Path(__file__).resolve().parent.parent / "data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / _LOG_FILENAME

    def record(self, *, task_id: str = "", task_type: str = "general",
               goal: str = "", actions: Optional[list] = None,
               result: Any = None, blockers: Optional[list] = None,
               outcome: str = OUTCOME_OK, duration_ms: int = 0,
               meta: Optional[dict] = None) -> Optional[dict]:
        """记录一次任务执行。返回写入的日志条目；写入失败返回 None（不抛异常）。"""
        if outcome not in (OUTCOME_OK, OUTCOME_PARTIAL, OUTCOME_FAIL):
            outcome = OUTCOME_PARTIAL
        entry: dict = {
            "id": uuid.uuid4().hex[:12],
            "task_id": task_id or uuid.uuid4().hex[:12],
            "ts": time.time(),
            "task_type": str(task_type or "general"),
            "goal": str(goal or ""),
            "actions": [dict(a) for a in (actions or [])],
            "result": str(result)[:2000] if result is not None else "",
            "outcome": outcome,
            "blockers": [_norm_blocker(b) for b in (blockers or [])],
            "duration_ms": int(duration_ms),
            "meta": dict(meta) if meta else {},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return entry
        except Exception as e:
            logger.warning("执行日志写入失败（已静默降级）: %s", e)
            return None

    def load_recent(self, limit: int = 1000) -> list[dict]:
        """读取最近 limit 条日志（时间正序）。日志文件损坏时返回已解析的部分。"""
        out: list[dict] = []
        if not self.path.exists():
            return out
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("日志行解析失败（跳过）: %.60s", line)
        except Exception as e:
            logger.warning("执行日志读取失败: %s", e)
        return out[-limit:]

    def load_after(self, ts: float, limit: int = 1000) -> list[dict]:
        """复盘游标：读取 ts 之后（ts >= 阈值）的日志，未复盘的新日志归它管。"""
        return [e for e in self.load_recent(limit) if float(e.get("ts") or 0) > ts]

    def count(self) -> int:
        return len(self.load_recent(10 ** 9))
