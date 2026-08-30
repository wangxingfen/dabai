"""执行力自我迭代（exec_loop_impl）—— 执行日志/复盘/策略库闭环 + 行为自洽守则。

合并自原 execution_loop 技能（含 behavior_strategy 纯提示词技能）。
行为自洽守则已并入 skill.json 的 prompt，本文件保持原 execution_loop 实现。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证能在任何启动方式下找到 dabai 根目录下的 execution_loop 包
_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from execution_loop import hooks  # noqa: E402

PROMPT = "【技能 执行力自我迭代】执行日志→复盘→策略库闭环：动手前 strategy_lookup 查策略，完成后 execution_record 登记，定期 execution_review 沉淀策略。"


async def _review(args: dict) -> str:
    return hooks.execution_review(args or {})


async def _lookup(args: dict) -> str:
    return hooks.strategy_lookup(args or {})


async def _record(args: dict) -> str:
    return hooks.execution_record(args or {})


async def _feedback(args: dict) -> str:
    return hooks.strategy_feedback(args or {})


HANDLERS = {
    "execution_review": _review,
    "strategy_lookup": _lookup,
    "execution_record": _record,
    "strategy_feedback": _feedback,
}