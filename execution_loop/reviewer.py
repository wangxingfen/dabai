"""复盘提炼（Reviewer）——闭环的第 2 环：把日志中的卡点提炼成可复用策略。

流程：
  1. 取「未复盘」的执行日志（复盘游标 last_review_ts，持久化在 data/review_history.json）；
  2. 提取每条日志的卡点（blockers），归一化成 (scene, symptom) 指纹；
  3. 按指纹聚类：同场景同卡点出现多次 = 稳定复发的坑，优先沉淀；
  4. 模板化生成策略规则（启发式给出"应对动作"），可选注入 llm_distill 回调润色；
  5. 写入策略库（store 内部做同场景同关键词策略合并去重），推进复盘游标。

llm_distill：可选的 LLM 提炼回调，签名 llm_distill(blocker: dict) -> str（润色后的规则文本）。
不传则用纯规则模板，保证离线可用。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .logger import ExecutionLogger
from .store import StrategyStore

logger = logging.getLogger("execution_loop.reviewer")

_REVIEW_FILE = "review_history.json"

# 常见卡点 → 可操作应对建议（启发式，纯离线也能给出可执行策略）
HEURISTIC_SUGGESTIONS: list[tuple[tuple[str, ...], str]] = [
    (("403", "反爬", "拦截", "验证码", "封禁", "robot"),
     "先补 User-Agent/Cookie/请求头伪装，遇拦截先退避重试或更换入口（镜像/接口），不要硬闯"),
    (("超时", "timeout", "连接失败", "网络不通", "断连"),
     "先跑最小连通性用例确认能连上，再全量执行；网络操作加超时+指数退避重试"),
    (("不存在", "找不到", "not found", "缺失", "404"),
     "执行前先确认目标真实存在（列目录/查状态/搜路径），存在了再操作，别猜路径"),
    (("权限", "denied", "拒绝访问", "无权限", "forbidden"),
     "先检查当前角色权限是否够，不够先走申请/换通道，别重复硬试"),
    (("格式", "解析失败", "parse", "json 错误", "语法错误"),
     "先校验输入格式与样例，解析失败时保留原始片段便于定位，再批量处理"),
    (("依赖", "未安装", "module not found", "import", "缺库"),
     "先确认运行环境与依赖版本，环境问题先修环境（装依赖/切解释器）再写逻辑"),
    (("内存", "内存不足", "oom", "溢出", "超长"),
     "批量/大输入任务先切小份试跑，确认单份可行再逐步放大规模"),
]


def _norm_fingerprint(text: str) -> str:
    """卡点指纹：小写 + 去空白，用于聚类去重。"""
    return re.sub(r"[\s\u3000]+", " ", str(text or "").strip().lower())


def _pick_suggestion(text: str) -> str:
    low = str(text or "")
    for keys, advice in HEURISTIC_SUGGESTIONS:
        if any(k.lower() in low.lower() for k in keys):
            return advice
    return "先复现最小用例确认根因，再针对根因处理；同类卡点复现多次时优先查环境/输入，其次查流程"


class Reviewer:
    def __init__(self, logger_: Optional[ExecutionLogger] = None,
                 store: Optional[StrategyStore] = None,
                 base_dir: Optional[Path | str] = None,
                 llm_distill: Optional[Callable[[dict], str]] = None,
                 max_strategies_per_review: int = 10):
        if base_dir is None and (logger_ is None or store is None):
            base_dir = Path(__file__).resolve().parent.parent / "data"
        self.logger = logger_ or ExecutionLogger(base_dir=base_dir)
        self.store = store or StrategyStore(base_dir=base_dir)
        self.llm_distill = llm_distill
        self.max_strategies_per_review = max_strategies_per_review
        self._review_path = (base_dir if base_dir else self.logger.base_dir) / _REVIEW_FILE

    # ---------- 复盘游标 ----------
    def _last_review_ts(self) -> float:
        try:
            if self._review_path.exists():
                with open(self._review_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return float(data.get("last_review_ts") or 0)
        except Exception:
            pass
        return 0.0

    def _advance_cursor(self, ts: float) -> None:
        try:
            tmp = str(self._review_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"last_review_ts": ts, "reviewed_at": time.time()}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, self._review_path)
        except Exception as e:
            logger.warning("复盘游标写入失败: %s", e)

    # ---------- 复盘主流程 ----------
    def review(self, since: Optional[float] = None) -> dict:
        """执行一次复盘：日志卡点 → 聚类 → 策略提炼入库。

        返回报告：{"reviewed_logs": n, "blocker_clusters": [...], "strategies_added": [...]}
        """
        cursor = since if since is not None else self._last_review_ts()
        logs = self.logger.load_after(cursor)
        reviewed: list[dict] = [e for e in logs
                                if e.get("outcome") in ("fail", "partial") or e.get("blockers")]
        # ---- 1. 提取卡点并聚类 ----
        clusters: dict[tuple, dict] = {}   # (scene, symptom_fp) -> {scene, symptom, cause, log_ids, count}
        for entry in reviewed:
            scene = str(entry.get("task_type") or "general")
            goal = str(entry.get("goal") or "")
            for b in entry.get("blockers") or []:
                symptom = str(b.get("symptom") or "")
                if not symptom:
                    continue
                fp = (scene, _norm_fingerprint(symptom))
                c = clusters.setdefault(fp, {
                    "scene": scene, "symptom": symptom[:120],
                    "cause": str(b.get("cause") or "")[:200],
                    "goal_examples": [], "log_ids": [], "count": 0})
                c["log_ids"].append(str(entry.get("id") or ""))
                c["count"] += 1
                if goal and len(c["goal_examples"]) < 2:
                    c["goal_examples"].append(goal[:60])
        # ---- 2. 聚类 → 策略规则 ----
        added: list[dict] = []
        made = 0
        for (scene, _fp), c in sorted(clusters.items(), key=lambda kv: -kv[1]["count"]):
            if made >= self.max_strategies_per_review:
                break
            title, rule, keywords = self._build_strategy(c)
            if self.llm_distill is not None:
                try:
                    polished = str(self.llm_distill({**c, "draft_rule": rule}) or "").strip()
                    if polished:
                        rule = polished
                except Exception as e:
                    logger.warning("LLM 提炼失败（退回模板规则）: %s", e)
            item = self.store.add(scene=scene, title=title, rule=rule,
                                  trigger_keywords=keywords,
                                  source_log_ids=c["log_ids"],
                                  source_count=c["count"])
            existing = item.get("source_log_ids") or []
            merged = len(existing) > len(set(c["log_ids"]))
            added.append({"id": item.get("id"), "scene": scene,
                          "title": item.get("title"), "rule": item.get("rule"),
                          "merged": merged})
            made += 1
        # ---- 3. 推进复盘游标（本次扫描过的日志全部跳过，含无卡点的成功日志） ----
        if logs:
            latest_ts = max(float(e.get("ts") or 0) for e in logs)
            if latest_ts > cursor:
                self._advance_cursor(latest_ts)
        return {"reviewed_logs": len(reviewed), "cursor": self._last_review_ts(),
                "blocker_clusters": [
                    {"scene": c["scene"], "symptom": c["symptom"], "count": c["count"],
                     "goal_examples": c["goal_examples"], "cause": c["cause"]}
                    for c in clusters.values()],
                "strategies_added": added}

    # ---------- 规则模板 ----------
    def _build_strategy(self, c: dict) -> tuple[str, str, list]:
        """由卡点聚类生成 (标题, 规则文本, 触发关键词)。"""
        scene, symptom, cause, n = c["scene"], c["symptom"], c["cause"], c["count"]
        advice = _pick_suggestion(symptom + " " + cause)
        if n >= 2:
            title = f"{scene}：{symptom[:24]}（复现 {n} 次）"
        else:
            title = f"{scene}：{symptom[:24]}"
        if n >= 2:
            rule = (f"场景「{scene}」同类卡点已复现 {n} 次：{symptom}"
                    + (f"。根因线索：{cause}" if cause else "")
                    + f"。执行这类任务前先做：{advice}。")
        else:
            rule = (f"场景「{scene}」出现过卡点：{symptom}"
                    + (f"。根因线索：{cause}" if cause else "")
                    + f"。下次同类任务先做：{advice}。")
        keywords = [scene] + re.findall(r"[\w\u4e00-\u9fff]{2,}", symptom)[:8]
        keywords = list(dict.fromkeys(kw for kw in keywords if kw))[:10]
        return title, rule, keywords
