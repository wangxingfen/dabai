"""自动检索（Retriever）——闭环的第 4 环：执行同类任务前自动检索策略库。

打分模型（完全离线，可插拔扩展）：
    score = 4.0 * scene 命中 + 2.0 * 关键词 Jaccard + 0.3 * log2(1+历史命中) + 口碑修正
- scene 命中：任务类型与策略场景精确相等 4 分；包含关系 2 分；
- 关键词 Jaccard：策略 trigger_keywords（中文按 2-gram，英文按单词）与
  「任务目标 + 场景」词袋的重叠率；
- 历史命中：hit_count 越多权重略高（实战验证过的策略优先）；
- 口碑修正：bad 明显多于 good 的策略降权。
只返回 enabled 的策略；top_k 条，附带命中的关键词与来源信息，方便注入提示词。
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from .store import StrategyStore

_ENG_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _bag(text: str) -> set[str]:
    """中英混排词袋：英文按单词、中文按双字组切（覆盖中文关键词的常见形态）。"""
    out: set[str] = set()
    low = str(text or "").lower()
    for w in _ENG_WORD.findall(low):
        if len(w) >= 2:
            out.add(w)
    # 中文双字组
    cjk = "".join(_CJK.findall(low))
    if cjk:
        out.update(cjk[i:i + 2] for i in range(len(cjk) - 1))
    return out


class StrategyRetriever:
    def __init__(self, store: Optional[StrategyStore] = None,
                 base_dir: Optional[Any] = None):
        self.store = store or StrategyStore(base_dir=base_dir)

    def retrieve(self, scene: str, goal_text: str = "", top_k: int = 3,
                 min_score: float = 0.5) -> list[dict]:
        """按任务场景 + 目标文本检索策略，返回按分数降序的命中列表。

        每项：{"strategy": {...}, "score": float, "scene_exact": bool,
               "matched_keywords": [...], "advice": "可直接注入执行的策略要点"}
        """
        scene = str(scene or "general").strip()
        goal_bag = _bag(goal_text + " " + scene)
        scored: list[dict] = []
        for s in self.store.list():
            if not s.get("enabled", True):
                continue
            s_scene = str(s.get("scene") or "")
            scene_exact = (s_scene == scene)
            scene_partial = (s_scene and (s_scene in scene or scene in s_scene))
            # 关键词 Jaccard
            kw_bag: set[str] = set()
            for k in (s.get("trigger_keywords") or []) + [s_scene, s.get("title") or ""]:
                kw_bag |= _bag(k)
            inter = goal_bag & kw_bag
            jac = len(inter) / len(kw_bag) if kw_bag else 0.0
            # 口碑
            good = int(s.get("good") or 0)
            bad = int(s.get("bad") or 0)
            reputation = 1.0 if bad == 0 else max(0.0, (good - bad) / (good + bad + 1))
            score = (4.0 if scene_exact else (2.0 if scene_partial else 0.0)) \
                + 2.0 * jac + 0.3 * math.log2(1 + int(s.get("hit_count") or 0)) \
                + reputation * 0.8
            if bad >= 2 and bad >= good:
                score -= 1.2   # 多次反馈失效的策略明显降权
            # 场景隔离硬门槛：场景完全无关时，仅靠口碑/命中加分不许入选；
            # 必须有关键词实质重叠（Jaccard >= 0.2）才允许跨场景借鉴
            if not scene_exact and not scene_partial:
                if not inter or jac < 0.2:
                    continue
            if score < min_score:
                continue
            scored.append({
                "strategy": s,
                "score": round(score, 2),
                "scene_exact": scene_exact,
                "matched_keywords": sorted(inter)[:6],
            })
        scored.sort(key=lambda x: -x["score"])
        for item in scored[:top_k]:
            s = item["strategy"]
            kw = "、".join(item["matched_keywords"]) or "（场景）"
            item["advice"] = (f"[策略] {s.get('title')}：{s.get('rule')} "
                              f"[命中：{kw}，已有 {s.get('hit_count') or 0} 次实战]")
        return scored[:top_k]

    def build_notes(self, scene: str, goal_text: str = "", top_k: int = 3) -> str:
        """把检索结果拼成一段可直接注入执行上下文的策略提示文本（无结果返回空串）。"""
        hits = self.retrieve(scene, goal_text, top_k=top_k)
        if not hits:
            return ""
        lines = ["【历史策略提示（执行同类任务自动检索，优先遵循）】"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h['advice']}")
        return "\n".join(lines) + "\n"
