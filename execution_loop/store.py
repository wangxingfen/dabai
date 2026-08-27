"""策略库（Strategy Store）——闭环的第 3 环：策略规则的持久化与分类存储。

数据文件：data/strategies.json，形如：
    {"version": 1, "strategies": [ {id, scene, title, rule, trigger_keywords,
                                    source_log_ids, source_count, hit_count,
                                    good, bad, enabled, created_at, updated_at, last_used_at} ]}

- 按场景（scene = 任务类型）分类；
- add 时同场景 + 规则文本前缀相似 → 合并（累计 source_log_ids / source_count），
  防止反复复盘产生重复策略（策略爆炸）；
- hit / feedback 记录每条策略的实战效果，供检索排序与裁剪。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("execution_loop.store")

_STRATEGY_FILE = "strategies.json"


class StrategyStore:
    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else (
            Path(__file__).resolve().parent.parent / "data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / _STRATEGY_FILE
        self._lock = threading.Lock()
        self._strategies: list[dict] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        try:
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    ls = data.get("strategies")
                    if isinstance(ls, list):
                        self._strategies = [s for s in ls if isinstance(s, dict)]
        except Exception as e:
            logger.warning("策略库读取失败（以空库启动）: %s", e)
            self._strategies = []

    def _save(self) -> None:
        """原子落盘：先写 tmp 再 os.replace，避免写一半损坏文件。"""
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "updated_at": time.time(),
                       "strategies": self._strategies}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ---------- 写入 ----------
    def add(self, *, scene: str, title: str, rule: str,
            trigger_keywords: Optional[list] = None,
            source_log_ids: Optional[list] = None,
            source_count: int = 1) -> dict:
        """新增策略；同场景且规则文本前 40 字相似 → 与既有策略合并计数。"""
        scene = str(scene or "general").strip()
        rule = str(rule or "").strip()
        if not rule:
            raise ValueError("策略规则文本不能为空")
        with self._lock:
            new_kw = set(trigger_keywords or [])
            for s in self._strategies:
                if not (s.get("scene") == scene and s.get("enabled", True)):
                    continue
                s_kw = set(s.get("trigger_keywords") or [])
                # 同场景且触发关键词交集>=2 -> 同一卡点反复沉淀，合并计数不新增条目
                if s_kw and new_kw and len(s_kw & new_kw) >= 2:
                    logs = set(s.get("source_log_ids") or [])
                    logs.update(source_log_ids or [])
                    s["source_log_ids"] = sorted(logs)
                    s["source_count"] = int(s.get("source_count") or 1) + max(1, int(source_count or 1))
                    s["updated_at"] = time.time()
                    self._save()
                    return s
            entry: dict = {
                "id": uuid.uuid4().hex[:12],
                "scene": scene,
                "title": str(title or "").strip() or rule[:30],
                "rule": rule,
                "trigger_keywords": [str(k).strip() for k in (trigger_keywords or []) if str(k).strip()],
                "source_log_ids": sorted(set(source_log_ids or [])),
                "source_count": max(1, int(source_count or 1)),
                "hit_count": 0,
                "good": 0,
                "bad": 0,
                "enabled": True,
                "created_at": time.time(),
                "updated_at": time.time(),
                "last_used_at": 0.0,
            }
            self._strategies.append(entry)
            self._save()
            return entry

    def hit(self, strategy_id: str) -> None:
        """策略被自动检索引用（实战命中计数）。"""
        with self._lock:
            for s in self._strategies:
                if s.get("id") == strategy_id:
                    s["hit_count"] = int(s.get("hit_count") or 0) + 1
                    s["last_used_at"] = time.time()
                    self._save()
                    return

    def feedback(self, strategy_id: str, good: bool) -> bool:
        """效果反馈：good=True 记为有效（下次同场景加分），False 记为失效（降权）。"""
        with self._lock:
            for s in self._strategies:
                if s.get("id") == strategy_id:
                    key = "good" if good else "bad"
                    s[key] = int(s.get(key) or 0) + 1
                    s["updated_at"] = time.time()
                    self._save()
                    return True
        return False

    def set_enabled(self, strategy_id: str, enabled: bool) -> bool:
        with self._lock:
            for s in self._strategies:
                if s.get("id") == strategy_id:
                    s["enabled"] = bool(enabled)
                    s["updated_at"] = time.time()
                    self._save()
                    return True
        return False

    # ---------- 读取 ----------
    def get(self, strategy_id: str) -> Optional[dict]:
        for s in self._strategies:
            if s.get("id") == strategy_id:
                return s
        return None

    def list(self, scene: Optional[str] = None) -> list[dict]:
        ls = self._strategies
        if scene:
            ls = [s for s in ls if s.get("scene") == scene]
        return list(ls)

    def snapshot(self) -> dict:
        """策略库完整快照（供展示/管理）。"""
        return {"path": str(self.path), "count": len(self._strategies),
                "strategies": list(self._strategies)}
