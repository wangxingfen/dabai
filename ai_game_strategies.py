"""AI 游戏策略系统 —— 让 AI 在不同游戏中拥有不同的目标与决策逻辑。

设计理念：
- 每种游戏有独立的目标（MOBA=推塔拿人头，寻宝=找宝藏，沙盒=自由探索）
- 策略类封装游戏专属的决策树，复用统一的 BehaviorDecision 产出
- 行为引擎根据 scene_type 分发到对应策略；策略返回 None 时回退到默认好奇心逻辑
- 策略可访问感知引擎的原始快照数据（英雄/塔/野怪/宝藏等游戏特定信息）

策略优先级（高→低）：
1. 立即行动（check_immediate_action）：危险/机遇需打断当前行为
2. 游戏专属决策（decide）：基于游戏目标的战术决策
3. 默认好奇心决策：探索/漫步/小动作（由行为引擎处理）
"""

from __future__ import annotations

import math
import time
import logging
from typing import Optional

from ai_behavior_engine import BehaviorDecision, BehaviorType

logger = logging.getLogger("ai_strategy")


# ==================== 策略基类 ====================

class GameStrategy:
    """游戏策略基类。

    子类实现 decide() 和 check_immediate_action()，
    返回 BehaviorDecision 或 None（回退到默认逻辑）。
    """

    # 策略决策冷却（秒），避免 AI 话太多
    DECISION_COOLDOWN: float = 8.0
    # 立即行动冷却（秒）
    IMMEDIATE_COOLDOWN: float = 5.0
    # 是否抑制默认好奇心立即行动（MOBA 等竞技游戏不需要"扑向 POI"逻辑）
    SUPPRESS_DEFAULT_IMMEDIATE: bool = False
    # 是否抑制默认好奇心决策（MOBA 不需要"探索 POI"逻辑，完全由策略驱动）
    SUPPRESS_DEFAULT_DECIDE: bool = False

    def __init__(self, perception):
        self.perception = perception
        self._last_decision_time: float = 0
        self._last_immediate_time: float = 0
        # 跨决策记忆：追踪上次说过的话，避免重复
        self._last_speak_topic: str = ""
        self._last_speak_time: float = 0

    def decide(self, context) -> Optional[BehaviorDecision]:
        """游戏专属决策。返回 None 则回退到默认好奇心逻辑。"""
        return None

    def check_immediate_action(self) -> Optional[BehaviorDecision]:
        """立即行动检查（可打断当前行为）。返回 None 表示无紧急事项。"""
        return None

    def _can_decide(self) -> bool:
        """是否可做新决策（冷却检查）。"""
        now = time.time()
        if now - self._last_decision_time < self.DECISION_COOLDOWN:
            return False
        self._last_decision_time = now
        return True

    def _can_immediate(self) -> bool:
        now = time.time()
        if now - self._last_immediate_time < self.IMMEDIATE_COOLDOWN:
            return False
        self._last_immediate_time = now
        return True

    def _make_commentary(self, reason: str, speak_text: str, confidence: float = 0.7) -> BehaviorDecision:
        """生成纯解说决策（不移动，只说话）。"""
        self._last_speak_topic = reason
        self._last_speak_time = time.time()
        return BehaviorDecision(
            behavior=BehaviorType.IDLE_ACTION,
            reason=reason,
            speak_text=speak_text,
            confidence=confidence,
        )

    def _make_suggest_move(self, reason: str, speak_text: str,
                           x: float, z: float, label: str,
                           confidence: float = 0.75) -> BehaviorDecision:
        """生成建议移动决策。"""
        self._last_speak_topic = reason
        self._last_speak_time = time.time()
        return BehaviorDecision(
            behavior=BehaviorType.GO_TO_POI,
            reason=reason,
            target_x=x, target_z=z, target_label=label,
            speak_text=speak_text,
            confidence=confidence,
        )

    def reset(self):
        self._last_decision_time = 0
        self._last_immediate_time = 0
        self._last_speak_topic = ""
        self._last_speak_time = 0


# ==================== 大厅/沙盒策略（默认，回退到好奇心逻辑）====================

class LobbyStrategy(GameStrategy):
    """大厅策略：不做游戏专属决策，完全回退到好奇心驱动的探索逻辑。"""

    def decide(self, context) -> Optional[BehaviorDecision]:
        return None


class SandboxStrategy(GameStrategy):
    """沙盒策略：同大厅，好奇心驱动。"""

    def decide(self, context) -> Optional[BehaviorDecision]:
        return None


# ==================== 寻宝策略 ====================

class TreasureHuntStrategy(GameStrategy):
    """寻宝游戏策略：目标是找到所有宝藏/收集品。

    决策树：
    1. 有未收集的宝藏 → 前往最近的
    2. 刚收集到宝藏 → 庆祝
    3. 无目标 → 回退到好奇心探索
    """

    DECISION_COOLDOWN = 6.0

    def decide(self, context) -> Optional[BehaviorDecision]:
        if not self._can_decide():
            return None

        progress = self.perception.get_game_progress()
        objects = self.perception.get_game_objects()
        player = self.perception.get_player_state()

        # 优先级 1：寻找未收集的宝藏
        uncollected = self._find_uncollected_treasures(objects)
        if uncollected:
            target = uncollected[0]  # 最近的
            px = player.get("x", 0)
            pz = player.get("z", 0)
            collected = progress.get("collected", 0)
            total = progress.get("total_collectibles", 0)
            return self._make_suggest_move(
                reason=f"寻宝: 前往{target['label']}",
                speak_text=(
                    f"（你看到{target.get('direction', '附近')}约{target['distance']:.0f}米处有个"
                    f"{target['label']}！进度{collected}/{total}。表达兴奋并建议一起去拿。）"
                ),
                x=target["x"], z=target["z"],
                label=target["label"],
            )

        # 优先级 2：检查是否刚收集到（进度变化）
        if progress.get("treasure_found"):
            return self._make_commentary(
                reason="宝藏找到",
                speak_text="（太棒了！我们找到宝藏了！表达你的喜悦，说说这个宝藏看起来怎么样。）",
            )

        return None  # 回退到好奇心逻辑

    def check_immediate_action(self) -> Optional[BehaviorDecision]:
        """宝藏就在眼前 → 立刻去拿。"""
        if not self._can_immediate():
            return None
        objects = self.perception.get_game_objects()
        uncollected = self._find_uncollected_treasures(objects)
        if uncollected and uncollected[0]["distance"] < 5:
            target = uncollected[0]
            return self._make_suggest_move(
                reason=f"立刻拿取: {target['label']}",
                speak_text=f"（就在眼前！{target['label']}只有{target['distance']:.0f}米，快去拿！）",
                x=target["x"], z=target["z"],
                label=target["label"],
                confidence=0.95,
            )
        return None

    def _find_uncollected_treasures(self, objects: dict) -> list[dict]:
        """从游戏对象中找出未收集的宝藏，按距离排序。"""
        player = self.perception.get_player_state()
        px, pz = player.get("x", 0), player.get("z", 0)
        treasures = []
        for obj_type, obj_list in objects.items():
            if not isinstance(obj_list, list):
                continue
            for obj in obj_list:
                if obj.get("collected") or obj.get("found"):
                    continue
                # 识别宝藏/收集品类型
                label = obj.get("name", obj.get("id", obj_type))
                ot = obj_type.lower()
                if any(k in ot for k in ("treasure", "collectible", "clue", "item", "star")) or \
                   any(k in str(label).lower() for k in ("宝藏", "碎片", "星星", "线索", "宝箱")):
                    dx = obj.get("x", 0) - px
                    dz = obj.get("z", 0) - pz
                    dist = math.hypot(dx, dz)
                    treasures.append({
                        "x": obj.get("x", 0), "z": obj.get("z", 0),
                        "distance": dist, "label": label,
                        "direction": self._calc_direction(dx, dz, player.get("facing", 0)),
                    })
        treasures.sort(key=lambda t: t["distance"])
        return treasures

    @staticmethod
    def _calc_direction(dx: float, dz: float, facing: float) -> str:
        cosA, sinA = math.cos(-facing), math.sin(-facing)
        rx = dx * cosA - dz * sinA
        rz = dx * sinA + dz * cosA
        deg = (math.atan2(rx, rz) * 180 / math.pi + 360) % 360
        dirs = ["正前方", "右前方", "右方", "右后方", "正后方", "左后方", "左方", "左前方"]
        idx = int((deg + 22.5) // 45) % 8
        return dirs[idx]


# ==================== MOBA 推塔策略 ====================

class MobaStrategy(GameStrategy):
    """MOBA 5v5 推塔策略：目标是推塔、拿人头、赢比赛。

    AI 作为战术伙伴（一体双魂），感知全局战局并给出战术建议/解说。

    决策树（优先级高→低）：
    1. 玩家血量危险 + 敌人靠近 → 警告撤退
    2. 敌方英雄残血 + 玩家附近 → 建议追击
    3. 敌方塔被推/可推 → 建议推塔
    4. 暴君/主宰可用 → 建议拿龙
    5. 团战爆发（多人聚集）→ 团战解说
    6. 兵线压力 → 建议清线
    7. 默认 → 局势解说（比分、经济、装备）
    """

    DECISION_COOLDOWN = 10.0       # 解说间隔
    IMMEDIATE_COOLDOWN = 6.0       # 紧急警告间隔
    SUPPRESS_DEFAULT_IMMEDIATE = True  # MOBA 不使用"扑向 POI"逻辑
    SUPPRESS_DEFAULT_DECIDE = True     # MOBA 完全由策略驱动，不回退到好奇心探索
    DANGER_HP_RATIO = 0.3          # 血量低于30%视为危险
    KILL_HP_RATIO = 0.4            # 敌人血量低于40%可追击
    TEAMFIGHT_RADIUS = 10.0        # 团战判定半径
    OBJECTIVE_DIST = 8.0           # 目标可达距离

    def __init__(self, perception):
        super().__init__(perception)
        # 追踪上次比分，检测击杀事件
        self._last_blue_kills: int = 0
        self._last_red_kills: int = 0
        self._last_tower_count: dict = {"blue": 0, "red": 0}
        self._last_player_state: str = ""

    def decide(self, context) -> Optional[BehaviorDecision]:
        if not self._can_decide():
            return None

        progress = self.perception.get_game_progress()
        objects = self.perception.get_game_objects()
        player = self.perception.get_player_state()

        # 解析游戏状态
        state = self._analyze_moba_state(objects, player, progress)

        # 优先级 1：玩家危险 → 警告撤退
        decision = self._check_player_danger(state, player)
        if decision:
            return decision

        # 优先级 2：敌方残血 → 建议追击
        decision = self._check_enemy_low_hp(state, player)
        if decision:
            return decision

        # 优先级 3：可推塔 → 建议推塔
        decision = self._check_push_tower(state, player)
        if decision:
            return decision

        # 优先级 4：拿龙（暴君/主宰）
        decision = self._check_objective(state, player)
        if decision:
            return decision

        # 优先级 5：团战解说
        decision = self._check_teamfight(state, player)
        if decision:
            return decision

        # 优先级 6：击杀事件反应
        decision = self._check_kill_event(state, progress)
        if decision:
            return decision

        # 优先级 7：默认局势解说
        return self._default_commentary(state, progress)

    def check_immediate_action(self) -> Optional[BehaviorDecision]:
        """紧急情况：玩家极度危险或有大好机会。"""
        if not self._can_immediate():
            return None

        objects = self.perception.get_game_objects()
        player = self.perception.get_player_state()
        progress = self.perception.get_game_progress()
        state = self._analyze_moba_state(objects, player, progress)

        # 玩家血量极低 + 敌人很近 → 紧急撤退警告
        if state["player_hp_ratio"] < 0.2 and state["nearest_enemy_dist"] < 8:
            return self._make_commentary(
                reason="紧急撤退警告",
                speak_text=(
                    f"（危险！血量只剩{state['player_hp_ratio']*100:.0f}%，"
                    f"敌人{state['nearest_enemy_name']}只有{state['nearest_enemy_dist']:.0f}米！"
                    f"立刻警告玩家撤退回城，语气要急切！）"
                ),
                confidence=0.95,
            )

        # 敌方主水晶血量极低 → 冲刺推水晶
        if state.get("enemy_main_hp_ratio", 1) < 0.2:
            return self._make_commentary(
                reason="冲刺推水晶",
                speak_text=(
                    f"（敌方水晶只剩{state['enemy_main_hp_ratio']*100:.0f}%血量！"
                    f"这是最后冲刺，全力推水晶！表达极致的兴奋和急迫感。）"
                ),
                confidence=0.95,
            )

        return None

    # ==================== 状态分析 ====================

    def _analyze_moba_state(self, objects: dict, player: dict, progress: dict) -> dict:
        """从游戏对象中提取 MOBA 战局状态。"""
        px = player.get("x", 0)
        pz = player.get("z", 0)
        player_team = "blue"  # 玩家固定蓝方
        enemy_team = "red"

        state = {
            "player_hp": player.get("hp", 0),
            "player_max_hp": player.get("max_hp", 1),
            "player_hp_ratio": player.get("hp", 0) / max(1, player.get("max_hp", 1)),
            "player_level": player.get("level", 1),
            "player_gold": player.get("gold", 0),
            "player_hero": player.get("hero", ""),
            "match_time": progress.get("match_time", 0),
            "blue_kills": progress.get("blue_kills", 0),
            "red_kills": progress.get("red_kills", 0),
            "blue_towers": progress.get("blue_towers", 0),
            "red_towers": progress.get("red_towers", 0),
            "blue_main_hp": progress.get("blue_main_hp", 0),
            "red_main_hp": progress.get("red_main_hp", 0),
            "nearest_enemy": None,
            "nearest_enemy_dist": 999,
            "nearest_enemy_name": "",
            "low_hp_enemies": [],
            "enemy_main_hp_ratio": 1,
            "ally_main_hp_ratio": 1,
            "teamfight": None,
        }

        # 分析英雄
        heroes = objects.get("heroes", [])
        if isinstance(heroes, list):
            for h in heroes:
                if not isinstance(h, dict):
                    continue
                if h.get("team") == enemy_team and h.get("alive", True):
                    dx = h.get("x", 0) - px
                    dz = h.get("z", 0) - pz
                    d = math.hypot(dx, dz)
                    if d < state["nearest_enemy_dist"]:
                        state["nearest_enemy_dist"] = d
                        state["nearest_enemy"] = h
                        state["nearest_enemy_name"] = h.get("name", "敌方英雄")
                    hp_ratio = h.get("hp", 0) / max(1, h.get("max_hp", 1))
                    if hp_ratio < self.KILL_HP_RATIO and d < 12:
                        state["low_hp_enemies"].append({
                            "name": h.get("name", "敌方英雄"),
                            "hp_ratio": hp_ratio, "distance": d,
                            "x": h.get("x", 0), "z": h.get("z", 0),
                        })

        # 分析塔
        structures = objects.get("structures", [])
        if isinstance(structures, list):
            enemy_towers_low = []
            for s in structures:
                if not isinstance(s, dict) or s.get("team") != enemy_team:
                    continue
                hp_ratio = s.get("hp", 0) / max(1, s.get("max_hp", 1))
                if s.get("kind") == "main":
                    state["enemy_main_hp_ratio"] = hp_ratio
                elif hp_ratio < 0.3:
                    dx = s.get("x", 0) - px
                    dz = s.get("z", 0) - pz
                    d = math.hypot(dx, dz)
                    enemy_towers_low.append({
                        "id": s.get("id", ""), "hp_ratio": hp_ratio,
                        "distance": d, "x": s.get("x", 0), "z": s.get("z", 0),
                        "kind": s.get("kind", "tower"),
                    })
            state["enemy_towers_low"] = enemy_towers_low
            # 己方主水晶血量
            for s in structures:
                if isinstance(s, dict) and s.get("team") == player_team and s.get("kind") == "main":
                    state["ally_main_hp_ratio"] = s.get("hp", 0) / max(1, s.get("max_hp", 1))
        else:
            state["enemy_towers_low"] = []

        # 分析野怪（暴君/主宰）
        jungle = objects.get("jungle", [])
        state["objectives"] = []
        if isinstance(jungle, list):
            for j in jungle:
                if not isinstance(j, dict):
                    continue
                jtype = j.get("type", "")
                if jtype in ("tyrant", "overlord"):
                    dx = j.get("x", 0) - px
                    dz = j.get("z", 0) - pz
                    state["objectives"].append({
                        "type": jtype,
                        "name": "暴君" if jtype == "tyrant" else "主宰",
                        "distance": math.hypot(dx, dz),
                        "x": j.get("x", 0), "z": j.get("z", 0),
                        "hp_ratio": j.get("hp", 0) / max(1, j.get("max_hp", 1)),
                    })

        # 团战检测：附近有 >=3 个英雄
        nearby_heroes = []
        if isinstance(heroes, list):
            for h in heroes:
                if not isinstance(h, dict) or not h.get("alive", True):
                    continue
                dx = h.get("x", 0) - px
                dz = h.get("z", 0) - pz
                d = math.hypot(dx, dz)
                if d < self.TEAMFIGHT_RADIUS:
                    nearby_heroes.append({"team": h.get("team", ""), "name": h.get("name", ""), "dist": d})
        if len(nearby_heroes) >= 3:
            state["teamfight"] = {
                "total": len(nearby_heroes),
                "enemies": sum(1 for h in nearby_heroes if h["team"] == enemy_team),
                "allies": sum(1 for h in nearby_heroes if h["team"] == player_team),
            }

        return state

    # ==================== 决策分支 ====================

    def _check_player_danger(self, state: dict, player: dict) -> Optional[BehaviorDecision]:
        """玩家血量危险 + 敌人靠近 → 警告撤退。"""
        if state["player_hp_ratio"] >= self.DANGER_HP_RATIO:
            return None
        if state["nearest_enemy_dist"] > 8:
            return None
        return self._make_commentary(
            reason="血量危险警告",
            speak_text=(
                f"（注意！你的血量只剩{state['player_hp_ratio']*100:.0f}%，"
                f"敌方{state['nearest_enemy_name']}在{state['nearest_enemy_dist']:.0f}米外。"
                f"建议先撤退回城补血，别硬拼。用关切的语气提醒玩家。）"
            ),
            confidence=0.85,
        )

    def _check_enemy_low_hp(self, state: dict, player: dict) -> Optional[BehaviorDecision]:
        """敌方残血 → 建议追击。"""
        if not state.get("low_hp_enemies"):
            return None
        if state["player_hp_ratio"] < 0.4:
            return None  # 自己也残血，不追
        target = state["low_hp_enemies"][0]
        return self._make_suggest_move(
            reason=f"追击残血: {target['name']}",
            speak_text=(
                f"（敌方{target['name']}只剩{target['hp_ratio']*100:.0f}%血量，"
                f"距离{target['distance']:.0f}米！好机会，建议追击拿人头！兴奋地鼓励玩家。）"
            ),
            x=target["x"], z=target["z"],
            label=f"残血{target['name']}",
            confidence=0.8,
        )

    def _check_push_tower(self, state: dict, player: dict) -> Optional[BehaviorDecision]:
        """敌方塔残血 → 建议推塔。"""
        towers = state.get("enemy_towers_low", [])
        if not towers:
            return None
        # 选最近的残血塔
        towers.sort(key=lambda t: t["distance"])
        t = towers[0]
        if t["distance"] > 15:
            return None
        return self._make_suggest_move(
            reason=f"推塔: {t['id']}",
            speak_text=(
                f"（敌方{t['kind']}只剩{t['hp_ratio']*100:.0f}%血量！"
                f"距离{t['distance']:.0f}米，快去推掉它！这能为团队带来经济优势。）"
            ),
            x=t["x"], z=t["z"],
            label=f"残血{t['kind']}",
            confidence=0.78,
        )

    def _check_objective(self, state: dict, player: dict) -> Optional[BehaviorDecision]:
        """暴君/主宰可用 → 建议拿龙。"""
        if not state.get("objectives"):
            return None
        # 选最近的目标
        obj = min(state["objectives"], key=lambda o: o["distance"])
        if obj["distance"] > 15:
            return None
        # 比赛时间 < 2分钟不拿龙（前期太危险）
        if state["match_time"] < 120:
            return None
        return self._make_suggest_move(
            reason=f"拿龙: {obj['name']}",
            speak_text=(
                f"（{obj['name']}在{obj['distance']:.0f}米外，血量{obj['hp_ratio']*100:.0f}%。"
                f"现在是个拿{obj['name']}的好时机，全队增益很大！建议集合打龙。）"
            ),
            x=obj["x"], z=obj["z"],
            label=obj["name"],
            confidence=0.72,
        )

    def _check_teamfight(self, state: dict, player: dict) -> Optional[BehaviorDecision]:
        """团战爆发 → 团战解说。"""
        tf = state.get("teamfight")
        if not tf:
            return None
        ally = tf["allies"]
        enemy = tf["enemies"]
        if ally > enemy:
            msg = (f"（团战爆发！附近有{tf['total']}个英雄，我方{ally}人敌方{enemy}人，"
                   f"人数优势！建议果断开团！为玩家加油打气。）")
        elif ally < enemy:
            msg = (f"（团战爆发！附近有{tf['total']}个英雄，但我方只有{ally}人，"
                   f"敌方{enemy}人，人数劣势！建议先撤退等人集合。提醒玩家小心。）")
        else:
            msg = (f"（团战爆发！双方各{ally}人在附近交战，势均力敌！"
                   f"看操作了，建议找准时机切入。紧张地解说战况。）")
        return self._make_commentary(reason="团战解说", speak_text=msg, confidence=0.75)

    def _check_kill_event(self, state: dict, progress: dict) -> Optional[BehaviorDecision]:
        """检测击杀事件（比分变化）→ 反应。"""
        blue = state["blue_kills"]
        red = state["red_kills"]
        blue_delta = blue - self._last_blue_kills
        red_delta = red - self._last_red_kills
        self._last_blue_kills = blue
        self._last_red_kills = red

        if blue_delta > 0 and red_delta > 0:
            return self._make_commentary(
                reason="双方击杀",
                speak_text=f"（双方各拿一个人头！比分{blue}:{red}，战况激烈！点评这波交换。）",
            )
        elif blue_delta > 0:
            return self._make_commentary(
                reason="我方击杀",
                speak_text=f"（漂亮！我方拿到击杀！比分{blue}:{red}，干得好！为玩家庆祝。）",
            )
        elif red_delta > 0:
            return self._make_commentary(
                reason="敌方击杀",
                speak_text=f"（敌方拿到一个人头，比分{blue}:{red}。没关系，稳住继续打。安慰并鼓励玩家。）",
            )
        return None

    def _default_commentary(self, state: dict, progress: dict) -> Optional[BehaviorDecision]:
        """默认局势解说。"""
        m, s = divmod(int(state["match_time"]), 60)
        time_str = f"{m:02d}:{s:02d}"
        blue, red = state["blue_kills"], state["red_kills"]
        bt, rt = state["blue_towers"], state["red_towers"]

        # 经济/塔差分析
        if blue > red + 3 and bt >= rt:
            msg = (f"（比赛{time_str}，比分{blue}:{red}，我方塔{bt}敌方塔{rt}。"
                   f"我方优势！建议抱团推进，扩大优势。自信地分析局势。）")
        elif red > blue + 3 and rt >= bt:
            msg = (f"（比赛{time_str}，比分{blue}:{red}，我方塔{bt}敌方塔{rt}。"
                   f"敌方优势，需要稳住防守。建议清线发育，等待机会翻盘。冷静地分析。）")
        else:
            msg = (f"（比赛{time_str}，比分{blue}:{red}，双方塔数{bt}对{rt}。"
                   f"局势胶着，建议做好视野，找机会抓单或拿龙。客观分析当前局势。）")
        return self._make_commentary(reason="局势解说", speak_text=msg, confidence=0.6)


# ==================== 策略注册表 ====================

# 游戏 key → 策略类
_STRATEGY_REGISTRY: dict[str, type[GameStrategy]] = {
    "moba_5v5": MobaStrategy,
    "treasure_hunt": TreasureHuntStrategy,
    "treasure": TreasureHuntStrategy,
    "sandbox": SandboxStrategy,
    "lobby": LobbyStrategy,
}


def get_strategy_for_game(game_key: str) -> type[GameStrategy]:
    """根据游戏 key 获取策略类。未知游戏回退到大厅策略。"""
    return _STRATEGY_REGISTRY.get(game_key, LobbyStrategy)


def register_strategy(game_key: str, strategy_cls: type[GameStrategy]):
    """注册自定义游戏策略。"""
    _STRATEGY_REGISTRY[game_key] = strategy_cls
