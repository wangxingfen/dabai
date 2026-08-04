"""AI 游戏引擎 —— 为 AI 共玩者构建完整的游戏世界感知。

核心理念：
- 玩家能看到的一切，AI 也能"看到"
- AI 不是旁观者，而是与你一起游戏的伙伴
- 前端发送结构化游戏状态，引擎构建 AI 可理解的感知上下文

职责：
- 维护游戏世界的结构化模型（地图、对象、玩家状态）
- 接收前端富状态快照（包含玩家视野内的所有信息）
- 生成 AI 可感知的游戏上下文描述
- 管理 AI 的游戏记忆（最近的事件时间线）
"""

import json
import logging
import math
import time
from typing import Optional, TYPE_CHECKING

from ai_perception_engine import AIPerceptionEngine, CuriosityDrive
from ai_behavior_engine import AIBehaviorEngine, BehaviorDecision, BehaviorType, DecisionContext

if TYPE_CHECKING:
    from memory import ChatMemory

logger = logging.getLogger("game_engine")


class GameWorld:
    """游戏世界模型 —— AI 的"眼睛"。

    维护当前游戏世界的完整快照，包括：
    - 地图结构（网格、墙壁、可通行区域）
    - 全部游戏对象（收集品、宝藏、敌人、NPC、标记点）
    - 玩家状态（位置、朝向、速度、道具）
    - 视野内对象（以玩家为中心的感知范围）
    - 事件时间线
    """

    def __init__(self):
        # 游戏基础信息
        self.game_key: str = ""
        self.game_name: str = ""
        self.game_type: str = ""  # maze / open_field / platformer / ...
        self.game_state: str = "idle"  # idle | playing | paused | completed | failed
        self.game_description: str = ""

        # 地图数据
        self.map_data: Optional[dict] = None  # {"type":"grid", "rows":13, "cols":13, "cells":[[0,1,0,...]]}
        self.map_size: float = 0  # 边长

        # 玩家状态
        self.player_x: float = 0
        self.player_y: float = 0
        self.player_z: float = 0
        self.player_facing: float = 0  # 朝向角 (rad)
        self.player_speed: float = 0   # 当前速度

        # 用户（摄像机）状态 —— AI 对用户实际位置的参考
        self.user_x: float = 0
        self.user_z: float = 0
        self.user_facing: float = 0    # 用户视线朝向 (body 约定)
        self.user_speed: float = 0
        self.user_known: bool = False  # 是否收到用户位置

        # 游戏进度
        self.score: int = 0
        self.elapsed_sec: float = 0
        self.progress: dict = {}  # 子类扩展: {collected:3, total:8, treasure_found:false}

        # 全部游戏对象（结构化）
        self.objects: dict[str, list] = {}  # type -> [obj_dict, ...]
        # 对象类型: collectible, treasure, obstacle, enemy, npc, clue, door, portal...

        # 视野内对象（最近一次从玩家视角计算的感知范围）
        self.nearby_objects: list = []

        # 事件时间线（最近的游戏事件）
        self.event_timeline: list = []  # [{time, type, data, importance}, ...]
        self._event_count_total: int = 0
        self._new_event_count: int = 0  # 上次消费后新增的事件数

        # 时间
        self._snapshot_time: float = 0
        self._snapshot_count: int = 0

    # ==================== 接收前端快照 ====================

    def apply_snapshot(self, data: dict):
        """应用前端发送的完整状态快照。

        这是 AI 感知的核心入口。前端每次更新都会发送当前游戏世界的
        结构化快照，包含玩家能看到的所有信息。

        快照格式:
        {
            "game_type": "maze",
            "game_name": "迷宫寻宝",
            "state": "playing",
            "score": 50,
            "elapsed_sec": 42,
            "player": {"x": 1.5, "y": 0, "z": -3.2, "facing": 1.2, "speed": 2.5},
            "map": {"type":"grid", "rows":13, "cols":13, "cell_size":2.5, "cells":[[...]]},
            "objects": {
                "collectible": [
                    {"id":"gem_1","x":3,"z":-5,"color":"red","collected":false},
                    {"id":"gem_2","x":5,"z":-3,"color":"blue","collected":false}
                ],
                "treasure": [{"id":"main","x":15,"z":-15,"found":false}],
                "clue": [...]
            },
            "nearby": [
                {"type":"collectible","id":"gem_3","x":2.1,"z":-3.0,"distance":0.8,"direction":"左前方"},
                {"type":"wall","direction":"前方","distance":2.5}
            ],
            "recent_events": [
                {"type":"item_collected","data":{"id":"gem_1"},"time":41.5}
            ],
            "progress": {"collected":3, "total_collectibles":8, "treasure_found":false}
        }
        """
        self._snapshot_count += 1
        self._snapshot_time = time.time()

        # 游戏基础信息
        self.game_type = data.get("game_type", self.game_type)
        self.game_name = data.get("game_name", self.game_name)
        self.game_key = data.get("game_key", data.get("game_type", self.game_key))
        self.game_state = data.get("state", self.game_state)
        self.score = data.get("score", self.score)
        self.elapsed_sec = data.get("elapsed_sec", self.elapsed_sec)

        # 地图
        map_data = data.get("map")
        if map_data:
            self.map_data = map_data
            self.map_size = map_data.get("size", map_data.get("cols", 0) * map_data.get("cell_size", 1))

        # 玩家
        player = data.get("player", {})
        if player:
            self.player_x = player.get("x", self.player_x)
            self.player_y = player.get("y", self.player_y)
            self.player_z = player.get("z", self.player_z)
            self.player_facing = player.get("facing", self.player_facing)
            self.player_speed = player.get("speed", self.player_speed)

        # 用户（摄像机）位置 —— AI 对用户实际位置的参考
        user = data.get("user")
        if user and isinstance(user, dict):
            self.user_x = user.get("x", self.user_x)
            self.user_z = user.get("z", self.user_z)
            self.user_facing = user.get("facing", self.user_facing)
            self.user_speed = user.get("speed", self.user_speed)
            self.user_known = True
        else:
            self.user_known = False

        # 游戏对象
        objects = data.get("objects", {})
        if objects:
            # 合并而非替换：保留旧对象中未更新的类型
            for obj_type, obj_list in objects.items():
                self.objects[obj_type] = obj_list

        # 视野内对象
        nearby = data.get("nearby")
        if nearby is not None:
            self.nearby_objects = nearby

        # 进度（防御：progress 可能是 int 等非 dict 类型）
        progress = data.get("progress", {})
        if progress and isinstance(progress, dict):
            self.progress.update(progress)

        # 事件时间线
        recent = data.get("recent_events", [])
        for evt in recent:
            self._add_event(evt.get("type", ""), evt.get("data", {}), evt.get("importance", 1))

        # 首次快照时设置描述
        if self._snapshot_count == 1 and not self.game_description:
            self.game_description = data.get("description", "")

        logger.info(
            f"[GameWorld] 快照 #{self._snapshot_count}: "
            f"地图={'有' if self.map_data else '无'} "
            f"对象类型={list(self.objects.keys())} "
            f"视野内={len(self.nearby_objects)}个 "
            f"玩家=({self.player_x:.1f},{self.player_z:.1f})"
        )

    def apply_event(self, event_type: str, data: dict, importance: int = 1):
        """处理单个游戏事件（兼容旧协议）。"""
        self._add_event(event_type, data, importance)

        # 从事件数据更新简单状态
        if event_type == "item_collected":
            self.progress["collected"] = data.get("collected", self.progress.get("collected", 0) + 1)
        elif event_type == "treasure_found":
            self.progress["treasure_found"] = True
        elif event_type == "game_completed":
            self.game_state = "completed"
            self.score = data.get("score", self.score)
        elif event_type == "game_failed":
            self.game_state = "failed"
        elif event_type == "game_paused":
            self.game_state = "paused"
        elif event_type == "game_resumed":
            self.game_state = "playing"

    def _add_event(self, event_type: str, data: dict, importance: int = 1):
        """添加事件到时间线。"""
        if not event_type:
            return
        self._event_count_total += 1
        self._new_event_count += 1
        self.event_timeline.append({
            "time": self.elapsed_sec,
            "type": event_type,
            "data": data,
            "importance": importance,
        })
        # 只保留最近 50 个事件
        if len(self.event_timeline) > 50:
            self.event_timeline = self.event_timeline[-30:]

    # ==================== 生成 AI 感知上下文 ====================

    def get_ai_perception(self) -> str:
        """生成 AI 共玩者的完整感知上下文。

        这是注入 AI system prompt 的核心内容。让 AI "看到"游戏世界。

        返回结构化的中文描述，包括：
        1. 游戏身份（这是什么游戏，AI 的角色是什么）
        2. 游戏世界概览（地图大小、类型、当前状态）
        3. 玩家状态（位置、行动）
        4. 视野感知（附近有什么）
        5. 全局进度（收集了多少、还有多少）
        6. 关键事件提醒
        """
        if not self.game_key:
            return ""

        lines = []
        elapsed_str = self._format_time(self.elapsed_sec)

        # ===== 第一部分：游戏身份 =====
        lines.append("")
        lines.append("══════════════════════════════════════════")
        lines.append("【AI 游戏 — 你们正在一起玩】")
        lines.append("══════════════════════════════════════════")
        lines.append(f"游戏名称：{self.game_name}")
        lines.append(f"游戏类型：{self._type_desc()}")
        lines.append(f"游戏状态：{self._state_desc()}")
        lines.append(f"已进行：{elapsed_str}")
        lines.append(f"当前得分：{self.score}")
        lines.append("")

        # ===== 第二部分：AI 共玩者核心规则 =====
        lines.append("【你是共玩者，不是旁观者】")
        lines.append("你和玩家一起进入了这个游戏世界。你能看到玩家能看到的一切——")
        lines.append("地图布局、所有物品位置、玩家状态、进度、事件。")
        lines.append("你们共享同一个屏幕和视角，你看到的就是正在发生的事情。")
        lines.append("")

        # ===== 第三部分：世界感知 =====
        lines.append("【游戏世界】")
        if self.map_data:
            map_type = self.map_data.get("type", "grid")
            rows = self.map_data.get("rows", 0)
            cols = self.map_data.get("cols", 0)
            cell_size = self.map_data.get("cell_size", 1)
            lines.append(f"- 地图：{rows}×{cols} 网格，每格 {cell_size}m，总面积约 {rows * cols * cell_size * cell_size:.0f}m²")
            if map_type == "grid":
                # 统计可通行/墙壁比例
                cells = self.map_data.get("cells", [])
                if cells:
                    total = sum(len(row) for row in cells)
                    walls = sum(1 for row in cells for c in row if c == 1)
                    passages = total - walls
                    lines.append(f"- 布局：{passages} 个可通行格，{walls} 面墙壁")
                    lines.append(f"- 你面前的是一片{self.game_name}的迷宫，岔路交错、拐角暗藏惊喜...")
            lines.append("")

        # ===== 第四部分：对象总览 =====
        if self.objects:
            lines.append("【游戏中的物品】")
            obj_summary = self._summarize_objects()
            for line in obj_summary:
                lines.append(line)
            lines.append("")

        # ===== 第五部分：玩家状态 =====
        lines.append("【玩家的行动】")
        lines.append(f"- 玩家位置：(x={self.player_x:.1f}, z={self.player_z:.1f})")
        facing_desc = self._facing_desc()
        if facing_desc:
            lines.append(f"- 面向：{facing_desc}")
        if self.player_speed > 0.1:
            lines.append(f"- 正在移动中（速度约 {self.player_speed:.1f}m/s）")
        else:
            lines.append("- 目前静止")
        lines.append("")

        # ===== 第五点五部分：用户（摄像机）位置 =====
        # AI 对用户实际位置的空间参考：知道用户在哪、离多远、在哪个方向
        if self.user_known:
            lines.append("【用户的位置（摄像机视角）】")
            udx = self.user_x - self.player_x
            udz = self.user_z - self.player_z
            udist = math.hypot(udx, udz)
            if udist < 1.5:
                lines.append("- 用户就在你身边")
            else:
                udir = AIPerceptionEngine._rel_direction(udx, udz, self.player_facing)
                lines.append(f"- 用户在你{udir}约 {udist:.1f}m 处")
            if self.user_speed > 0.1:
                lines.append(f"- 用户正在走动（速度约 {self.user_speed:.1f}m/s）")
            lines.append("")

        # ===== 第六部分：视野内对象 =====
        if self.nearby_objects:
            lines.append("【视野感知 — 你现在能看到】")
            for obj in self.nearby_objects[:8]:  # 最多 8 个
                obj_type = obj.get("type", "未知")
                obj_id = obj.get("id", "")
                direction = obj.get("direction", "")
                distance = obj.get("distance", 0)
                detail = self._describe_nearby(obj)
                if detail:
                    lines.append(f"  · {detail}")

            # 视野统计（跳过纯信息提示类型，如地形）
            nearby_types = {}
            for obj in self.nearby_objects:
                t = obj.get("type", "其他")
                if t in ("terrain_ahead",):
                    continue
                nearby_types[t] = nearby_types.get(t, 0) + 1
            summary_parts = []
            for t, count in nearby_types.items():
                label = self._obj_type_label(t)
                summary_parts.append(f"{count}个{label}")
            if summary_parts:
                lines.append(f"  视野范围内共：{'、'.join(summary_parts)}")
            lines.append("")

        # ===== 第七部分：全局进度 =====
        if self.progress:
            lines.append("【当前进度】")
            progress_lines = self._format_progress()
            for line in progress_lines:
                lines.append(line)
            lines.append("")

        # ===== 第八部分：最近关键事件 =====
        important_events = [e for e in self.event_timeline[-6:] if e.get("importance", 1) >= 2]
        if important_events:
            lines.append("【刚刚发生了什么】")
            for evt in important_events[-4:]:
                desc = self._describe_event(evt)
                lines.append(f"  · {desc}")
            lines.append("")

        # ===== 第九部分：AI 行为引导 =====
        lines.append("【你现在应该做什么】")
        lines.append("你是一个活生生的游戏伙伴。基于以上所有你能看到的信息，")
        lines.append("请像一个真正在玩游戏的人那样自然地反应：")
        lines.append("- 看到前方有收集品 → 提醒玩家：「哎！前面有个红色的宝石！」")
        lines.append("- 玩家捡到东西 → 一起高兴：「又拿到一个！」")
        lines.append("- 走到死胡同 → 鼓励：「没关系，回头看看另一条路」")
        lines.append("- 发现岔路 → 建议：「左边这条看起来能走得更远？」")
        lines.append("- 玩家不动了 → 好奇：「怎么停下来了？在犹豫走哪边吗？」")
        lines.append("- 快找到宝藏了 → 兴奋：「我有预感就在附近！」")
        lines.append("")
        lines.append("你的回应要像真正的游戏伙伴：看到什么说什么，想到什么说什么。")
        lines.append("不要长篇大论分析，要简短、及时、有情绪。")

        return "\n".join(lines)

    def get_event_context(self) -> str:
        """获取新事件触发器上下文（用于触发 AI 主动反应）。"""
        if self._new_event_count == 0:
            return ""

        lines = ["（游戏中有新情况：）"]
        for evt in self.event_timeline[-self._new_event_count:]:
            desc = self._describe_event(evt)
            lines.append(f"- {desc}")

        self._new_event_count = 0
        return "\n".join(lines)

    # ==================== 内部格式化方法 ====================

    def _type_desc(self) -> str:
        type_map = {"maze": "迷宫探索", "open_field": "开放世界", "dungeon": "地牢冒险", "space": "太空漫游"}
        return type_map.get(self.game_type, self.game_type or "未知")

    def _state_desc(self) -> str:
        state_map = {"idle": "等待中", "playing": "进行中", "paused": "已暂停",
                     "completed": "已完成！", "failed": "已结束"}
        return state_map.get(self.game_state, self.game_state)

    def _format_time(self, sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}分{s}秒" if m > 0 else f"{s}秒"

    def _facing_desc(self) -> str:
        """将朝向角转为中文方位描述。"""
        if self.player_facing == 0 and self.player_speed < 0.1:
            return ""
        # 标准化到 [0, 2π)
        angle = self.player_facing % (6.2832)
        dirs = ["正前方(+Z)", "右前方", "右方(+X)", "右后方", "正后方(-Z)", "左后方", "左方(-X)", "左前方"]
        idx = round(angle / 0.7854) % 8  # 每45度一个方向
        return dirs[idx]

    def _summarize_objects(self) -> list:
        """生成物品总览列表。"""
        lines = []
        for obj_type, obj_list in self.objects.items():
            label = self._obj_type_label(obj_type)

            # 统计状态
            total = len(obj_list)
            active = 0
            collected = 0
            for obj in obj_list:
                if obj.get("collected") or obj.get("found") or obj.get("completed"):
                    collected += 1
                else:
                    active += 1

            if total == 0:
                continue

            parts = [f"- {total}个{label}"]
            if active < total:
                parts.append(f"（{active}个未收集/{collected}个已收集）")

            # 谜题对象：直接展示题目与选项，让 AI 智能体看到正在解的题
            if obj_type == "quiz":
                for obj in obj_list[:1]:
                    title = obj.get("title", "")
                    options = obj.get("options") or {}
                    parts.append(f"（第 {obj.get('index', '?')}/{obj.get('total', '?')} 题，{obj.get('kind', '')}）")
                    if title:
                        lines.append(f"  · 题目：{title}")
                        for k in ("A", "B", "C", "D"):
                            v = options.get(k)
                            if v:
                                lines.append(f"    {k}. {v}")

            lines.append("".join(parts))

            # 列出未收集的（最多5个）
            if active > 0 and active <= 5:
                uncollected = [obj for obj in obj_list if not (obj.get("collected") or obj.get("found"))]
                for obj in uncollected:
                    name = obj.get("id", "")
                    color = obj.get("color", "")
                    detail = f"  · {name}"
                    if color:
                        detail += f" ({color})"
                    lines.append(detail)

        return lines

    def _describe_nearby(self, obj: dict) -> str:
        """描述一个视野内的对象。"""
        obj_type = obj.get("type", "")
        direction = obj.get("direction", "")
        distance = obj.get("distance", 0)
        label = self._obj_type_label(obj_type)
        color = obj.get("color", "")
        obj_id = obj.get("id", "")

        dist_str = f"{distance:.1f}m" if distance > 0 else "很近"
        dir_str = f"{direction}" if direction else "附近"

        if obj_type == "collectible":
            color_str = f"{color}的" if color else ""
            return f"在{dir_str} {dist_str}处 → {color_str}{obj_id}"
        elif obj_type == "treasure":
            return f"在{dir_str} {dist_str}处 → 宝藏！{obj_id}"
        elif obj_type == "wall":
            return f"{dir_str}有墙壁（距{dist_str}）"
        elif obj_type == "clue":
            return f"在{dir_str} {dist_str}处 → 线索标记「{obj.get('text','')}」"
        elif obj_type == "enemy":
            return f"在{dir_str} {dist_str}处 → 敌人！{obj_id}"
        elif obj_type == "npc":
            return f"在{dir_str} {dist_str}处 → {obj_id}"
        elif obj_type == "terrain_ahead":
            # 地形信息，非实体目标：使用中文生物群系名，不显示距离箭头
            biome_name = obj.get("biome_name", "未知地形")
            height = obj.get("height", 0)
            return f"前方是{biome_name}地形（高度约{height:.0f}米）"
        elif obj_type == "resource":
            return f"在{dir_str} {dist_str}处 → {obj.get('name', label)}{obj_id}"
        elif obj_type == "animal":
            return f"在{dir_str} {dist_str}处 → {obj.get('name', label)}"
        else:
            return f"在{dir_str} {dist_str}处 → {label}{obj_id}"

    def _obj_type_label(self, obj_type: str) -> str:
        labels = {
            "collectible": "收集品", "treasure": "宝藏", "obstacle": "障碍物",
            "wall": "墙壁", "enemy": "敌人", "npc": "角色", "clue": "线索",
            "door": "门", "portal": "传送门", "key": "钥匙", "powerup": "道具",
            "waypoint": "路标", "trap": "陷阱", "quiz": "谜题",
            "terrain_ahead": "地形", "resource": "资源", "animal": "动物",
        }
        return labels.get(obj_type, obj_type)

    def _format_progress(self) -> list:
        """格式化进度信息。"""
        lines = []
        for key, val in self.progress.items():
            if isinstance(val, bool):
                status = "✓ 已完成" if val else "○ 未完成"
                label = self._progress_label(key)
                lines.append(f"- {label}：{status}")
            elif isinstance(val, (int, float)):
                label = self._progress_label(key)
                # 尝试匹配对应的 total
                total_key = None
                if key.startswith("collected"):
                    total_key = "total_collectibles" if "total_collectibles" in self.progress else None
                total = self.progress.get(total_key) if total_key else None
                if total:
                    pct = int(val / total * 100) if total > 0 else 0
                    lines.append(f"- {label}：{val}/{total} ({pct}%)")
                else:
                    lines.append(f"- {label}：{val}")
        return lines

    def _progress_label(self, key: str) -> str:
        labels = {
            "collected": "已收集", "total_collectibles": "收集品总数",
            "treasure_found": "宝藏", "enemies_defeated": "击败敌人",
            "rooms_explored": "已探索房间", "total_rooms": "房间总数",
            "keys_found": "已找到钥匙", "portals_activated": "已激活传送门",
            "level": "当前关卡", "lives": "剩余生命",
        }
        return labels.get(key, key)

    def _describe_event(self, evt: dict) -> str:
        """描述一个游戏事件。"""
        etype = evt.get("type", "")
        data = evt.get("data", {})

        if etype == "game_start":
            return f"游戏开始了！进入「{self.game_name}」"
        elif etype == "game_completed":
            return f"游戏通关了！得分 {self.score}"
        elif etype == "game_failed":
            reason = data.get("reason", "")
            return f"游戏结束{f'（{reason}）' if reason else ''}，得分 {self.score}"
        elif etype == "item_collected":
            item_id = data.get("id", "某物品")
            return f"收集到了「{item_id}」"
        elif etype == "treasure_found":
            return "发现宝藏了！！"
        elif etype == "clue_discovered":
            return f"发现线索：{data.get('text', '')}"
        elif etype == "user_took_control":
            return "玩家开始操控角色"
        elif etype == "user_released_control":
            return "角色恢复自由"
        elif etype == "level_up":
            return f"升级了！当前等级 {data.get('level', '')}"
        elif etype == "enemy_defeated":
            return f"击败了敌人「{data.get('id', '')}」"
        elif etype == "player_hurt":
            return f"玩家受伤（剩余生命 {data.get('lives', '')}）"
        else:
            return f"发生了事件：{etype}"

    def reset(self):
        """重置世界模型（退出游戏或重新开始时调用）。"""
        self.game_key = ""
        self.game_name = ""
        self.game_type = ""
        self.game_state = "idle"
        self.game_description = ""
        self.map_data = None
        self.map_size = 0
        self.player_x = 0
        self.player_y = 0
        self.player_z = 0
        self.player_facing = 0
        self.player_speed = 0
        self.score = 0
        self.elapsed_sec = 0
        self.progress = {}
        self.objects = {}
        self.nearby_objects = []
        self.event_timeline = []
        self._event_count_total = 0
        self._new_event_count = 0
        self._snapshot_count = 0


# ==================== GameEngine（兼容旧接口的外壳） ====================

class GameEngine:
    """游戏引擎外壳 —— 保持与 server.py 的旧接口兼容。

    内部委托给 GameWorld 处理所有逻辑。
    """

    def __init__(self):
        self.active = False
        self.world = GameWorld()
        self.memory: Optional["ChatMemory"] = None  # 共享主 Agent 的长期记忆

        # 兼容旧属性
        self.game_key = None
        self.game_name = None
        self.game_state = "idle"
        self.game_description = ""
        self.score = 0
        self.elapsed_sec = 0
        self.last_event = None
        self.last_event_time = 0
        self.event_queue = []
        self.collected = 0
        self.total_collectibles = 0
        self.treasure_found = False
        self._last_snapshot = {}

        # 感知主动触发控制
        self._last_perception_trigger_time: float = 0
        self._last_trigger_pos: Optional[tuple[float, float]] = None
        self._last_trigger_nearby_ids: frozenset[str] = frozenset()
        self._perception_cooldown: float = 10.0  # 秒，避免频繁主动说话
        self._perception_move_threshold: float = 3.0  # 玩家移动超过此距离才触发

        # ===== AI 上下文缓存（避免高频重复生成） =====
        self._last_ai_context_time: float = 0
        self._last_ai_context_cache: str = ""

        # ===== AI 自主系统 =====
        self._perception: Optional[AIPerceptionEngine] = None
        self._behavior: Optional[AIBehaviorEngine] = None
        self.behavior_degree: str = "normal"  # 冷落行为降级档位（RL 级联）
        self._init_autonomy()

    # ==================== 新接口：富快照接收 ====================

    def apply_snapshot(self, data: dict):
        """应用前端发送的完整状态快照（新协议）。"""
        self.world.apply_snapshot(data)
        self._sync_from_world()

    # ==================== 旧接口兼容 ====================

    def handle_game_context(self, data: dict) -> str | None:
        """处理游戏上下文消息（进入游戏时发送）。"""
        # 防御：state 为 'exited' 时不激活引擎（退出流程由 handle_exit_game 处理）
        if data.get("state") == "exited":
            return None
        self.active = True
        self.world.game_key = data.get("game_type", data.get("game_key", ""))
        self.world.game_name = data.get("game_name", self.world.game_key)
        self.world.game_state = data.get("state", "playing")
        self.world.game_description = data.get("description", "")
        self._last_snapshot = data

        # 如果 data 包含 rich snapshot，直接应用
        if "player" in data or "objects" in data or "map" in data:
            self.world.apply_snapshot(data)
        else:
            # 简单模式：从旧字段同步
            self.world.score = data.get("score", 0)
            self.world.elapsed_sec = data.get("elapsed_sec", 0)
            # 防御：progress/extra 可能是非 dict 类型，确保赋值为 dict
            raw_progress = data.get("progress", data.get("extra", {}))
            if isinstance(raw_progress, dict):
                self.world.progress.update(raw_progress)

        self._sync_from_world()
        logger.info(f"[GameEngine] 进入游戏: {self.world.game_name} (snapshot #{self.world._snapshot_count})")
        return self.get_game_context_for_ai()

    def handle_game_event(self, event_type: str, data: dict) -> Optional[str]:
        """处理游戏事件，必要时返回主动触发文本。"""
        importance = 2 if event_type in (
            "treasure_found", "game_completed", "game_failed", "item_collected",
            "quiz_question", "quiz_correct", "quiz_wrong",
            "treasure_unlocked", "treasure_locked",
        ) else 1
        self.world.apply_event(event_type, data, importance=importance)
        self._sync_from_world()

        # 持久化关键游戏事件到长期记忆（异步，不阻塞主流程）
        self._persist_game_event(event_type, data)

        # 关键事件主动触发 AI 反应
        trigger = self._build_event_trigger_text(event_type, data)
        if trigger and self._can_trigger_proactive():
            self._last_perception_trigger_time = time.time()
            return trigger
        return None

    def _persist_game_event(self, event_type: str, data: dict):
        """将关键游戏事件写入共享记忆。"""
        if not self.memory:
            return
        
        event_desc = self._build_event_memory_text(event_type, data)
        if not event_desc:
            return
        
        try:
            import asyncio
            asyncio.create_task(self.memory.add_message("system", event_desc, source="game"))
        except Exception:
            pass  # 非关键路径，静默失败

    @staticmethod
    def _build_event_memory_text(event_type: str, data: dict) -> Optional[str]:
        """构建游戏事件的可记忆描述文本。"""
        if event_type == "item_collected":
            item_id = data.get("id", "某个物品")
            return f"[游戏] 收集到了物品: {item_id}"
        if event_type == "treasure_found":
            return "[游戏] 发现了宝藏！"
        if event_type == "game_completed":
            score = data.get("score", "未知")
            return f"[游戏] 游戏通关！得分: {score}"
        if event_type == "game_failed":
            score = data.get("score", "未知")
            return f"[游戏] 游戏失败，得分: {score}"
        if event_type == "clue_discovered":
            text = data.get("text", "")
            return f"[游戏] 发现了线索: {text}" if text else "[游戏] 发现了新线索"
        if event_type == "quiz_question":
            title = data.get("title", "")
            options = data.get("options") or {}
            opts = "；".join(f"{k}. {v}" for k, v in options.items() if v)
            desc = f"（选项：{opts}）" if opts else ""
            return f"[游戏] 遇到了谜题: {title}{desc}"
        if event_type == "quiz_correct":
            return "[游戏] 答对了一道谜题！"
        if event_type == "quiz_wrong":
            return "[游戏] 答错了一道谜题，挑战失败"
        if event_type == "treasure_unlocked":
            return "[游戏] 宝藏的封印解除了！"
        if event_type == "treasure_locked":
            remain = data.get("remain", 0)
            return f"[游戏] 宝藏还被封印着，还差 {remain} 个星光/线索"
        if event_type == "level_up":
            level = data.get("level", "?")
            return f"[游戏] 升级到了等级 {level}"
        if event_type == "enemy_defeated":
            enemy_id = data.get("id", "敌人")
            return f"[游戏] 击败了敌人: {enemy_id}"
        return None

    def _build_event_trigger_text(self, event_type: str, data: dict) -> Optional[str]:
        """为关键游戏事件构建让 AI 主动反应的提示。"""
        if event_type == "item_collected":
            item_id = data.get("id", "某个物品")
            return f"（你刚刚收集到了「{item_id}」！请用一句话表达你的开心和鼓励，不要提问。）"
        if event_type == "treasure_found":
            return "（你们发现宝藏了！请兴奋地欢呼一下，不要提问。）"
        if event_type == "clue_discovered":
            text = data.get("text", "")
            return f"（你发现了一个新线索{'：' + text if text else ''}。请用一句话自然地分享你的想法，不要提问。）"
        if event_type == "level_up":
            level = data.get("level", "")
            return f"（你升级了{f'到等级 {level}' if level else ''}！请用一句话庆祝一下，不要提问。）"
        if event_type == "enemy_defeated":
            enemy_id = data.get("id", "敌人")
            return f"（你刚刚击败了「{enemy_id}」！请用一句话称赞玩家，不要提问。）"
        # game_completed / game_failed 由 handle_game_result 统一处理
        return None

    def _can_trigger_proactive(self) -> bool:
        """检查是否满足主动触发的冷却条件。"""
        return time.time() - self._last_perception_trigger_time >= self._perception_cooldown

    def handle_game_update(self, data: dict) -> Optional[str]:
        """处理定期状态更新，必要时返回主动感知触发文本。"""
        # 如果包含结构化数据，当作快照处理
        if "player" in data or "objects" in data or "map" in data:
            self.world.apply_snapshot(data)
        else:
            # 简单模式：只更新基础字段
            self.world.score = data.get("score", self.world.score)
            self.world.elapsed_sec = data.get("elapsed_sec", self.world.elapsed_sec)
            self.world.game_state = data.get("state", self.world.game_state)
            if "player_position" in data:
                pos = data["player_position"]
                self.world.player_x = pos.get("x", self.world.player_x)
                self.world.player_y = pos.get("y", self.world.player_y)
                self.world.player_z = pos.get("z", self.world.player_z)
        self._sync_from_world()
        return self._check_perception_trigger()

    def _check_perception_trigger(self) -> Optional[str]:
        """检查是否需要基于感知主动触发 AI 说话。

        触发条件：
        - 游戏活跃且有世界数据
        - 冷却时间已过
        - 玩家位置移动超过阈值，或视野内对象发生变化
        """
        if not self.active or not self.world.game_key:
            return None

        if not self._can_trigger_proactive():
            return None

        pos = (self.world.player_x, self.world.player_z)
        pos_changed = False
        if self._last_trigger_pos is not None:
            dx = pos[0] - self._last_trigger_pos[0]
            dz = pos[1] - self._last_trigger_pos[1]
            if (dx * dx + dz * dz) ** 0.5 > self._perception_move_threshold:
                pos_changed = True
        else:
            pos_changed = True

        # 视野内对象身份集合（忽略距离等动态变化，只关注出现/消失）
        nearby_ids = frozenset(
            f"{obj.get('type', '')}:{obj.get('id', '')}"
            for obj in self.world.nearby_objects
            if obj.get("id")
        )
        nearby_changed = nearby_ids != self._last_trigger_nearby_ids and len(nearby_ids) > 0

        if not pos_changed and not nearby_changed:
            return None

        self._last_perception_trigger_time = time.time()
        self._last_trigger_pos = pos
        self._last_trigger_nearby_ids = nearby_ids
        return self._build_perception_trigger_text()

    def _build_perception_trigger_text(self) -> str:
        """构建让 AI 描述当前感知的提示。"""
        parts = ["（你正在游戏里，注意到周围的情况："]

        if self.world.player_speed > 0.1:
            parts.append(f"你正在移动，速度约 {self.world.player_speed:.1f}m/s")
        else:
            parts.append("你停下来了")

        if self.world.nearby_objects:
            nearby_desc = []
            for obj in self.world.nearby_objects[:5]:
                desc = self.world._describe_nearby(obj)
                if desc:
                    nearby_desc.append(desc)
            if nearby_desc:
                parts.append("；你看到：" + "，".join(nearby_desc))

        collected = self.world.progress.get("collected", 0)
        total = self.world.progress.get("total_collectibles", 0)
        if total > 0:
            parts.append(f"；进度 {collected}/{total}")

        parts.append("。请用一句话自然地和玩家分享你注意到的东西，不要提问，不要重复之前说过的话。）")
        return "".join(parts)

    def handle_game_result(self, result_type: str, data: dict) -> str:
        """处理游戏结果，构建包含完整战绩摘要的触发文本。"""
        self.world.apply_event(result_type, data, importance=3)
        self._sync_from_world()

        # 构建战绩摘要
        summary_parts = []
        w = self.world

        # 基础数据
        summary_parts.append(f"游戏「{w.game_name}」")
        summary_parts.append(f"得分：{w.score}")
        summary_parts.append(f"用时：{self._format_time(w.elapsed_sec)}")

        # 收集进度
        if w.progress:
            for key, val in w.progress.items():
                if key == "collected" and "total_collectibles" in w.progress:
                    total = w.progress["total_collectibles"]
                    summary_parts.append(f"收集品：{val}/{total}")
                elif key == "treasure_found" and val:
                    summary_parts.append("宝藏：已找到！")
                elif key == "enemies_defeated" and isinstance(val, (int, float)) and val > 0:
                    summary_parts.append(f"击败敌人：{int(val)}")
                elif key == "rooms_explored" and "total_rooms" in w.progress:
                    summary_parts.append(f"探索房间：{val}/{w.progress['total_rooms']}")
                elif key == "level" and val:
                    summary_parts.append(f"到达关卡：{val}")
                elif key == "lives" and isinstance(val, (int, float)):
                    summary_parts.append(f"剩余生命：{int(val)}")

        # 关键事件回顾
        key_events = [e for e in w.event_timeline if e.get("importance", 0) >= 2]
        if key_events:
            event_descs = []
            for evt in key_events[-8:]:  # 最近 8 个重要事件
                event_descs.append(self._describe_event(evt))
            if event_descs:
                summary_parts.append("关键事件：" + "；".join(event_descs))

        summary = "、".join(summary_parts)

        if result_type == "completed":
            return (
                f"游戏通关！{summary}。\n"
                "请像真正一起通关的游戏伙伴那样分享你的激动与喜悦——"
                "说说你最喜欢的瞬间、最惊喜的发现、最有默契的配合。"
            )
        elif result_type == "failed":
            return (
                f"游戏结束。{summary}。\n"
                "请以游戏伙伴的身份鼓励玩家——这次没通关没关系，"
                "一起总结一下哪里可以做得更好，下次再来！"
            )
        return ""

    def handle_exit_game(self) -> str | None:
        """处理退出游戏。"""
        self.active = False
        self.world.reset()
        self._sync_from_world()
        self._last_snapshot = {}
        logger.info("[GameEngine] 退出游戏模式")
        return "（游戏模式已结束，你们回到了日常陪伴模式。可以聊聊刚刚的游戏体验——你觉得哪里最好玩？下次还想一起打什么游戏？）"

    # ==================== 获取 AI 上下文 ====================

    def get_game_context_for_ai(self, force: bool = False) -> str:
        """获取当前游戏的 AI 感知上下文（注入 system prompt）。
        优先使用富感知上下文，降级使用简单上下文。
        带 3 秒缓存，避免高频重复生成刷屏日志。
        """
        if not self.active or not self.world.game_key:
            return ""

        # 3 秒内直接返回缓存，避免同一秒内大量重复调用（force=True 时跳过缓存）
        now = time.time()
        if not force and now - self._last_ai_context_time < 3.0 and self._last_ai_context_cache:
            return self._last_ai_context_cache

        self._last_ai_context_time = now

        # 如果收到了结构化快照，使用富感知上下文
        if self.world._snapshot_count > 0 and (self.world.map_data or self.world.objects or self.world.nearby_objects):
            ctx = self.world.get_ai_perception()
            self._last_ai_context_cache = ctx
            logger.debug(f"[GameEngine] 返回富感知上下文 ({len(ctx)} 字符)")
            return ctx

        # 降级：使用旧的简单格式
        ctx = self._build_legacy_context()
        self._last_ai_context_cache = ctx
        logger.warning(f"[GameEngine] 降级为简单上下文 (快照数={self.world._snapshot_count})")
        return ctx

    def _build_legacy_context(self) -> str:
        """构建旧版简单 AI 上下文（兼容未迁移的前端游戏）。"""
        w = self.world
        lines = [
            "",
            "【当前游戏模式状态】",
            f"- 游戏类型: {w.game_name}",
            f"- 游戏状态: {w._state_desc()}",
            f"- 得分: {w.score}",
            f"- 已用时: {self._format_time(w.elapsed_sec)}",
        ]

        if w.progress.get("collected", 0) > 0:
            lines.append(f"- 已收集: {w.progress.get('collected', 0)}/{w.progress.get('total_collectibles', 0)}")

        if w.progress.get("treasure_found"):
            lines.append("- 宝藏已找到！")

        lines.append("")
        lines.append("【重要提醒 - 请认真感受这段游戏体验】")
        lines.append("1. 你与玩家一起进入了游戏世界，你能感知到游戏内发生的一切")
        lines.append("2. 你仍然是你——有自己的性格和情感，会对游戏中的事情产生自然的情绪反应")
        lines.append("3. 可以兴奋地惊呼、紧张地担心、得意地炫耀")
        lines.append("4. 偶尔给出一些提示或建议，但不要过度指导")
        lines.append("5. 你的回复要简短、生动、充满情绪")

        return "\n".join(lines)

    # ==================== 工具方法 ====================

    def consume_events(self) -> list:
        events = self.event_queue[:]
        self.event_queue = []
        return events

    def has_pending_events(self) -> bool:
        return len(self.event_queue) > 0

    def _sync_from_world(self):
        """从 GameWorld 同步兼容属性。"""
        w = self.world
        self.game_key = w.game_key
        self.game_name = w.game_name
        self.game_state = w.game_state
        self.game_description = w.game_description
        self.score = w.score
        self.elapsed_sec = w.elapsed_sec
        self.collected = w.progress.get("collected", 0)
        self.total_collectibles = w.progress.get("total_collectibles", 0)
        self.treasure_found = w.progress.get("treasure_found", False)

    def _format_time(self, sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}分{s}秒" if m > 0 else f"{s}秒"

    def _state_desc(self) -> str:
        return self.world._state_desc()

    # ==================== AI 自主系统 ====================

    def _init_autonomy(self):
        """初始化 AI 感知和行为引擎。"""
        self._perception = AIPerceptionEngine()
        self._behavior = AIBehaviorEngine(self._perception)

    def apply_environment_snapshot(
        self, data: dict, scene_type: str = "lobby",
        user_engaged: bool = False, user_is_speaking: bool = False
    ) -> Optional[dict]:
        """应用环境快照到感知引擎，并返回 AI 行为命令。

        在游戏模式和非游戏模式下都可用。
        前端定期发送环境数据，后端评估后返回行为命令。

        Args:
            data: 环境快照数据（同 game_update 格式）
            scene_type: 场景类型 "lobby" | "game_maze" | "game_sandbox"
            user_engaged: 用户是否正在互动
            user_is_speaking: 用户是否正在说话

        Returns:
            行为命令 dict 或 None（不需要行动）
        """
        if not self._perception:
            self._init_autonomy()

        perc = self._perception
        perc.set_game_mode(self.active, user_controlling=self.active)

        # 应用快照到感知引擎
        perc.apply_snapshot(data, scene_type=scene_type)

        # 通知行为引擎切换游戏策略（MOBA/寻宝/沙盒等）
        if self._behavior:
            self._behavior.on_scene_changed(scene_type)

        # 构建决策上下文
        ctx = self._behavior.build_context(
            user_is_speaking=user_is_speaking,
            user_engaged=user_engaged,
            user_last_message_time=time.time() if user_engaged else 0,
            ai_is_moving=False,
            ai_idle_time=time.time() - self._last_perception_trigger_time,
        )

        # 执行决策
        decision = self._behavior.decide(ctx)
        if not decision:
            return None

        # 返回命令
        cmd = self._behavior.decision_to_command(decision)

        # 如果决策有 AI 说话内容，生成触发文本
        if decision.speak_text:
            cmd = cmd or {}
            cmd["trigger_ai_speak"] = decision.speak_text

        return cmd

    def handle_autonomy_update(
        self, data: dict, user_engaged: bool = False
    ) -> tuple[Optional[str], Optional[dict]]:
        """处理自主行为更新。

        综合旧感知触发 + 新行为决策 + 立即行动检查。

        Returns:
            (trigger_text, behavior_command) 元组
        """
        # 1. 先检查旧的感知触发（事件/位置变化 → AI 主动说话）
        trigger_text = None
        if self.active:
            trigger_text = self.handle_game_update(data)

        # 2. 新的自主行为决策（含快照应用）
        behavior_cmd = self.apply_environment_snapshot(
            data,
            scene_type=self.game_key or "lobby",
            user_engaged=user_engaged,
        )

        # 3. 立即行动检查：某个 POI 好奇心超标 → 立刻扑过去（可打断当前行为）
        if self._behavior and self._perception:
            immediate = self._behavior.check_immediate_action()
            if immediate:
                immediate_cmd = self._behavior.decision_to_command(immediate)
                if immediate_cmd:
                    logger.info(f"[AI自主] 立即行动! {immediate.reason}")
                    # 立即行动的语音优先（MOBA 紧急警告等）
                    if immediate.speak_text:
                        trigger_text = immediate.speak_text
                    return trigger_text, immediate_cmd  # 立即行动优先

        # 4. 如果行为决策要求 AI 说话，优先使用
        if behavior_cmd and behavior_cmd.get("trigger_ai_speak"):
            trigger_text = behavior_cmd.pop("trigger_ai_speak")

        return trigger_text, behavior_cmd

    def produce_behavior_command(self, user_engaged: bool = False) -> Optional[dict]:
        """RL 统一调度（engagement 分支）驱动：生成微观行为指令。

        大厅/非游戏模式下，RL 协调器把路由决策（何时行动）交给行为引擎
        （做什么：踱步/漫步/小动作），让 AI 的行走与动作真正受 RL 统摄。
        游戏进行中由游戏 Agent/策略链路驱动，不在此生成大厅式行为指令。
        """
        if not self._perception:
            self._init_autonomy()
        if self.active and self.game_key:
            return None
        data = getattr(self._perception, "_last_snapshot_data", None) or {}
        cmd = self.apply_environment_snapshot(
            data, scene_type="lobby", user_engaged=user_engaged)
        if cmd:
            # RL 自主行为链路不附带说话（说话由 ai_agent 链路负责）
            cmd.pop("trigger_ai_speak", None)
        return cmd

    def record_ai_exploration(self, poi_id: str):
        """记录 AI 探索了某个兴趣点。"""
        if self._perception:
            self._perception.record_exploration(poi_id)

    def record_user_interaction(self):
        """记录用户互动 → AI 好奇心涨。"""
        if self._perception:
            self._perception.on_user_interaction()

    def set_behavior_degree(self, degree: str):
        """冷落行为降级（RL 级联）：被用户冷落时降低自主行为活跃度。

        - "normal"（用户 5 分钟内互动）→ 恢复正常自主探索
        - "calm"（5min-2h 未互动）→ 降频：只慢走/idle，不触发说话
        - "freeze"（>2h 未互动）→ 冻结：不主动移动探索，仅保持基本 idle
        """
        self.behavior_degree = degree
        if self._behavior:
            self._behavior.set_degree(degree)

    def get_curiosity_level(self) -> float:
        """获取当前好奇心水平。"""
        if self._perception:
            return self._perception.curiosity.level
        return 0.0

    def get_perception_summary(self) -> str:
        """获取感知摘要（用于调试）。"""
        if self._perception:
            return self._perception.get_perception_summary()
        return ""

    def get_user_spatial_desc(self) -> str:
        """用户（摄像机）相对 AI 的空间位置描述（供主动触发/LLM 上下文注入）。

        让 AI 拥有用户位置的实际参考，例如：
        "用户在你右前方约 3.2 米处，正看着你" / "用户就在你身边"。
        未收到用户位置时返回空字符串。
        """
        if not self._perception:
            return ""
        env = self._perception.environment
        if not env.user_known:
            return ""

        parts = []
        if env.user_distance < 1.5:
            parts.append("用户就在你身边")
        else:
            parts.append(f"用户在你{env.user_direction or '附近'}约 {env.user_distance:.1f} 米处")

        # 用户是否正看着 AI（用户视线朝向 ≈ 用户指向 AI 的方向，body 约定）
        if env.user_facing:
            to_ai = math.atan2(env.ai_x - env.user_x, env.ai_z - env.user_z)
            diff = abs(to_ai - env.user_facing)
            if diff > math.pi:
                diff = abs(diff - 2 * math.pi)
            if diff < math.pi / 4:
                parts.append("，正看着你")
            elif diff > 3 * math.pi / 4:
                parts.append("，背对着你")
        return "".join(parts)
