"""游戏技能 —— 启动小游戏（返回 __screen_command__ 由前端拉起游戏世界）。"""
from __future__ import annotations

import json


def launch(args: dict) -> str:
    return json.dumps({"__screen_command__": True, "tool": "launch_game", "args": {
        "game_key": args.get("game_key"),
    }}, ensure_ascii=False)


HANDLERS = {
    "launch_game": launch,
}
