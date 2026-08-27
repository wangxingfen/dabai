# 界面模式技能（interface）

切换应用交互方式、在屏幕上显示提示消息。

## 工具
- switch_app_mode(mode) —— 切换运行模式
  - auto_voice：自动对话（边说边录自动发送）
  - press_voice：按住说话
  - lock_screen：锁屏防误触（仅语音对话，角色照常活动）
  - normal：普通模式（解锁）
- show_screen_toast(message) —— 屏幕 Toast（简短有趣，不超过 30 字）