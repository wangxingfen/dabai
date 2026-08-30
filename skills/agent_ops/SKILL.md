# 智能体指挥（agent_ops）

把任务委派给手下智能体（dsh/codex/opencode）并查看进展。触发：复杂/跨系统/多步骤任务，或用户点名 DSH。

## 工具
- `delegate_agent_task` 委派任务（先产出任务规范：目标/范围/验收/步骤/回滚）
- `list_agent_tasks` 查看任务中心进展

## 规则
- 点名 DSH 必须用 dsh；所有委派先请用户确认
- 画图与音乐绝不委派；委派前先查任务中心，同一任务连续失败两次以上停止自动重试
- 清理只删白名单临时文件，git 已跟踪文件禁删

详细文档：references/guide.md
