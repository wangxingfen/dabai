# 长任务与批量任务技能（tasks）

让大白把**多步骤长任务**与**批量任务**交给 harness 任务系统后台执行：提交即返回、
后台推进、断点续跑、失败重试、并发受控、随时可查可取消。

## 使用心法（重要）

1. **用户说得越短，越要先规划**。用户往往三言两语（"帮我整理点睡前歌"）——
   先调用 `harness_flow_plan(goal=用户原话)`：规划器会**复述对任务的理解**并拆解成
   通过校验的步骤，还会**自评一遍**（风险/遗漏/更优路径，必要时给出修订版）；
   你把 understanding + critique 讲给用户听，确认或 autostart 直接开跑。
2. **goal 必填且用用户原话**。goal 会持久化并自动注入每个 llm 步骤
   （【任务目标】+【已完成步骤摘要】）——流程跑到一半、甚至服务重启续跑后，
   每一步都记得自己为什么而做。
3. **提交即校验，错误当场暴露**：工具名不存在 / 缺必填参数 / 参数类型不符会在
   提交时被拒绝（拼错的工具名会自动纠正并在返回里说明），不会跑到一半才失败。
   拿不准工具用法时先 `skill_help("技能名")`。
4. **危险动作必须加确认门**。删除/覆盖/发送/花钱/对外发布类步骤加
   `"confirm":"危险性说明"`——执行到该步会暂停等用户批准（对话里用
   harness_task_confirm 或管理页按钮）。宁多问一句，不做不可挽回的事。
5. **失败会自动反思**。步骤重试耗尽后，任务系统会先请决策器评估：能否换工具/
   换参数/换路径绕过（自动修订剩余步骤），不行再按 abort/continue 策略收场。
   你查进度看到"反思修订"属于正常自救，不用重复提交。
6. **提交后不要重复提交**：用 harness_task_status 查进度（默认返回紧凑摘要省
   token，需要完整结果传 detail=true）；harness_task_list 会自动带回
   「已完成待汇报」——不查也会在下一次列任务时知道哪些完成了。
7. **需求有歧义先问再做**：用户的请求缺少关键信息（对象/范围/偏好）时，
   先向用户提 1-2 个具体问题，再规划——不要基于猜测开跑长任务。

## 工具

| 工具 | 用途 |
|------|------|
| harness_flow_plan(goal, hints?, autostart?, policy?) | 规划：理解复述 + 步骤方案 + 自评（校验/纠错；autostart 直接开跑） |
| harness_flow_submit(name, goal, steps, policy?, timeout?) | 提交多步骤 DAG 流程（长任务） |
| harness_batch_submit(name, tool, items, concurrency?) | 提交批量任务（工具并行 map 到 N 组参数） |
| harness_task_status(task_id) | 查询进度与结果（步骤/条目粒度，含任务目标与待确认状态） |
| harness_task_list(state?, limit?) | 列出最近任务 |
| harness_task_confirm(task_id, approve, note?) | 批准/拒绝等待确认的危险步骤 |
| harness_task_cancel(task_id) | 取消（流程连带取消未完成步骤） |
| harness_task_retry(task_id) | 重试（流程保留已成功步骤，只补跑失败部分；危险步骤需重新确认） |

## 步骤（steps）写法

每步 `{id, deps?, action, ...}`，action 四类：

```
{"id":"search", "action":"tool",  "tool":"music_search", "args":{"keyword":"..."}}
{"id":"pick",   "action":"llm",   "system":"选歌助手", "prompt":"从里选一首: {{search.result}}", "deps":["search"]}
{"id":"all",    "action":"batch", "tool":"music_search", "items":[{...},{...}], "concurrency":3}
```

- `deps` 声明依赖（DAG，自动并行无依赖分支）；环依赖会被拒绝。
- `{{步骤id.result}}` 在后续步骤的 prompt / args 里引用前步结果（长文本自动截断）。
- llm 步骤**自动携带**全局上下文：任务目标 + 当前进度 + 已完成步骤结果摘要
  （无需手写，也因此在长流程/重启续跑后不会失忆）。
- 关键产出步骤可加 `"expect":"结果应满足什么"`——执行后自动语义校验，
  答非所问会被拦截重试（比 result_guard 的文案反查更强一级）。
- 步骤级可用 `timeout` / `max_attempts` 覆盖默认值。

## 失败策略 policy

- `abort`（默认）：任一步失败 → 取消其余步骤，流程失败。
- `continue`：只放弃失败分支，独立分支继续执行完。

## 执行保证

- **持久化断点续跑**：流程定义与每步完成状态实时落盘（harness_tasks.json）；
  服务重启 / 热重载后自动恢复，已成功步骤不重算（at-least-once 补跑未完成部分）。
- **重试与退避**：步骤失败按 2s→4s 退避重试（默认 2 次），可按步覆盖。
- **并发受控**：队列 worker 上限 + 批量批内信号量，不会打爆下游。
- **统一监督**：工具步骤经 harness 稳定路由 + runtime 熔断/超时/计量；llm 步骤走
  plan 渠道（用量记入运行时统计）。

## 适用判断

- 一次能说完的单个工具调用 → 直接用原工具，不必提交任务。
- 多步骤 / 步骤有依赖 / 需要 LLM 中间加工 / 批量同构操作 → 用本技能。
- 提交后用 harness_task_status 汇报进度，不要重复提交同一任务。
