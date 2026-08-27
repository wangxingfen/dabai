# 执行力自我迭代技能（execution_loop）

让大白拥有**执行日志 → 复盘提炼 → 策略库 → 自动检索**的自我迭代闭环：
每一次任务执行（尤其失败）都沉淀为经验，下次同类任务动手前自动拿来就用。

## 使用心法（重要）

1. **动手前先查策略**。同类任务失败过一次以上，先调
   `strategy_lookup(task_type=..., goal=...)`——goal 用用户原话或忠实复述
   （关键词重叠率影响检索得分）。返回的策略要点应在执行中主动遵循；
   提示「暂无策略」则按正常流程做，完成后记得登记。
2. **任务结束必登记，失败更必登记**。调 `execution_record(task_type, goal,
   outcome, blockers, actions, result)`；outcome 如实填 ok/partial/fail。
   **失败时 blockers 务必写清**：优先传 `{stage, symptom, cause}` 对象
   （哪个阶段/什么现象/根因线索），卡点质量直接决定复盘能提炼出什么策略。
3. **定期或连续失败后复盘**。调 `execution_review()`（无参数）把未复盘日志里的
   卡点聚类提炼成策略入库。复盘按游标增量执行，每条日志只复盘一次，
   可每日/每周跑，也可在连续卡壳后立即跑一次再重试任务。
4. **给策略反馈效果**。执行中某条策略确实帮上忙 → `strategy_feedback(
   strategy_id, good=true)`；照做仍失败 → `good=false`。多次失效的策略会被
   降权直至不再检索，避免错误经验反复污染执行。
5. **task_type 命名保持稳定**。用简短一致的场景名（如 web_scrape / file_batch /
   shell_deploy / video_merge），命名漂移会让策略检索不到。宁可复用已有类型，
   也不要每次发明新叫法。
6. **闭环永不拖垮主流程**：所有落盘原子写、失败静默降级，登记/复盘出错只返回
   文本提示，不影响任务本身。

## 工具

| 工具 | 用途 |
|------|------|
| strategy_lookup(task_type, goal?) | 动手前检索策略库，返回可遵循的历史策略要点 |
| execution_record(task_type, goal, outcome, blockers?, actions?, result?) | 任务完成后登记执行日志（目标/动作/结果/卡点） |
| execution_review() | 复盘：未复盘日志的卡点聚类提炼成策略入库，返回报告 |
| strategy_feedback(strategy_id, good) | 策略效果反馈：true=有效提权，false=失效降权 |

## 典型节奏

```
# ① 动手前（同类任务失败过）
strategy_lookup(task_type="web_scrape", goal="抓取某新闻站点列表页")

# ② 执行任务（遵循返回的策略要点）……

# ③ 完成后登记
execution_record(
  task_type="web_scrape", goal="抓取某新闻站点列表页", outcome="fail",
  blockers=[{"stage": "请求", "symptom": "403 反爬拦截", "cause": "缺 User-Agent 伪装"}],
  actions=[{"name": "http_get", "args": {"url": "..."}, "ok": false}],
  result="首页被拦截，未取到列表")

# ④ 攒了几条日志后复盘
execution_review()          # → 沉淀策略：web_scrape 场景先补 UA/Cookie 再请求

# ⑤ 下次同类任务回到 ①，策略已在库里等着
```

## 机制速览（便于解释与信任）

- **复盘提炼**：卡点按（场景×卡点指纹）聚类，复现 ≥2 次的坑优先沉淀；
  无 LLM 也能用启发式模板给应对建议。
- **策略去重**：同场景关键词交集 ≥2 的策略自动合并计数，不会越攒越重复。
- **检索打分**：场景精确命中 4 分 / 包含 2 分 + 关键词重叠率 + 历史命中加成 +
  good/bad 口碑修正；场景完全无关的策略须关键词实质重叠才可跨场景借鉴，
  防止错误场景的经验互相污染。
- **数据落点**：`data/execution_logs.jsonl`（执行日志）、`data/strategies.json`
  （策略库）、`data/review_history.json`（复盘游标），均为原子落盘。

## 适用判断

- 一次性的简单问答/单步调用 → 不必登记，直接做。
- 多步骤任务、批量任务、容易失败的脏活（爬取/下载/部署/转码/环境操作）→
  值得走闭环：动手前 lookup，结束后 record。
- 与 harness 任务系统的 lessons 经验互补不冲突：lessons 是流程级一句话经验，
  本闭环是**结构化、按场景检索、带效果评分**的执行策略库。
