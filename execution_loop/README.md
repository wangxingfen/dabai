# execution_loop —— 「大白」的执行力自我迭代闭环

> **执行日志 → 复盘提炼 → 策略沉淀 → 自动检索调用**：让每一次任务执行都变成
> 下一次执行的经验，形成可自我迭代的执行力闭环。

```
  任务执行
    执行前：自动检索策略库（第 4 环）→ 注入执行上下文（strategy_notes）
    执行后：自动记录 目标/动作/结果/卡点（第 1 环）
                    |
                    v
  定期复盘：卡点聚类 → 提炼策略规则（第 2 环）
                    |
                    v
  策略库：按任务类型分类持久化 + 效果评分（第 3 环）
```

## 1. 目录结构

    execution_loop/
    ├── logger.py      执行日志（JSONL 追加写：goal/actions/result/blockers/outcome）
    ├── reviewer.py    复盘提炼（卡点聚类 → 模板化策略；可选 llm_distill 回调润色）
    ├── store.py       策略库（strategies.json：场景分类 + 合并去重 + hit/good/bad 评分）
    ├── retriever.py   自动检索（场景匹配 + 关键词 Jaccard + 历史命中 + 口碑，top-k 打分）
    ├── agent.py       ExecutionLoop 闭环总入口（retrieve → run → record 一条龙）
    ├── hooks.py       与「大白」运行时接线的桥（技能工具 + 自动记录/检索接线点）
    └── README.md
    examples/
    └── demo_iteration.py   三轮闭环可运行演示
    data/                   运行时数据（自动生成）
    ├── execution_logs.jsonl   执行日志
    ├── strategies.json        策略库
    └── review_history.json    复盘游标

## 2. 四个环节

| 环节 | 模块 | 说明 |
|---|---|---|
| 1 执行日志 | logger | 每次任务结束自动记录 task_type/goal/actions/result/blockers/outcome/耗时；卡点支持字符串或 {stage,symptom,cause}；写入失败静默降级 |
| 2 复盘提炼 | reviewer | 取未复盘日志 → 卡点按（场景×指纹）聚类 → 同卡点复现≥2 次的优先 → 模板+启发式建议生成规则；可传 llm_distill(blocker)->str 用 LLM 润色；复盘游标保证每条日志只复盘一次 |
| 3 策略库 | store | strategies.json 原子落盘；按 scene（任务类型）分类；同场景同关键词（交集≥2）自动合并计数，不产生重复策略；hit/good/bad 记录实战效果 |
| 4 自动调用 | retriever | 执行前按 scene + goal 打分检索（场景精确 4 分/部分 2 分 + 关键词 Jaccard + log2(命中) + 口碑）；场景完全无关必须关键词 Jaccard≥0.2 才允许跨场景借鉴，防污染 |

## 3. API 速查

    from execution_loop import ExecutionLoop

    loop = ExecutionLoop()                       # 默认用 D:/AI/dabai/data

    # 闭环一条龙：自动检索 → 注入 → 执行 → 记日志
    r = loop.execute(task_type="web_scrape", goal="抓取新闻", run=my_executor)
    # run(ctx) 契约：ctx = {task_id, task_type, goal, strategy_notes, strategies, strategy_hits}
    # 返回 {"outcome": "ok"|"partial"|"fail", "result": ..., "actions": [...], "blockers": [...]}
    # 执行器从 ctx["strategy_notes"] 读取本次自动检索到的策略要点并遵循。

    loop.review()                                # 定期复盘：卡点 → 策略入库（返回复盘报告）
    loop.lookup("web_scrape", "抓取股票报价")     # 主动查策略（不执行）
    loop.feedback(strategy_id, good=True)        # 效果反馈：有效/失效（多次失效会降权）
    loop.snapshot()                              # 日志统计 + 策略库概览

    # 可选：LLM 提炼策略（不传则离线模板，开箱即用）
    def distill(blocker):                        # blocker 含 scene/symptom/cause/count/draft_rule
        return call_llm("把以下复盘要点润色成可执行策略规则：" + str(blocker))
    loop2 = ExecutionLoop(llm_distill=distill)
    loop2.review()

## 4. 接入「大白」（零侵入，无需改核心）

1. **对话内工具（已就绪）**：skills/execution_loop 技能注册了 4 个工具：
   execution_review（复盘）/ strategy_lookup（动手前查策略）/ execution_record（登记任务）/
   strategy_feedback（效果反馈）。重启 server.py 或调 /api/harness/reload 热重载后，
   「大白」在对话中即可使用（on_demand 披露，配合内置 skill_help）。
2. **任务完成自动记录（可选一行）**：在任务执行点追加
       from execution_loop.hooks import record_dabai_task
       record_dabai_task(task_type=..., goal=..., outcome=..., blockers=...)
3. **执行前自动注入（可选一行）**：在决策循环里把
       from execution_loop.hooks import strategy_notes_for
       notes = strategy_notes_for(task_type, goal)
   的结果拼进 LLM 上下文。

所有落盘均用 tmp + os.replace 原子写，失败静默降级——闭环永不拖垮主流程。

## 5. 与 harness 已有经验记忆（lessons）的关系

harness/tasks.py 已有 _remember_lesson（扁平经验文本，60 条上限，覆盖 flow/batch 任务）。
本闭环与其**互补不冲突**，差异在于：

| | harness lessons | execution_loop |
|---|---|---|
| 形态 | 一句话经验文本 | 结构化：场景分类 + 规则 + 关键词 + 效果评分 |
| 数据来源 | 任务系统终态自动提炼 | 任何任务的执行日志（目标/动作/结果/卡点） |
| 复盘 | 无（直接入库） | 聚类去重 + 启发式/LLM 提炼 + 游标 |
| 检索 | 全量注入前 8 条 | 按场景+关键词打分 top-k |
| 效果反馈 | 无 | good/bad 反馈驱动排序与裁剪 |

## 6. 演示

    python examples/demo_iteration.py

三轮演示：空库失败 ×2 → 复盘提炼策略 → 同类任务自动检索并成功；
交叉验证另一场景（批量改名）互不污染；数据落在 data/demo/ 便于复查。

## 7. 注意事项

- **卡点质量决定策略质量**：日志的 blockers 字段是复盘唯一输入，登记时写清
  symptom（现象）与 cause（根因线索）会让提炼出的策略更可用。
- **自动调用是“建议”不是“强制”**：检索结果以 strategy_notes 注入上下文，
  执行决策仍由主 Agent/任务逻辑自己掌握；反馈 bad 可对失效策略降权直至停用。
- **定期复盘**：复盘按游标增量执行，可每日/每周或在连续失败后手工触发
  （execution_review 工具即为此设计）。
- **场景命名**：task_type 请保持稳定命名（如 web_scrape / file_batch / shell_deploy），
  命名漂移会导致策略无法被检索命中。
- **Windows 控制台**：脚本输出建议设 PYTHONIOENCODING=utf-8（emoji 在 GBK 代码页会报错）。
