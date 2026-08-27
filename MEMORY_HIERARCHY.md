# 大白分层记忆上下文管理方案
## 0. 当前实现状态（2026-08-25 已落地）
- `memory.py:build_hierarchical_context` 四层打包 + token 预算已实现并由 `agent.py` 调用；
- `record_context_stats` 已接入 `agent.py` 普通对话主循环：每轮对话结束把
  raw/packed 估算与 LLM 返回的真实 prompt 用量写入 `context_stats` 表；
- `settings.json -> memory` 段已加入分层配置（默认值与代码常量一致，可随时覆盖）；
- 新增 `memory_benchmark.py`：基于 `chat_memory.db` 真实数据回放，输出旧规则 vs
  分层的 token 节省对比（真实会话实测平均节省约 70%+）。

## 1. 背景与现状调研

大白当前的上下文由三部分拼装（`agent.py._chat_stream_normal` / `_chat_stream_game`）：

| 层 | 现状实现 | 位置 |
| --- | --- | --- |
| 短期记忆 | 服务端连接级 `history`（最多 200 条→裁 100）+ 记忆库最近 10 条消息；Agent 用滞回窗口（满 12 轮裁回 8 轮）保持前缀缓存稳定 | `server.py` / `agent.py:_stable_history_view` / `memory.py:get_context_messages` |
| 长期摘要 | 每 10 条消息（≥20 条后）触发 LLM 增量摘要，最多注入 3 条、合并限 600 字 | `memory.py:maybe_summarize` / `_generate_summary` |
| 长期关键信息 | LLM/关键词提取用户信息入 `user_memories`，常驻注入 top-2 | `memory.py:extract_and_save_memories` / `get_user_memories` |
| 检索召回 | 2-gram 关键词检索 用户记忆+摘要+最近消息，注入上限 800 字符 | `memory.py:recall_memories` / `build_recall_block` |

调研发现的问题：

1. **预算常量是死代码**：`CONTEXT_TOKEN_BUDGET_RATIO` / `MAX_PROMPT_TOKENS` / `MEMORY_BLOCK_RATIO`
   已定义但从未被引用；`RECALL_MAX_TOKENS` 定义了但 `build_recall_block` 只按字符截断。
2. **上下文可能无限膨胀**：短期窗口只按「轮数」截断（12 轮），单轮超长（实测最长消息 1360 字符）
   不会被压缩；摘要固定 3 条 600 字，不随会话增长降级。
3. **没有量化统计**：`context_stats` 表已建但 `record_context_stats` 从未实现，无法衡量 token 节省。

## 2. 分层方案设计

在保留现有「静态大前缀在前、动态尾巴在后」的缓存友好排布前提下，给每层加 token 预算：

```
messages[0]  system 人设/资源/技能（静态前缀，不变）
messages[1]  system 长期摘要块（最新在前，预算 summary_max_tokens）
messages[2]  system 常驻长期记忆块（按 importance，预算 long_term_max_tokens）
messages[3]  system 易变状态（形象/场景/音乐/媒体/子智能体）
messages[4]  system 相关召回块（按相关性，预算 recall_max_tokens，强制生效）
messages[5..]       短期窗口（最近轮次，预算 short_term_max_tokens，由新到旧截断）
messages[-1] user 当前输入
```

各层预算（可在 `settings.json -> memory` 段覆盖，默认值）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `hierarchical_packing` | `true` | 总开关；`false` 完全回退旧行为 |
| `token_budget_ratio` | `0.5` | 提示词预算 = context_window × ratio（`max_prompt_tokens>0` 时优先用固定值） |
| `short_term_max_tokens` | `1200` | 短期窗口预算（由新到旧截断，最旧优先被挤出） |
| `short_term_max_chars_per_round` | `500` | 单轮历史最大字符，超出截断加 `…` |
| `summary_max_tokens` | `200` | 长期摘要块预算 |
| `summary_max_chars_per_item` | `120` | 单条摘要最大字符 |
| `long_term_max_tokens` | `150` | 常驻长期记忆块预算 |
| `long_term_top_k` | `2` | 常驻长期记忆条数（保持记忆侧稳定排序） |
| `recall_max_tokens` | `200` | 召回块预算（与 `recall_max_chars` 取更小者，强制生效） |
| `record_stats` | `true` | 每轮写入 `context_stats`（raw/packed/actual 用量） |

### 量化口径

每轮对话写入一行 `context_stats`：

- `raw_tokens`：旧规则（不压缩）下会注入的估算 token —— 完整滞回窗口 +
  全部摘要 + 常驻记忆 + 召回字符上限；
- `packed_tokens`：分层预算实际注入的估算 token；
- `actual_prompt_tokens`：LLM 返回的真实 prompt usage（LLM 调用后回填）；
- `history_rounds` / `summaries` / `memories` / `recall_items`：各层实际条数。

## 3. 改动文件与回滚

- `memory.py`：新增分层打包、预算执行、`record_context_stats`；旧函数保持签名兼容。
- `agent.py`：普通/游戏模式的消息组装改为调用分层打包；回退开关关闭时行为与旧版一致。
- `settings.json`：`memory` 段新增分层配置。
- 新增 `memory_benchmark.py`：基于 `chat_memory.db` 真实数据回放，输出节省对比。

回滚方式：

1. `settings.json -> memory.hierarchical_packing = false`（软回退，不改代码）；
2. 恢复编辑前备份 `memory.py.bak-hierarchical-*` / `agent.py.bak-hierarchical-*` /
   `settings.json.bak-hierarchical-*`（硬回滚）；
3. 数据库只新增行（`context_stats`），不改旧表结构，无需迁移。

## 4. 预期收益

- 长会话（>20 轮）从「全部历史 + 固定摘要」变为「摘要预算 + 最新窗口预算」，
  单轮 prompt token 从随会话线性增长变为有上界；
- 超长消息不再整条塞入，单轮 1360 字符的消息被截到 500 字符；
- 召回/摘要/记忆三层合计默认约 550 token 封顶，历史窗口 1200 token 封顶；
- 量化数据落库，可持续校准估算误差。
