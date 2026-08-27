"""execution_loop —— 「大白」的执行力自我迭代闭环。

执行日志 → 复盘提炼 → 策略沉淀 → 自动检索调用

- logger.ExecutionLogger    第 1 环：结构化执行日志（目标/动作/结果/卡点）
- reviewer.Reviewer         第 2 环：卡点聚类 → 策略规则提炼
- store.StrategyStore       第 3 环：策略库持久化（按任务类型分类 + 效果评分）
- retriever.StrategyRetriever 第 4 环：执行前按场景/关键词打分检索
- agent.ExecutionLoop       闭环总入口：retrieve → run → record 一条龙
- hooks                     与「大白」运行时接线的桥（技能工具 + 自动记录点）

快速开始：
    from execution_loop import ExecutionLoop
    loop = ExecutionLoop()
    result = loop.execute(task_type="web_scrape", goal="抓取新闻",
                          run=lambda ctx: {"outcome": "ok", "result": "..."})
    loop.review()   # 定期复盘沉淀策略
"""
from .agent import ExecutionLoop, default_loop
from .logger import ExecutionLogger, OUTCOME_OK, OUTCOME_PARTIAL, OUTCOME_FAIL
from .reviewer import Reviewer
from .retriever import StrategyRetriever
from .store import StrategyStore

__all__ = ["ExecutionLoop", "ExecutionLogger", "Reviewer", "StrategyRetriever",
           "StrategyStore", "OUTCOME_OK", "OUTCOME_PARTIAL", "OUTCOME_FAIL",
           "default_loop"]
__version__ = "1.0.0"
