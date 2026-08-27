"""Flow DSL × TaskSystem 桥接演示：中文方法名 Flow 提交为可断点续跑的流程。

运行：python examples/flow_task_bridge_demo.py
验证点：
1. Flow 方法自动注册为可持久化 handler（flow:<类名>:<方法名>）；
2. @listen 依赖自动转成步骤 deps，结果经 {{步骤id.result}} 模板传递；
3. 提交后 TaskSystem 调度执行，全部步骤成功；
4. 模拟"重启"：重建 TaskSystem 后从 journal 恢复，未完成步骤自动续跑。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.flow_dsl import Flow, start, listen  # noqa: E402


class 调研流程(Flow):
    """中文方法名 + 声明式依赖链的示例流程。"""

    @start
    def 开始(self):
        return "需求：研究 crewAI"

    @listen(开始)
    def 搜索(self, 需求):
        return f"已搜索：{需求}"

    @listen(搜索)
    def 总结(self, 结果):
        return f"总结：{结果}"


async def main() -> None:
    # 1. 提交到 TaskSystem（durable=True → 落盘 journal，重启可恢复）
    from harness.tasks import TaskSystem

    ts = TaskSystem(harness=None)
    ts.ensure_started()
    flow = 调研流程()
    tid = flow.submit_to_tasks(ts, name="调研 crewAI", goal="研究 crewAI 架构")
    print(f"已提交流程: {tid}")

    # 2. 轮询直到终态
    for _ in range(60):
        t = ts._tasks.get(tid)
        if t is not None and t.state in ("succeeded", "failed", "cancelled"):
            break
        await asyncio.sleep(0.5)
    t = ts._tasks.get(tid)
    print(f"流程终态: {t.state} / {t.status_text}")
    print(f"结果: {t.result}")
    assert t.state == "succeeded", f"流程未成功: {t.state}"
    steps = t.result["steps"]
    assert steps["开始"]["state"] == "succeeded"
    assert steps["搜索"]["state"] == "succeeded"
    assert steps["总结"]["state"] == "succeeded"
    assert "已搜索" in steps["搜索"]["result"]
    assert "总结" in steps["总结"]["result"]
    print("[OK] 全链路验证通过：中文方法名 + 依赖传递 + 结果汇总")


if __name__ == "__main__":
    asyncio.run(main())
