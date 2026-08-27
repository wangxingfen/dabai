"""Flow DSL 演示：声明式多步骤流程（借鉴 crewAI @start/@listen）。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.flow_dsl import Flow, start, listen


class 调研流程(Flow):
    @start
    def 开始(self):
        print("[1] 开始：定义需求")
        return "研究 crewAI 的 Flow 架构"

    @listen(开始)
    def 搜索(self, 需求):
        print(f"[2] 搜索：{需求}")
        return f"已搜索到资料：{需求}"

    @listen(搜索)
    def 总结(self, 资料):
        print(f"[3] 总结：{资料}")
        return f"最终结论：{资料} 的要点已整理"


def main():
    flow = 调研流程()
    flow.kickoff()
    print("\n=== 各步骤结果 ===")
    for name, res in flow.results().items():
        print(f"  {name}: {res}")


if __name__ == "__main__":
    main()