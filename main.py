import asyncio
from research_system.core.graph import run_research_stream


async def main():
    print("=" * 50)
    print("  🔬 四Agent协作研究系统")
    print("=" * 50)

    topic = input("\n请输入研究主题: ").strip()
    if not topic:
        print("主题不能为空")
        return

    print()

    # 流式输出进度
    final_state = None
    async for msg_type, content in run_research_stream(topic):
        if msg_type == "status":
            print(content)
        elif msg_type == "result":
            final_state = content

    if not final_state:
        return

    # 简单问题：直接显示回答
    if not final_state.get("is_complex"):
        print("\n" + "=" * 50)
        print(final_state.get("final_report", ""))
        return

    # 复杂课题：显示研究总结
    print("\n" + "=" * 50)

    if final_state.get("plan"):
        print("📋 子问题:")
        for q in final_state["plan"].get("sub_questions", []):
            print(f"  - {q}")

    if final_state.get("sources"):
        print(f"\n🔍 共找到 {len(final_state['sources'])} 条资料")

    score = final_state.get("quality_score", {}).get("score", 0)
    passed = final_state.get("quality_score", {}).get("passed", False)
    print(f"\n🔎 质量评分: {score}/5.0 {'✅ 通过' if passed else '❌ 未通过'}")

    if final_state.get("debate_round", 0) > 0:
        print(f"   辩论轮次: {final_state['debate_round']}")

    print("\n" + "=" * 50)
    print("📝 最终报告:")
    print(final_state.get("final_report") or final_state.get("report_draft", ""))


if __name__ == "__main__":
    asyncio.run(main())
