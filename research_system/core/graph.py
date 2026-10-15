import asyncio
from langgraph.graph import StateGraph, END
from research_system.core.state import ResearchState
from research_system.agents.classifier import classify_topic, direct_answer_agent
from research_system.agents.planner import planner_agent
from research_system.agents.searcher import searcher_agent
from research_system.agents.analyst import analyst_agent
from research_system.agents.critic import critic_agent
from research_system.agents.rewriter import rewrite_agent
from research_system.config.settings import MAX_DEBATE_ROUNDS


def build_research_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    workflow.add_node("classifier", classify_topic)
    workflow.add_node("direct_answer", direct_answer_agent)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("searcher", searcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("rewriter", rewrite_agent)
    workflow.add_node("finalize", lambda s: s)

    def routing_decision(state: ResearchState) -> str:
        if state.get("is_complex"):
            return "complex"
        return "simple"

    workflow.add_conditional_edges(
        "classifier",
        routing_decision,
        {"simple": "direct_answer", "complex": "planner"},
    )

    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "analyst")
    workflow.add_edge("analyst", "critic")

    def debate_decision(state: ResearchState) -> str:
        if state["quality_score"].get("passed", False):
            return "pass"
        elif state["debate_round"] >= MAX_DEBATE_ROUNDS:
            return "force_exit"
        else:
            return "rewrite"

    workflow.add_conditional_edges(
        "critic",
        debate_decision,
        {
            "pass": "finalize",
            "rewrite": "rewriter",
            "force_exit": "finalize",
        },
    )

    workflow.add_edge("rewriter", "critic")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("finalize", END)

    workflow.set_entry_point("classifier")

    return workflow.compile()


def _make_initial_state(topic: str, history: list = None) -> ResearchState:
    return {
        "topic": topic,
        "is_complex": None,
        "plan": None,
        "sources": [],
        "search_strategy": "",
        "search_evaluation": "",
        "report_draft": "",
        "review_feedback": "",
        "quality_score": {},
        "debate_round": 0,
        "final_report": "",
        "history": history or [],
        "progress": [],
    }


async def run_research(topic: str, history: list = None) -> ResearchState:
    """原始非流式接口（用于 gradio 等场景，保持向后兼容）"""
    graph = build_research_graph()
    result = await graph.ainvoke(_make_initial_state(topic, history))
    return result


async def run_research_stream(topic: str, history: list = None):
    """流式接口：yield (type, content) 元组

    type:
      - "status"  → 进度消息
      - "result"  → 最终状态 (ResearchState dict)
    """
    graph = build_research_graph()
    initial_state = _make_initial_state(topic, history)

    last_progress_count = 0
    final_state = {**initial_state}

    async for event in graph.astream(initial_state, stream_mode="updates"):
        for _, update in event.items():
            # 逐条 yield 新增的进度消息
            progress_msgs = update.get("progress", [])
            for msg in progress_msgs[last_progress_count:]:
                yield ("status", msg)
            last_progress_count = len(progress_msgs)

            # 合并所有状态变更
            final_state.update(update)

    yield ("result", final_state)
