from research_system.core.state import ResearchState
from research_system.core.llm import llm_call
from research_system.config.prompts import REWRITER_PROMPT


def rewrite_agent(state: ResearchState) -> ResearchState:
    round_num = state["debate_round"] + 1
    state["progress"].append(f"🔄 正在根据评审意见修改报告（第 {round_num} 轮）...")
    revised = llm_call(
        "你是一个专业的研究编辑。",
        REWRITER_PROMPT.format(
            feedback=state["review_feedback"],
            report=state["report_draft"],
        ),
        temperature=0.7,
    )
    state["report_draft"] = revised
    state["debate_round"] += 1
    state["progress"].append(f"✅ 第 {round_num} 轮修改完成")
    return state
