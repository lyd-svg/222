from research_system.core.state import ResearchState
from research_system.core.llm import llm_call
from research_system.config.prompts import CLASSIFIER_PROMPT, DIRECT_ANSWER_PROMPT


def classify_topic(state: ResearchState) -> ResearchState:
    state["progress"].append("⏳ 正在分析问题类型...")
    result = (
        llm_call("", CLASSIFIER_PROMPT.format(question=state["topic"]))
        .strip()
        .lower()
    )
    is_complex = result == "complex"
    state["is_complex"] = is_complex
    if is_complex:
        state["progress"].append("✅ 确定为复杂课题，进入研究流程")
    else:
        state["progress"].append("✅ 简单问题，直接回答")
    return state


def direct_answer_agent(state: ResearchState) -> ResearchState:
    state["progress"].append("💬 正在生成回答...")
    answer = llm_call(
        "你是一个有帮助的AI助手。",
        DIRECT_ANSWER_PROMPT.format(
            question=state["topic"],
        ),
        temperature=0.5,
    )
    state["final_report"] = answer
    return state
