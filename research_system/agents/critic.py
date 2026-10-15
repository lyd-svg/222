import json
import re
from research_system.core.state import ResearchState
from research_system.core.llm import llm_call
from research_system.config.prompts import CRITIC_PROMPT
from research_system.config.settings import QUALITY_PASS_THRESHOLD


def _parse_score(response: str) -> dict:
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', response)
        score = float(score_match.group(1)) if score_match else 3.0
        return {
            "score": score,
            "feedback": response,
            "dimensions": {
                "accuracy": score,
                "completeness": score,
                "consistency": score,
                "citations": score,
            },
        }


def critic_agent(state: ResearchState) -> ResearchState:
    state["progress"].append("🔎 正在评审报告质量...")
    response = llm_call(
        "你是一个严格的学术审稿人。",
        CRITIC_PROMPT.format(report=state["report_draft"]),
        temperature=0.3,
    )
    parsed = _parse_score(response)
    score = parsed.get("score", 3.0)
    passed = score >= QUALITY_PASS_THRESHOLD
    state["quality_score"] = {
        "score": score,
        "passed": passed,
        "dimensions": parsed.get("dimensions", {}),
    }
    state["review_feedback"] = parsed.get("feedback", response)

    round_info = f"（第 {state['debate_round']} 轮评审）" if state["debate_round"] > 0 else ""
    if passed:
        state["progress"].append(f"🔎 质量评分 {score}/5.0，✅ 通过{round_info}")
    else:
        state["progress"].append(f"🔎 质量评分 {score}/5.0，❌ 未通过{round_info}")
    return state
