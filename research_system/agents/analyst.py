from research_system.core.state import ResearchState
from research_system.core.llm import llm_call
from research_system.config.prompts import WRITER_PROMPT


def analyst_agent(state: ResearchState) -> ResearchState:
    state["progress"].append("📝 正在撰写研究报告...")
    sources_text = "\n\n".join(
        [
            f"[{s['source']}] {s['title']}\n{s['content']}\n来源: {s['url']}"
            for s in state["sources"]
        ]
    )
    report = llm_call(
        "你是一个专业的研究报告撰写专家。",
        WRITER_PROMPT.format(
            sources=sources_text,
        ),
        temperature=0.7,
    )
    state["report_draft"] = report
    state["progress"].append("✅ 报告初稿完成")
    return state
