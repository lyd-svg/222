import asyncio
from research_system.core.state import ResearchState
from research_system.config.settings import (
    TAVILY_ENABLED,
    CHROMA_ENABLED,
    GITHUB_ENABLED,
)
from research_system.tools import (
    tavily_search,
    chroma_retriever,
    github_search,
)


async def searcher_agent(state: ResearchState) -> ResearchState:
    plan = state.get("plan", {})
    topic = state.get("topic", "")

    keywords = plan.get("keywords", [topic])
    sub_questions = plan.get("sub_questions", [])
    # 合并关键词和子问题作为搜索查询
    search_queries = keywords + sub_questions

    enabled_tools = []
    if TAVILY_ENABLED:
        enabled_tools.append("Tavily")
    if CHROMA_ENABLED:
        enabled_tools.append("知识库")
    if GITHUB_ENABLED:
        enabled_tools.append("GitHub")

    tools_str = "、".join(enabled_tools) if enabled_tools else "无可用工具"
    state["progress"].append(f"🔍 正在通过 {tools_str} 并行搜索...")

    tasks = []
    if TAVILY_ENABLED:
        tasks.append(tavily_search.search(search_queries))
    if CHROMA_ENABLED:
        tasks.append(chroma_retriever.search(search_queries))
    if GITHUB_ENABLED:
        tasks.append(github_search.search(search_queries))

    if not tasks:
        state["sources"] = []
        state["progress"].append("⚠️ 没有可用的搜索工具")
        return state

    results = await asyncio.gather(*tasks)

    seen = set()
    merged = []
    for batch in results:
        for item in batch:
            dedup_key = item["title"] + item["content"][:50]
            if dedup_key not in seen:
                seen.add(dedup_key)
                merged.append(item)

    state["sources"] = merged
    state["progress"].append(f"✅ 共找到 {len(merged)} 条相关资料")
    return state
