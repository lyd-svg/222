import json
import re
from research_system.core.state import ResearchState
from research_system.core.llm import llm_call
from research_system.config.prompts import PLANNER_PROMPT


def _extract_json(text: str) -> dict | None:
    """从 LLM 回复中提取 JSON，支持 ```json 包裹"""
    # 尝试解析整段
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def planner_agent(state: ResearchState) -> ResearchState:
    state["progress"].append("📋 正在制定研究计划...")
    topic = state["topic"]
    response = llm_call(
        "", PLANNER_PROMPT.format(topic=topic), temperature=0.5
    )
    parsed = _extract_json(response)
    if parsed and all(k in parsed for k in ("sub_questions", "keywords", "search_depth", "required_sources")):
        state["plan"] = parsed
    else:
        # JSON 解析失败时让 LLM 重新生成，要求严格输出
        retry = llm_call(
            "你必须输出合法的 JSON，不要包含其他文字。",
            f"主题：{topic}\n\n"
            f"请输出JSON格式（不要包含```json标记）：\n"
            f'{{\n'
            f'  "sub_questions": ["子问题1", "子问题2", ...],\n'
            f'  "keywords": ["关键词1", "关键词2", ...],\n'
            f'  "search_depth": "浅度/中等/深度",\n'
            f'  "required_sources": ["知识库查询", "联网搜索"]\n'
            f'}}',
            temperature=0.3,
        )
        parsed = _extract_json(retry)
        if parsed:
            state["plan"] = parsed
        else:
            # 最终 fallback：用 LLM 分析主题生成合理的关键词
            keywords = llm_call(
                "提取 3-5 个中文关键词。",
                f"从以下研究主题中提取核心关键词：{topic}",
                temperature=0.3,
            )
            state["plan"] = {
                "sub_questions": [f"什么是{topic}", f"{topic}的最新进展", f"{topic}的关键技术"],
                "keywords": [k.strip() for k in keywords.replace("\n", ",").split(",") if k.strip()],
                "search_depth": "中等",
                "required_sources": ["知识库搜索", "联网搜索github与tavily"],
            }
    sub_count = len(state["plan"].get("sub_questions", []))
    state["progress"].append(f"✅ 研究计划完成，共 {sub_count} 个子问题")
    return state
