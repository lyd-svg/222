"""ChromaDB 知识库检索（复用 knowledge_builder 的层级检索）"""

import asyncio
from research_system.knowledge_builder import query_knowledge_base


async def search(keywords: list[str]) -> list[dict]:
    """搜索本地知识库，返回统一格式的结果"""
    try:
        query = " ".join(keywords)
        results = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: query_knowledge_base(query, n_results=3)
            ),
            timeout=15.0,
        )

        items = []
        for r in results:
            items.append({
                "title": r.get("section_title", "知识库条目"),
                "content": r["text"],
                "source": "Chroma知识库",
                "url": f"https://arxiv.org/abs/{r['paper_id']}",
                "metadata": {
                    "paper_id": r.get("paper_id", ""),
                    "hierarchy": r.get("hierarchy", ""),
                    "chunks_used": r.get("chunks_used", []),
                    "total_tokens": r.get("total_tokens", 0),
                },
            })
        return items
    except Exception as e:
        print(f"Chroma 检索失败: {e}")
        return []
