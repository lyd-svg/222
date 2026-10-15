import asyncio
from tavily import TavilyClient
from research_system.config.settings import TAVILY_API_KEY

_client = None


def get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


async def search(keywords: list[str]) -> list[dict]:
    if not TAVILY_API_KEY:
        return []

    client = get_client()
    query = " ".join(keywords[:3])
    response = await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.search(query=query, search_depth="advanced", max_results=5),
        ),
        timeout=15.0,
    )
    results = []
    for r in response.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "source": "Tavily",
                "url": r.get("url", ""),
            }
        )
    return results
