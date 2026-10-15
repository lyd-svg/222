import httpx


async def search(keywords: list[str]) -> list[dict]:
    query = "+".join(keywords[:3])
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=5"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append(
                {
                    "title": item.get("full_name", ""),
                    "content": item.get("description", "") or "",
                    "source": "GitHub",
                    "url": item.get("html_url", ""),
                }
            )
        return results
