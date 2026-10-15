"""搜索工具独立测试脚本
用法:
    python test_tools.py
    python test_tools.py "量子计算"   # 自定义关键词
    python test_tools.py --single tavily  # 只测试单个工具
"""

import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from research_system.config.settings import (
    TAVILY_ENABLED,
    CHROMA_ENABLED,
    GITHUB_ENABLED,
    TAVILY_API_KEY,
    CHROMA_PERSIST_DIR,
)
from research_system.tools import (
    tavily_search,
    chroma_retriever,
    github_search,
)


def print_header(text: str):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


async def test_tool(name: str, tool_module, keywords: list[str], enabled: bool, reason: str = ""):
    print(f"\n>> 测试: {name}")
    print(f"  关键词: {keywords}")
    if not enabled:
        print(f"  [跳过] {reason}")
        return {"name": name, "status": "skipped"}

    start = time.time()
    try:
        results = await asyncio.wait_for(
            tool_module.search(keywords), timeout=15.0
        )
        elapsed = time.time() - start
        print(f"  状态: 成功 ({elapsed:.1f}s)")
        print(f"  结果数: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"    [{i}] {r['title']}")
            print(f"        来源: {r['source']}")
            print(f"        链接: {r['url']}")
            content_preview = r['content'][:80].replace('\n', ' ')
            print(f"        摘要: {content_preview}...")
        return {"name": name, "status": "ok", "count": len(results), "elapsed": elapsed}
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  状态: 超时 ({elapsed:.1f}s) - 15秒未响应")
        return {"name": name, "status": "timeout"}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  状态: 失败 ({elapsed:.1f}s)")
        print(f"  错误: {e}")
        return {"name": name, "status": "error", "error": str(e)}


async def main():
    # 从命令行参数获取关键词，默认用 "深度学习"
    keywords = sys.argv[1:] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["深度学习"]
    keywords = [k for k in keywords if not k.startswith("--")]
    if not keywords:
        keywords = ["深度学习"]

    single_tool = None
    if "--single" in sys.argv:
        idx = sys.argv.index("--single")
        if idx + 1 < len(sys.argv):
            single_tool = sys.argv[idx + 1]

    print_header("Research System 工具测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  关键词:   {keywords}")
    print(f"  Tavily API Key: {'已配置' if TAVILY_API_KEY else '未配置'}")
    print(f"  ChromaDB 路径: {CHROMA_PERSIST_DIR}")

    tools = []
    if single_tool is None or single_tool == "tavily":
        tools.append(("Tavily 网页搜索", tavily_search, TAVILY_ENABLED, "未配置 API Key" if not TAVILY_API_KEY else ""))
    if single_tool is None or single_tool == "chroma":
        tools.append(("Chroma 知识库", chroma_retriever, CHROMA_ENABLED, ""))
    if single_tool is None or single_tool == "github":
        tools.append(("GitHub 仓库搜索", github_search, GITHUB_ENABLED, ""))

    if not tools:
        print(f"\n未知工具: {single_tool}，可选: tavily, chroma, github")
        return

    print_header(f"开始测试 ({len(tools)} 个工具)")
    results = await asyncio.gather(*[
        test_tool(name, mod, keywords, enabled, reason)
        for name, mod, enabled, reason in tools
    ])

    print_header("测试汇总")
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] not in ("ok", "skipped"))
    for r in results:
        icon = {"ok": "[OK]", "skipped": "[--]", "timeout": "[!!]", "error": "[!!]"}.get(r["status"], "[??]")
        extra = ""
        if r["status"] == "ok":
            extra = f" ({r['count']} 条结果, {r['elapsed']:.1f}s)"
        elif r["status"] == "error":
            extra = f" ({r.get('error', '')})"
        elif r["status"] == "skipped":
            extra = " (已禁用)"
        print(f"  {icon} {r['name']}: {r['status']}{extra}")
    print(f"\n总计: {len(results)} 个工具 | OK: {ok_count} | 跳过: {skipped} | 失败: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
