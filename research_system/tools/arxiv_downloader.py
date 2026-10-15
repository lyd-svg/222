"""arXiv LaTeX 源码下载器

用法:
    import asyncio
    from research_system.tools.arxiv_downloader import search_arxiv_papers, download_latex_source

    async def main():
        # 搜索论文
        papers = await search_arxiv_papers("transformer attention mechanism", max_results=5)

        # 下载 LaTeX 源码
        for paper in papers:
            result = await download_latex_source(paper["arxiv_id"], paper["title"])
            print(f"已下载: {result['save_dir']}")
"""

import os
import re
import tarfile
import io
import httpx
from typing import Optional

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIX_EPS = "https://arxiv.org/e-print"
HEADERS = {"User-Agent": "ResearchSystem/1.0"}
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arxiv_math")


def _sanitize_filename(name: str) -> str:
    """清理文件名，移除非法字符"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.strip()[:200]


async def search_arxiv_papers(
    query: str, max_results: int = 10
) -> list[dict]:
    """通过 arXiv API 搜索论文，返回元信息列表"""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(ARXIV_API_BASE, params=params, headers=HEADERS)

    resp.raise_for_status()

    papers = []
    # arXiv API 返回 XML，用简单正则提取关键字段
    xml = resp.text
    entries = xml.split("<entry>")[1:]

    for entry in entries:
        arxiv_id = (
            entry.split("<id>")[1].split("</id>")[0].strip().split("/abs/")[-1]
        )
        title = entry.split("<title>")[1].split("</title>")[0].strip()
        title = re.sub(r"\s+", " ", title)
        summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
        summary = re.sub(r"\s+", " ", summary)[:500]

        published = entry.split("<published>")[1].split("</published>[0]")[0].strip()[:10]
        # 提取作者
        authors = re.findall(r"<name>([^<]+)</name>", entry)

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "published": published,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
            }
        )

    return papers


async def download_latex_source(
    arxiv_id: str,
    title: str,
    save_dir: Optional[str] = None,
) -> dict:
    """下载 arXiv 论文的 LaTeX 源码并解压到本地

    参数:
        arxiv_id: arXiv ID (如 "2401.12345" 或 "cs/9901001")
        title: 论文标题（用于目录名）
        save_dir: 保存根目录，默认 research_system/arxiv_math/

    返回:
        {
            "arxiv_id": str,
            "title": str,
            "save_dir": str,        # 解压目录
            "tex_files": [str],     # 所有 .tex 文件路径
            "file_count": int,      # 总文件数
        }
    """
    save_root = save_dir or DEFAULT_SAVE_DIR
    folder_name = f"{arxiv_id.replace('/', '_')}_{_sanitize_filename(title)[:40]}"
    target_dir = os.path.join(save_root, folder_name)

    # 本地已存在则跳过
    if os.path.exists(target_dir):
        tex_files = _find_tex_files(target_dir)
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "save_dir": target_dir,
            "tex_files": tex_files,
            "file_count": len(tex_files),
            "from_cache": True,
        }

    url = f"{ARXIX_EPS}/{arxiv_id}"

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=HEADERS)

    if resp.status_code == 403:
        raise PermissionError(f"arXiv 拒绝访问 {arxiv_id}，可能论文未开源源码")
    resp.raise_for_status()

    os.makedirs(target_dir, exist_ok=True)

    # arXiv 返回 tar.gz 或单个 .tex 文件
    tex_files = []

    if resp.headers.get("content-type", "").startswith("application/x-eprint"):
        # tar.gz 包
        try:
            tar = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
            tar.extractall(path=target_dir)
            tar.close()
        except tarfile.ReadError:
            # 不是 tar 包，可能是单个 .tex
            _save_as_single_file(target_dir, arxiv_id, resp.text)
    elif resp.text.strip().startswith("\\") or resp.text.strip().startswith("%"):
        # 单个 .tex 文件
        _save_as_single_file(target_dir, arxiv_id, resp.text)
    else:
        # 尝试当成 tar 解压（有的 arXiv 包 content-type 不标准）
        try:
            tar = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
            tar.extractall(path=target_dir)
            tar.close()
        except tarfile.ReadError:
            _save_as_single_file(target_dir, arxiv_id, resp.text)

    _remove_pdf_tex_files(target_dir)
    tex_files = _find_tex_files(target_dir)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "save_dir": target_dir,
        "tex_files": tex_files,
        "file_count": len(tex_files),
        "from_cache": False,
    }


def _save_as_single_file(target_dir: str, arxiv_id: str, content: str):
    """将单个 .tex 内容写入文件"""
    tex_path = os.path.join(target_dir, f"{arxiv_id.replace('/', '_')}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)


def _find_tex_files(directory: str) -> list[str]:
    """递归查找目录下所有 .tex 文件"""
    tex_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".tex"):
                tex_files.append(os.path.join(root, f))
    return tex_files


def _remove_pdf_tex_files(directory: str):
    """删除被误标为 .tex 的 PDF 文件"""
    for root, _, files in os.walk(directory):
        for f in files:
            if not f.endswith(".tex"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "rb") as fh:
                    head = fh.read(30)
                if head.startswith(b"%PDF"):
                    os.remove(path)
                    print(f"  -> 删除 PDF 伪装文件: {path}")
            except Exception:
                pass


async def download_papers_by_query(
    query: str,
    max_results: int = 5,
    save_dir: Optional[str] = None,
) -> list[dict]:
    """搜索并下载多篇论文的 LaTeX 源码

    参数:
        query: 搜索关键词
        max_results: 最大下载篇数
        save_dir: 保存根目录

    返回:
        每篇论文的下载结果列表
    """
    papers = await search_arxiv_papers(query, max_results)
    results = []

    for paper in papers:
        try:
            result = await download_latex_source(
                paper["arxiv_id"], paper["title"], save_dir
            )
            results.append(result)
            print(f"  OK {paper['arxiv_id']} -> {result['save_dir']}")
        except Exception as e:
            print(f"  x {paper['arxiv_id']} 下载失败: {e}")

    return results


# 命令行入口：python -m research_system.tools.arxiv_downloader "your query"
if __name__ == "__main__":
    import asyncio

    async def cli():
        import sys

        query = " ".join(sys.argv[1:]) or "machine learning"
        print(f"搜索并下载: {query}")
        results = await download_papers_by_query(query, max_results=3)
        print(f"\n共下载 {len(results)} 篇论文")

    asyncio.run(cli())
