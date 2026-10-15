"""LaTeX 源码知识库构建器

三步流程:
  1. 结构解析 — 扫描 arxiv_math/，展开 \input/\include，按 section 层级建文档树
  2. 块识别 —  识别 atomic 块（equation/theorem/proof/...），不可切分
  3. 递归切分 — text 块按 MAX_CHUNK 递归切分，atomic 块整体保留

用法:
    from research_system.knowledge_builder import build_knowledge_base
    build_knowledge_base(incremental=True)
"""

import os
import re
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_system.config.settings import CHROMA_PERSIST_DIR
from research_system.tools.embeddings import MultilingualEmbeddingFunction

# ── 常量 ──────────────────────────────────────────────────────────────────

ARXIV_MATH_DIR = os.path.join(os.path.dirname(__file__), "arxiv_math")
COLLECTION_NAME = "latex_papers"

MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP = 50

# 不可切分的 LaTeX 环境（原子块）
ATOMIC_ENVIRONMENTS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multiline", "multiline*", "flalign", "flalign*",
    "theorem", "lemma", "proof", "definition", "corollary",
    "proposition", "remark", "example", "conjecture", "claim",
    "algorithm", "algorithmic",
    "lstlisting", "verbatim", "minted",
}

# 层级映射（数字越小越上层）
LEVEL_MAP = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}

# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass
class TextBlock:
    """单个内容块"""
    text: str
    block_type: str  # "text" | "atomic"
    env_type: str = ""  # 当 atomic 时记录环境名


@dataclass
class LatexNode:
    """文档树节点"""
    title: str
    level: str  # "section" | "subsection" | ...
    content: str  # 原始文本（不含子节点）
    children: list = field(default_factory=list)
    label: str = ""
    env_type: str = ""


@dataclass
class Chunk:
    """最终的 chunk，准备入库"""
    text: str
    metadata: dict


# ── Step 1: 扫描 & 展开 ──────────────────────────────────────────────────


def _scan_paper_dirs(base_dir: str) -> list[dict]:
    """扫描 arxiv_math/，返回每篇论文的信息"""
    papers = []
    if not os.path.isdir(base_dir):
        return papers

    for entry in sorted(os.listdir(base_dir)):
        paper_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(paper_dir):
            continue

        tex_files = _find_tex_files(paper_dir)
        if not tex_files:
            continue

        # 优先用 main.tex，其次用最大的 .tex，最后用第一个
        main = next((f for f in tex_files if os.path.basename(f).startswith("main")), None)
        if not main:
            main = max(tex_files, key=lambda f: os.path.getsize(f))

        papers.append({
            "dir": paper_dir,
            "main_tex": main,
            "all_tex": tex_files,
            "paper_id": entry.split("_")[0] if "_" in entry else entry[:20],
            "paper_title": _extract_paper_title(entry),
        })

    return papers


def _find_tex_files(directory: str) -> list[str]:
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".tex"):
                result.append(os.path.join(root, f))
    return result


def _extract_paper_title(entry_name: str) -> str:
    """从目录名提取论文标题"""
    parts = entry_name.split("_", 1)
    if len(parts) > 1:
        return parts[1][:120]
    return entry_name


def _is_valid_tex(filepath: str) -> bool:
    """检查文件是否是真正的 .tex 文件（不是 PDF 伪装）"""
    try:
        with open(filepath, "rb") as f:
            head = f.read(30)
        return not head.startswith(b"%PDF")
    except Exception:
        return False


def _expand_tex(filepath: str) -> str:
    """递归展开 \input/\include，返回完整 LaTeX 源码"""
    if not os.path.isfile(filepath):
        return ""

    if not _is_valid_tex(filepath):
        print(f"    -> 跳过（非 LaTeX 文件）")
        return ""

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""

    base_dir = os.path.dirname(filepath)

    def _resolve_filename(name: str) -> Optional[str]:
        # 去掉多余空格
        name = name.strip()
        for ext in ["", ".tex"]:
            # 尝试相对于当前目录
            path = os.path.join(base_dir, name + ext)
            if os.path.isfile(path):
                return path
            # 尝试相对于 arxiv_math 根目录
            path2 = os.path.join(ARXIV_MATH_DIR, name + ext)
            if os.path.isfile(path2):
                return path2
        return None

    def _inline(m: re.Match) -> str:
        cmd = m.group(1)  # input 或 include
        fname = m.group(2) or m.group(3) or ""
        resolved = _resolve_filename(fname)

        # 处理 \input{dir/file}
        if not resolved:
            alt = fname.replace("\\", "/")
            if "/" in alt:
                for ext in ["", ".tex"]:
                    path3 = os.path.join(base_dir, alt + ext)
                    if os.path.isfile(path3):
                        resolved = path3
                        break

        if resolved and os.path.isfile(resolved):
            return _expand_tex(resolved)
        return f"% MISSING \\{cmd}{{{fname}}}"

    # \input{file} 和 \include{file}
    content = re.sub(
        r'\\(input|include)\{([^}]+)\}',
        _inline,
        content,
    )
    # \input file（无大括号）
    content = re.sub(
        r'\\(input)\s+([a-zA-Z_/][a-zA-Z0-9_/.\\-]*)',
        _inline,
        content,
    )
    return content


# ── Step 2: 结构解析 ─────────────────────────────────────────────────────


def _parse_structure(content: str) -> LatexNode:
    """将 LaTeX 源码解析为文档树"""
    root = LatexNode(title="Document", level="document", content="")
    stack = [root]
    prev_end = 0

    section_pattern = re.compile(
        r'\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)'
        r'(?:\*)?'  # 允许带星号 \section*
        r'(?:\[([^\]]*)\])?'  # 可选短标题 [toc-title]
        r'\{(.*?)\}',  # 标题正文 {title}
        re.DOTALL,
    )

    for m in section_pattern.finditer(content):
        level_name = m.group(1)
        short_title = m.group(2) or ""
        title = m.group(3).strip()

        # 当前节点之前的正文部分
        pre_text = content[prev_end:m.start()].strip()
        if pre_text:
            stack[-1].content += pre_text + "\n\n"

        # 清理标题中的 LaTeX 命令
        clean_title = _clean_latex_text(title)
        new_node = LatexNode(
            title=clean_title,
            level=level_name,
            content="",
        )

        # 根据层级弹出栈
        new_level = LEVEL_MAP.get(level_name, 99)
        while len(stack) > 1:
            parent_level = LEVEL_MAP.get(stack[-1].level, -1)
            if new_level > parent_level:
                break
            stack.pop()

        stack[-1].children.append(new_node)
        stack.append(new_node)
        prev_end = m.end()

    # 剩余正文
    remaining = content[prev_end:].strip()
    if remaining:
        stack[-1].content += remaining

    return root


def _clean_latex_text(text: str) -> str:
    """清理标题/文本中的 LaTeX 命令"""
    # 移除 \label, \ref, \cite
    text = re.sub(r'\\(label|ref|cite)\{([^}]*)\}', r'\2', text)
    # 移除 \textbf{}, \textit{} 等格式命令但保留内容
    text = re.sub(r'\\(textbf|textit|emph|texttt|textsf|textsc|textup)\{([^}]*)\}', r'\2', text)
    # 移除脚注
    text = re.sub(r'\\footnote\{[^}]*\}', '', text)
    # 移除连续空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Step 2.5: 块识别 ─────────────────────────────────────────────────────


def _extract_blocks(text: str) -> list[TextBlock]:
    """将节点文本拆为块序列，标记 atomic / text"""
    blocks = []

    # 匹配所有 \begin{env}...\end{env}
    env_pattern = re.compile(
        r'\\begin\{(' + '|'.join(ATOMIC_ENVIRONMENTS) + r')\}'
        r'(.*?)'
        r'\\end\{\1\}',
        re.DOTALL,
    )

    pos = 0
    for m in env_pattern.finditer(text):
        # 前面的普通文本
        if m.start() > pos:
            pre = text[pos:m.start()].strip()
            if pre:
                blocks.append(TextBlock(text=pre, block_type="text"))

        env_name = m.group(1)
        env_content = m.group(0).strip()
        blocks.append(TextBlock(text=env_content, block_type="atomic", env_type=env_name))
        pos = m.end()

    # 剩余文本
    remaining = text[pos:].strip()
    if remaining:
        blocks.append(TextBlock(text=remaining, block_type="text"))

    return blocks


# ── Step 3: 递归切分 ─────────────────────────────────────────────────────


def _build_metadata(
    paper_id: str,
    paper_title: str,
    hierarchy: list[str],
    level: str,
    section_title: str,
    chunk_idx: int,
    total_chunks: int,
    file_hash: str,
    has_math: bool = False,
    token_count: int = 0,
) -> dict:
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "hierarchy": " > ".join(hierarchy),
        "level": level,
        "section_title": section_title,
        "chunk_idx": chunk_idx,
        "total_chunks": total_chunks,
        "has_math": str(has_math),
        "file_hash": file_hash,
        "token_count": token_count,
    }


def _token_count(text: str) -> int:
    """粗略估算 token 数（中英文统一按字符/2）"""
    return max(1, len(text) // 2)


def _chunk_blocks(
    blocks: list[TextBlock],
    paper_id: str,
    paper_title: str,
    hierarchy: list[str],
    level: str,
    section_title: str,
    file_hash: str,
) -> list[Chunk]:
    """将块序列组装为最终 chunk"""
    chunks: list[Chunk] = []
    current_text = ""
    chunk_counter = 0

    def _flush():
        nonlocal current_text, chunk_counter
        if not current_text.strip():
            return
        chunk_counter += 1
        meta = _build_metadata(
            paper_id=paper_id,
            paper_title=paper_title,
            hierarchy=hierarchy,
            level=level,
            section_title=section_title,
            chunk_idx=chunk_counter,
            total_chunks=0,  # 最后再更新
            file_hash=file_hash,
            has_math=False,
            token_count=_token_count(current_text),
        )
        chunks.append(Chunk(text=current_text.strip(), metadata=meta))
        current_text = ""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "。",
            "！",
            "？",
            " ",
        ],
        length_function=_token_count,
    )

    for block in blocks:
        block_tokens = _token_count(block.text)

        if block.block_type == "atomic":
            # atomic 块：flush 当前累积，独立成 chunk
            _flush()
            chunk_counter += 1
            meta = _build_metadata(
                paper_id=paper_id,
                paper_title=paper_title,
                hierarchy=hierarchy,
                level=level,
                section_title=section_title,
                chunk_idx=chunk_counter,
                total_chunks=0,
                file_hash=file_hash,
                has_math=True,
                token_count=block_tokens,
            )
            chunks.append(Chunk(text=block.text, metadata=meta))
            continue

        # text 块
        combined = current_text + ("\n\n" if current_text else "") + block.text
        combined_tokens = _token_count(combined)

        if combined_tokens <= MAX_CHUNK_TOKENS:
            current_text = combined
        else:
            # 当前累积 flush
            if current_text.strip():
                _flush()
            # 如果块本身 > MAX，用 splitter 切分
            if block_tokens > MAX_CHUNK_TOKENS:
                split_docs = text_splitter.split_text(block.text)
                for piece in split_docs:
                    chunk_counter += 1
                    meta = _build_metadata(
                        paper_id=paper_id,
                        paper_title=paper_title,
                        hierarchy=hierarchy,
                        level=level,
                        section_title=section_title,
                        chunk_idx=chunk_counter,
                        total_chunks=0,
                        file_hash=file_hash,
                        token_count=_token_count(piece),
                    )
                    chunks.append(Chunk(text=piece, metadata=meta))
            else:
                current_text = block.text

    _flush()

    # 更新 total_chunks
    for c in chunks:
        c.metadata["total_chunks"] = chunk_counter

    return chunks


def _chunk_node(
    node: LatexNode,
    paper_id: str,
    paper_title: str,
    hierarchy: list[str],
    file_hash: str,
) -> list[Chunk]:
    """递归切分文档树节点"""
    content_tokens = _token_count(node.content)

    # 叶子节点 + 内容小于 MIN → 不做拆分，直接返回
    if not node.children and content_tokens < MIN_CHUNK_TOKENS:
        blocks = _extract_blocks(node.content)
        return _chunk_blocks(
            blocks, paper_id, paper_title,
            hierarchy, node.level, node.title, file_hash,
        )

    # 有子节点 → 递归处理子节点
    if node.children:
        chunks = []
        for child in node.children:
            child_hierarchy = hierarchy + [child.title]
            chunks.extend(_chunk_node(
                child, paper_id, paper_title, child_hierarchy, file_hash,
            ))
        return chunks

    # 叶子节点且内容 ≥ MIN → 按块切分
    blocks = _extract_blocks(node.content)
    return _chunk_blocks(
        blocks, paper_id, paper_title,
        hierarchy, node.level, node.title, file_hash,
    )


# ── Step 4: ChromaDB 入库 ────────────────────────────────────────────────


def _get_file_hash(filepath: str) -> str:
    """计算文件 SHA256"""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""


def _get_processed_hashes() -> set:
    """从 ChromaDB 读取已处理文件的 hash"""
    import chromadb
    ef = MultilingualEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        coll = client.get_collection(COLLECTION_NAME, embedding_function=ef)
        existing = coll.get(include=["metadatas"])
        return {
            m["file_hash"]
            for m in (existing["metadatas"] or [])
            if m and "file_hash" in m
        }
    except Exception:
        return set()


def _ingest_chunks(chunks: list[Chunk]):
    """将 chunks 写入 ChromaDB"""
    import chromadb
    ef = MultilingualEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)

    if not chunks:
        return

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(str(uuid.uuid4()))
        documents.append(c.text)
        # ChromaDB metadata 只能存简单类型
        meta = dict(c.metadata)
        meta["has_math"] = meta.get("has_math") == "True"
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)


# ── Main ──────────────────────────────────────────────────────────────────


def build_knowledge_base(incremental: bool = True) -> dict:
    """构建 LaTeX 知识库

    参数:
        incremental: 增量模式（只处理新文件）

    返回:
        {"papers": int, "chunks": int, "new_files": int, "skipped": int}
    """
    import chromadb

    papers = _scan_paper_dirs(ARXIV_MATH_DIR)

    if not papers:
        print(f"未在 {ARXIV_MATH_DIR} 找到 .tex 文件")
        return {"papers": 0, "chunks": 0, "new_files": 0, "skipped": 0}

    processed_hashes = set()
    if incremental:
        processed_hashes = _get_processed_hashes()

    total_chunks = 0
    new_files = 0
    skipped = 0

    for paper in papers:
        # 计算主文件 hash（作为论文的唯一标识）
        h = _get_file_hash(paper["main_tex"])
        if not h:
            skipped += 1
            continue

        if incremental and h in processed_hashes:
            print(f"  - 跳过（已有）: {paper['paper_id']}")
            skipped += 1
            continue

        print(f"  + 处理: {paper['paper_id']} - {paper['paper_title'][:50]}...")

        # 展开 \input/\include
        full_content = _expand_tex(paper["main_tex"])

        # 解析结构
        root = _parse_structure(full_content)

        # 递归切分
        chunks = _chunk_node(
            root,
            paper_id=paper["paper_id"],
            paper_title=paper["paper_title"],
            hierarchy=[],
            file_hash=h,
        )

        if not chunks:
            print(f"    -> 无有效内容")
            continue

        # 入库
        _ingest_chunks(chunks)
        total_chunks += len(chunks)
        new_files += 1

        print(f"    -> {len(chunks)} chunks")

    stats = {
        "papers": len(papers),
        "chunks": total_chunks,
        "new_files": new_files,
        "skipped": skipped,
    }

    # 汇总
    print(f"\n{'=' * 40}")
    print(f"论文总数: {stats['papers']}")
    print(f"新增入库: {stats['new_files']}")
    print(f"跳过（已有）: {stats['skipped']}")
    print(f"总 chunks: {stats['chunks']}")
    print(f"{'=' * 40}")
    return stats


def query_knowledge_base(
    query: str,
    n_results: int = 3,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """查询知识库，按层级展开上下文

    流程:
        1. 向量检索 top-n
        2. 对每个命中，取同 (paper_id + hierarchy) 下所有 chunk
        3. 从命中块向两边展开，累积到 max_tokens 停
        4. 按 chunk_idx 排序拼接返回

    参数:
        query: 搜索关键词
        n_results: 向量检索条数
        max_tokens: 上下文展开上限（默认 800）
    """
    import chromadb

    ef = MultilingualEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        print(f"集合 '{COLLECTION_NAME}' 不存在，请先运行 build_knowledge_base()")
        return []

    hits = collection.query(query_texts=[query], n_results=n_results)

    items = []
    seen = set()
    if not hits["documents"]:
        return items

    for i in range(len(hits["ids"][0])):
        meta = hits["metadatas"][0][i] if hits["metadatas"] else {}
        hit_doc = hits["documents"][0][i] if hits["documents"] else ""
        paper_id = meta.get("paper_id", "")
        hierarchy = meta.get("hierarchy", "")

        # 去重：同一篇论文的同一节只展开一次
        dedup_key = f"{paper_id}||{hierarchy}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        hit_idx = meta.get("chunk_idx", 1)

        if not paper_id or not hierarchy:
            items.append({
                "text": hit_doc,
                "hierarchy": hierarchy,
                "paper_id": paper_id,
                "chunks_used": [hit_idx],
                "total_tokens": _token_count(hit_doc),
            })
            continue

        # 取同 hierarchy 下所有 chunk
        section_data = collection.get(
            where={"$and": [{"paper_id": paper_id}, {"hierarchy": hierarchy}]},
            include=["documents", "metadatas"],
        )

        # 按 chunk_idx 排序
        indexed = []
        for j in range(len(section_data["ids"])):
            m = section_data["metadatas"][j]
            indexed.append({
                "idx": m.get("chunk_idx", 1),
                "doc": section_data["documents"][j],
            })
        indexed.sort(key=lambda x: x["idx"])

        # 找到命中块的位置
        pos = None
        for p, entry in enumerate(indexed):
            if entry["idx"] == hit_idx:
                pos = p
                break
        if pos is None:
            pos = 0

        # 从命中块向两边展开，累积到 max_tokens
        selected = []
        total = 0
        left = pos
        right = pos + 1

        while total < max_tokens and (left >= 0 or right < len(indexed)):
            # 优先取左边（靠近命中块）
            if left >= 0 and (right >= len(indexed) or (pos - left) <= (right - pos - 1)):
                entry = indexed[left]
                if total + _token_count(entry["doc"]) <= max_tokens:
                    selected.append(entry)
                    total += _token_count(entry["doc"])
                left -= 1
            elif right < len(indexed):
                entry = indexed[right]
                if total + _token_count(entry["doc"]) <= max_tokens:
                    selected.append(entry)
                    total += _token_count(entry["doc"])
                right += 1
            else:
                break

        # 按 chunk_idx 排序
        selected.sort(key=lambda x: x["idx"])
        context = "\n\n".join(s["doc"] for s in selected)
        chunks_used = [s["idx"] for s in selected]

        items.append({
            "text": context,
            "hierarchy": hierarchy,
            "paper_id": paper_id,
            "paper_title": meta.get("paper_title", ""),
            "section_title": meta.get("section_title", ""),
            "has_math": meta.get("has_math", False),
            "chunks_used": chunks_used,
            "total_tokens": total,
            "score": hits["distances"][0][i] if hits.get("distances") else None,
        })

    return items


# ── CLI ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        query = " ".join(sys.argv[2:]) or "deep learning"
        print(f"查询: {query}\n")
        results = query_knowledge_base(query)
        for r in results:
            print(f"[{r['paper_id']}] {r['section_title']}")
            print(f"  层级: {r['hierarchy']}")
            print(f"  展开: 共 {r['chunks_used']} ({r['total_tokens']} tokens)")
            print(f"  内容: {r['text'][:200]}")
            print()
    else:
        # python -m research_system.knowledge_builder [--no-incremental]
        incremental = "--no-incremental" not in sys.argv
        if not incremental:
            print("全量重建模式\n")
        else:
            print("增量模式（跳过已有文件）\n")
        build_knowledge_base(incremental=incremental)

        # 小验证
        print("\n验证查询 'deep learning':")
        results = query_knowledge_base("deep learning", n_results=3)
        for r in results:
            print(f"  [{r['paper_id']}] {r['section_title']}: {r['text'][:80]}...")
