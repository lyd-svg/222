# 四Agent协作研究系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于LangGraph构建多Agent协作研究系统，支持多轮对话。

**Architecture:** 主题分类器决定走直接回答还是四Agent流水线（规划→搜索→分析→质检+辩论循环），Gradio聊天界面驱动多轮交互。

**Tech Stack:** Python, LangGraph, DeepSeek API, Tavily API, ChromaDB, Gradio

---

## 文件结构

```
d:\项目二\research_system\
├── config/
│   ├── __init__.py
│   ├── settings.py          # API Key、模型配置、工具开关
│   └── prompts.py           # 所有prompt模板
├── core/
│   ├── __init__.py
│   ├── state.py             # ResearchState TypedDict
│   └── graph.py             # LangGraph图构建 + 运行时
├── agents/
│   ├── __init__.py
│   ├── classifier.py        # 主题分类器
│   ├── planner.py           # 规划Agent
│   ├── searcher.py          # 搜索Agent（编排4个工具）
│   ├── analyst.py           # 分析Agent
│   ├── critic.py            # 质检Agent
│   └── rewriter.py          # 修改Agent
├── tools/
│   ├── __init__.py
│   ├── tavily_search.py     # Tavily联网搜索
│   ├── chroma_retriever.py  # Chroma知识库
│   ├── arxiv_search.py      # ArXiv论文搜索
│   └── github_search.py     # GitHub项目搜索
├── web/
│   ├── __init__.py
│   └── gradio_app.py        # Gradio多轮对话界面
├── main.py                  # CLI入口
├── run_web.py               # Web启动
├── requirements.txt
└── .env
```

---

### Task 1: 项目脚手架 & 配置文件

**Files:**
- Create: `d:\项目二\requirements.txt`
- Create: `d:\项目二\.env`
- Create: `d:\项目二\research_system\__init__.py`
- Create: `d:\项目二\research_system\config\__init__.py`
- Create: `d:\项目二\research_system\config\settings.py`
- Create: `d:\项目二\research_system\config\prompts.py`

- [ ] **Step 1: 创建 requirements.txt**

```text
langgraph>=0.4.0
langchain-core>=0.3.0
openai>=1.0.0
tavily-python>=0.5.0
chromadb>=0.6.0
arxiv>=2.1.0
httpx>=0.28.0
gradio>=5.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: 创建 .env**

```text
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-chat
TAVILY_API_KEY=tvly-your-tavily-api-key
OUTPUT_LANGUAGE=zh

# 工具开关（true/false）
TAVILY_ENABLED=true
CHROMA_ENABLED=true
ARXIV_ENABLED=true
GITHUB_ENABLED=true

# Ollama备选（无需API Key）
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=deepseek-coder
```

- [ ] **Step 3: 创建 settings.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 工具开关
TAVILY_ENABLED = os.getenv("TAVILY_ENABLED", "true").lower() == "true"
CHROMA_ENABLED = os.getenv("CHROMA_ENABLED", "true").lower() == "true"
ARXIV_ENABLED = os.getenv("ARXIV_ENABLED", "true").lower() == "true"
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "true").lower() == "true"

# 质检阈值
QUALITY_PASS_THRESHOLD = 4.0
MAX_DEBATE_ROUNDS = 3

# 输出语言
OUTPUT_LANGUAGE = os.getenv("OUTPUT_LANGUAGE", "zh")
```

- [ ] **Step 4: 创建 prompts.py**

```python
CLASSIFIER_PROMPT = """你是一个研究主题分类器。判断用户的问题是"简单查询"还是"复杂研究"。

简单查询：事实性问答、简单解释、代码片段、定义说明——不需要多步调研即可回答。
复杂研究：需要对比分析、趋势研究、多来源调研、报告撰写——需要搜集资料才能完整回答。

只需输出一个词：simple 或 complex

用户问题：{question}"""

PLANNER_PROMPT = """你是一个研究规划专家。将以下研究主题拆解为结构化研究计划。

主题：{topic}

请输出JSON格式（不要包含```json标记）：
{{
  "sub_questions": ["子问题1", "子问题2", ...],
  "keywords": ["关键词1", "关键词2", ...],
  "search_depth": "浅度/中等/深度",
  "required_sources": ["来源类型1", "来源类型2"]
}}"""

ANALYST_PROMPT = """你是一个研究分析师。基于以下研究计划和收集到的资料，撰写一份结构化研究报告。

研究计划：{plan}
收集资料：{sources}

要求：
1. 包含摘要、目录、正文、结论
2. 交叉验证不同来源的信息
3. 标注每个论断的资料来源（标注格式：[来源：标题]）
4. 对矛盾观点进行对比分析
5. 使用{language}撰写

报告格式：Markdown"""

CRITIC_PROMPT = """你是一个严格的审稿人。审查以下研究报告并给出质量评分。

评分标准（1-5分）：
1. 事实准确性 - 论断是否有来源支撑？
2. 完整性 - 是否覆盖了必要的方面？
3. 逻辑一致性 - 前后文是否矛盾？
4. 引用质量 - 来源是否可靠且引用规范？

请输出JSON格式（不要包含```json标记）：
{{
  "score": 4.5,
  "feedback": "具体的审查意见...",
  "dimensions": {{
    "accuracy": 4.5,
    "completeness": 4.0,
    "consistency": 5.0,
    "citations": 4.0
  }}
}}

报告：{report}"""

REWRITER_PROMPT = """你是一个研究编辑。根据以下审查反馈修改研究报告。

审查反馈：{feedback}
原报告：{report}

请基于审查意见逐条改进报告，保持原有结构但提升质量。使用{language}撰写。"""

DIRECT_ANSWER_PROMPT = """请回答以下用户问题。回答要准确、简洁、有依据。

问题：{question}

使用{language}回答。"""
```

- [ ] **Step 5: 创建目录和 __init__.py 文件**

创建 `research_system/`、`research_system/config/`、`research_system/core/`、`research_system/agents/`、`research_system/tools/`、`research_system/web/` 目录及各自的 `__init__.py`。

---

### Task 2: LLM客户端 & 核心State

**Files:**
- Create: `d:\项目二\research_system\core\__init__.py`
- Create: `d:\项目二\research_system\core\state.py`
- Create: `d:\项目二\research_system\core\llm.py`

- [ ] **Step 1: 创建 core/llm.py**

```python
from openai import OpenAI
from research_system.config.settings import (
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
)

_client = None

def get_llm() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _client

def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    client = get_llm()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
```

- [ ] **Step 2: 创建 core/state.py**

```python
from typing import TypedDict, Optional

class ResearchState(TypedDict):
    topic: str                    # 用户原始输入
    is_complex: Optional[bool]    # 分类结果
    plan: Optional[dict]          # 研究计划
    sources: list                 # 搜索到的资料
    report_draft: str             # 报告草稿
    review_feedback: str          # 质检反馈
    quality_score: dict           # {score, passed, dimensions}
    debate_round: int             # 辩论轮次
    final_report: str             # 最终输出
    history: list                 # 对话历史 [{role, content}]
```

---

### Task 3: 搜索工具

**Files:**
- Create: `d:\项目二\research_system\tools\__init__.py`
- Create: `d:\项目二\research_system\tools\tavily_search.py`
- Create: `d:\项目二\research_system\tools\chroma_retriever.py`
- Create: `d:\项目二\research_system\tools\arxiv_search.py`
- Create: `d:\项目二\research_system\tools\github_search.py`

所有工具统一接口：
```python
async def search(keywords: list[str]) -> list[dict]:
    """返回 [{title, content, source, url}]"""
```

- [ ] **Step 1: 创建 tavily_search.py**

```python
from tavily import TavilyClient
from research_system.config.settings import TAVILY_API_KEY

_client = None

def get_client():
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client

async def search(keywords: list[str]) -> list[dict]:
    client = get_client()
    query = " ".join(keywords[:3])
    response = client.search(query=query, search_depth="advanced", max_results=5)
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "source": "Tavily",
            "url": r.get("url", ""),
        })
    return results
```

- [ ] **Step 2: 创建 chroma_retriever.py**

```python
import chromadb
from research_system.config.settings import CHROMA_PERSIST_DIR

_client = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client

async def search(keywords: list[str]) -> list[dict]:
    try:
        client = get_client()
        collection = client.get_or_create_collection("research_knowledge")
        query = " ".join(keywords)
        results = collection.query(query_texts=[query], n_results=3)
        items = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                items.append({
                    "title": metadata.get("title", "知识库条目"),
                    "content": doc,
                    "source": "Chroma知识库",
                    "url": metadata.get("url", ""),
                })
        return items
    except Exception as e:
        print(f"Chroma搜索失败: {e}")
        return []
```

- [ ] **Step 3: 创建 arxiv_search.py**

```python
import arxiv

async def search(keywords: list[str]) -> list[dict]:
    query = " AND ".join(keywords[:3])
    search = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for r in search.results():
        results.append({
            "title": r.title,
            "content": r.summary[:500],
            "source": "ArXiv",
            "url": r.entry_id,
        })
    return results
```

- [ ] **Step 4: 创建 github_search.py**

```python
import httpx

async def search(keywords: list[str]) -> list[dict]:
    query = "+".join(keywords[:3])
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=5"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("full_name", ""),
                "content": item.get("description", "") or "",
                "source": "GitHub",
                "url": item.get("html_url", ""),
            })
        return results
```

---

### Task 4: Agent节点

**Files:**
- Create: `d:\项目二\research_system\agents\__init__.py`
- Create: `d:\项目二\research_system\agents\classifier.py`
- Create: `d:\项目二\research_system\agents\planner.py`
- Create: `d:\项目二\research_system\agents\searcher.py`
- Create: `d:\项目二\research_system\agents\analyst.py`
- Create: `d:\项目二\research_system\agents\critic.py`
- Create: `d:\项目二\research_system\agents\rewriter.py`

- [ ] **Step 1: 创建 classifier.py**

```python
import json
from research_system.core.llm import llm_call
from research_system.config.prompts import CLASSIFIER_PROMPT
from research_system.core.state import ResearchState

def classify_topic(state: ResearchState) -> ResearchState:
    result = llm_call("", CLASSIFIER_PROMPT.format(question=state["topic"])).strip().lower()
    state["is_complex"] = (result == "complex")
    return state
```

- [ ] **Step 2: 创建 planner.py**

```python
import json
from research_system.core.llm import llm_call
from research_system.config.prompts import PLANNER_PROMPT
from research_system.core.state import ResearchState

def planner_agent(state: ResearchState) -> ResearchState:
    response = llm_call("", PLANNER_PROMPT.format(topic=state["topic"]), temperature=0.5)
    try:
        state["plan"] = json.loads(response.strip())
    except json.JSONDecodeError:
        state["plan"] = {
            "sub_questions": [state["topic"]],
            "keywords": state["topic"].split(),
            "search_depth": "中等",
            "required_sources": ["网页搜索"],
        }
    return state
```

- [ ] **Step 3: 创建 searcher.py**

```python
import asyncio
from research_system.core.state import ResearchState
from research_system.config.settings import (
    TAVILY_ENABLED, CHROMA_ENABLED, ARXIV_ENABLED, GITHUB_ENABLED
)
from research_system.tools import (
    tavily_search, chroma_retriever, arxiv_search, github_search
)

async def searcher_agent(state: ResearchState) -> ResearchState:
    keywords = state["plan"].get("keywords", [state["topic"]])
    tasks = []

    if TAVILY_ENABLED:
        tasks.append(tavily_search.search(keywords))
    if CHROMA_ENABLED:
        tasks.append(chroma_retriever.search(keywords))
    if ARXIV_ENABLED:
        tasks.append(arxiv_search.search(keywords))
    if GITHUB_ENABLED:
        tasks.append(github_search.search(keywords))

    if not tasks:
        state["sources"] = []
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
    return state
```

- [ ] **Step 4: 创建 analyst.py**

```python
from research_system.core.llm import llm_call
from research_system.config.prompts import ANALYST_PROMPT
from research_system.config.settings import OUTPUT_LANGUAGE
from research_system.core.state import ResearchState

def analyst_agent(state: ResearchState) -> ResearchState:
    sources_text = "\n\n".join(
        [f"[{s['source']}] {s['title']}\n{s['content']}\n来源: {s['url']}" for s in state["sources"]]
    )
    report = llm_call(
        "你是一个专业的研究分析师。",
        ANALYST_PROMPT.format(
            plan=state["plan"],
            sources=sources_text,
            language=OUTPUT_LANGUAGE,
        ),
        temperature=0.7,
    )
    state["report_draft"] = report
    return state
```

- [ ] **Step 5: 创建 critic.py**

```python
import json
import re
from research_system.core.llm import llm_call
from research_system.config.prompts import CRITIC_PROMPT
from research_system.config.settings import QUALITY_PASS_THRESHOLD
from research_system.core.state import ResearchState

def _parse_score(response: str) -> dict:
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', response)
        score = float(score_match.group(1)) if score_match else 3.0
        return {
            "score": score,
            "feedback": response,
            "dimensions": {"accuracy": score, "completeness": score, "consistency": score, "citations": score},
        }

def critic_agent(state: ResearchState) -> ResearchState:
    response = llm_call(
        "你是一个严格的学术审稿人。",
        CRITIC_PROMPT.format(report=state["report_draft"]),
        temperature=0.3,
    )
    parsed = _parse_score(response)
    state["quality_score"] = {
        "score": parsed.get("score", 3.0),
        "passed": parsed.get("score", 3.0) >= QUALITY_PASS_THRESHOLD,
        "dimensions": parsed.get("dimensions", {}),
    }
    state["review_feedback"] = parsed.get("feedback", response)
    return state
```

- [ ] **Step 6: 创建 rewriter.py**

```python
from research_system.core.llm import llm_call
from research_system.config.prompts import REWRITER_PROMPT
from research_system.config.settings import OUTPUT_LANGUAGE
from research_system.core.state import ResearchState

def rewrite_agent(state: ResearchState) -> ResearchState:
    revised = llm_call(
        "你是一个专业的研究编辑。",
        REWRITER_PROMPT.format(
            feedback=state["review_feedback"],
            report=state["report_draft"],
            language=OUTPUT_LANGUAGE,
        ),
        temperature=0.7,
    )
    state["report_draft"] = revised
    state["debate_round"] += 1
    return state
```

---

### Task 5: LangGraph 工作流

**Files:**
- Create: `d:\项目二\research_system\core\graph.py`

- [ ] **Step 1: 创建 graph.py**

```python
from langgraph.graph import StateGraph, END
from research_system.core.state import ResearchState
from research_system.agents.classifier import classify_topic
from research_system.agents.planner import planner_agent
from research_system.agents.searcher import searcher_agent
from research_system.agents.analyst import analyst_agent
from research_system.agents.critic import critic_agent
from research_system.agents.rewriter import rewrite_agent
from research_system.config.settings import QUALITY_PASS_THRESHOLD, MAX_DEBATE_ROUNDS

def build_research_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    workflow.add_node("classifier", classify_topic)
    workflow.add_node("direct_answer", lambda s: s)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("searcher", searcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("rewriter", rewrite_agent)
    workflow.add_node("finalize", lambda s: s)

    def routing_decision(state: ResearchState) -> str:
        if state.get("is_complex"):
            return "complex"
        return "simple"

    workflow.add_conditional_edges(
        "classifier",
        routing_decision,
        {
            "simple": "direct_answer",
            "complex": "planner",
        }
    )

    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "analyst")
    workflow.add_edge("analyst", "critic")

    def debate_decision(state: ResearchState) -> str:
        if state["quality_score"].get("passed", False):
            return "pass"
        elif state["debate_round"] >= MAX_DEBATE_ROUNDS:
            return "force_exit"
        else:
            return "rewrite"

    workflow.add_conditional_edges(
        "critic",
        debate_decision,
        {
            "pass": "finalize",
            "rewrite": "rewriter",
            "force_exit": "finalize",
        }
    )

    workflow.add_edge("rewriter", "critic")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("finalize", END)

    workflow.set_entry_point("classifier")

    return workflow.compile()
```

---

### Task 6: 直接回答节点 + 入口函数

**Files:**
- Modify: `d:\项目二\research_system\agents\classifier.py`（添加 direct_answer 函数）

- [ ] **Step 1: 在 classifier.py 中添加 direct_answer 函数**

```python
from research_system.config.prompts import DIRECT_ANSWER_PROMPT
from research_system.config.settings import OUTPUT_LANGUAGE

def direct_answer_agent(state: ResearchState) -> ResearchState:
    answer = llm_call(
        "你是一个有帮助的AI助手。",
        DIRECT_ANSWER_PROMPT.format(
            question=state["topic"],
            language=OUTPUT_LANGUAGE,
        ),
        temperature=0.5,
    )
    state["final_report"] = answer
    return state
```

- [ ] **Step 2: 创建 graph.py 中的研究运行函数**

在 graph.py 中添加：
```python
import asyncio

async def run_research(topic: str, history: list = None) -> ResearchState:
    graph = build_research_graph()
    initial_state: ResearchState = {
        "topic": topic,
        "is_complex": None,
        "plan": None,
        "sources": [],
        "report_draft": "",
        "review_feedback": "",
        "quality_score": {},
        "debate_round": 0,
        "final_report": "",
        "history": history or [],
    }
    result = await graph.ainvoke(initial_state)
    return result
```

---

### Task 7: Gradio 多轮对话界面

**Files:**
- Create: `d:\项目二\research_system\web\gradio_app.py`
- Create: `d:\项目二\run_web.py`

- [ ] **Step 1: 创建 gradio_app.py**

```python
import gradio as gr
import time
from research_system.core.graph import run_research

CONVERSATION_HISTORY = []

async def respond(message: str, history: list):
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY.append({"role": "user", "content": message})

    # 先显示"正在分析..."
    history.append([message, "⏳ 正在分析问题类型..."])
    yield history, ""

    result = await run_research(message, CONVERSATION_HISTORY)

    if not result.get("is_complex"):
        # 简单问题直接回答
        history[-1][1] = result.get("final_report", "无法回答")
        CONVERSATION_HISTORY.append({"role": "assistant", "content": result["final_report"]})
        yield history, ""
        return

    # 复杂问题：分阶段显示进度
    history[-1][1] = "📋 **研究计划**\n\n" + "\n".join(
        [f"- {q}" for q in result["plan"].get("sub_questions", [])]
    )
    yield history, ""
    await asyncio.sleep(0.5)

    # 搜索阶段
    source_count = len(result.get("sources", []))
    history[-1][1] += f"\n\n🔍 **搜索阶段** — 共找到 {source_count} 条资料\n"
    for s in result["sources"][:5]:
        history[-1][1] += f"- [{s['source']}] {s['title']}\n"
    yield history, ""
    await asyncio.sleep(0.5)

    # 报告生成
    history[-1][1] += f"\n\n📝 **报告生成中...**"
    yield history, ""
    await asyncio.sleep(0.5)

    # 质检阶段
    score = result["quality_score"].get("score", 0)
    passed = result["quality_score"].get("passed", False)
    status = "✅ 通过" if passed else "🔄 需要修改"
    history[-1][1] += f"\n\n🔎 **质量检查** — 评分: {score}/5.0 — {status}"
    if result["debate_round"] > 0:
        history[-1][1] += f" (辩论轮次: {result['debate_round']})"
    yield history, ""
    await asyncio.sleep(0.5)

    # 最终报告
    final = result.get("final_report") or result.get("report_draft", "")
    history[-1][1] = final
    CONVERSATION_HISTORY.append({"role": "assistant", "content": final})
    yield history, ""


def clear_chat():
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []
    return [], ""


def build_ui():
    with gr.Blocks(title="四Agent协作研究系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🔬 AI 协作研究系统")
        gr.Markdown("输入研究主题，系统将自动规划→搜索→分析→质检，生成研究报告。")

        chatbot = gr.Chatbot(
            label="对话",
            height=600,
            bubble_full_width=False,
            render_markdown=True,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="输入研究主题或问题",
                placeholder="例如：2026年AI Agent框架对比...",
                scale=8,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话")
            examples = gr.Examples(
                examples=[
                    "2026年AI Agent框架有哪些新趋势？",
                    "LangGraph和CrewAI有什么区别？",
                    "用Python写一个FastAPI的REST API示例",
                ],
                inputs=msg,
            )

        send_event = msg.submit(
            respond, [msg, chatbot], [chatbot, msg]
        )
        send_btn.click(
            respond, [msg, chatbot], [chatbot, msg]
        )
        clear_btn.click(clear_chat, None, [chatbot, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: 创建 run_web.py**

```python
from research_system.web.gradio_app import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

### Task 8: 主入口与验证

**Files:**
- Create: `d:\项目二\main.py`

- [ ] **Step 1: 创建 main.py**

```python
import asyncio
from research_system.core.graph import run_research


async def main():
    print("=" * 50)
    print("  四Agent协作研究系统")
    print("=" * 50)

    topic = input("\n请输入研究主题: ").strip()
    if not topic:
        print("主题不能为空")
        return

    print("\n⏳ 正在分析问题...")
    result = await run_research(topic)

    if not result["is_complex"]:
        print("\n" + "=" * 50)
        print(result["final_report"])
        return

    print("\n📋 研究计划:")
    for q in result["plan"].get("sub_questions", []):
        print(f"  - {q}")

    print(f"\n🔍 共找到 {len(result['sources'])} 条资料")
    for s in result["sources"][:3]:
        print(f"  - [{s['source']}] {s['title']}")

    score = result["quality_score"].get("score", 0)
    passed = result["quality_score"].get("passed", False)
    print(f"\n🔎 质量评分: {score}/5.0 {'✅ 通过' if passed else '🔄 修改'}")
    if result["debate_round"] > 0:
        print(f"   辩论轮次: {result['debate_round']}")

    print("\n" + "=" * 50)
    print("📝 最终报告:")
    print(result.get("final_report") or result.get("report_draft", ""))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 验证项目结构**

运行 `python -c "from research_system.core.graph import build_research_graph; print('Graph OK')"` 确认导入无误。

---

## Spec覆盖检查

- ✅ 主题分类器 → classifier.py
- ✅ 规划Agent → planner.py
- ✅ 搜索Agent（4工具并行）→ searcher.py + tools/
- ✅ 分析Agent → analyst.py
- ✅ 质检Agent（一次调用，评分+审查）→ critic.py
- ✅ 辩论循环（打回修改）→ rewriter.py + graph.py条件边
- ✅ 三种路径（简单/复杂一次过/复杂辩论）→ graph.py路由
- ✅ 多轮对话 → gradio_app.py + history传递
- ✅ 工具可开关 → settings.py环境变量控制
