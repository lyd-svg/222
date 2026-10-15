# 四Agent协作研究系统 - 设计文档

## 概述

基于 LangGraph 构建的多Agent协作研究系统，支持多轮对话。用户提出问题后，系统通过主题分类器判断难易程度，复杂问题进入四Agent流水线（规划→搜索→分析→质检+辩论循环），最终输出研究报告。

## 技术栈

- **LLM**: DeepSeek API（默认）/ Ollama 本地部署（备选）
- **工作流引擎**: LangGraph
- **搜索工具**: Tavily（联网搜索）、ChromaDB（本地知识库）、ArXiv（论文）、GitHub（开源项目）
- **Web界面**: Gradio（支持多轮对话）

## 项目结构

```
research_system/
├── config/
│   ├── __init__.py
│   ├── settings.py          # API Key、模型选择、工具开关
│   └── prompts.py           # 所有 prompt 模板
├── core/
│   ├── __init__.py
│   ├── state.py             # ResearchState 定义
│   └── graph.py             # LangGraph 图构建
├── agents/
│   ├── __init__.py
│   ├── classifier.py        # 主题分类器（入口决策）
│   ├── planner.py           # Agent1: 规划Agent
│   ├── searcher.py          # Agent2: 搜索Agent
│   ├── analyst.py           # Agent3: 分析Agent
│   ├── critic.py            # Agent4: 质检Agent
│   └── rewriter.py          # 辩论循环修改Agent
├── tools/
│   ├── __init__.py
│   ├── tavily_search.py     # Tavily 联网搜索
│   ├── chroma_retriever.py  # Chroma 知识库检索
│   ├── arxiv_search.py      # ArXiv 论文搜索
│   └── github_search.py     # GitHub 项目搜索
├── web/
│   ├── __init__.py
│   └── gradio_app.py        # Gradio 多轮对话界面
├── main.py                  # CLI 入口
├── run_web.py               # Web 启动脚本
├── requirements.txt
└── .env
```

## 核心数据流

### State 定义

```python
class ResearchState(TypedDict):
    topic: str              # 用户原始输入
    is_complex: bool        # 分类结果
    plan: dict              # 研究计划（sub_questions, keywords, search_depth, required_sources）
    sources: list           # 搜索到的资料列表 [{title, content, source, url}]
    report_draft: str       # 报告草稿（Markdown）
    review_feedback: str    # 质检反馈
    quality_score: dict     # {score: float, passed: bool}
    debate_round: int       # 当前辩论轮次（0开始）
    final_report: str       # 最终输出
    history: list           # 对话历史 [{role, content}]
```

### 工作流

```
classifier → "simple" → 直接回答 → END
classifier → "complex" → planner → searcher → analyst → critic
    critic → passed=True  → finalize → END
    critic → passed=False → rewriter → critic（辩论循环）
    critic → debate_round≥3 → 强制输出 → END
```

## Agent 详细设计

### 主题分类器 (classifier)
- 输入：用户问题 + 对话历史
- 输出："simple" 或 "complex"
- 逻辑：一次LLM调用判断
- simple：事实问答、简单解释
- complex：需要调研、对比分析、研究报告

### Agent1 - 规划Agent (planner)
- 输入：用户主题
- 输出：结构化研究计划
- 包含：sub_questions（3-5个子问题）、keywords、search_depth、required_sources

### Agent2 - 搜索Agent (searcher)
- 输入：研究计划中的 keywords
- 输出：合并去重后的资料列表
- 实现：asyncio.gather 并行调用所有已启用的搜索工具
- 每个工具可独立在 settings.py 中开关

### Agent3 - 分析Agent (analyst)
- 输入：研究计划 + 资料列表
- 输出：结构化 Markdown 报告（含摘要、正文、结论、来源标注）

### Agent4 - 质检Agent (critic)
- 输入：报告草稿
- 输出：{score: float, feedback: str, passed: bool}
- 逻辑：一次LLM调用打分+审查
- score ≥ 阈值(默认4.0/5.0) → passed=True
- score < 阈值 → passed=False，附带修改意见

### 修改Agent (rewriter)
- 输入：报告草稿 + 质检反馈
- 输出：修改后的新草稿
- 逻辑：LLM根据反馈意见逐条修改

## 搜索工具设计

统一接口：`async def search(keywords: list[str]) -> list[dict]`
统一返回：`[{"title": str, "content": str, "source": str, "url": str}]`

| 工具 | 实现方式 | 默认 |
|---|---|---|
| Tavily | `tavily` Python SDK | 开 |
| ChromaDB | 本地持久化 Chroma，支持文档注入 | 开（可关） |
| ArXiv | `arxiv` Python SDK | 开（可关） |
| GitHub | `httpx` 调 GitHub 公开 REST API | 开（可关） |

Searcher 内使用 `asyncio.gather` 并行执行，结果合并去重。

## Gradio 多轮对话界面

- 主界面：`gr.Chatbot` 显示对话历史
- 输入：`gr.Textbox` + "发送"按钮
- 每条复杂查询的响应带有可展开的研究过程详情
- 对话上下文传递给后续Agent，支持追问
- 快捷键：查看当前研究详情、清空对话

## 质量阈值

- 评分范围：1.0 - 5.0
- 通过阈值：4.0
- 最大辩论轮次：3
- 超过最大轮次强制输出

## 非功能性需求

- 搜索阶段耗时：2-4秒（并行）
- 报告生成：5-10秒
- 辩论循环每轮：+6-12秒
- 所有 API Key 通过 .env 配置
- 每个搜索工具可通过配置文件独立启用/禁用
