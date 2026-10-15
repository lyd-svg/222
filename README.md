# 四 Agent 协作研究系统

<p align="center">
  <b>基于 LangGraph + DeepSeek 的多 Agent 协作研究系统</b>
  <br>
  <i>自然语言驱动 · 多源并行搜索 · 辩论式质检 · 报告自动生成</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LLM-DeepSeek-4A90D9?logo=openai" alt="DeepSeek">
  <img src="https://img.shields.io/badge/Orchestration-LangGraph-FF6F00" alt="LangGraph">
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-3ECF8E?logo=chroma" alt="ChromaDB">
  <img src="https://img.shields.io/badge/UI-Gradio-FCC624" alt="Gradio">
</p>

---

## 📋 目录

- [项目概述](#-项目概述)
- [系统架构](#-系统架构)
- [六 Agent 与分工](#-六-agent-与分工)
- [共享状态](#-共享状态)
- [辩论循环](#-辩论循环)
- [技术特色](#-技术特色)
- [前置条件](#-前置条件)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [本地知识库](#-本地知识库)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)

---

## 🎯 项目概述

本项目是一个**多 Agent 协作研究系统**。输入一个研究主题，系统自动完成 **规划 → 并行搜索 → 分析 → 辩论质检** 全流程，输出结构化研究报告，将耗时数小时的深度研究工作压缩到数十秒。

**核心能力：**

| 能力 | 说明 |
|------|------|
| 🧠 **智能分流** | 分类器自动判断问题难易：简单问答直接回复，复杂课题进入完整研究流水线 |
| 📋 **自动规划** | 将研究主题拆解为子问题，提取搜索关键词，制定搜索策略 |
| 🔍 **多源并行搜索** | 同时搜索网页（Tavily）、本地知识库（ChromaDB）、GitHub 开源项目 |
| 📝 **报告生成** | 基于收集到的资料撰写结构化 Markdown 报告，含摘要、正文、结论 |
| 🔎 **辩论式质检** | 评审 Agent 从四个维度评分，不达标则打回修改，最多迭代 3 轮 |
| 💬 **双入口** | 提供 CLI 命令行和 Gradio Web 界面两种使用方式 |

---

## 🏗️ 系统架构

系统采用 **6 个 Agent 节点** 的流水线架构，通过 **LangGraph StateGraph** 编排，支持条件分支路由和辩论回环。

```
用户输入
   │
   ▼
┌──────────────────────┐
│    主题分类器         │ ─── 简单 ──→ [直接回答] ──→ 结束
│    Classifier         │
└─────────┬────────────┘
          │ 复杂
          ▼
┌──────────────────────┐
│    规划 Agent         │  →  拆解为子问题与关键词
│    Planner            │
└─────────┬────────────┘
          │ 研究计划
          ▼
┌──────────────────────┐     ┌─────────────┐
│    搜索 Agent         │ ──→ │  Tavily 网页 │
│    Searcher           │ ──→ │  ChromaDB   │ ← asyncio 并行
│    (asyncio.gather)   │ ──→ │  GitHub     │
└─────────┬────────────┘     └─────────────┘
          │ 搜索结果
          ▼
┌──────────────────────┐
│    分析 Agent         │  →  撰写结构化报告初稿
│    Analyst            │
└─────────┬────────────┘
          │ report_draft
          ▼
┌──────────────────────┐
│    评审 Agent         │  →  四维度评分（/5.0）
│    Critic             │
└──┬───────────┬───────┘
   │           │
   │           └── 不通过 ──→ [改写 Agent] ──→ 再评审
   │                           ↑ 最多 3 轮 ──┘
   │
   └── 通过 / 超限 ──→ [终稿] ──→ 结束
```

**三种执行路径：**

```
路径 A（简单问题）：
  用户提问 → 分类器 → "简单" → 直接回答 → 结束

路径 B（一次通过）：
  用户提问 → 分类器 → "复杂" → 规划 → 搜索 → 分析 → 评审(通过) → 终稿

路径 C（辩论后通过）：
  用户提问 → 分类器 → "复杂" → 规划 → 搜索 → 分析 →
  评审(不通过) → 改写 → 评审… → 终稿 / 强制退出
```

---

## 🤖 六 Agent 与分工

| Agent | 输入 | 输出 | 职责 |
|-------|------|------|------|
| **Classifier** | 用户问题 `topic` | `is_complex`（布尔值） | 判断问题难易，路由到简单/复杂分支 |
| **Planner** | `topic` | 结构化研究计划（子问题、关键词、搜索深度、来源要求） | 分解研究主题为可执行的搜索策略 |
| **Searcher** | `plan.keywords` + `plan.sub_questions` | 合并去重的 `sources` 列表 | 多工具并行搜索，汇总结果 |
| **Analyst** | 研究计划 + 搜索结果 | `report_draft`（Markdown） | 基于资料撰写结构化报告 |
| **Critic** | `report_draft` | `quality_score` + `review_feedback` | 从四维度评分，驱动辩论循环 |
| **Rewriter** | `report_draft` + 评审意见 | 修订后的 `report_draft` | 根据反馈逐条修改报告 |

每个节点通过 LangGraph 的条件边路由，评审不通过时触发 **Critic → Rewriter → Critic** 回环。

---

## 📦 共享状态

节点之间通过 `ResearchState`（TypedDict）传递数据：

| 字段 | 类型 | 说明 | 写入者 |
|------|------|------|--------|
| `topic` | `str` | 用户原始输入 | 初始状态 |
| `is_complex` | `bool` | 是否为复杂课题 | Classifier |
| `plan` | `dict` | 结构化研究计划 `{sub_questions, keywords, search_depth, required_sources}` | Planner |
| `sources` | `list[dict]` | 搜索结果 `[{title, content, source, url}]` | Searcher |
| `report_draft` | `str` | 报告草稿（复杂路径的最终输出所在） | Analyst → Rewriter |
| `review_feedback` | `str` | 评审反馈意见 | Critic |
| `quality_score` | `dict` | `{score, passed, dimensions}` | Critic |
| `debate_round` | `int` | 当前辩论轮次（从 0 开始） | Rewriter |
| `final_report` | `str` | 最终回答（仅简单路径使用） | direct_answer |
| `history` | `list` | 对话历史 `[{role, content}]` | Web UI |
| `progress` | `list[str]` | 流式进度消息队列 | 各 Agent |

> 注：简单路径输出 `final_report`；复杂路径输出 `report_draft`（经 Analyst 撰写 → Critic 评审 → Rewriter 迭代后作为最终结果）。

---

## 🔎 辩论循环

Critic Agent 从四个维度对报告打分（满分 5.0）：

| 维度 | 说明 |
|------|------|
| **事实准确性** | 论断是否有来源支撑，是否存在幻觉 |
| **完整性** | 是否覆盖研究计划中的所有子问题 |
| **逻辑一致性** | 前后文是否矛盾，论证是否连贯 |
| **引用质量** | 来源是否可靠，引用格式是否规范 |

评分流程：

```
Critic 评审 → score ≥ 4.0? ──→ 通过 → 终稿
               │ 否
               ▼
          debate_round < 3? ──→ 是 → Rewriter 修改 → 再评审
               │ 否
               ▼
           强制输出终稿
```

阈值和轮次均可通过 `.env` 配置。

---

## ✨ 技术特色

### 🧩 多源并行搜索
搜索 Agent 通过 `asyncio.gather` 同时调用多个搜索工具，耗时由最慢的工具决定（通常 2-4 秒），而非串行累加。每个工具可通过 `.env` 独立开关。

### 🔄 辩论式质检
引入"评审-改写"闭环：Critic 给出结构化评分和具体修改意见，Rewriter 逐条改进后再次提交评审。这种"同行评议"机制有效提升报告质量。

### 🧠 智能路由
Classifier 通过 LLM 判断问题复杂度，简单问答（如"Python 怎么读写文件"）绕过完整流水线直接回答，节省时间和 token。复杂课题（如"2026 年 AI Agent 框架对比"）才进入研究流程。

### 📊 流式进度反馈
两种入口均支持流式输出，实时显示当前执行阶段（分类中 → 规划中 → 搜索中 → 撰写中 → 评审中…），避免用户等待时无反馈。

### 🗃️ 本地知识库 RAG
支持将 arXiv 论文 LaTeX 源码构建为 ChromaDB 向量知识库，实现 RAG 检索增强。构建器智能识别论文结构（section 层级）、保护数学公式等原子块不被切分。

---

## 📋 前置条件

1. **Python 3.10+**
2. **DeepSeek API Key** — [申请地址](https://platform.deepseek.com/api_keys)
3. **（可选）Tavily API Key** — 用于网页搜索，[注册](https://tavily.com)

---

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone <repo-url> && cd d:\项目二

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY
# 可选：填入 TAVILY_API_KEY 并设置工具开关

# 4. CLI 模式运行
python main.py

# 5. （或）Web 模式运行
python run_web.py
# 浏览器打开 http://localhost:7860
```

---

## 🎮 使用指南

### CLI 模式

```bash
python main.py
```

在提示符后输入研究主题，系统会实时显示进度并输出最终报告。

### Web 模式

```bash
python run_web.py
```

浏览器访问 `http://localhost:7860`，在聊天界面输入问题，支持流式进度展示和多轮对话。

示例问题：
```
2026年AI Agent框架有哪些新趋势？
LangGraph和CrewAI有什么区别？
用Python写一个FastAPI的REST API示例
```

### 环境变量

所有配置通过 `.env` 管理（参考 `.env.example`）：

```ini
# DeepSeek API 配置（必填）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
DEEPSEEK_MODEL=deepseek-chat

# Tavily 搜索 API（可选，需注册）
TAVILY_API_KEY=tvly-your-tavily-api-key

# 工具开关（默认开启）
TAVILY_ENABLED=true
CHROMA_ENABLED=true
GITHUB_ENABLED=true
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QUALITY_PASS_THRESHOLD` | `4.0` | 质量评分通过线（满分 5.0） |
| `MAX_DEBATE_ROUNDS` | `3` | 辩论循环最大迭代次数 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | 向量数据库持久化目录 |

---

## 📚 本地知识库

将 arXiv 论文 LaTeX 源码构建为向量知识库，研究流程中通过 ChromaDB 检索增强。

```bash
# 增量模式（默认）：只处理新增论文
python -m research_system.knowledge_builder

# 全量重建模式
python -m research_system.knowledge_builder --no-incremental

# 查询测试
python -m research_system.knowledge_builder query "deep learning"
```

论文源码存放在 `research_system/arxiv_math/` 目录，按论文 ID 分目录存放。

---

## 📁 项目结构

```
d:\项目二\
├── main.py                          # 🚀 CLI 入口：交互式研究系统
├── run_web.py                       # 🌐 Web 入口：启动 Gradio 界面
├── requirements.txt                 # 📦 Python 依赖清单
├── .env.example                     # 🔑 环境变量模板
├── .gitignore

├── research_system/
│   ├── agents/
│   │   ├── classifier.py            # 🧠 主题分类节点
│   │   ├── planner.py               # 📋 研究规划节点（输出 JSON 计划）
│   │   ├── searcher.py              # 🔍 并行搜索节点（asyncio.gather）
│   │   ├── analyst.py               # 📝 报告撰写节点
│   │   ├── critic.py                # 🔎 质量评审节点
│   │   └── rewriter.py              # ✏️  报告改写节点（辩论循环）
│   │
│   ├── core/
│   │   ├── state.py                 # 📦 共享状态定义 (ResearchState)
│   │   ├── llm.py                   # 🤖 LLM 客户端封装（OpenAI 兼容 SDK）
│   │   └── graph.py                 # 🔗 LangGraph 图构建与流式执行
│   │
│   ├── tools/
│   │   ├── tavily_search.py         # 🌐 Tavily 网页搜索
│   │   ├── chroma_retriever.py      # 🗄️  ChromaDB 向量检索
│   │   ├── github_search.py         # 🐙 GitHub 搜索
│   │   ├── arxiv_downloader.py      # 📄 arXiv 论文源码下载（知识库构建用）
│   │   └── embeddings.py            # 🔢 嵌入模型封装（fastembed）
│   │
│   ├── config/
│   │   ├── settings.py              # ⚙️  配置读取（环境变量）
│   │   └── prompts.py               # 💬 LLM 提示词模板
│   │
│   ├── web/
│   │   └── gradio_app.py            # 🖥️  Gradio 聊天界面
│   │
│   └── knowledge_builder.py         # 🏗️  arXiv LaTeX 知识库构建与查询
│
├── chroma_db/                       # 🗄️  向量数据库持久化（gitignored）
├── research_system/arxiv_math/      # 📄  arXiv LaTeX 源码（gitignored）
├── docs/                            # 📚 设计文档
└── .claude/                         # 🤖 Claude Code 配置
```

---

## 🧰 技术栈

| 层面 | 技术 |
|------|------|
| **语言** | Python 3.10+ |
| **LLM** | DeepSeek（兼容 OpenAI SDK） |
| **工作流引擎** | LangGraph StateGraph |
| **向量数据库** | ChromaDB + fastembed |
| **网页搜索** | Tavily API |
| **代码搜索** | GitHub REST API |
| **论文源码** | arXiv API（仅知识库构建） |
| **Web UI** | Gradio |
| **依赖管理** | python-dotenv |

---

## 📄 许可证

MIT
