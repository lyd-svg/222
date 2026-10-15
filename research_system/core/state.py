from typing import TypedDict, Optional, Any


class ResearchState(TypedDict):
    topic: str                    # 用户原始输入
    is_complex: Optional[bool]    # 分类结果
    plan: Optional[dict]          # 研究计划
    sources: list                 # 搜索到的资料
    search_strategy: str          # 搜索策略（searcher 输出）
    search_evaluation: str        # 搜索结果评估（searcher 输出）
    report_draft: str             # 报告草稿
    review_feedback: str          # 质检反馈
    quality_score: dict           # {score, passed, dimensions}
    debate_round: int             # 辩论轮次
    final_report: str             # 最终输出
    history: list                 # 对话历史 [{role, content}]
    progress: list[str]           # 进度消息队列（用于流式输出）
