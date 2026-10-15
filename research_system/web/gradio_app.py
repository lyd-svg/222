import gradio as gr
from research_system.core.graph import run_research_stream

CONVERSATION_HISTORY = []


async def respond(message: str, history: list):
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY.append({"role": "user", "content": message})

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, ""

    status_lines = []

    final_state = None
    async for msg_type, content in run_research_stream(message, CONVERSATION_HISTORY):
        if msg_type == "status":
            status_lines.append(content)
            # 用纯文本逐行显示进度
            history[-1]["content"] = "```\n" + "\n".join(status_lines) + "\n```"
            yield history, ""
        elif msg_type == "result":
            final_state = content

    if not final_state:
        return

    # 简单问题：直接显示回答
    if not final_state.get("is_complex"):
        history[-1]["content"] = final_state.get("final_report", "")
        CONVERSATION_HISTORY.append(
            {"role": "assistant", "content": final_state["final_report"]}
        )
        yield history, ""
        return

    # 构建研究结果展示
    parts = []

    # 进度概要
    parts.append("## 📋 研究进度\n")
    for line in status_lines:
        parts.append(f"- {line}")

    # 研究计划
    if final_state.get("plan"):
        sub_questions = final_state["plan"].get("sub_questions", [])
        if sub_questions:
            parts.append("\n## 📋 研究计划\n")
            for q in sub_questions:
                parts.append(f"- {q}")

    # 搜索结果
    sources = final_state.get("sources", [])
    if sources:
        parts.append(f"\n## 🔍 搜索结果 — 共 {len(sources)} 条资料\n")
        for s in sources[:5]:
            parts.append(f"- **[{s['source']}]** [{s['title']}]({s['url']})")
        if len(sources) > 5:
            parts.append(f"- *...及其他 {len(sources) - 5} 条*")

    # 质量评分
    score = final_state.get("quality_score", {}).get("score", 0)
    passed = final_state.get("quality_score", {}).get("passed", False)
    rounds = final_state.get("debate_round", 0)

    parts.append(f"\n## 🔎 质量检查\n")
    parts.append(f"- **评分**: {score}/5.0")
    parts.append(f"- **结果**: {'✅ 通过' if passed else '❌ 未通过'}")
    if rounds > 0:
        parts.append(f"- **辩论轮次**: {rounds}")

    # 最终报告
    final_report = final_state.get("final_report") or final_state.get("report_draft", "")
    if final_report:
        parts.append(f"\n## 📝 最终报告\n")
        parts.append(final_report)

    history[-1]["content"] = "\n\n".join(parts)
    CONVERSATION_HISTORY.append(
        {"role": "assistant", "content": final_report}
    )
    yield history, ""


def clear_chat():
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = []
    return [], ""


def build_ui():
    with gr.Blocks(title="四Agent协作研究系统") as demo:
        gr.Markdown("## 🔬 AI 协作研究系统")
        gr.Markdown(
            "输入研究主题，系统将自动**规划 → 搜索 → 分析 → 质检**，生成研究报告。"
        )

        chatbot = gr.Chatbot(
            label="对话",
            height=600,
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
            gr.Examples(
                examples=[
                    "2026年AI Agent框架有哪些新趋势？",
                    "LangGraph和CrewAI有什么区别？",
                    "用Python写一个FastAPI的REST API示例",
                ],
                inputs=msg,
            )

        send_event = msg.submit(respond, [msg, chatbot], [chatbot, msg])
        send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, None, [chatbot, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
