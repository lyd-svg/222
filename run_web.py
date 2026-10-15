import gradio as gr
from research_system.web.gradio_app import build_ui

if __name__ == "__main__":
    demo = build_ui()
    port = 7860
    print(f"\n🌐 在浏览器中访问: http://localhost:{port}")
    print("   (0.0.0.0 是绑定地址，请在浏览器中使用 localhost 或 127.0.0.1)\n")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft())
