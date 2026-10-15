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
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "true").lower() == "true"

# 质检阈值
QUALITY_PASS_THRESHOLD = 4.0
MAX_DEBATE_ROUNDS = 3
