from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

load_dotenv(".env")


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"环境变量 {key} 未设置")
    return value


DEEPSEEK_API_KEY = require_env("DEEPSEEK_API_KEY")
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.3
)

TAVILY_API_KEY = require_env("TAVILY_API_KEY")