import os
from agent import Agent
from config import DEEPSEEK_API_KEY
from openai import OpenAI


def chat_loop():
    print("进入多轮对话，输入 exit/quit 结束。")

    # 复用同一个 Agent 实例以保持对话记忆（进程内）
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)
    agent = Agent(client=client, verbose=True)

    while True:
        query = input("\n你：").strip()
        if query.lower() in {"exit", "quit"}:
            print("结束对话。")
            break
        if not query:
            continue
        ans = agent.get_completion(query)
        print("\nAI：")
        print(ans)


if __name__ == "__main__":
    chat_loop()