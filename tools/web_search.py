import requests
from langchain_tavily import TavilySearch
from config import TAVILY_API_KEY

search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5
)

def _fallback_request(query: str, timeout: float):
    """在 langchain_tavily 调用异常时直接请求 Tavily API，增加超时保护。"""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 5,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results")
    except Exception:
        return None


def web_search(query: str, timeout: float = 8.0):
    """
    搜索互联⽹获取最新信息。
    - 输⼊：搜索关键词（字符串）
    - 输出：相关⽹⻚内容摘要
    - 适⽤场景：需要实时数据、最新新闻、当前事件
    - 不适⽤场景：历史数据、个⼈信息、数学计算  
    返回精简后的摘要字符串，避免将完整 JSON 直接给模型。
    仅取前 5 个结果，提炼标题和内容（可用时附上 URL）。
    """
    try:
        resp = search.invoke({"query": query}, config={"timeout": timeout})
        results = resp.get("results") if isinstance(resp, dict) else None
    except Exception:
        results = _fallback_request(query, timeout)

    if not results:
        return "搜索失败或超时，或未找到相关结果。"

    lines = []
    for idx, item in enumerate(results[:5], 1):
        title = (item.get("title") or item.get("url") or "无标题").strip()
        snippet = (
            item.get("content")
            or item.get("snippet")
            or item.get("description")
            or ""
        ).strip()
        url = item.get("url")

        # 确保 URL 可点击（加 markdown 链接）
        display_title = title
        link = url
        if link and not link.startswith(("http://", "https://")):
            link = "https://" + link.lstrip("/")
        if link:
            display_title = f"[{title}]({link})"

        line = f"{idx}. {display_title}"
        if snippet:
            line += f"：{snippet}"
        lines.append(line)

    return "搜索结果摘要：\n" + "\n".join(lines)
