import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
from mcp import types  # type: ignore
from mcp.client.session import ClientSession  # type: ignore
from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore

# 默认调用超时（秒）与结果裁剪长度
DEFAULT_CALL_TIMEOUT = 20
DEFAULT_RESULT_MAX_CHARS = 1200


class MCPClient:
    """简单的 MCP stdio 客户端封装，用于列出工具并执行工具调用。"""

    def __init__(
        self,
        command: str = "python",
        args: Optional[List[str]] = None,
        cwd: Optional[str | Path] = None,
        env: Optional[Dict[str, str]] = None,
        call_timeout: int = DEFAULT_CALL_TIMEOUT,
        result_max_chars: int = DEFAULT_RESULT_MAX_CHARS,
    ):
        self.server_params = StdioServerParameters(
            command=command,
            args=args or ["mcp_server.py"],
            cwd=str(cwd or Path(__file__).resolve().parent),
            env=env,
        )
        self.call_timeout = call_timeout
        self.result_max_chars = result_max_chars
        self._tool_cache: List[types.Tool] = []

    def fetch_tools(self) -> List[types.Tool]:
        """直接连接 MCP server 获取工具列表。"""
        try:
            tools = anyio.run(self._list_tools_once)
            self._tool_cache = tools or []
            print(f"[MCP] list_tools fetched: {[t.name for t in self._tool_cache]}")
            return self._tool_cache
        except Exception as exc:
            print(f"⚠️ 获取 MCP 工具失败：{exc}")
            self._tool_cache = []
            return []

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """将 MCP 工具信息转换为 OpenAI 工具 schema。"""
        tools = self._tool_cache or self.fetch_tools()
        schemas: List[Dict[str, Any]] = []
        for tool in tools:
            params = tool.inputSchema or {"type": "object", "properties": {}}
            if "type" not in params:
                params = {"type": "object", **params}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": params,
                    },
                }
            )
        return schemas

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> str:
        """执行 MCP 工具调用并返回裁剪后的文本结果。"""
        timeout_sec = timeout or self.call_timeout
        try:
            result = anyio.run(self._call_tool_once, name, arguments or {}, timeout_sec)
        except Exception as exc:
            if exc.__class__.__name__ == "TimeoutError":
                print(f"⏱️ 工具调用超时：{name}")
                return "工具调用超时，请换一种方式或缩短查询"
            print(f"❌ 工具调用失败：{name} -> {exc}")
            return f"工具调用失败：{exc}"

        formatted = self._format_result(result)
        print(f"[MCP] call_tool {name} args={arguments} -> {formatted[:80]}{'...' if len(formatted) > 80 else ''}")
        return formatted

    async def _list_tools_once(self) -> List[types.Tool]:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools

    async def _call_tool_once(self, name: str, arguments: Dict[str, Any], timeout_sec: int) -> types.CallToolResult:
        async with stdio_client(self.server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                with anyio.fail_after(timeout_sec):
                    return await session.call_tool(name, arguments or {})

    def _format_result(self, result: types.CallToolResult) -> str:
        """提取文本并裁剪，避免直接返回大对象。"""
        parts: List[str] = []
        for item in result.content or []:
            item_type = getattr(item, "type", "")
            if item_type == "text":
                text_val = getattr(item, "text", "")
                if text_val:
                    parts.append(text_val.strip())
            elif item_type == "image":
                mime = getattr(item, "mimeType", "") or "image/*"
                parts.append(f"[image {mime} 内容已省略]")
            elif item_type == "audio":
                mime = getattr(item, "mimeType", "") or "audio/*"
                parts.append(f"[audio {mime} 内容已省略]")
            elif item_type == "resource_link":
                uri = getattr(item, "uri", "") or getattr(item, "resource", "") or ""
                display = f"[resource_link {uri}]" if uri else "[resource_link 内容已省略]"
                parts.append(display)
            else:
                parts.append(f"[{item_type or 'content'} 内容已省略]")

        if result.structuredContent:
            try:
                parts.append(json.dumps(result.structuredContent, ensure_ascii=False))
            except Exception:
                parts.append(str(result.structuredContent))

        text = "\n".join(p for p in parts if p).strip()
        if not text:
            text = "(空结果)"
        return self._clip(text)

    def _clip(self, text: str) -> str:
        if len(text) <= self.result_max_chars:
            return text
        return text[: self.result_max_chars].rstrip() + f"... (已截断，原始长度 {len(text)})"

