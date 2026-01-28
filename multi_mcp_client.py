import copy
from typing import Any, Dict, List, Optional

from mcp_client import MCPClient


class MultiMCPClient:
    """
    聚合多个 MCP stdio server。
    - 每个 server 用一个 MCPClient 管理。
    - 工具名加前缀：{server_name}::{tool_name}，避免重名。
    """

    def __init__(self, servers: List[Dict[str, Any]]):
        """
        servers: [{name, command, args, cwd?, env?, timeout?, result_max_chars?}]
        """
        self.clients: Dict[str, MCPClient] = {}
        for server in servers:
            name = server["name"]
            self.clients[name] = MCPClient(
                command=server.get("command", "python"),
                args=server.get("args") or ["mcp_server.py"],
                cwd=server.get("cwd"),
                env=server.get("env"),
                call_timeout=server.get("timeout", 20),
                result_max_chars=server.get("result_max_chars", 1200),
            )

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for server_name, client in self.clients.items():
            for tool in client.get_openai_tools():
                tool_copy = copy.deepcopy(tool)
                # OpenAI tools 名称要求匹配 ^[a-zA-Z0-9_-]+$，不能包含冒号
                tool_copy["function"]["name"] = f"{server_name}__{tool_copy['function']['name']}"
                schemas.append(tool_copy)
        print(f"[MCP] aggregated tools: {[t['function']['name'] for t in schemas]}")
        return schemas

    def call_tool(self, prefixed_name: str, arguments: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> str:
        if "__" not in prefixed_name:
            return f"工具名称缺少前缀：{prefixed_name}"
        server_name, tool_name = prefixed_name.split("__", 1)
        client = self.clients.get(server_name)
        if not client:
            return f"未找到 MCP server：{server_name}"
        return client.call_tool(tool_name, arguments or {}, timeout=timeout)

