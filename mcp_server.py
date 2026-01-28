import json
import inspect
from typing import Any, Dict, List

import anyio
from mcp import types
from mcp.server import Server, stdio

from tools.web_search import web_search
from tools.datetime import get_current_datetime
from tools.file import (
    list_dir,
    read_file,
    write_file,
    append_file,
    delete_file,
    rename_file,
    make_dir,
)
from tools.cookbook_rag import rag_rebuild_index, rag_search, rag_read_file


def _python_type_to_json_schema(param: inspect.Parameter) -> Dict[str, Any]:
    """将 Python 类型简单映射到 JSON Schema（常见标注）。"""
    annotation = param.annotation
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    json_type = type_map.get(annotation, "string")
    return {"type": json_type}


def _build_tool_schema(func) -> types.Tool:
    """把本地函数转换为 MCP Tool 定义。"""
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        properties[name] = _python_type_to_json_schema(param)
        if param.default is inspect._empty:
            required.append(name)

    description = (func.__doc__ or "").strip() or f"调用 {func.__name__}"
    return types.Tool(
        name=func.__name__,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        outputSchema=None,
    )


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_current_datetime": get_current_datetime,
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "delete_file": delete_file,
    "rename_file": rename_file,
    "make_dir": make_dir,
    "rag_search": rag_search,
    "rag_rebuild_index": rag_rebuild_index,
    "rag_read_file": rag_read_file,
}

server = Server("myagent-mcp", instructions="myagentbymcp 工具通过 MCP 暴露给模型使用。")


@server.list_tools()
async def handle_list_tools():
    return [_build_tool_schema(func) for func in TOOL_FUNCTIONS.values()]


@server.call_tool()
async def handle_call_tool(tool_name: str, arguments: Dict[str, Any]):
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return [
            types.TextContent(type="text", text=f"未知工具：{tool_name}"),
        ]

    try:
        result = func(**(arguments or {}))
    except Exception as exc:
        return [
            types.TextContent(type="text", text=f"工具执行失败：{exc}"),
        ]

    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False)
        except Exception:
            text = str(result)

    return [
        types.TextContent(type="text", text=text),
    ]


async def main():
    init_options = server.create_initialization_options()
    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    anyio.run(main)

