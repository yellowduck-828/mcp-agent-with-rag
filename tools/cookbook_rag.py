"""
RAG 工具适配层：供 MCP 暴露，调用 rag 模块完成检索、索引重建，以及只读访问菜谱源文件。
"""

from pathlib import Path

from rag.retrieval import rag_search_tool, rebuild_index_tool


RAG_BASE = (Path(__file__).resolve().parent.parent / "rag").resolve()
RAG_DATA = (RAG_BASE / "data").resolve()


def _safe_rag_path(rel_path: str) -> Path:
    """限制访问 rag/data 下的文件，防止越界。"""
    candidate = (RAG_DATA / rel_path).resolve()
    if not str(candidate).startswith(str(RAG_DATA)):
        raise PermissionError("仅允许读取 rag/data 下的文件")
    return candidate


def rag_search(query: str, top_k: int = 5):
    """基于菜谱数据集的 RAG 检索，返回命中的片段与上下文。"""
    return rag_search_tool(query=query, top_k=top_k)


def rag_rebuild_index():
    """重建菜谱向量索引。"""
    return rebuild_index_tool()


def rag_read_file(path: str) -> str:
    """读取 rag/data 下的菜谱 markdown，提供只读访问。"""
    try:
        real = _safe_rag_path(path)
    except Exception as exc:
        return f"读取失败：{exc}"
    if not real.exists():
        return f"读取失败：文件不存在：{path}"
    try:
        return real.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return real.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return f"读取失败：{exc}"
