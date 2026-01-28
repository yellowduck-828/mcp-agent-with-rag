import os
from typing import List

# 🔒 安全沙箱根目录
WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")


def _ensure_workspace():
    """保证 workspace 存在"""
    os.makedirs(WORKSPACE, exist_ok=True)


def _safe_path(path: str) -> str:
    """
    将用户传入路径转换为 workspace 内安全路径。
    如果尝试越界访问，将抛出异常。
    """
    _ensure_workspace()

    # 支持相对路径
    real = os.path.realpath(os.path.join(WORKSPACE, path))
    base = os.path.realpath(WORKSPACE)

    if not real.startswith(base):
        raise PermissionError("禁止访问 workspace 目录之外的路径")

    return real


def list_dir(path: str = ".") -> List[str]:
    """
    列出 workspace 中某个目录下的文件和子目录。

    给模型的说明：
    - 当你需要查看有哪些文件/目录时，使用该工具。
    - path 是 workspace 下的相对路径，如 "." 或 "data/"
    """
    try:
        real = _safe_path(path)
        return os.listdir(real)
    except Exception as e:
        return [f"错误：{e}"]


def read_file(path: str) -> str:
    """
    读取 workspace 中的文本文件内容。

    给模型的说明：
    - 当用户需要你读取某个文件内容时调用。
    - 仅支持文本文件。
    - 如果文件不存在，请如实反馈。
    """
    try:
        real = _safe_path(path)
        with open(real, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败：{e}"


def write_file(path: str, content: str) -> str:
    """
    将内容写入 workspace 中的文件（覆盖写入）。

    给模型的说明：
    - 当需要新建/替换文件内容时调用。
    - 注意：此操作会覆盖原内容。
    """
    try:
        real = _safe_path(path)

        # 确保目录存在
        os.makedirs(os.path.dirname(real), exist_ok=True)

        with open(real, "w", encoding="utf-8") as f:
            f.write(content)

        return f"写入成功：{path}"
    except Exception as e:
        return f"写入失败：{e}"


def append_file(path: str, content: str) -> str:
    """
    以追加方式写入内容。

    给模型的说明：
    - 当你只想在文件末尾增加内容时使用。
    """
    try:
        real = _safe_path(path)

        os.makedirs(os.path.dirname(real), exist_ok=True)

        with open(real, "a", encoding="utf-8") as f:
            f.write(content)

        return f"追加成功：{path}"
    except Exception as e:
        return f"追加失败：{e}"


def delete_file(path: str) -> str:
    """
    删除 workspace 中的文件。
    """
    try:
        real = _safe_path(path)
        os.remove(real)
        return f"删除成功：{path}"
    except Exception as e:
        return f"删除失败：{e}"


def rename_file(src: str, dst: str) -> str:
    """
    重命名/移动 workspace 中的文件。
    """
    try:
        real_src = _safe_path(src)
        real_dst = _safe_path(dst)

        os.makedirs(os.path.dirname(real_dst), exist_ok=True)

        os.rename(real_src, real_dst)

        return f"已将 {src} 重命名为 {dst}"
    except Exception as e:
        return f"重命名失败：{e}"


def make_dir(path: str) -> str:
    """
    在 workspace 中创建新目录。
    """
    try:
        real = _safe_path(path)
        os.makedirs(real, exist_ok=True)
        return f"目录创建成功：{path}"
    except Exception as e:
        return f"创建失败：{e}"
