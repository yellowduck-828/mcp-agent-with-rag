import json
import os
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Tuple, List, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from fastapi.responses import StreamingResponse

from agent import Agent
from config import DEEPSEEK_API_KEY
from tools.file import _safe_path, WORKSPACE
from backend.schemas import ChatRequest, ChatResponse

app = FastAPI(title="MyAgent Backend")

# 允许本地前端访问，可按需收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简单内存会话：session_id -> (agent, last_used_ts)
SESSION_TTL = 60 * 30  # 30 分钟不活跃自动清理
sessions: Dict[str, Tuple[Agent, float]] = {}
# 会话中断标记：session_id -> threading.Event
session_cancel_flags: Dict[str, threading.Event] = {}

# 持久化文件：保存每个 session 的摘要与最近轮次
SESSION_FILE = Path(__file__).parent / "chat_sessions.json"
MAX_RECENT_MESSAGES = 6  # 3 轮(user+assistant)


def _load_session_store() -> Dict[str, Dict[str, Any]]:
    if not SESSION_FILE.exists():
        return {}
    try:
        with SESSION_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_session_store(data: Dict[str, Dict[str, Any]]):
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compact_session(session: Dict[str, Any]):
    """保证 recent 最多 6 条消息（3 轮），更早内容并入 summary。"""
    recent: List[Dict[str, Any]] = session.get("recent", [])
    summary_parts: List[str] = []
    while len(recent) > MAX_RECENT_MESSAGES:
        oldest_pair = recent[:2]
        recent = recent[2:]
        summary_parts.append("\n".join(f"{m.get('role')}: {m.get('content')}" for m in oldest_pair))
    if summary_parts:
        merged = "\n".join(summary_parts)
        prev = session.get("summary") or ""
        session["summary"] = f"{prev}\n{merged}".strip()
    session["recent"] = recent


def _append_history(sid: str, user_msg: str, assistant_msg: str, tools: List[str]):
    data = _load_session_store()
    session = data.get(sid, {"summary": "", "recent": []})
    session["recent"] = session.get("recent", [])
    session["recent"].append({"role": "user", "content": user_msg, "tools": []})
    session["recent"].append({"role": "assistant", "content": assistant_msg, "tools": tools or []})
    _compact_session(session)
    data[sid] = session
    _save_session_store(data)


def _hydrate_agent_from_store(agent: Agent, sid: str):
    data = _load_session_store()
    session = data.get(sid)
    if not session:
        return
    summary = session.get("summary")
    recent = session.get("recent", [])
    if summary:
        agent.messages.append({"role": "system", "content": f"以下是该会话到目前为止的摘要：\n{summary}"})
    for msg in recent:
        agent.messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})


def _delete_session_store(sid: str):
    data = _load_session_store()
    if sid in data:
        data.pop(sid, None)
        _save_session_store(data)


def _get_client():
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)


def _get_agent(session_id: str) -> Agent:
    now = time.time()
    # 清理过期会话
    expired = [sid for sid, (_, ts) in sessions.items() if now - ts > SESSION_TTL]
    for sid in expired:
        sessions.pop(sid, None)
        session_cancel_flags.pop(sid, None)

    if session_id in sessions:
        agent, _ = sessions[session_id]
        sessions[session_id] = (agent, now)
        return agent

    agent = Agent(client=_get_client(), verbose=False)
    _hydrate_agent_from_store(agent, session_id)
    sessions[session_id] = (agent, now)
    return agent


def _get_cancel_flag(session_id: str) -> threading.Event:
    flag = session_cancel_flags.get(session_id)
    if flag is None:
        flag = threading.Event()
        session_cancel_flags[session_id] = flag
    return flag


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    sid = payload.session_id or str(uuid.uuid4())
    agent = _get_agent(sid)
    cancel_flag = _get_cancel_flag(sid)
    cancel_flag.clear()

    result = agent.get_completion(payload.message, return_details=True, stop_event=cancel_flag)
    reply = result["content"] if isinstance(result, dict) else str(result)
    tools = result.get("tools", []) if isinstance(result, dict) else []
    tool_results = result.get("tool_results", []) if isinstance(result, dict) else []

    # 追加历史并持久化
    _append_history(sid, payload.message, reply, tools)

    # 更新最后使用时间
    if sid in sessions:
        sessions[sid] = (agent, time.time())

    return ChatResponse(session_id=sid, reply=reply, tools=tools, tool_results=tool_results)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    sid = payload.session_id or str(uuid.uuid4())
    agent = _get_agent(sid)
    cancel_flag = _get_cancel_flag(sid)
    cancel_flag.clear()

    def streamer():
        for chunk in agent.stream_completion(payload.message, stop_event=cancel_flag):
            if chunk is None:
                break
            # SSE 格式
            yield f"data: {chunk}\n\n"

    headers = {"X-Session-Id": sid}
    return StreamingResponse(streamer(), media_type="text/event-stream", headers=headers)


@app.get("/health")
def health():
    return {"ok": True}


@app.delete("/chat/session/{session_id}")
def delete_session(session_id: str):
    # 删除内存 Agent
    sessions.pop(session_id, None)
    # 删除中断标记
    session_cancel_flags.pop(session_id, None)
    # 删除持久化记录
    _delete_session_store(session_id)
    return {"ok": True, "session_id": session_id}


@app.post("/chat/session/{session_id}/cancel")
def cancel_session(session_id: str):
    """标记会话为中断，后台 Agent 将在下一次检查时退出循环。"""
    flag = _get_cancel_flag(session_id)
    flag.set()
    return {"ok": True, "session_id": session_id}


@app.get("/workspace/list")
def workspace_list(path: str = "."):
    """列出 workspace 指定目录的文件与子目录。"""
    real = _safe_path(path)
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail="路径不存在")
    items: List[Dict[str, Any]] = []
    for entry in os.scandir(real):
        stat = entry.stat()
        items.append(
            {
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {"path": path, "abs_path": os.path.realpath(real), "items": items}


@app.post("/workspace/upload")
async def workspace_upload(path: str = ".", file: UploadFile = File(...)):
    """上传文件到 workspace 指定目录。"""
    real_dir = _safe_path(path)
    os.makedirs(real_dir, exist_ok=True)
    dest = os.path.join(real_dir, file.filename)
    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"写入失败: {exc}")
    return {"ok": True, "path": path, "filename": file.filename, "dest": os.path.realpath(dest)}

