from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话标识，不传则新建")
    message: str = Field(..., description="用户输入")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tools: List[str] = []
    tool_results: List[str] = []

