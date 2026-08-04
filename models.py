from pydantic import BaseModel
from typing import Any

class ChatRequest(BaseModel):
    message: str
    language: str | None = None
    stream: bool = False
    session_id: int | None = None  # None = use global session; set to branch session id for branch chat


class BranchChatRequest(BaseModel):
    context_messages: list[dict[str, Any]]  # [{role: str, text: str}, ...]
    language: str | None = None


class DeviceRegisterRequest(BaseModel):
    fcm_token: str
