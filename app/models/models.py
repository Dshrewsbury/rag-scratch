from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class BaseResponse(BaseModel):
    timestamp: str

    @classmethod
    def now(cls) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class MessageRequest(BaseModel):
    conversation_id: str
    content: str

class AgentResponse(BaseResponse):
    conversation_id: str
    message: str
    response: str

class HealthResponse(BaseResponse):
    status: str
    model: str

class ModelConfigData(BaseModel):
    provider: str = "openai"   # Options: "openai", "ollama", "anthropic"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    api_base: Optional[str] = None

class ModelConfigResponse(BaseResponse):
    provider: str
    model: str
    api_key_set: bool
    api_base: Optional[str]