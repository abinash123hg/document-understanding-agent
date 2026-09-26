"""Pydantic models."""
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentMeta(BaseModel):
    id: str
    filename: str
    file_type: str
    status: Literal["PROCESSING", "INDEXED", "FAILED"]
    extraction_method: Literal["digital", "ocr"] = "digital"
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: str


class Source(BaseModel):
    filename: str
    chunk_index: int
    text: str
    score: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    error: Optional[str] = None
