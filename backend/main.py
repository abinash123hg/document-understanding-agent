"""
Document Understanding Agent - Main FastAPI application
Strict document-only answers (no cross-document leakage)
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import settings
from backend import processor, retriever, storage, llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Understanding Agent",
    version="1.1.0",
    description="Private document Q&A – answers only from the selected document"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".png", ".jpg", ".jpeg"}


class ChatRequest(BaseModel):
    question: str
    document_name: Optional[str] = None


class Source(BaseModel):
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    document_name: Optional[str] = None


@app.get("/")
def root():
    return {
        "status": "running",
        "model": settings.llm_model,
        "mode": "document-only"
    }


@app.get("/health")
def health():
    return {
        "backend": "ok",
        "ollama": llm.is_ollama_ready(),
        "model": settings.llm_model,
        "chunk_count": len(storage.get_chunks())
    }


@app.post("/upload")
async def upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    temp_path = settings.uploads_dir / f"{uuid.uuid4()}{ext}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = processor.process_upload(temp_path, file.filename)

        if result["status"] == "FAILED":
            raise HTTPException(422, result.get("error", "Could not extract text"))

        return {
            "filename": file.filename,
            "document_name": file.filename,
            "status": "INDEXED",
            "extraction_method": result.get("extraction_method"),
            "chunk_count": result.get("chunk_count", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/documents/upload")
async def documents_upload(file: UploadFile):
    return await upload(file)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # Retrieve ONLY from the selected document
    sources = retriever.retrieve(
        query=question,
        top_k=settings.top_k,
        document_name=request.document_name
    )

    # No relevant chunks found → stop here
    if not sources:
        return ChatResponse(
            answer="I could not find the answer in this document.",
            sources=[],
            document_name=request.document_name
        )

    # Dynamic evidence context:
    normalized_question = question.lower()
    narrow_terms = (
        "what is", "define", "definition", "quote", "one sentence",
        "state only", "list every", "list all", "exact"
    )
    broad_terms = (
        "summarize", "summary", "across the document", "throughout",
        "all sections", "multi-part", "comprehensive", "in detail"
    )

    if any(term in normalized_question for term in broad_terms):
        context_limit = 8
    elif any(term in normalized_question for term in narrow_terms):
        context_limit = 3
    else:
        context_limit = 6

    sources = sources[:min(context_limit, len(sources))]

    try:
        answer = llm.generate_answer(question, sources)
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(
            503,
            "The local language model is unavailable. Please try again."
        )

    # Safe score conversion: handle None gracefully
    def safe_score(s: dict) -> float:
        sc = s.get("score")
        if sc is None:
            return 0.0
        try:
            return float(sc)
        except (TypeError, ValueError):
            return 0.0

    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                filename=s.get("filename", ""),
                chunk_index=s.get("chunk_index", 0),
                score=safe_score(s),
                text=s.get("text", "")[:300]
            )
            for s in sources
        ],
        document_name=request.document_name
    )


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    storage.clear_document(document_name)
    return {"status": "deleted", "document_name": document_name}