"""
Document Understanding Agent for Handwritten Text Recognition.

Answers are drawn only from the selected document, and only from text that was
actually extracted from it. Uploads never block the event loop: the file is
streamed to disk and recognition runs in a worker thread.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import handwriting, llm, processor, retriever, storage
from backend.config import settings
from backend.handwriting import HandwritingModelError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}
READ_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(
    title="Document Understanding Agent for Handwritten Text Recognition",
    version="2.0.0",
    description="Private document Q&A - answers only from the selected document",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    document_name: Optional[str] = None


class Source(BaseModel):
    filename: str
    chunk_index: int
    chunk_id: str = ""
    page_number: Optional[int] = None
    content_type: str = "digital_text"
    extraction_method: str = "unknown"
    score: float = 0.0
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    document_name: Optional[str] = None
    refusal_reason: Optional[
        Literal[
            "no_evidence",
            "low_handwriting_confidence",
            "untraceable_citation",
            "unsupported_claim",
            "not_answerable",
        ]
    ] = None


@app.get("/")
def root():
    return {
        "status": "running",
        "llm_model": settings.llm_model,
        "handwriting_model": settings.htr_model,
        "mode": "document-only",
    }


@app.get("/health")
def health():
    return {
        "backend": "ok",
        "ollama": llm.is_ollama_reachable(),
        "llm_model_pulled": llm.is_ollama_ready(),
        "llm_model": settings.llm_model,
        "handwriting_model": settings.htr_model,
        "handwriting_loaded": handwriting.is_loaded(),
        "document_count": len(storage.list_documents()),
        "chunk_count": len(storage.get_chunks()),
    }


@app.get("/documents")
def documents():
    return {"documents": storage.list_documents()}


async def _save_limited(file: UploadFile, temp_path: Path) -> int:
    """Stream to disk and abort the moment the limit is crossed, rather than
    buffering an oversized upload in memory first."""
    written = 0

    with open(temp_path, "wb") as handle:
        while True:
            block = await file.read(READ_CHUNK)
            if not block:
                break
            written += len(block)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    413,
                    f"File exceeds the {settings.max_upload_size_mb} MB limit.",
                )
            handle.write(block)

    return written


@app.post("/upload")
async def upload(file: UploadFile):
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(400, "No filename provided")

    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    temp_path = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        size = await _save_limited(file, temp_path)

        # Recognition is CPU-bound; keep it off the event loop so other
        # requests are not stalled while a scanned PDF is being read.
        result = await run_in_threadpool(processor.process_upload, temp_path, name)

        if result.get("status") != "INDEXED":
            raise HTTPException(422, result.get("error", "Could not extract text"))

        return {
            "filename": name,
            "document_name": name,
            "status": "INDEXED",
            "extraction_method": result.get("extraction_method"),
            "content_type": result.get("content_type"),
            "chunk_count": result.get("chunk_count", 0),
            "page_count": result.get("page_count", 0),
            "ocr_page_count": result.get("ocr_page_count", 0),
            "bytes_received": size,
        }
    except HTTPException:
        raise
    except HandwritingModelError as error:
        logger.exception("Handwriting model unavailable")
        raise HTTPException(503, str(error))
    except ValueError as error:
        raise HTTPException(422, str(error))
    except Exception as error:
        logger.exception("Upload processing failed")
        raise HTTPException(500, f"Processing failed: {error}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as error:
            # A reader that still holds the handle must not replace the real
            # outcome of the request with a cleanup failure.
            logger.warning("Could not remove temp upload %s: %s", temp_path.name, error)


@app.post("/documents/upload")
async def documents_upload(file: UploadFile):
    return await upload(file)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    # Retrieval is scoped to the selected document and fails closed when no
    # document is selected, so an answer can never come from another file.
    if llm.is_document_request(question):
        # A request about the document as a whole has no answer-bearing chunk, so
        # ranking it against the request returns arbitrary slides. Coverage of the
        # selected document is what this kind of request needs instead.
        retrieved = retriever.document_overview(
            document_name=request.document_name,
        )
    else:
        retrieved = retriever.retrieve(
            query=question,
            top_k=settings.top_k,
            document_name=request.document_name,
        )

    try:
        answer, refusal_reason, trusted = llm.generate_answer_result(
            question, retrieved
        )
    except Exception as error:
        logger.exception("LLM generation failed")
        raise HTTPException(
            503,
            f"The local language model is unavailable: {error}",
        )

    # Sources are the excerpts that were labelled [S1]..[Sn] for the model, in
    # that order. The raw retrieval list is never returned: it also carries
    # chunks that were filtered out, so its index 1 is not the excerpt a printed
    # [S2] refers to, and a reader following that citation lands on the wrong
    # passage.
    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                filename=str(source.get("filename", "")),
                chunk_index=int(source.get("chunk_index", 0)),
                chunk_id=str(source.get("chunk_id", "")),
                page_number=source.get("page_number"),
                content_type=str(source.get("content_type", "digital_text")),
                extraction_method=str(source.get("extraction_method", "unknown")),
                score=float(source.get("score") or 0.0),
                text=str(source.get("text", ""))[:400],
            )
            for source in trusted
        ],
        document_name=request.document_name,
        refusal_reason=refusal_reason,
    )


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    storage.clear_document(document_name)
    # The vector index holds a separate copy of the text. Forgetting this leaves
    # deleted content retrievable, which reads as grounded evidence to the
    # verifier because the stale text really is in the source excerpts.
    retriever.purge_document(document_name)
    return {"status": "deleted", "document_name": document_name}
