"""
Configuration for the Document Understanding Agent.

Every value can be overridden by an environment variable or a local .env file.
No secrets belong here: the project talks to a local Ollama server and needs no
API keys.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local language model (Ollama).
    # qwen2.5:3b is the ceiling for this laptop: 1.5b drops citation labels often
    # enough to be refused on its own output, and 7b is 4.7 GB, which blows the
    # ~5-6 GB budget once TrOCR is loaded alongside it.
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"

    # Handwriting recognition (TrOCR)
    htr_model: str = "microsoft/trocr-base-handwritten"
    htr_max_tokens: int = 64
    # Lines scoring below this are still kept, but the page is flagged so the
    # answer stage can refuse rather than build on unreadable OCR.
    htr_min_confidence: float = 0.30
    # Lines decoded per TrOCR call. Larger batches do not help on CPU and hold
    # more pixel tensors in memory at once.
    trocr_line_batch: int = 4

    # PDF handling. Embedded text is always tried first because it is instant;
    # only pages with no usable text are rasterized and recognised.
    pdf_dpi: int = 150
    pdf_max_dimension: int = 2000
    # Hard ceiling on the slow path, so a huge scanned PDF cannot hang an upload.
    # Above this the upload is rejected with instructions to split the file.
    pdf_max_ocr_pages: int = 40
    # Pages rasterized and recognised together. Two keeps peak memory down
    # without leaving the CPU idle the way fully serial pages would.
    ocr_pages_at_a_time: int = 2
    # Printed scans are read by the local Tesseract binary first, which is orders
    # of magnitude faster than TrOCR; TrOCR only sees pages Tesseract garbles.
    printed_scan_ocr: bool = True
    # fastNlMeansDenoising is the single slowest step in preprocessing and the
    # recognition accuracy it buys on these scans is marginal, so it is opt-in.
    denoise_images: bool = False

    # CORS. The page is served from 5500 (or Live Server's 5500), and opening it
    # straight from the backend's own port must work too.
    cors_origins: str = (
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:8000,http://127.0.0.1:8000"
    )

    # Upload limits
    max_upload_size_mb: int = 100

    # Retrieval
    top_k: int = 6
    chunk_size: int = 400
    chunk_overlap: int = 60
    # Cross-encoder candidates are kept relative to the best score for the query
    # rather than against an absolute cutoff: on chunked PDFs the score for the
    # answer-bearing excerpt can sit near -10 while unrelated excerpts sit near
    # -11, so only the gap between them carries information. Measured on scanned
    # slides: at 1.5 only 1-2 excerpts survived when 4 were asked for, which put
    # the answer-bearing chunk outside the window the model was shown.
    rerank_margin: float = 2.5
    # Candidate pool reaching the reranker. A fixed pool covers 8% of a 150-chunk
    # document but 0.8% of a 1,500-chunk one, so it grows with the document and
    # stops at rerank_candidates_max - that ceiling is what keeps a book inside
    # the laptop's RAM and the rerank inside a few seconds.
    rerank_candidates: int = 24
    rerank_candidates_max: int = 40
    # Evidence window for a request about the document as a whole, which is wider
    # than top_k on purpose: no chunk is the summary, so the model has to see
    # enough of the document to describe it. Measured on a 13-page scanned slide
    # deck with qwen2.5:3b - at 6 chunks (2.6k characters) the model answered
    # "summarize this document" with the refusal string every time, and at 10
    # (4.1k) it produced the deck's actual topics.
    overview_chunks: int = 10
    # Each factual claim in a draft is verified on its own pass. The cap is a CPU
    # budget, not a trust budget: on this laptop every extra check is another
    # second or two, so a claim past the cap is left unverified and dropped from
    # the answer rather than passed silently.
    max_claims_per_answer: int = 5

    # Paths
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
