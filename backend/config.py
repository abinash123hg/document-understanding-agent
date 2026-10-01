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

    # Local language model (Ollama)
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:1.5b"

    # Handwriting recognition (TrOCR)
    htr_model: str = "microsoft/trocr-base-handwritten"
    htr_max_tokens: int = 64
    # Lines scoring below this are still kept, but the page is flagged so the
    # answer stage can refuse rather than build on unreadable OCR.
    htr_min_confidence: float = 0.30

    # PDF handling. Embedded text is always tried first because it is instant;
    # only pages with no usable text are rasterized and recognised.
    pdf_dpi: int = 200
    pdf_max_dimension: int = 2000
    # Hard ceiling on the slow path, so a huge scanned PDF cannot hang an upload.
    pdf_max_ocr_pages: int = 50

    # CORS
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"

    # Upload limits
    max_upload_size_mb: int = 50

    # Retrieval
    top_k: int = 4
    chunk_size: int = 400
    chunk_overlap: int = 60
    min_rerank_score: float = -6.0

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
