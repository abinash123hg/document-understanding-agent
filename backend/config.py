"""Configuration - all localhost, no API keys."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"
    max_upload_size_mb: int = 10
    top_k: int = 4
    min_similarity: float = 0.10
    chunk_size: int = 500
    chunk_overlap: int = 80
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
