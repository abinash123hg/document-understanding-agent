"""
Configuration – all settings for the Document Understanding Agent
Optimized for models under 2GB
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # Ollama settings
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:3b"          # change to qwen2.5:1.5b if you want faster

    # CORS
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500"

    # Upload limits
    max_upload_size_mb: int = 15

    # Retrieval settings (important for small models)
    # Change these values
    top_k: int = 4
    min_similarity: float = 0.05          # ← change to 0.05
    chunk_size: int = 400
    chunk_overlap: int = 60
    # Paths
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()

# Create folders if they do not exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)