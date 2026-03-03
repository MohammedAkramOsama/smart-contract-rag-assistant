"""
app/core/config.py

Centralised configuration management.
All settings are loaded from environment variables or a .env file.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(0.2, alias="GEMINI_TEMPERATURE")

    # ── Local Ollama embeddings ──────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field("nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    # ── Chroma vector store ──────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        BASE_DIR / "data" / "chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        "smart_contracts", alias="CHROMA_COLLECTION_NAME"
    )

    # ── Upload storage ───────────────────────────────────────────────────────
    upload_dir: Path = Field(BASE_DIR / "data" / "uploads", alias="UPLOAD_DIR")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(5, alias="RETRIEVAL_TOP_K")

    # ── Text splitting ───────────────────────────────────────────────────────
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(100, alias="CHUNK_OVERLAP")

    # ── FastAPI server ───────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")

    # ── Gradio ───────────────────────────────────────────────────────────────
    gradio_host: str = Field("0.0.0.0", alias="GRADIO_HOST")
    gradio_port: int = Field(7860, alias="GRADIO_PORT")
    api_base_url: str = Field("http://localhost:8000", alias="API_BASE_URL")

    @property
    def embedding_model(self):
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=self.ollama_base_url,
            model=self.ollama_embed_model,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once, reused on every call)."""
    return Settings()  # type: ignore[call-arg]