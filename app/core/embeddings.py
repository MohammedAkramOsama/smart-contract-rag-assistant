"""
app/core/embeddings.py

Factory for local Ollama embeddings (nomic-embed-text).
Remote embeddings are explicitly forbidden by the spec.
"""

from functools import lru_cache

from langchain_community.embeddings import OllamaEmbeddings

from app.core.config import get_settings
from app.core.logging import log


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Return a cached OllamaEmbeddings instance.

    Returns:
        OllamaEmbeddings configured with the local nomic-embed-text model.

    Raises:
        ConnectionError: If the Ollama server is unreachable.
    """
    settings = get_settings()
    log.info(
        "Initialising Ollama embeddings: model=%s base_url=%s",
        settings.ollama_embed_model,
        settings.ollama_base_url,
    )
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
