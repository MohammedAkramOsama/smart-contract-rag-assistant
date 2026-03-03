"""
app/core/llm.py

Factory functions for LLM and embedding instances.
Gemini = answers
Ollama = retrieval understanding
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings

from app.core.config import get_settings
from app.core.logging import log


# ===================== LLM (Gemini) =====================

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached Gemini chat model"""
    settings = get_settings()

    log.info(
        "Initialising Gemini LLM: model=%s temperature=%s",
        settings.gemini_model,
        settings.gemini_temperature,
    )

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        google_api_key=settings.google_api_key,
        convert_system_message_to_human=True,
    )


# ===================== Embeddings (Ollama) =====================

@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Return local embedding model used for semantic search"""

    settings = get_settings()

    log.info(
        "Initialising Ollama embeddings: model=%s url=%s",
        settings.ollama_embed_model,
        settings.ollama_base_url,
    )

    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )