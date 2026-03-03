"""
app/main.py

FastAPI application factory.
Registers all routers and exposes LangServe chains at /retriever and /generator.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda

from app.api import routes_chat, routes_health, routes_upload
from app.core.config import get_settings
from app.core.logging import log, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Application lifespan: runs setup on startup and teardown on shutdown."""
    setup_logging()
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    log.info("Smart Contract Assistant API started on port %d", settings.api_port)
    yield
    log.info("Smart Contract Assistant API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured FastAPI instance.
    """
    app = FastAPI(
        title="Smart Contract Assistant API",
        description=(
            "RAG-powered contract analysis: upload, chat, summarize, and evaluate "
            "legal contracts via Gemini + Ollama + Chroma."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Core API routes ───────────────────────────────────────────────────────
    app.include_router(routes_health.router)
    app.include_router(routes_upload.router)
    app.include_router(routes_chat.router)

    # ── LangServe chains ──────────────────────────────────────────────────────
    _mount_langserve_routes(app)

    return app


def _mount_langserve_routes(app: FastAPI) -> None:
    """Mount LangServe chain endpoints at /retriever and /generator.

    Args:
        app: The FastAPI application instance.
    """
    try:
        from langserve import add_routes  # type: ignore[import]
        from app.core.llm import get_llm
        from app.pipelines.ingestion import get_vectorstore
        from app.core.config import get_settings

        settings = get_settings()

        # Retriever chain
        def _retriever_runnable(query: str) -> list[dict]:  # type: ignore[type-arg]
            vs = get_vectorstore()
            docs = vs.similarity_search(query, k=settings.retrieval_top_k)
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

        retriever_chain = RunnableLambda(_retriever_runnable)
        add_routes(app, retriever_chain, path="/retriever")
        log.info("LangServe route mounted: /retriever")

        # Generator chain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        generator_prompt = ChatPromptTemplate.from_template(
            "Answer this question about a contract: {question}"
        )
        generator_chain = generator_prompt | get_llm() | StrOutputParser()
        add_routes(app, generator_chain, path="/generator")
        log.info("LangServe route mounted: /generator")

    except ImportError:
        log.warning("langserve not installed – skipping /retriever and /generator routes.")


app = create_app()
