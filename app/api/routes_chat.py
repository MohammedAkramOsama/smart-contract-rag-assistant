"""
app/api/routes_chat.py

Chat, summary, and reset endpoints.

  POST /chat    – ask a question about the uploaded contract
  POST /summary – generate a structured contract summary
  POST /reset   – clear conversation memory
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logging import log
from app.pipelines.retrieval import answer_question, clear_memory
from app.pipelines.summarization import summarize_contract

router = APIRouter(tags=["Chat"])


# ── Request / Response models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request schema for /chat."""

    question: str = Field(..., min_length=1, description="The user's question.")
    session_id: str = Field("default", description="Session identifier for conversation memory.")


class ChatResponse(BaseModel):
    """Response schema for /chat."""

    answer: str
    sources: list[str]
    session_id: str


class SummaryResponse(BaseModel):
    """Response schema for /summary."""

    summary: str


class ResetRequest(BaseModel):
    """Request schema for /reset."""

    session_id: str = Field("default", description="Session to reset.")


class ResetResponse(BaseModel):
    """Response schema for /reset."""

    message: str
    session_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about the loaded contract",
)
def chat(request: ChatRequest) -> ChatResponse:
    """Run the RAG retrieval pipeline and return a grounded answer.

    Args:
        request: ChatRequest with question and optional session_id.

    Returns:
        ChatResponse with answer text, source citations, and session_id.

    Raises:
        HTTPException 500: If the retrieval pipeline encounters an error.
    """
    log.info("[/chat] session=%s question=%r", request.session_id, request.question[:80])
    try:
        result = answer_question(
            question=request.question,
            session_id=request.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Chat pipeline failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline error: {exc}",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )


@router.post(
    "/summary",
    response_model=SummaryResponse,
    summary="Generate a structured summary of the loaded contract",
)
def summary() -> SummaryResponse:
    """Run the summarization pipeline and return structured markdown output.

    Returns:
        SummaryResponse containing markdown-formatted summary.

    Raises:
        HTTPException 422: No document loaded.
        HTTPException 500: Pipeline failure.
    """
    log.info("[/summary] Summarization requested")
    try:
        result = summarize_contract()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Summarization pipeline failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization error: {exc}",
        ) from exc

    return SummaryResponse(summary=result["summary"])


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Clear the conversation memory for a session",
)
def reset(request: ResetRequest) -> ResetResponse:
    """Erase the conversation history for the given session.

    Args:
        request: ResetRequest with the session_id to clear.

    Returns:
        ResetResponse confirming the reset.
    """
    log.info("[/reset] Clearing memory for session '%s'", request.session_id)
    clear_memory(session_id=request.session_id)
    return ResetResponse(
        message=f"Conversation memory cleared for session '{request.session_id}'.",
        session_id=request.session_id,
    )
