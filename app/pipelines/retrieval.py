"""
app/pipelines/retrieval.py

Retrieval pipeline (main RAG loop):
  similarity search → rerank → build context → LLM answer → guardrails → citations.

Conversational memory is maintained per session via an in-process store
(thread-safe dict keyed by session_id).
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


from app.core.config import get_settings
from app.core.llm import get_llm
from app.core.logging import log
from app.pipelines.ingestion import get_vectorstore
from app.utils.citations import build_context_with_citations, inject_citation_markers
from app.utils.guardrails import (
    guardrail_off_topic_response,
    is_contract_related,
    validate_answer,
    LEGAL_DISCLAIMER,
    NOT_FOUND_RESPONSE,
)

# ── In-process session memory store ──────────────────────────────────────────
_memory_lock = threading.Lock()
_session_store: dict[str, ChatMessageHistory] = {}

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a legal contract analysis assistant.
Answer ONLY from the context provided below.
If the answer is not present, say exactly: "This information is not found in the document."
Cite specific sections using the [N] markers already in the context.
Never fabricate information, clauses, or legal interpretations.

Context:
{context}"""

_HUMAN = "Question: {question}"

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("placeholder", "{history}"), ("human", _HUMAN)]
)


# ── Session helpers ───────────────────────────────────────────────────────────


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """Return (or create) a ChatMessageHistory for *session_id*.

    Args:
        session_id: Unique identifier for the chat session.

    Returns:
        ChatMessageHistory bound to this session.
    """
    with _memory_lock:
        if session_id not in _session_store:
            _session_store[session_id] = ChatMessageHistory()
            log.debug("Created new session history for '%s'", session_id)
        return _session_store[session_id]


def clear_memory(session_id: str = "default") -> None:
    """Erase the conversation history for the given session.

    Args:
        session_id: The session whose memory should be erased.
    """
    with _memory_lock:
        if session_id in _session_store:
            del _session_store[session_id]
            log.info("Memory cleared for session '%s'", session_id)


# ── Main retrieval function ───────────────────────────────────────────────────


def answer_question(
    question: str,
    session_id: str = "default",
) -> dict[str, Any]:
    """Run the full RAG retrieval pipeline for a single question.

    Steps:
      1. Guardrail: reject off-topic questions.
      2. Similarity search in Chroma (top-k).
      3. Build numbered context with citation markers.
      4. Generate grounded answer via Gemini with session history.
      5. Post-process: guardrails + citations.

    Args:
        question: The user's question.
        session_id: Conversation session identifier for memory isolation.

    Returns:
        A dict with keys:
          - ``answer``: Final formatted answer string.
          - ``sources``: List of source metadata strings.
    """
    settings = get_settings()
    log.info("[Session %s] Question: %r", session_id, question[:120])

    # ── Step 1: Input guardrails ──────────────────────────────────────────────
    if not is_contract_related(question, context_available=True):
        return {
            "answer": guardrail_off_topic_response(question),
            "sources": [],
        }

    # ── Step 2: Similarity search ─────────────────────────────────────────────
    vectorstore = get_vectorstore()

    if not _store_has_documents(vectorstore):
        return {
            "answer": NOT_FOUND_RESPONSE + LEGAL_DISCLAIMER,
            "sources": [],
        }

    retrieved_docs: list[Document] = vectorstore.similarity_search(
        question, k=settings.retrieval_top_k
    )

    if not retrieved_docs:
        return {
            "answer": NOT_FOUND_RESPONSE + LEGAL_DISCLAIMER,
            "sources": [],
        }

    # ── Step 3: Build numbered context ────────────────────────────────────────
    context_text, citation_map = build_context_with_citations(retrieved_docs)

    # ── Step 4: LLM generation with session memory ────────────────────────────
    llm = get_llm()

    chain = CHAT_PROMPT | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    raw_answer: str = chain_with_history.invoke(
        {"question": question, "context": context_text},
        config={"configurable": {"session_id": session_id}},
    )

    # ── Step 5: Output guardrails + citations ─────────────────────────────────
    validated = validate_answer(raw_answer)
    final_answer = inject_citation_markers(validated, citation_map)

    sources = [citation_map[i] for i in sorted(citation_map)]
    log.info("[Session %s] Answer generated (%d chars)", session_id, len(final_answer))

    return {"answer": final_answer, "sources": sources}


# ── Utility ───────────────────────────────────────────────────────────────────


def _store_has_documents(vectorstore: Any) -> bool:
    """Return True if the Chroma collection contains at least one document.

    Args:
        vectorstore: A Chroma instance.

    Returns:
        True if documents exist, False otherwise.
    """
    try:
        return vectorstore._collection.count() > 0
    except Exception:
        return False


def retrieve_similar_chunks(query: str, k: int | None = None) -> list[Document]:
    """Return the top-k most similar document chunks for *query*.

    Args:
        query: Search query string.
        k: Number of results (defaults to settings.retrieval_top_k).

    Returns:
        List of LangChain Documents ordered by similarity.

    Raises:
        ValueError: If no documents have been ingested.
    """
    settings = get_settings()
    vectorstore = get_vectorstore()
    if not _store_has_documents(vectorstore):
        raise ValueError("No document has been ingested yet.")
    return vectorstore.similarity_search(query, k=k or settings.retrieval_top_k)