"""
app/pipelines/summarization.py

Summarization pipeline: generate bullet-point key insights,
a short narrative summary, and risk highlights from the contract.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import get_llm
from app.core.logging import log
from app.pipelines.ingestion import get_vectorstore

# ── Prompt templates ─────────────────────────────────────────────────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior legal analyst. Analyse the contract excerpts below and produce:

1. **Key Points** – exactly 5 bullet points capturing the most important obligations, rights, and terms.
2. **Short Summary** – 3–5 sentences summarising the contract's purpose and scope.
3. **Risk Highlights** – bullet points identifying clauses that carry legal, financial, or operational risk.

Respond in clean markdown. Do NOT add information not present in the excerpts.

Contract Excerpts:
{context}
"""
)


def summarize_contract(top_k: int = 20) -> dict[str, str]:
    """Generate a structured summary of the loaded contract.

    Retrieves the top-*k* most representative chunks from the vector store,
    then asks Gemini to produce structured output.

    Args:
        top_k: Number of document chunks to include as context.

    Returns:
        A dict with a single key ``summary`` containing the markdown text.

    Raises:
        RuntimeError: If the vector store is empty (no document loaded).
    """
    log.info("Summarization pipeline started (top_k=%d)", top_k)

    vectorstore = get_vectorstore()

    # Use a broad query to surface the most representative chunks
    docs = vectorstore.similarity_search(
        query="contract terms obligations parties payment termination liability",
        k=top_k,
    )

    if not docs:
        raise RuntimeError(
            "No document has been ingested yet. Please upload a contract first."
        )

    context = "\n\n---\n\n".join(
        f"[Chunk {d.metadata.get('chunk_index', '?')}]\n{d.page_content}" for d in docs
    )

    llm = get_llm()
    chain = _SUMMARY_PROMPT | llm | StrOutputParser()
    summary = chain.invoke({"context": context})

    log.info("Summarization complete (%d chars)", len(summary))
    return {"summary": summary}
