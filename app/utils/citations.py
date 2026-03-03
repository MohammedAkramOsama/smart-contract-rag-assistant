"""
app/utils/citations.py

Citation utilities: inject source markers [1], [2]… into LLM answers
and build a formatted references section from retrieved documents.
"""

from __future__ import annotations

import re
from langchain_core.documents import Document

from app.core.logging import log


def build_context_with_citations(docs: list[Document]) -> tuple[str, dict[int, str]]:
    """Convert retrieved documents into a numbered context block.

    Args:
        docs: Retrieved LangChain Documents (already reranked).

    Returns:
        A tuple of:
          - context_text: Formatted string with [N] markers for each chunk.
          - citation_map: Mapping of citation number → source metadata string.
    """
    citation_map: dict[int, str] = {}
    context_parts: list[str] = []

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        marker = f"[{idx}]"
        citation_map[idx] = f"{marker} Source: *{source}* (chunk {chunk_idx})"
        context_parts.append(f"{marker}\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)
    log.debug("Built context with %d citation(s)", len(citation_map))
    return context_text, citation_map


def format_references(citation_map: dict[int, str]) -> str:
    """Format the citation map into a markdown references section.

    Args:
        citation_map: Mapping of number → citation string.

    Returns:
        Markdown string listing all sources.
    """
    if not citation_map:
        return ""
    lines = ["", "---", "**References:**"]
    for num in sorted(citation_map):
        lines.append(citation_map[num])
    return "\n".join(lines)


def inject_citation_markers(answer: str, citation_map: dict[int, str]) -> str:
    """Append a references block to the answer if citations exist.

    The LLM is prompted to use [N] markers naturally; this function
    ensures the references section is always appended.

    Args:
        answer: LLM-generated answer text.
        citation_map: Citation number → source string.

    Returns:
        Answer with a references section appended.
    """
    refs = format_references(citation_map)
    if refs:
        return answer + "\n" + refs
    return answer
