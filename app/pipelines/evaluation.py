"""
app/pipelines/evaluation.py

Evaluation pipeline: compute RAG quality metrics:
  - context_relevance  – how relevant are the retrieved chunks to the question?
  - groundedness       – is the answer supported by the context?
  - answer_completeness – does the answer address all aspects of the question?

Metrics are computed via an LLM judge (Gemini) to avoid external API calls.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import get_llm
from app.core.logging import log

# ── Prompt templates ─────────────────────────────────────────────────────────

_EVAL_PROMPT = ChatPromptTemplate.from_template(
    """You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.
Score the following on three dimensions. Respond ONLY with a JSON object.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Scoring rubric (0.0 – 1.0):
- context_relevance: Are the retrieved passages relevant to the question?
- groundedness: Is the answer strictly supported by the context (no hallucinations)?
- answer_completeness: Does the answer fully address the question given the context?

Return EXACTLY this JSON (no markdown, no explanation):
{{"context_relevance": <float>, "groundedness": <float>, "answer_completeness": <float>, "notes": "<brief explanation>"}}
"""
)


def evaluate_response(
    question: str,
    context: str,
    answer: str,
) -> dict[str, object]:
    """Run LLM-judge evaluation on a single RAG response.

    Args:
        question: The original user question.
        context: The concatenated retrieved document chunks.
        answer: The generated answer to evaluate.

    Returns:
        A dict with keys:
          - ``context_relevance`` (float)
          - ``groundedness`` (float)
          - ``answer_completeness`` (float)
          - ``notes`` (str)
          - ``error`` (str, only present on parse failure)
    """
    import json
    import re

    log.info("Evaluation pipeline: scoring response for question %r", question[:80])

    llm = get_llm()
    chain = _EVAL_PROMPT | llm | StrOutputParser()

    raw: str = chain.invoke(
        {"question": question, "context": context[:4000], "answer": answer[:2000]}
    )

    # Strip potential markdown code fences
    clean = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`").strip()

    try:
        metrics: dict[str, object] = json.loads(clean)
        log.info(
            "Evaluation scores – relevance=%.2f groundedness=%.2f completeness=%.2f",
            metrics.get("context_relevance", 0),
            metrics.get("groundedness", 0),
            metrics.get("answer_completeness", 0),
        )
        return metrics
    except json.JSONDecodeError as exc:
        log.error("Evaluation JSON parse error: %s | raw=%r", exc, raw[:200])
        return {
            "context_relevance": 0.0,
            "groundedness": 0.0,
            "answer_completeness": 0.0,
            "notes": "Evaluation failed – could not parse LLM output.",
            "error": str(exc),
        }
