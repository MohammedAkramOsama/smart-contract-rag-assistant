"""
app/utils/text_splitter.py

Semantic text chunking using LangChain's RecursiveCharacterTextSplitter.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import log


def split_text(text: str, source: str = "unknown") -> list[Document]:
    """Split raw text into overlapping semantic chunks.

    Args:
        text: Full document text extracted by file_parsers.
        source: Original filename or identifier (stored in metadata).

    Returns:
        List of LangChain Document objects ready for embedding.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": idx},
        )
        for idx, chunk in enumerate(raw_chunks)
    ]
    log.info(
        "Split document '%s' into %d chunks (size=%d overlap=%d)",
        source,
        len(documents),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return documents
