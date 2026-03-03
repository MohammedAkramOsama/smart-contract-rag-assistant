"""
Structured document ingestion pipeline
PDF → elements → semantic chunks → embeddings
"""

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from unstructured.partition.pdf import partition_pdf

from app.core.llm import get_embeddings
from app.core.config import get_settings
from app.core.logging import log

_vectorstore = None


# ===================== PDF Parsing =====================

def parse_pdf(file_path: str) -> List[Document]:
    """Parse PDF into structured semantic elements"""

    elements = partition_pdf(
        filename=file_path,
        strategy="fast",
        chunking_strategy="by_title",
    )
    docs = []
    for el in elements:
        if not getattr(el, "text", None):
            continue

        text = el.text.strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(file_path).name,
                    "type": el.category
                }
            )
        )

    log.info("Parsed %s semantic elements", len(docs))
    return docs


# ===================== Chunking =====================

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Create semantic chunks suitable for RAG"""

    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = splitter.split_documents(docs)

    # Stamp each chunk with its index so citations can display "chunk N"
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i + 1

    log.info("Created %s chunks", len(chunks))
    return chunks


# ===================== Parallel Embedding =====================

def _embed_chunks_parallel(embeddings, chunks: List[Document], workers: int = 4) -> List[List[float]]:
    """Embed all chunks concurrently using a thread pool.

    Ollama processes requests sequentially by default; running multiple
    threads keeps the model busy and cuts total embedding time by ~(workers)x.

    Args:
        embeddings: LangChain embeddings instance.
        chunks: Document chunks to embed.
        workers: Number of parallel threads (default 4, tune to your machine).

    Returns:
        List of embedding vectors in the same order as chunks.
    """
    texts = [c.page_content for c in chunks]
    ordered = [None] * len(texts)

    def _embed_one(args):
        idx, text = args
        return idx, embeddings.embed_query(text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_embed_one, (i, t)): i for i, t in enumerate(texts)}
        done = 0
        for future in as_completed(futures):
            idx, vec = future.result()
            ordered[idx] = vec
            done += 1
            if done % 10 == 0:
                log.info("  Embedded %d / %d chunks...", done, len(texts))

    return ordered


# ===================== Ingestion =====================

def ingest_document(file_path: str):
    """Main ingestion entry"""

    global _vectorstore

    log.info("Parsing document...")
    docs = parse_pdf(file_path)

    log.info("Chunking document...")
    chunks = chunk_documents(docs)

    log.info("Creating embeddings index...")
    embeddings = get_embeddings()
    settings = get_settings()
    persist_dir = str(settings.chroma_persist_dir)

    # Wipe the existing collection so stale docs from previous uploads
    # never bleed into retrieval results.
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "langchain"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        log.info("Cleared existing ChromaDB collection '%s'", collection_name)

    # Embed in parallel (much faster than sequential Ollama calls)
    log.info("Embedding %d chunks with parallel workers...", len(chunks))
    vectors = _embed_chunks_parallel(embeddings, chunks, workers=4)

    # Insert pre-computed vectors directly — no redundant re-embedding
    _vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    _vectorstore._collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors,
        documents=[c.page_content for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    _vectorstore.persist()

    log.info("Document indexed successfully")

    return {
        "chunks": len(chunks),
        "collection": _vectorstore._collection.name,
    }


# ===================== Getter =====================

def get_vectorstore():
    return _vectorstore