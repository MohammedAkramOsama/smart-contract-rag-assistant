"""
Structured document ingestion pipeline
PDF → elements → semantic chunks → embeddings
"""

from pathlib import Path
from typing import List

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

    # elements = partition_pdf(
    #     filename=file_path,
    #     strategy="hi_res",
    #     infer_table_structure=True,
    #     chunking_strategy="by_title",
    # )
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

    # Delete any existing collection so stale documents from previous
    # uploads never bleed into retrieval results.
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection_name = "langchain"  # LangChain's default Chroma collection name
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        log.info("Cleared existing ChromaDB collection '%s'", collection_name)

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
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