# Smart Contract RAG Assistant – Project Summary

## Project Overview

This project implements a modular Retrieval-Augmented Generation (RAG) system using LangChain.  
The system allows users to upload a document (PDF/DOCX) and perform document-grounded question answering and summarization.

The core objective is to ensure that answers are strictly derived from the uploaded document, reducing hallucinations and improving reliability.

---

## System Architecture

The system follows a modular architecture:

Document → Parsing → Chunking → Embedding → Vector Store → Retrieval → LLM → Grounded Response

Components are separated into:
- Ingestion pipeline
- Retrieval pipeline
- Summarization pipeline
- Guardrails & validation
- API layer (FastAPI)
- UI layer (Gradio)

---

## Why RAG?

Traditional LLMs may hallucinate or generate unsupported answers.  
RAG solves this by retrieving relevant document chunks before generation, forcing the model to answer using external context.

This makes the system:
- More accurate
- Explainable (with citations)
- Suitable for legal document analysis

---

## Technology Choices & Justification

### 1) LangChain
LangChain was chosen because it provides:
- Modular pipeline abstraction
- Built-in retrievers
- Memory management
- Integration with multiple LLMs and vector databases

It enables clean separation between ingestion, retrieval, and generation stages.

---

### 2) Gemini 2.5 Flash (LLM)
Gemini 2.5 Flash was selected due to:
- Strong reasoning capabilities
- Large context window
- Fast response time
- Free academic usage tier

It is used strictly for answer generation after retrieval.

---

### 3) Ollama + nomic-embed-text (Embeddings)
Local embeddings were used to:
- Preserve document privacy
- Avoid external embedding API costs
- Allow offline vector indexing

The `nomic-embed-text` model provides strong semantic similarity performance.

---

### 4) ChromaDB (Vector Database)
ChromaDB was selected because:
- Lightweight and easy to deploy
- Supports persistent storage
- Fully compatible with LangChain
- Suitable for academic RAG systems

---

### 5) FastAPI
FastAPI provides:
- Clean REST API structure
- Automatic documentation
- Easy backend/frontend separation

---

### 6) Gradio
Gradio was used to:
- Quickly build an interactive UI
- Demonstrate document upload + chat
- Visualize grounded answers

---

## Key Features

- Document upload & ingestion
- Semantic chunking
- Vector similarity search
- Grounded answer generation
- Citation support
- Guardrails against off-topic queries
- Structured contract summarization
- Modular, non-notebook implementation

---

## Conclusion

This project demonstrates a complete modular RAG architecture built using LangChain.  
It ensures that answers are derived strictly from uploaded documents and provides explainable, grounded responses suitable for contract analysis.