# Smart Contract Assistant

A production-style **RAG (Retrieval-Augmented Generation)** system for analysing Upload a PDF or DOCX, then chat with it or generate an instant structured summary.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (`langchain-google-genai`) |
| Embeddings | Local Ollama `nomic-embed-text` |
| Vector Store | ChromaDB (persistent) |
| Backend API | FastAPI + LangServe |
| Frontend | Gradio |

---

## Project Structure

```
smart_contract_assistant/
├── app/
│   ├── api/          # FastAPI routers
│   ├── core/         # Config, LLM, embeddings, logging
│   ├── pipelines/    # Ingestion, retrieval, summarization, evaluation
│   ├── utils/        # File parsers, text splitter, guardrails, citations
│   └── main.py       # FastAPI app factory
├── frontend/
│   └── gradio_app.py # Gradio UI
├── data/
│   ├── uploads/      # Saved contract files
│   └── chroma_db/    # Persisted vector store
├── test_data/       # Sample PDF files for testing
├── run_server.py     # Start the API server
├── run_ui.py         # Start the Gradio frontend
├── requirements.txt
└── .env.example
```

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** – installed and running with the embed model pulled:
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```
3. **Google Gemini API key** – [Get one here](https://aistudio.google.com/app/apikey)

---

## Setup

```bash
# 1. Clone / navigate to the project
cd smart_contract_assistant_

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
# Edit .env and set GOOGLE_API_KEY=<your_key>
```

---

## Running the Application

Open **two terminal windows**:

**Terminal 1 – API Server**
```bash
python run_server.py
# API available at http://localhost:8000
# Docs at        http://localhost:8000/docs
```

**Terminal 2 – Gradio UI**
```bash
python run_ui.py
# UI available at http://localhost:7860
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status |
| `POST` | `/upload` | Upload & ingest a contract |
| `POST` | `/chat` | Ask a question (RAG) |
| `POST` | `/summary` | Generate contract summary |
| `POST` | `/reset` | Clear conversation memory |
| `POST` | `/retriever/invoke` | LangServe retriever chain |
| `POST` | `/generator/invoke` | LangServe generator chain |

---

## Guardrails

- Off-topic questions are rejected with a clear message.
- Answers are strictly grounded in retrieved context.
- Source citations `[1]`, `[2]` … are appended to every answer.
- A legal disclaimer is added to all responses.

---

## Evaluation

Call the evaluation pipeline programmatically:

```python
from app.pipelines.evaluation import evaluate_response

metrics = evaluate_response(
    question="What are the payment terms?",
    context="<retrieved chunks>",
    answer="<generated answer>",
)
print(metrics)
# {'context_relevance': 0.9, 'groundedness': 0.95, 'answer_completeness': 0.85, 'notes': '...'}
```
