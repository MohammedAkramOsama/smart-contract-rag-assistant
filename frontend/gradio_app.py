"""
frontend/gradio_app.py

Modern Gradio UI for the Smart Contract Assistant.
Compatible with Gradio v4.36+ / v5 / v6.

Layout:
  Left panel  – document upload + reset conversation
  Right panel – chatbot conversation with streaming

Backend endpoints (via requests):
  POST /upload   – ingest a contract file
  POST /ask      – ask a question (RAG)
  POST /reset    – clear conversation memory

BACKEND_URL is read from the environment variable BACKEND_URL
(default: http://127.0.0.1:8000).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Generator

import gradio as gr
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND_URL: str = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT: float = 120.0

# ── Helper ────────────────────────────────────────────────────────────────────


def _post_json(endpoint: str, **payload: object) -> dict:
    """Send a POST request with a JSON body to the backend.

    Args:
        endpoint: Path relative to BACKEND_URL (e.g. "/ask").
        **payload: JSON body fields.

    Returns:
        Parsed JSON response dict, or {"error": "..."} on failure.
    """
    url = BACKEND_URL + endpoint
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"error": detail}
    except requests.ConnectionError:
        return {
            "error": (
                f"Cannot connect to backend at {BACKEND_URL}. "
                "Make sure the server is running."
            )
        }
    except requests.Timeout:
        return {"error": "Request timed out. The server may be overloaded."}


# ── Core functions ─────────────────────────────────────────────────────────────


def upload_document(file: str | None) -> str:
    """Upload a PDF or DOCX file and trigger the ingestion pipeline.

    Args:
        file: Temporary filepath provided by Gradio after the user selects a file.

    Returns:
        A status message string (shown in the UI).
    """
    if file is None:
        return "⚠️ Please select a PDF or DOCX file before uploading."

    file_path = Path(file)
    suffix = file_path.suffix.lower()
    mime_map = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    mime = mime_map.get(suffix, "application/octet-stream")

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                BACKEND_URL + "/upload",
                files={"file": (file_path.name, f, mime)},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return f"❌ Upload failed: {detail}"
    except requests.ConnectionError:
        return f"❌ Cannot connect to backend at {BACKEND_URL}."
    except requests.Timeout:
        return "❌ Upload timed out."

    return (
        f"✅ **{data.get('filename', file_path.name)}** ingested successfully!\n\n"
        f"- Chunks stored: **{data.get('chunks', '?')}**\n"
        f"- Collection: `{data.get('collection', '?')}`\n\n"
        "You can now ask questions in the chat panel ➜"
    )


def ask_question(
    message: str,
    history: list[dict],
    session_id: str,
) -> Generator[tuple[list[dict], str], None, None]:
    """Send a question to the backend and stream the response into the chat.

    Simulates streaming by yielding partial updates while the full response
    is being assembled. Uses POST /ask (as per the spec).

    Args:
        message: The user's input text.
        history: Current Gradio chat history (list of role/content dicts).
        session_id: Opaque session identifier for conversation memory.

    Yields:
        Tuples of (updated_history, empty_input_string).
    """
    if not message.strip():
        yield history, ""
        return

    # Append user message immediately
    history = history + [{"role": "user", "content": message}]
    yield history, ""

    # Call backend
    result = _post_json("/chat", question=message, session_id=session_id)

    if "error" in result:
        answer = f"❌ {result['error']}"
    else:
        answer = result.get("answer", "(no answer returned)")

    # Stream the answer character-by-character for a live feel
    accumulated = ""
    history_with_reply = history + [{"role": "assistant", "content": ""}]
    for char in answer:
        accumulated += char
        history_with_reply[-1] = {"role": "assistant", "content": accumulated}
        yield history_with_reply, ""

    yield history_with_reply, ""


def reset_chat(session_id: str) -> tuple[list, str, str]:
    """Clear server-side memory and wipe the local chat history.

    Args:
        session_id: The current session ID to clear.

    Returns:
        Empty history, new session_id, and a status string.
    """
    _post_json("/reset", session_id=session_id)
    new_session = str(uuid.uuid4())
    return [], new_session, "🗑️ Conversation cleared. New session started."


# ── Gradio UI ─────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance ready to `launch()`.
    """
    with gr.Blocks(title="Smart Contract Assistant") as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 📄 Smart Contract Assistant
            Upload a contract (PDF or DOCX) on the left, then chat with it on the right.

            *Powered by Gemini · Ollama (nomic-embed-text) · ChromaDB · LangChain*
            """
        )

        # ── Shared state ───────────────────────────────────────────────────────
        session_id_state = gr.State(str(uuid.uuid4()))

        # ── Two-column layout ──────────────────────────────────────────────────
        with gr.Row():

            # ── LEFT PANEL ─────────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 📁 Upload Document")

                file_input = gr.File(
                    label="Select PDF or DOCX",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
                upload_btn = gr.Button("Upload & Ingest ⬆", variant="primary")
                upload_status = gr.Markdown(
                    value="*No document loaded yet.*", label="Status"
                )

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Session")
                reset_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")
                reset_status = gr.Markdown(value="")

            # ── RIGHT PANEL ────────────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat With Contract")

                chatbot = gr.Chatbot(
                    label="Contract Assistant",
                    height=500,
                    show_label=False,
                                        
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask anything about the contract…",
                        label="Your question",
                        lines=1,
                        scale=8,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1)

        # ── Event bindings ─────────────────────────────────────────────────────

        upload_btn.click(
            fn=upload_document,
            inputs=[file_input],
            outputs=[upload_status],
        )

        send_btn.click(
            fn=ask_question,
            inputs=[chat_input, chatbot, session_id_state],
            outputs=[chatbot, chat_input],
        )

        chat_input.submit(
            fn=ask_question,
            inputs=[chat_input, chatbot, session_id_state],
            outputs=[chatbot, chat_input],
        )

        reset_btn.click(
            fn=reset_chat,
            inputs=[session_id_state],
            outputs=[chatbot, session_id_state, reset_status],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        show_error=True,
        share=False,
    )
