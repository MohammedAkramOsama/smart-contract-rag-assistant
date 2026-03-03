"""
run_ui.py

Entry point to start the Gradio frontend UI.

Usage:
    python run_ui.py

The FastAPI backend must already be running (python run_server.py).
BACKEND_URL can be set via environment variable (default: http://127.0.0.1:8000).
"""

import os

from frontend.gradio_app import build_ui

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        show_error=True,
        share=False,
    )
