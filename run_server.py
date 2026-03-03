"""
run_server.py

Entry point to start the FastAPI backend server.

Usage:
    python run_server.py
"""

import uvicorn
from app.core.config import get_settings
from app.core.logging import setup_logging

if __name__ == "__main__":
    setup_logging()
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
