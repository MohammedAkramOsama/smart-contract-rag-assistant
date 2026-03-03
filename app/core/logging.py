"""
app/core/logging.py

Centralised logging configuration using Loguru.
Call setup_logging() once at application start-up.
"""

import sys
from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """Configure Loguru for the entire application.

    Args:
        log_level: Minimum log level to emit (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "logs/app.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        serialize=False,
    )
    logger.info("Logging initialised at level %s", log_level)


# Re-export the logger so other modules can do: from app.core.logging import log
log = logger
