"""
app/utils/file_parsers.py

Document parsing utilities for PDF (PyMuPDF) and DOCX (python-docx).
Returns raw extracted text as a single string.
"""

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

from app.core.logging import log


def parse_pdf(path: Path) -> str:
    """Extract all text from a PDF file.

    Args:
        path: Absolute path to the PDF file.

    Returns:
        Concatenated plain-text content of all pages.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    log.info("Parsing PDF: %s", path.name)
    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text}")
    full_text = "\n\n".join(pages)
    log.debug("PDF parsed: %d pages, %d chars", len(pages), len(full_text))
    return full_text


def parse_docx(path: Path) -> str:
    """Extract all text from a DOCX file.

    Args:
        path: Absolute path to the DOCX file.

    Returns:
        Concatenated plain-text content of all paragraphs.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")

    log.info("Parsing DOCX: %s", path.name)
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    log.debug("DOCX parsed: %d paragraphs, %d chars", len(paragraphs), len(full_text))
    return full_text


def parse_document(path: Path) -> str:
    """Dispatch to the correct parser based on file extension.

    Args:
        path: Absolute path to PDF or DOCX file.

    Returns:
        Extracted and lightly cleaned plain text.

    Raises:
        ValueError: If the file type is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        raw = parse_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Expected .pdf or .docx.")

    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph boundaries.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    import re

    # Collapse 3+ consecutive newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
