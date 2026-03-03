"""
app/api/routes_upload.py

Upload endpoint: POST /upload
Accepts a PDF or DOCX file, saves it to disk, and runs the ingestion pipeline.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import log
from app.pipelines.ingestion import ingest_document

router = APIRouter(tags=["Upload"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


class UploadResponse(BaseModel):
    """Schema for the upload/ingestion response."""

    message: str
    filename: str
    chunks: int
    collection: str


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a contract document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or DOCX contract file"),
) -> UploadResponse:
    """Accept a contract file, persist it, and run the ingestion pipeline.

    Args:
        file: Multipart file upload (PDF or DOCX).

    Returns:
        UploadResponse with chunk count and storage details.

    Raises:
        HTTPException 400: Unsupported file type.
        HTTPException 422: Extraction produced no text.
        HTTPException 500: Internal pipeline failure.
    """
    settings = get_settings()
    original_name = Path(file.filename or "upload")
    suffix = original_name.suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save with a unique name to avoid collisions
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = settings.upload_dir / unique_name

    try:
        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        log.info("File saved: %s → %s", original_name.name, dest_path)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {exc}",
        ) from exc

    try:
        result = ingest_document(dest_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected ingestion failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return UploadResponse(
        message="Document ingested successfully.",
        filename=original_name.name,
        chunks=int(result["chunks"]),
        collection=str(result["collection"]),
    )
