"""Pydantic request/response models for the Docent API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")


class Citation(BaseModel):
    source: str = Field(..., description="Source filename from authoritative metadata")
    chunk_id: str = Field(..., description="Retrieved chunk identifier")
    quote: str = Field(..., description="Direct quote or excerpt from the cited chunk")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    citations: list[Citation] = Field(
        default_factory=list, description="Validated citations corresponding to retrieved chunks"
    )
    grounded: bool = Field(
        ..., description="Whether the answer is grounded in documents"
    )


# ── Document management models ────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    """Metadata for a single indexed document returned by GET /documents."""

    filename: str = Field(..., description="Source filename as stored in Qdrant payload")
    chunk_count: int = Field(..., description="Number of indexed chunks for this document")
    doc_hash: str = Field(..., description="SHA-256 hash of the document at ingestion time")


class UploadResponse(BaseModel):
    """Response returned by POST /documents."""

    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(
        ...,
        description="'ingested' | 'updated' | 'unchanged'",
    )
    chunks_indexed: int = Field(
        ..., description="Total number of Qdrant chunks now belonging to this file"
    )
    message: str = Field(..., description="Human-readable result message")


class DeleteResponse(BaseModel):
    """Response returned by DELETE /documents/{name}."""

    filename: str = Field(..., description="Name of the deleted file")
    chunks_deleted: int = Field(..., description="Number of Qdrant chunks removed")
    file_deleted: bool = Field(..., description="Whether the source file was removed from disk")
    message: str = Field(..., description="Human-readable result message")

