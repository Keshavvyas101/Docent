"""Pydantic request/response models for the Docent API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")


class Citation(BaseModel):
    source: str = Field(..., description="Source filename")
    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Excerpt from the source chunk")
    relevance_score: float = Field(..., description="Cosine similarity score")


class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    citations: list[Citation] = Field(
        default_factory=list, description="Source chunks used"
    )
    grounded: bool = Field(
        ..., description="Whether the answer is grounded in documents"
    )
