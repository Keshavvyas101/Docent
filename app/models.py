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

