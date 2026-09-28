"""RAG schemas (Phase 10).

ResumeChunk carries section-aware text with provenance metadata.
RAGAnswer pairs a grounded answer with its source chunks.
"""

from pydantic import BaseModel, Field


class ResumeChunk(BaseModel):
    """A single searchable resume chunk."""

    text: str = Field(min_length=1)
    section: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with a cosine similarity score."""

    text: str = Field(min_length=1)
    section: str = Field(min_length=1)
    source: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)


class RAGSource(BaseModel):
    """Source attribution for a generated answer."""

    section: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class RAGAnswer(BaseModel):
    """Grounded answer with source attribution."""

    answer: str = Field(min_length=1)
    sources: list[RAGSource] = Field(default_factory=list)
