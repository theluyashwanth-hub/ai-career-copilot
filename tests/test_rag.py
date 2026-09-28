"""Tests for Phase 10 resume RAG (embeddings/model calls mocked)."""

import numpy as np
import pytest

from app.rag.chunker import chunk_resume
from app.rag.retriever import (
    NOT_AVAILABLE_ANSWER,
    answer_resume_question,
    retrieve,
)
from app.rag.vector_store import VectorStore, build_store
from app.schemas.rag import RAGAnswer, ResumeChunk

VOCAB = ["redis", "python", "internship", "sql", "docker", "testing"]


class FakeEmbeddings:
    """Deterministic keyword-presence embeddings (no network)."""

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if word in lowered else 0.0 for word in VOCAB]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChatModel:
    """Mock chat model returning fixed text; records prompts/calls."""

    def __init__(self, reply: str = "Mocked answer.", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return _FakeMessage(self.reply)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


SAMPLE_RESUME = """Jane Doe
jane@example.com

Summary:
Backend engineer with Python experience.

Experience:
Software Engineering Intern at Acme Corp
- Built caching layer using Redis for session storage
- Wrote SQL queries and Docker test setups

Projects:
Cache Dashboard - Python and Redis monitoring tool

Skills:
Python, SQL, Docker, Redis, Testing
"""


def _build_sample_store() -> VectorStore:
    chunks = chunk_resume(SAMPLE_RESUME, "resume.pdf")
    return build_store(chunks, FakeEmbeddings())


# --- Chunking ---


def test_chunking_splits_sections_with_metadata():
    chunks = chunk_resume(SAMPLE_RESUME, "resume.pdf")
    sections = {c.section for c in chunks}
    assert {"summary", "experience", "projects", "skills"} <= sections
    for chunk in chunks:
        assert chunk.source == "resume.pdf"
        assert chunk.chunk_id.startswith("resume.pdf#")
        assert chunk.text.strip()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunking_long_section_windows_with_overlap():
    body = "Experience:\n" + " ".join(f"word{i}" for i in range(500))
    chunks = chunk_resume(body, "long.pdf", max_words=100, overlap=20)
    assert len(chunks) > 1
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    assert all(len(c.text.split()) <= 100 for c in chunks)


def test_chunking_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        chunk_resume("   ", "resume.pdf")


def test_chunking_empty_source_rejected():
    with pytest.raises(ValueError, match="source"):
        chunk_resume(SAMPLE_RESUME, "  ")


# --- Vector store ---


def test_store_search_ranks_relevant_chunk_first():
    store = _build_sample_store()
    results = store.search("What projects did I build using Redis?", k=3)
    assert results
    top = results[0]
    assert "redis" in top.text.lower()
    assert top.chunk_id
    assert top.section
    assert -1.0 <= top.score <= 1.0


def test_store_search_empty_store_returns_empty():
    store = VectorStore(FakeEmbeddings())
    assert store.search("anything") == []
    assert len(store) == 0


def test_store_search_empty_query_rejected():
    store = _build_sample_store()
    with pytest.raises(ValueError, match="empty"):
        store.search("   ")


def test_store_add_empty_rejected():
    store = VectorStore(FakeEmbeddings())
    with pytest.raises(ValueError, match="No documents"):
        store.add_documents([])


def test_store_metadata_preserved_through_search():
    chunks = [
        ResumeChunk(text="Redis caching work", section="projects", source="r.pdf", chunk_id="r.pdf#projects:0"),
        ResumeChunk(text="Unrelated cooking hobby", section="general", source="r.pdf", chunk_id="r.pdf#general:0"),
    ]
    store = build_store(chunks, FakeEmbeddings())
    top = store.search("tell me about redis", k=1)[0]
    assert top.section == "projects"
    assert top.chunk_id == "r.pdf#projects:0"


def test_store_save_load_roundtrip(tmp_path):
    store = _build_sample_store()
    saved = store.save(tmp_path / "rag-index")
    assert saved.exists()
    loaded = VectorStore.load(tmp_path / "rag-index", FakeEmbeddings())
    assert len(loaded) == len(store)
    assert loaded.dimension == store.dimension
    top = loaded.search("redis projects", k=1)[0]
    assert "redis" in top.text.lower()
    assert top.section


def test_store_load_missing_directory_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved vector store"):
        VectorStore.load(tmp_path / "does-not-exist", FakeEmbeddings())


# --- Retrieval + Q&A ---


def test_retrieve_filters_below_threshold():
    store = _build_sample_store()
    results = retrieve(store, "redis", k=4, score_threshold=0.0)
    assert results
    strict = retrieve(store, "redis", k=4, score_threshold=0.99)
    assert all(r.score >= 0.99 for r in strict)


def test_answer_grounded_with_sources():
    store = _build_sample_store()
    chat = FakeChatModel(reply="You built a Redis caching layer.")
    result = answer_resume_question("What did I build using Redis?", store, chat_model=chat)
    assert isinstance(result, RAGAnswer)
    assert result.answer == "You built a Redis caching layer."
    assert result.sources, "expected source attribution"
    assert all(s.chunk_id and s.section and s.excerpt for s in result.sources)
    assert "redis" in chat.prompts[0].lower()


def test_no_relevant_results_without_llm_call():
    store = _build_sample_store()
    chat = FakeChatModel(reply="Should never be used.")
    result = answer_resume_question(
        "xylophone quantum banana", store, chat_model=chat, score_threshold=0.99
    )
    assert result.answer == NOT_AVAILABLE_ANSWER
    assert result.sources == []
    assert chat.calls == 0


def test_answer_empty_query_rejected():
    store = _build_sample_store()
    with pytest.raises(ValueError, match="empty"):
        answer_resume_question("  ", store, chat_model=FakeChatModel())


def test_answer_empty_store_rejected():
    with pytest.raises(ValueError, match="index is empty"):
        answer_resume_question("redis?", VectorStore(FakeEmbeddings()), chat_model=FakeChatModel())


def test_answer_model_failure_wrapped():
    store = _build_sample_store()
    chat = FakeChatModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="Resume Q&A failed"):
        answer_resume_question("redis?", store, chat_model=chat)


def test_answer_missing_api_key_rejected():
    store = _build_sample_store()
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        answer_resume_question("redis?", store, api_key="")
