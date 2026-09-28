"""Resume RAG retrieval + grounded Q&A (Phase 10).

Retrieves relevant resume chunks from a VectorStore and asks
OpenRouter to answer ONLY from those chunks, with source attribution
built deterministically from the retrieved chunks. The LLM never
sees the full resume and must not invent information.
"""

from typing import Any

from app.ai.openrouter_client import DEFAULT_OPENROUTER_MODEL, get_chat_model
from app.rag.vector_store import VectorStore
from app.schemas.rag import RAGAnswer, RAGSource, RetrievedChunk

DEFAULT_TOP_K = 4
DEFAULT_SCORE_THRESHOLD = 0.2
EXCERPT_CHARS = 240

NOT_AVAILABLE_ANSWER = "That information is not available in the uploaded resume."

RAG_QA_INSTRUCTIONS = """Answer the question below using ONLY the resume excerpts provided. Rules:
- Ground every claim in the excerpts. NEVER invent facts, numbers, skills, or experience.
- If the excerpts do not contain the answer, reply with exactly: {not_available}
- Keep the answer concise (a few sentences plus short bullets where helpful).
- Mention the section names your answer comes from.

Resume excerpts:
---
{context}
---

Question: {query}
"""


def retrieve(
    store: VectorStore,
    query: str,
    k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[RetrievedChunk]:
    """Return chunks scoring at or above the similarity threshold."""
    results = store.search(query, k=k)
    return [r for r in results if r.score >= score_threshold]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.section} | {c.chunk_id}]\n{c.text}" for c in chunks)


def _to_sources(chunks: list[RetrievedChunk]) -> list[RAGSource]:
    return [
        RAGSource(
            section=c.section,
            chunk_id=c.chunk_id,
            excerpt=c.text[:EXCERPT_CHARS] + ("..." if len(c.text) > EXCERPT_CHARS else ""),
        )
        for c in chunks
    ]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def answer_resume_question(
    query: str,
    store: VectorStore,
    *,
    chat_model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
    k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> RAGAnswer:
    """Retrieve resume chunks and generate an attributed answer.

    Returns a not-available answer WITHOUT calling the LLM when no
    chunk passes the similarity threshold.

    Raises:
        ValueError: Empty query, missing API key.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not query or not query.strip():
        raise ValueError("Query is empty. Ask a question about the resume.")
    if store is None or len(store) == 0:
        raise ValueError("Resume index is empty. Upload a resume to build it first.")

    chunks = retrieve(store, query.strip(), k=k, score_threshold=score_threshold)
    if not chunks:
        return RAGAnswer(answer=NOT_AVAILABLE_ANSWER, sources=[])

    model = chat_model if chat_model is not None else get_chat_model(
        api_key=api_key, model_name=model_name
    )
    prompt = RAG_QA_INSTRUCTIONS.format(
        not_available=NOT_AVAILABLE_ANSWER,
        context=_format_context(chunks),
        query=query.strip(),
    )
    try:
        raw = model.invoke(prompt)
        text = _content_to_text(raw.content if hasattr(raw, "content") else raw).strip()
    except Exception as exc:
        raise RuntimeError(f"Resume Q&A failed: {exc}") from exc
    if not text:
        raise RuntimeError("Resume Q&A failed: empty response from the model.")
    return RAGAnswer(answer=text, sources=_to_sources(chunks))
