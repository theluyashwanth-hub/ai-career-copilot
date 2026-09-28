"""Section-aware resume chunking (Phase 10).

Splits resume text along standard section headings, then word-window
chunks long sections with overlap. Every chunk keeps provenance
metadata: section, source, chunk_id.
"""

from app.schemas.rag import ResumeChunk
from app.services.ats_analyzer import SECTION_ALIASES

_ALIAS_TO_SECTION = {
    alias: name for name, aliases in SECTION_ALIASES.items() for alias in aliases
}

GENERAL_SECTION = "general"


def _split_by_section(text: str) -> list[tuple[str, str]]:
    """Group resume text into (section, body) pairs in document order."""
    groups: list[tuple[str, list[str]]] = []
    current: str | None = None
    for raw_line in text.split("\n"):
        heading = raw_line.strip().rstrip(":").strip()
        if heading.lower() in _ALIAS_TO_SECTION and len(heading) <= 30:
            current = _ALIAS_TO_SECTION[heading.lower()]
            groups.append((current, []))
        elif current is not None:
            groups[-1][1].append(raw_line)
        else:
            if not groups or groups[-1][0] != GENERAL_SECTION:
                groups.append((GENERAL_SECTION, []))
            groups[-1][1].append(raw_line)
    return [(section, "\n".join(lines).strip()) for section, lines in groups]


def _window_chunks(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """Split words into overlapping windows (last window may be shorter)."""
    if len(words) <= max_words:
        return [words]
    step = max(1, max_words - overlap)
    windows: list[list[str]] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            windows.append(window)
        if start + max_words >= len(words):
            break
    return windows


def chunk_resume(
    resume_text: str,
    source: str,
    *,
    max_words: int = 200,
    overlap: int = 40,
) -> list[ResumeChunk]:
    """Chunk resume text into section-aware overlapping windows.

    Args:
        resume_text: Extracted plain text (ResumeDocument.text).
        source: Provenance label, e.g. the uploaded filename.
        max_words: Maximum words per chunk.
        overlap: Words shared between consecutive windows.

    Raises:
        ValueError: Empty text/source, or no extractable content.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume before building the index.")
    if not source or not source.strip():
        raise ValueError("Chunk source is empty.")
    if max_words < 20:
        raise ValueError("max_words must be at least 20.")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must satisfy 0 <= overlap < max_words.")

    chunks: list[ResumeChunk] = []
    for section, body in _split_by_section(resume_text.strip()):
        words = body.split()
        if not words:
            continue
        for index, window in enumerate(_window_chunks(words, max_words, overlap)):
            chunks.append(
                ResumeChunk(
                    text=" ".join(window),
                    section=section,
                    source=source.strip(),
                    chunk_id=f"{source.strip()}#{section}:{index}",
                )
            )
    if not chunks:
        raise ValueError("No chunkable content found in the resume text.")
    return chunks
