"""FAISS vector store abstraction (Phase 10).

The ONLY module allowed to touch FAISS. Everything else uses
VectorStore (add_documents / search / save / load) plus
build_store, so the backend can later be replaced (Qdrant,
pgvector) without touching callers.

Persistence is isolated to save()/load(): the index plus a JSON
sidecar (texts, metadata, dimension). In-memory use is the
default; callers must not assume the filesystem is writable.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from app.schemas.rag import ResumeChunk, RetrievedChunk

_INDEX_FILENAME = "index.faiss"
_DOCS_FILENAME = "store.json"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class VectorStore:
    """Backend-agnostic vector store over a FAISS inner-product index."""

    def __init__(self, embeddings, dimension: int | None = None) -> None:
        self._embeddings = embeddings
        self._dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self._texts)

    @property
    def dimension(self) -> int | None:
        """Embedding dimension, known after the first add_documents()."""
        return self._dimension

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            self._dimension = dimension
            self._index = faiss.IndexFlatIP(dimension)
        elif dimension != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: index is {self._dimension}, got {dimension}."
            )

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Embed and index texts; returns positional document ids."""
        clean = [t for t in (t.strip() for t in texts) if t]
        if not clean:
            raise ValueError("No documents to add.")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length.")
        vectors = _normalize(np.asarray(self._embeddings.embed_documents(clean), dtype=np.float32))
        self._ensure_index(vectors.shape[1])
        metas = metadatas if metadatas is not None else [{} for _ in clean]
        start = len(self._texts)
        self._index.add(vectors)
        self._texts.extend(clean)
        self._metadatas.extend(metas)
        return [str(start + i) for i in range(len(clean))]

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return up to k chunks ranked by cosine similarity (may be empty)."""
        if not query or not query.strip():
            raise ValueError("Query is empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        if self._index is None or len(self._texts) == 0:
            return []
        vector = _normalize(
            np.asarray([self._embeddings.embed_query(query.strip())], dtype=np.float32)
        )
        k = min(k, len(self._texts))
        scores, indices = self._index.search(vector, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            meta = self._metadatas[idx]
            results.append(
                RetrievedChunk(
                    text=self._texts[idx],
                    section=str(meta.get("section", "general")),
                    source=str(meta.get("source", "")) or "resume",
                    chunk_id=str(meta.get("chunk_id", f"chunk:{idx}")),
                    score=float(max(-1.0, min(1.0, score))),
                )
            )
        return results

    def save(self, directory: str | Path) -> Path:
        """Persist index + documents sidecar into directory."""
        if self._index is None:
            raise ValueError("Nothing to save: the store is empty.")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILENAME))
        (path / _DOCS_FILENAME).write_text(
            json.dumps(
                {
                    "dimension": self._dimension,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path, embeddings) -> "VectorStore":
        """Load a store previously written by save()."""
        path = Path(directory)
        index_file, docs_file = path / _INDEX_FILENAME, path / _DOCS_FILENAME
        if not index_file.exists() or not docs_file.exists():
            raise FileNotFoundError(f"No saved vector store found in '{path}'.")
        payload = json.loads(docs_file.read_text(encoding="utf-8"))
        store = cls(embeddings, dimension=payload["dimension"])
        store._index = faiss.read_index(str(index_file))
        store._texts = payload["texts"]
        store._metadatas = payload["metadatas"]
        return store


def build_store(chunks: list[ResumeChunk], embeddings) -> VectorStore:
    """Chunk list -> populated VectorStore (metadata preserved)."""
    if not chunks:
        raise ValueError("No chunks to index.")
    store = VectorStore(embeddings)
    store.add_documents(
        [c.text for c in chunks],
        [{"section": c.section, "source": c.source, "chunk_id": c.chunk_id} for c in chunks],
    )
    return store
