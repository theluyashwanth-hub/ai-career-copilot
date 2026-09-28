"""Resume RAG layer (Phase 10).

Chunking, OpenRouter embeddings, FAISS vector store, and retrieval
live here. Callers must use these abstractions and never touch
FAISS directly, so the backend can later be swapped (Qdrant,
pgvector) without touching the rest of the app.
"""
