"""OpenRouter embeddings via LangChain (OpenAI-compatible).

All embedding-client construction lives here. Tests inject fake
embedding objects instead; the vector store only relies on the
embed_documents / embed_query interface.
"""

from app.ai.openrouter_client import DEFAULT_EMBEDDING_MODEL, get_embeddings_model

__all__ = ["DEFAULT_EMBEDDING_MODEL", "get_embeddings_model"]
