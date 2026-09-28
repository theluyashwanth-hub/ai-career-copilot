"""Career Chat page (Phase 10).

Thin Streamlit wrapper around the app.rag layer (chunking,
embeddings, FAISS vector store, retrieval, grounded Q&A).
No FAISS or LLM logic lives here; the index lives in
session_state (in-memory, safe for read-only filesystems).
"""

import streamlit as st

from app.core.config import settings
from app.rag.chunker import chunk_resume
from app.rag.embeddings import get_embeddings_model
from app.rag.retriever import answer_resume_question
from app.rag.vector_store import VectorStore, build_store
from app.services.resume_parser import ResumeDocument, parse_resume

CHAT_HISTORY_KEY = "chat_history"
RAG_STORE_KEY = "rag_store"
RAG_STORE_ID_KEY = "rag_store_id"


def _ensure_resume() -> ResumeDocument | None:
    """Return the session resume, offering an inline upload fallback."""
    doc = st.session_state.get("resume_document")
    if isinstance(doc, ResumeDocument):
        return doc
    st.info("Upload a resume to chat with it. You can also upload on the Resume page.")
    uploaded = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        help="Supported formats: PDF (.pdf), Word (.docx).",
        key="chat_uploader",
    )
    if uploaded is None:
        return None
    try:
        doc = parse_resume(uploaded.getvalue(), uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        return None
    st.session_state["resume_file_id"] = f"{uploaded.name}:{uploaded.size}"
    st.session_state["resume_document"] = doc
    st.session_state["resume_error"] = None
    return doc


def _ensure_store(doc: ResumeDocument) -> VectorStore | None:
    """Build (once per resume) the in-memory FAISS index."""
    file_id = st.session_state.get("resume_file_id", doc.filename)
    store = st.session_state.get(RAG_STORE_KEY)
    if isinstance(store, VectorStore) and st.session_state.get(RAG_STORE_ID_KEY) == file_id:
        return store
    if not settings.openrouter_api_key:
        st.error("OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example.")
        return None
    with st.spinner("Indexing your resume for chat..."):
        try:
            chunks = chunk_resume(doc.text, doc.filename)
            fresh = build_store(chunks, get_embeddings_model())
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
            return None
        except Exception as exc:  # defensive: never crash the page
            st.error(f"Could not index resume: {exc}")
            return None
    st.session_state[RAG_STORE_KEY] = fresh
    st.session_state[RAG_STORE_ID_KEY] = file_id
    st.session_state.pop(CHAT_HISTORY_KEY, None)
    return fresh


def _render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    with st.chat_message(role):
        st.write(content)
        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(f"**{src['section']}** (`{src['chunk_id']}`)")
                    st.caption(src["excerpt"])


def render_chat_page() -> None:
    """Render the Career Chat page."""
    st.header("Career Chat")
    st.write("Ask questions about your uploaded resume. Answers cite their sources.")

    doc = _ensure_resume()
    if doc is None:
        return
    store = _ensure_store(doc)
    if store is None:
        return
    st.caption(f"Chatting with: {doc.filename} ({len(store)} chunks indexed)")

    history = st.session_state.setdefault(CHAT_HISTORY_KEY, [])
    for message in history:
        _render_message(message["role"], message["content"], message.get("sources"))

    prompt = st.chat_input("e.g. What projects did I build using Redis?")
    if not prompt:
        return
    history.append({"role": "user", "content": prompt})
    _render_message("user", prompt)
    with st.spinner("Searching your resume..."):
        try:
            result = answer_resume_question(prompt, store)
        except (ValueError, RuntimeError) as exc:
            history.append({"role": "assistant", "content": f"Sorry — {exc}", "sources": []})
            st.rerun()
        except Exception as exc:  # defensive: never crash the page
            history.append({"role": "assistant", "content": f"Sorry — chat failed: {exc}", "sources": []})
            st.rerun()
    sources = [s.model_dump() for s in result.sources]
    history.append({"role": "assistant", "content": result.answer, "sources": sources})
    st.rerun()
