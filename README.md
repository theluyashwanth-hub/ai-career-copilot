# AI Career Copilot 💼

> Your AI-powered career assistant — parse resumes, score ATS compatibility, match jobs, optimize bullets, analyze SWOT, practice interviews, chat with your resume, and orchestrate everything with a copilot agent.

Built with **Python, Streamlit, LangChain, LangGraph, FAISS, PostgreSQL, and OpenRouter LLMs**.

---

## Table of Contents

- [1. What Is This?](#1-what-is-this)
- [2. Who Is It For?](#2-who-is-it-for)
- [3. Features at a Glance](#3-features-at-a-glance)
- [4. How It Works (Architecture)](#4-how-it-works-architecture)
- [5. Tech Stack](#5-tech-stack)
- [6. Project Structure](#6-project-structure)
- [7. Prerequisites](#7-prerequisites)
- [8. Quickstart (5 Minutes)](#8-quickstart-5-minutes)
- [9. Environment Variables](#9-environment-variables)
- [10. Running the App](#10-running-the-app)
- [11. Usage Guide (Page by Page)](#11-usage-guide-page-by-page)
- [12. Core Concepts](#12-core-concepts)
- [13. Database & Persistence (Phase 12)](#13-database--persistence-phase-12)
- [14. Testing](#14-testing)
- [15. Configuration & Customization](#15-configuration--customization)
- [16. Troubleshooting & FAQ](#16-troubleshooting--faq)
- [17. Limitations & Roadmap](#17-limitations--roadmap)
- [18. Contributing](#18-contributing)
- [19. License & Acknowledgements](#19-license--acknowledgements)

---

## 1. What Is This?

**AI Career Copilot** is a full-stack, multi-page Streamlit web app that helps job seekers understand and improve their resumes.

You upload a resume once (`PDF` / `DOCX`), and the app can:

1. Extract and structure it into a clean profile
2. Score it like an Applicant Tracking System (ATS)
3. Compare it against any job description
4. Rewrite weak bullets, find SWOT insights, and generate interview questions
5. Let you **chat with your resume** using RAG
6. Orchestrate all of the above through a **LangGraph copilot agent**

Key design principle: **never hallucinate**. Missing info is shown as `null` / empty, numbers are never invented, and every chat answer cites its source chunk.

---

## 2. Who Is It For?

- **Job seekers** — check ATS score, tailor resume to a job, prepare for interviews
- **Students / beginners** — learn what recruiters and ATS systems look for
- **Developers** — example of clean layered Python: `UI → Services → AI/RAG/Agent → Schemas → DB`
- **AI learners** — practical example of structured LLM output, RAG with FAISS, and controlled LangGraph agents (no autonomous tools)

No ML background needed to *use* it. Python basics needed to *extend* it.

---

## 3. Features at a Glance

| # | Page (Sidebar) | What You Give | What You Get |
|---|----------------|---------------|--------------|
| 1 | **Copilot** (landing) | Natural question, e.g. "How well do I match this job?" | Routed answer with scores, gaps, recommendations |
| 2 | **Resume** | `.pdf` / `.docx` upload | Extracted text + metadata + structured AI profile (8 sections) |
| 3 | **ATS Analysis** | Resume (reused or fresh upload) | Overall 0-100 + 6 categories, strengths, weaknesses, missing keywords, fixes |
| 4 | **Job Description** | Pasted job posting text | Structured job: title, company, required/preferred skills, responsibilities, keywords |
| 5 | **Job Match** | Analyzed resume + analyzed job | Match score, matched/missing skills, experience gaps, recommendations |
| 6 | **Resume Optimizer** | 1 bullet (+ optional JD) | Improved bullet + explanation + missing metrics + action verbs + X-Y-Z breakdown |
| 7 | **SWOT Analysis** | Resume profile (ATS + Match optional) | Strengths, Weaknesses, Opportunities, Threats — each with `(evidence)` |
| 8 | **Interview Coach** | Resume (+ job for role-specific Qs) | 8 questions → answer → feedback → next → final summary |
| 9 | **Career Chat** | Question about your resume | Grounded answer + sources (section, chunk id, excerpt) |

All AI pages require `OPENROUTER_API_KEY`. Without it the UI shows a clear error and blocks the call.

---

## 4. How It Works (Architecture)

```text
                    ┌──────────────┐
                    │  Streamlit   │
                    │  app/ui/*.py │  thin: session_state + display only
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     app/services/   app/agent/    app/rag/
     deterministic   LangGraph     FAISS RAG
     rules + blend   state machine chunk → embed → retrieve
              │            │            │
              └────────────┼────────────┘
                           ▼
                      app/ai/
              OpenRouter LLMs via LangChain
              (one client factory only)
                           ▼
                    app/schemas/
              Pydantic models (validation)
                           ▼
                       app/db/
              PostgreSQL + SQLAlchemy + Alembic
              (only place that imports SQLAlchemy)
```

**Layer rules (enforced by convention):**

- `app/ui` never calls LLMs or SQL directly — it calls services / agent / RAG.
- `app/services` holds deterministic logic (regex, overlap, scoring) and *blending*, never builds LLM clients.
- `app/ai` holds all LLM prompts. Only `app/ai/openrouter_client.py` builds `ChatOpenAI` / `OpenAIEmbeddings`.
- `app/schemas` holds Pydantic models only, no logic.
- `app/db` is the only place importing SQLAlchemy. Everything else uses repository classes.
- `app/rag` owns FAISS. Only chunk *metadata* is stored in Postgres — the index is rebuilt per resume.

**Copilot agent flow (Phase 11):**

```text
intent → resume → job → [match?] → [ats?] → recommend → respond → END
```

No autonomous tools, no filesystem / shell execution. It just routes and reuses the existing ATS / match services.

---

## 5. Tech Stack

| Concern | Choice | Version / Notes |
|---------|--------|-----------------|
| Language | Python | `>=3.11` |
| UI | Streamlit | `>=1.37` (`file_uploader`, `chat_input`, `session_state`) |
| LLM provider | OpenRouter (OpenAI-compatible) via LangChain | `langchain>=0.3`, `langchain-openai>=0.3`, `openai>=1.0`, default model `openrouter/free` |
| Orchestration | LangGraph | `>=1.2` (controlled graph only) |
| Embeddings | OpenRouter `openai/text-embedding-3-small` | via `OpenAIEmbeddings` |
| Vector DB | FAISS-CPU | `>=1.8` (`IndexFlatIP` + cosine, in-memory + `index.faiss`/`store.json`) |
| Validation | Pydantic + pydantic-settings | `>=2.0` |
| Resume parsing | PyMuPDF (`fitz`), python-docx | PDF incl. tables |
| Database | PostgreSQL via SQLAlchemy 2.0 + psycopg + Alembic | `sqlalchemy>=2.0`, `alembic>=1.13`, `psycopg[binary]>=3.1` |
| Package manager | uv | `uv sync`, `uv run` |
| Tests | pytest | `>=8.0`, 14 suites |
| Config | python-dotenv | `.env` file, never committed |

> Note: older docs mention `GEMINI_API_KEY`. That is now a **deprecated fallback**. Use `OPENROUTER_API_KEY`.

---

## 6. Project Structure

```text
ai-career-copilot/
├── app/
│   ├── main.py               # Streamlit entrypoint, routes 9 pages
│   ├── agent/                # Phase 11 — LangGraph copilot
│   │   ├── state.py          # CopilotState, Intent (analyze_resume/match_job/improve/prepare_role/general)
│   │   ├── tools.py          # classify_intent, run_ats/match, recommendations, response
│   │   ├── nodes.py          # intent/resume/job/match/ats/recommend/respond steps
│   │   └── graph.py          # build_graph(), run_copilot()
│   ├── ai/                   # All LLM calls (OpenRouter via LangChain)
│   │   ├── openrouter_client.py  # get_chat_model/get_embeddings_model/invoke_structured
│   │   ├── resume_analyzer.py    # text → ResumeProfile
│   │   ├── ats_advisor.py        # semantic ATS signals
│   │   ├── job_analyzer.py       # JD text → JobProfile
│   │   ├── match_advisor.py      # resume-vs-job semantics
│   │   ├── bullet_optimizer.py   # bullet rewrite (never invents numbers)
│   │   ├── swot_analyzer.py      # evidence-grounded SWOT
│   │   └── interview_coach.py    # question gen + answer feedback
│   ├── core/
│   │   └── config.py         # Settings from .env (OpenRouter, DB URLs)
│   ├── db/                   # Phase 12 — ONLY SQLAlchemy importer
│   │   ├── base.py           # Declarative Base
│   │   ├── models.py         # 10 tables: User, Resume, ResumeProfile, JobDescription, ...
│   │   ├── session.py        # engine/session helpers
│   │   └── repositories.py   # User/Resume/Job/Analysis/Interview/RagDocument repos
│   ├── rag/                  # Phase 10 — resume chat
│   │   ├── chunker.py        # section-aware overlapping windows
│   │   ├── embeddings.py     # re-export embeddings factory
│   │   ├── vector_store.py   # FAISS wrapper + save/load
│   │   └── retriever.py      # retrieve() + answer_resume_question()
│   ├── schemas/              # Pydantic models, no logic
│   │   ├── resume_profile.py, ats_analysis.py, job.py, job_match.py
│   │   ├── bullet.py, swot.py, interview.py, rag.py
│   ├── services/             # Deterministic + blend logic
│   │   ├── resume_parser.py  # PDF/DOCX → text + metadata
│   │   ├── ats_analyzer.py   # rules + combine_ats_analysis (0.6-0.7 deterministic weight)
│   │   ├── job_matcher.py    # overlap + combine_job_match (55% skills + 15% kw + 30% semantic)
│   │   └── interview_coach.py# session state transitions
│   ├── ui/                   # Streamlit pages (thin wrappers)
│   │   ├── components.py, copilot_page.py, resume_page.py, profile_view.py
│   │   ├── ats_page.py, job_page.py, match_page.py, optimizer_page.py
│   │   ├── swot_page.py, interview_page.py, chat_page.py
│   └── utils/                # placeholder
├── tests/                    # 14 pytest files (mock-model injection, no network)
├── alembic/                  # env.py + versions/43461288f92c_phase_12_initial_schema.py
├── alembic.ini
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

---

## 7. Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **PostgreSQL** (only for Phase 12 persistence; app works without it for other phases)
- **OpenRouter API key** — get one at https://openrouter.ai (free tier works; default model is `openrouter/free`)

---

## 8. Quickstart (5 Minutes)

```bash
# 1. Install dependencies (creates .venv/)
uv sync

# 2. Copy env template
# Windows PowerShell:
Copy-Item .env.example .env
# macOS / Linux:
cp .env.example .env
```

Edit `.env` and set at minimum:

```text
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Then:

```bash
# 3. Run the app
streamlit run app/main.py
# or: uv run streamlit run app/main.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The landing page is **Copilot**.

**With database (optional but recommended):**

```bash
createdb ai_career_copilot
# set DATABASE_URL in .env, then:
uv run alembic upgrade head
```

**Run tests:**

```bash
pytest
# or targeted:
uv run pytest tests/test_persistence.py
```

---

## 9. Environment Variables

| Variable | Required? | Default | Purpose |
|----------|-----------|---------|---------|
| `OPENROUTER_API_KEY` | **Yes** (for all AI) | `""` | Chat + embeddings auth |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | Override endpoint |
| `OPENROUTER_MODEL` | No | `openrouter/free` | Chat model name |
| `OPENROUTER_EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Embeddings model |
| `DATABASE_URL` | For Phase 12 | `postgresql+psycopg://postgres:postgres@localhost:5432/ai_career_copilot` | App DB |
| `TEST_DATABASE_URL` | No | `sqlite+pysqlite:///:memory:` | Tests (no server needed) |
| `GEMINI_API_KEY` | Deprecated | `""` | Legacy fallback if OpenRouter key empty |

Rules: never commit `.env` (gitignored). `Settings` in `app/core/config.py` loads from `.env` with `extra="ignore"`. The `.env.example` file ships with a dummy key — replace it.

---

## 10. Running the App

- Entry point: `app/main.py::main()` — `st.set_page_config` → header + sidebar → route to one of 9 `render_*_page()` functions.
- Sidebar order: `Copilot, Resume, ATS Analysis, Job Description, Job Match, Resume Optimizer, SWOT Analysis, Interview Coach, Career Chat`.
- Footer shows whether `OPENROUTER_API_KEY` is configured.
- State is kept in `st.session_state` (`resume_document`, `resume_profile`, `job_profile`, `rag_store`, `copilot_history`, etc.). Use **Clear uploaded resume** on the Resume page to reset.
- Cache-busting keys: `file_id = {filename}:{size}` for resume, `match_key = {file_id}::{job_hash}` and `swot_key` so re-analysis triggers only when inputs change.

---

## 11. Usage Guide (Page by Page)

### Copilot (landing, Phase 11)

1. Analyze a resume on **Resume** page first; optionally analyze a job on **Job Description**.
2. Open **Copilot** and ask, e.g.:
   - "Analyze my resume."
   - "How well do I match this job?"
   - "What should I improve?"
   - "Prepare me for this role."
3. The LangGraph machine classifies intent, reuses ATS/match services as needed, and returns markdown with scores, gaps, recommendations. Errors become `Sorry — …`, never a crash.

### Resume (Phases 2–3)

1. Open **Resume** → drag `.pdf` or `.docx`.
2. See filename, type, size, word/char counts + extracted text in an expander.
3. Click **Analyze Resume** (needs API key) → structured profile in 8 sections: Personal Info, Summary, Education, Experience, Projects, Skills, Certifications, Achievements. Unknown = null/empty.

### ATS Analysis (Phase 4)

1. Open **ATS Analysis** (reuses uploaded resume or upload inline).
2. Click **Analyze ATS**.
3. Review overall score + 6 categories (formatting, keywords, experience, skills, projects, education), strengths, weaknesses, missing keywords, recommendations. Deterministic rules (contact, sections, measurable achievements, action verbs, `|`/emoji/formatting checks) are blended 60-70% with LLM semantics.

### Job Description (Phase 5)

1. Open **Job Description** → paste full posting → **Analyze Job**.
2. Review title, company, required/preferred skills, responsibilities, qualifications, experience, keywords. No matching happens here.

### Job Match (Phase 6)

1. Need both analyzed resume + analyzed job.
2. Open **Job Match** → **Run Match Analysis**.
3. Formula: `0.55*skills + 0.15*keywords + 0.30*semantic`. Skills use exact + normalized comparison (`js→javascript` etc.); transferable skills are filtered to what the resume actually contains.

### Resume Optimizer (Phase 7)

1. Open **Resume Optimizer** → enter one bullet + optional JD → **Optimize**.
2. Get improved bullet, explanation, `missing_metrics` (kinds of numbers worth adding — never invented), action verbs, X-Y-Z (Accomplishment, Measurement, Method).

### SWOT Analysis (Phase 8)

1. Need resume profile; ATS + Match optional for richer output.
2. Open **SWOT** → **Generate SWOT** → 4 quadrants, each bullet cites `(evidence)`. Opportunities/threats use job-match data only when available.

### Interview Coach (Phase 9)

1. Need resume profile (+ job for role-specific Qs).
2. Open **Interview Coach** → **Start Interview** (8 Qs: behavioral, technical, resume-specific, role-specific).
3. Answer one at a time → **Submit Answer** → feedback (strengths, weaknesses, missing points, clarity, relevance, improvement) → **Next** → **Finish** for summary. No medical/psychological claims.

### Career Chat (Phase 10)

1. Upload resume (needs key for embeddings) → auto `chunk_resume → embed → build FAISS store` (in-memory per `file_id`).
2. Ask e.g. "What projects did I build using Redis?" → grounded answer + **Sources** expander (section, chunk id, excerpt). Unknown → "not available in the uploaded resume", never invented. Save/load of index supported via `vector_store.py`.

---

## 12. Core Concepts

- **Deterministic-led blending:** ATS and Match never rely on LLM alone. Rules compute the base score; LLM adds nuance with a fixed weight. This keeps scores stable and explainable.
- **Anti-hallucination:** extraction prompts forbid invention; matcher filters semantic skills to the resume set; optimizer suggests *kinds* of metrics instead of inventing numbers; RAG returns `NOT_AVAILABLE` when retrieval threshold (default 0.2) fails.
- **Structured output:** all LLM calls go through `invoke_structured()` — tries tool-calling → `json_mode` → plain invoke + manual JSON extraction + Pydantic validation. Handles free-tier models that return plain text.
- **Section-aware RAG:** `chunker.py` splits by resume sections with overlapping word windows (`max_words=200, overlap=40`, `chunk_id={source}#{section}:{i}`); retrieval is cosine over L2-normalized FAISS `IndexFlatIP`, top-k=4.
- **Controlled agent:** `app/agent/graph.py` is a fixed state machine, not an autonomous loop. Intent keywords (`MATCH/JOB/IMPROVE/PREPARE/RESUME`) decide whether match/ATS nodes run.

---

## 13. Database & Persistence (Phase 12)

PostgreSQL via SQLAlchemy, versioned with Alembic. Stores users, resumes, profiles, jobs, analyses, interview sessions, and RAG chunk metadata.

**Tables (10):** `users`, `resumes` (`file_key` unique), `resume_profiles` (1-1), `job_descriptions`, `job_profiles` (1-1), `analyses` (`kind=ats/match/swot/bullet/copilot`, `result_json`, `score`), `interview_sessions`, `interview_questions`, `interview_answers`, `rag_documents` (chunk metadata only — FAISS index itself is rebuilt).

**Key rules:**

- Only `app/db` may import SQLAlchemy. UI/services/AI/RAG/agent use `UserRepository`, `ResumeRepository`, `JobRepository`, `AnalysisRepository`, `InterviewRepository`, `RagDocumentRepository`.
- No auth — `UserRepository.get_or_create_default()` (`local@ai-career-copilot`).

```bash
createdb ai_career_copilot
# set DATABASE_URL in .env
uv run alembic upgrade head        # single revision 43461288f92c
uv run pytest tests/test_persistence.py
```

Tests default to SQLite in-memory (`TEST_DATABASE_URL`), so no server is needed. Point `TEST_DATABASE_URL` at Postgres to test the real backend.

---

## 14. Testing

```bash
pytest                        # all suites (testpaths=["tests"])
uv run pytest tests/test_persistence.py
```

| File | Covers |
|------|--------|
| `test_config.py` | Settings / env loading |
| `test_resume_parser.py` | PDF/DOCX extraction |
| `test_resume_analyzer.py` | Profile extraction (mock LLM) |
| `test_ats_analyzer.py` | Deterministic rules + blend |
| `test_job_analyzer.py` | JD structuring |
| `test_job_matcher.py` | Overlap + blend, anti-hallucination filter |
| `test_bullet_optimizer.py` | Bullet rewrite |
| `test_swot_analyzer.py` | SWOT grounding |
| `test_interview_coach.py` | Q-gen, sessions, transitions |
| `test_rag.py` | Chunking, FAISS store, retrieval |
| `test_agent.py` | LangGraph routing + nodes |
| `test_persistence.py` | Repositories on SQLite / Postgres |
| `test_ui.py` | Page rendering smoke |

AI tests inject fake `model` objects (no network). Persistence tests use `TEST_DATABASE_URL`.

---

## 15. Configuration & Customization

- Change models in `.env`: `OPENROUTER_MODEL`, `OPENROUTER_EMBEDDING_MODEL`, `OPENROUTER_BASE_URL`.
- Reasoning: `openrouter_reasoning_enabled=True` by default adds `extra_body={"reasoning":{"enabled":True}}` so reasoning models return `reasoning_details`.
- RAG tuning: `chunk_resume(max_words, overlap)`, `retrieve(k=4, threshold=0.2)`, `generate_interview_questions(num_questions=8)` — edit call sites in `app/ui/chat_page.py` / `app/rag/*`.
- Scoring weights: `combine_ats_analysis()` in `app/services/ats_analyzer.py`, `combine_job_match()` in `app/services/job_matcher.py`.
- Add a page: create `app/ui/my_page.py::render_my_page()`, add to `NAV_ITEMS` in `app/ui/components.py`, route in `app/main.py`.

---

## 16. Troubleshooting & FAQ

| Symptom | Cause / Fix |
|---------|-------------|
| `OPENROUTER_API_KEY is not configured` | `.env` missing or not loaded. `Copy-Item .env.example .env`, set key, restart Streamlit. |
| `GEMINI_API_KEY` mentioned in old docs | Stale. Use `OPENROUTER_API_KEY`; Gemini is only a legacy fallback. |
| Empty / generic AI output | Free-tier model rate limit or plain-text response. `invoke_structured()` already retries via `json_mode` + manual parse; try again or set a stronger `OPENROUTER_MODEL`. |
| Chat says "not available in the uploaded resume" | Correct behavior — retrieval threshold not met. Rephrase or check the resume actually contains it. |
| `DATABASE_URL` connection fails | Postgres not running / DB not created. `createdb ai_career_copilot`, check `postgresql+psycopg://...`, run `alembic upgrade head`. |
| Tests need no DB server? | Yes — default `TEST_DATABASE_URL=sqlite+pysqlite:///:memory:`. |
| Stale resume / match results | Cache keys (`file_id`, `match_key`, `swot_key`) — re-upload or change inputs, or **Clear uploaded resume**. |
| DOCX tables missing | `parse_docx()` includes tables; if still missing, the file may use text boxes (unsupported — export to PDF). |

---

## 17. Limitations & Roadmap

**Current limitations:**

- No authentication — all rows belong to a default local user.
- FAISS index is in-memory per session (only metadata persisted); large resumes re-embed on reload.
- Single resume + single job at a time; no multi-file comparison dashboard.
- English-focused prompts; no PDF export of reports yet.

**Possible next steps:**

- Auth + multi-user workspaces
- Persistent vector store (pgvector / Qdrant) + multi-resume chat
- Cover-letter generator, LinkedIn optimizer, salary estimator
- Export ATS / Match / SWOT reports to PDF
- Docker + CI + hosted demo

---

## 18. Contributing

1. Fork / branch, then `uv sync`.
2. Keep layer rules: LLM clients only in `app/ai/openrouter_client.py`, SQLAlchemy only in `app/db`, UI stays thin.
3. Add / update tests under `tests/` (mock LLMs, never hit network in tests).
4. Run `pytest` before pushing.
5. Never commit `.env`, `.venv/`, or uploaded resumes.

---

## 19. License & Acknowledgements

- License: specify here (e.g. MIT — add a `LICENSE` file if distributing).
- Built with Streamlit, LangChain, LangGraph, FAISS, SQLAlchemy, Alembic, PyMuPDF, OpenRouter.
- Secrets belong in `.env` — never hardcoded.

Happy job hunting! 🚀
