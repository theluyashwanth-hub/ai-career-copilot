"""Engine / session helpers (Phase 12).

Centralizes SQLAlchemy engine and session construction so the rest of
the application only ever receives a ``Session`` object (or uses the
repository classes) and never builds engines itself.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base


def create_engine_for_url(url: str, *, echo: bool = False) -> Engine:
    """Create an engine for ``url``.

    In-memory SQLite (the default test backend) uses a ``StaticPool``
    so all sessions share the single backing connection.
    """
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        if url.endswith(":memory:"):
            return create_engine(
                url,
                echo=echo,
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        return create_engine(url, echo=echo, connect_args=connect_args)
    return create_engine(url, echo=echo, pool_pre_ping=True)


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Return an engine for ``url`` (defaults to ``settings.database_url``)."""
    return create_engine_for_url(url or settings.database_url, echo=echo)


def init_db(engine: Engine) -> None:
    """Create all tables (used by tests / local bootstrap).

    Production schema changes go through Alembic migrations; this helper
    is a convenience for fresh databases and the test suite.
    """
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to ``engine``."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def db_session(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a transactional session (commit on success, rollback on error)."""
    maker = session_factory(engine or get_engine())
    session = maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
