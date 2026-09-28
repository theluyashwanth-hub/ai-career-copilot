"""SQLAlchemy declarative base (Phase 12).

Single ``Base`` for all ORM models so Alembic autogenerate sees the
full metadata via ``Base.metadata``.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all persistent entities."""
