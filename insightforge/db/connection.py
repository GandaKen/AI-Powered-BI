"""SQLAlchemy engine and session factory."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def _get_engine(database_url: str):
    global _engine
    if _engine is None:
        _engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory(database_url: str) -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        engine = _get_engine(database_url)
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session(database_url: str):
    """Yield a transactional session that auto-commits on success."""
    factory = get_session_factory(database_url)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session error")
        raise
    finally:
        session.close()
