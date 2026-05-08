"""
memory/store.py — Chroma vector store + SQLAlchemy conversation metadata (Cluster 03)

Two-layer memory:
  1. Chroma — vector embeddings of code chunks (semantic search)
  2. SQLite via SQLAlchemy — conversation history, suggestions, accepted edits

SDKs: chromadb, SQLAlchemy
"""
import json
import logging
import os
import time
from datetime import datetime, UTC
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()


# ── ORM models ────────────────────────────────────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(String(64), nullable=False)
    role        = Column(String(16), nullable=False)   # user | assistant | system
    content     = Column(Text, nullable=False)
    model       = Column(String(64), nullable=True)
    tokens      = Column(Integer, nullable=True)
    latency_ms  = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(UTC))
    __table_args__ = (Index("ix_conv_session", "session_id"),)


class Suggestion(Base):
    __tablename__ = "suggestions"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(64), nullable=False)
    file_path    = Column(String(512), nullable=True)
    original     = Column(Text, nullable=True)
    suggested    = Column(Text, nullable=False)
    accepted     = Column(Boolean, default=False)
    feedback     = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=lambda: datetime.now(UTC))
    __table_args__ = (Index("ix_sugg_accepted", "accepted"),)


class IndexedFile(Base):
    __tablename__ = "indexed_files"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    file_path    = Column(String(512), unique=True, nullable=False)
    file_hash    = Column(String(64), nullable=True)
    chunk_count  = Column(Integer, default=0)
    indexed_at   = Column(DateTime, default=lambda: datetime.now(UTC))
    __table_args__ = (Index("ix_file_path", "file_path"),)


# ── Store ─────────────────────────────────────────────────────────────────────

class SovereignStore:
    """Combined Chroma + SQLAlchemy memory store."""

    def __init__(self, config: dict):
        db_path = os.path.expanduser(config.get("db", {}).get("path", "~/.sovereign/sovereign.db"))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("SovereignStore initialized: %s", db_path)

    def _sess(self):
        return self._Session()

    # ── Conversations ─────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str,
                    model: Optional[str] = None, tokens: Optional[int] = None,
                    latency_ms: Optional[float] = None) -> int:
        with self._sess() as sess:
            row = Conversation(session_id=session_id, role=role, content=content,
                               model=model, tokens=tokens, latency_ms=latency_ms)
            sess.add(row)
            sess.commit()
            return row.id

    def get_history(self, session_id: str, limit: int = 20) -> list:
        with self._sess() as sess:
            rows = (sess.query(Conversation)
                    .filter_by(session_id=session_id)
                    .order_by(Conversation.created_at.desc())
                    .limit(limit).all())
            return [{"role": r.role, "content": r.content, "model": r.model,
                     "latency_ms": r.latency_ms, "created_at": str(r.created_at)}
                    for r in reversed(rows)]

    # ── Suggestions ───────────────────────────────────────────────────────────

    def add_suggestion(self, session_id: str, suggested: str,
                       file_path: Optional[str] = None,
                       original: Optional[str] = None) -> int:
        with self._sess() as sess:
            row = Suggestion(session_id=session_id, suggested=suggested,
                             file_path=file_path, original=original)
            sess.add(row)
            sess.commit()
            return row.id

    def accept_suggestion(self, suggestion_id: int, feedback: Optional[str] = None) -> None:
        with self._sess() as sess:
            row = sess.get(Suggestion, suggestion_id)
            if row:
                row.accepted = True
                row.feedback = feedback
                sess.commit()

    def get_accepted_suggestions(self, limit: int = 100) -> list:
        with self._sess() as sess:
            rows = (sess.query(Suggestion).filter_by(accepted=True)
                    .order_by(Suggestion.created_at.desc()).limit(limit).all())
            return [{"id": r.id, "file_path": r.file_path,
                     "original": r.original, "suggested": r.suggested,
                     "feedback": r.feedback} for r in rows]

    def accepted_count(self) -> int:
        with self._sess() as sess:
            return sess.query(Suggestion).filter_by(accepted=True).count()

    # ── Indexed files ─────────────────────────────────────────────────────────

    def record_indexed_file(self, file_path: str, chunk_count: int,
                            file_hash: Optional[str] = None) -> None:
        with self._sess() as sess:
            row = sess.query(IndexedFile).filter_by(file_path=file_path).first()
            if row:
                row.chunk_count = chunk_count
                row.file_hash = file_hash
                row.indexed_at = datetime.now(UTC)
            else:
                sess.add(IndexedFile(file_path=file_path, chunk_count=chunk_count,
                                     file_hash=file_hash))
            sess.commit()

    def get_indexed_files(self) -> list:
        with self._sess() as sess:
            return [{"file_path": r.file_path, "chunk_count": r.chunk_count,
                     "indexed_at": str(r.indexed_at)} for r in sess.query(IndexedFile).all()]

    def stats(self) -> dict:
        with self._sess() as sess:
            return {
                "conversations": sess.query(Conversation).count(),
                "suggestions": sess.query(Suggestion).count(),
                "accepted_suggestions": sess.query(Suggestion).filter_by(accepted=True).count(),
                "indexed_files": sess.query(IndexedFile).count(),
            }
