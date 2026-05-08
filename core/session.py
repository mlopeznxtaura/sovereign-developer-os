"""
core/session.py — Conversation + suggestion history session (Cluster 03)

Manages a single developer session: conversation turns, pending suggestions,
and the context window fed to the LLM.

SDKs: SQLAlchemy (via memory/store.py)
"""
import time
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeveloperSession:
    """
    Manages one developer interaction session.
    Tracks messages, pending suggestions, and feeds history to the agent.
    """

    def __init__(self, store, config: dict, session_id: Optional[str] = None):
        self._store = store
        self._cfg = config
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self._pending_suggestions = []
        self._start_time = time.time()
        logger.info("Session started: %s", self.session_id)

    def add_user_message(self, content: str) -> int:
        return self._store.add_message(self.session_id, "user", content)

    def add_assistant_message(self, content: str, latency_ms: Optional[float] = None,
                               model: Optional[str] = None) -> int:
        return self._store.add_message(self.session_id, "assistant", content,
                                       model=model, latency_ms=latency_ms)

    def get_history(self, limit: int = 20) -> list:
        return self._store.get_history(self.session_id, limit=limit)

    def build_message_history(self, limit: int = 10) -> list:
        """Return history in LangChain/OpenAI message format."""
        from langchain_core.messages import HumanMessage, AIMessage
        history = self.get_history(limit=limit)
        messages = []
        for h in history:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                messages.append(AIMessage(content=h["content"]))
        return messages

    def add_suggestion(self, suggested: str, file_path: Optional[str] = None,
                       original: Optional[str] = None) -> int:
        sid = self._store.add_suggestion(self.session_id, suggested,
                                          file_path=file_path, original=original)
        self._pending_suggestions.append({"id": sid, "suggested": suggested,
                                           "file_path": file_path})
        return sid

    def accept_suggestion(self, suggestion_id: int, feedback: Optional[str] = None) -> None:
        self._store.accept_suggestion(suggestion_id, feedback=feedback)
        self._pending_suggestions = [s for s in self._pending_suggestions
                                      if s["id"] != suggestion_id]
        logger.info("Suggestion %d accepted", suggestion_id)

    def get_pending_suggestions(self) -> list:
        return list(self._pending_suggestions)

    def stats(self) -> dict:
        return {
            "session_id": self.session_id,
            "duration_s": round(time.time() - self._start_time, 1),
            "pending_suggestions": len(self._pending_suggestions),
            "store_stats": self._store.stats(),
        }
