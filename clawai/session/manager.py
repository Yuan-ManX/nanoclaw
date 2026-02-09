"""
Session management for conversation history.

Design principles (ClawAI style):
- Session = pure data object
- Manager = IO + lifecycle orchestration
- Unified runtime paths
- Append-only + atomic persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from clawai.utils.helpers import (
    RUNTIME_PATHS,
    now_iso,
    safe_filename,
)


# ===========================
# Session Object
# ===========================

@dataclass(slots=True)
class Session:
    """
    A single conversation session.

    Notes:
    - messages are append-only
    - metadata is free-form extensible
    """

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, role: str, content: str, **extra: Any) -> None:
        """Append a new message."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": now_iso(),
            **extra
        })
        self.updated_at = now_iso()

    def history(self, max_messages: int = 50) -> list[dict[str, str]]:
        """Return recent messages in LLM format."""
        msgs = self.messages[-max_messages:]
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    def clear(self) -> None:
        """Clear session history."""
        self.messages.clear()
        self.updated_at = now_iso()


# ===========================
# Session Manager
# ===========================

class SessionManager:
    """
    Persistent session lifecycle manager.

    Responsibilities:
    - session loading
    - caching
    - persistence
    - lifecycle cleanup
    """

    def __init__(self):
        self.sessions_dir = RUNTIME_PATHS.sessions
        self._cache: dict[str, Session] = {}

    # ---------- internal helpers ----------

    def _path(self, key: str) -> Path:
        safe = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe}.jsonl"

    # ---------- core APIs ----------

    def get(self, key: str) -> Session:
        """Get or create a session."""
        if key in self._cache:
            return self._cache[key]

        session = self._load(key) or Session(key=key)
        self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        """Persist session atomically."""
        path = self._path(session.key)
        tmp = path.with_suffix(".tmp")

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "_type": "metadata",
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "metadata": session.metadata,
                }) + "\n")

                for msg in session.messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")

            tmp.replace(path)
            self._cache[session.key] = session

        except Exception as e:
            logger.exception(f"Session save failed: {session.key}: {e}")

    def delete(self, key: str) -> bool:
        """Delete a session."""
        self._cache.pop(key, None)
        path = self._path(key)

        if path.exists():
            path.unlink()
            return True
        return False

    def list(self) -> list[dict[str, Any]]:
        """List all sessions."""
        sessions: list[dict[str, Any]] = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                with open(path, encoding="utf-8") as f:
                    meta = json.loads(f.readline())

                if meta.get("_type") == "metadata":
                    sessions.append({
                        "key": path.stem.replace("_", ":"),
                        "created_at": meta.get("created_at"),
                        "updated_at": meta.get("updated_at"),
                        "path": str(path)
                    })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at") or "", reverse=True)

    # ---------- load ----------

    def _load(self, key: str) -> Session | None:
        path = self._path(key)
        if not path.exists():
            return None

        try:
            messages: list[dict[str, Any]] = []
            metadata: dict[str, Any] = {}
            created_at: str | None = None

            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = data.get("created_at")
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or now_iso(),
                updated_at=now_iso(),
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Failed to load session {key}: {e}")
            return None
  
