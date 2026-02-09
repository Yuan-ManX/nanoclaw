"""
Runtime utility helpers.

Design principles (ClawAI style):
- Centralized path management
- Pure functional utilities
- Predictable IO boundaries
- Strong naming semantics
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final


# ===========================
# Path System
# ===========================

@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """
    Centralized runtime path manager.

    This ensures:
    - unified directory layout
    - predictable filesystem structure
    - easy environment switching
    """

    root: Path

    @classmethod
    def default(cls) -> "RuntimePaths":
        return cls(root=Path.home() / ".clawai")

    def ensure(self) -> "RuntimePaths":
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.memory.mkdir(parents=True, exist_ok=True)
        self.skills.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def memory(self) -> Path:
        return self.workspace / "memory"

    @property
    def skills(self) -> Path:
        return self.workspace / "skills"


# Global singleton (ClawAI standard pattern)
RUNTIME_PATHS: Final[RuntimePaths] = RuntimePaths.default().ensure()


# ===========================
# Clock Utilities
# ===========================

def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def now_iso() -> str:
    """Return current timestamp in ISO-8601 format."""
    return datetime.now().isoformat()


def now_ms() -> int:
    """Return current unix time in milliseconds."""
    return int(datetime.now().timestamp() * 1000)


# ===========================
# String Utilities
# ===========================

_UNSAFE_CHARS = '<>:"/\\|?*'


def truncate(s: str, max_len: int = 120, suffix: str = "...") -> str:
    """Truncate a string with suffix."""
    if len(s) <= max_len:
        return s
    return s[: max_len - len(suffix)] + suffix


def safe_filename(name: str) -> str:
    """Convert arbitrary string to filesystem-safe filename."""
    for ch in _UNSAFE_CHARS:
        name = name.replace(ch, "_")
    return name.strip()


# ===========================
# Session Helpers
# ===========================

def parse_session_key(key: str) -> tuple[str, str]:
    """
    Parse session key:  channel:chat_id

    Raises:
        ValueError: invalid key format
    """
    if ":" not in key:
        raise ValueError(f"Invalid session key format: {key}")

    channel, chat_id = key.split(":", 1)
    return channel, chat_id


def build_session_key(channel: str, chat_id: str) -> str:
    """Build normalized session key."""
    return f"{channel}:{chat_id}"


# ===========================
# Directory Helpers
# ===========================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
