"""
ClawAI Memory Store
------------------
Persistent memory system for the agent.

Memory Types:
- Long-term memory: stable knowledge (MEMORY.md)
- Daily memory: execution notes and short-term facts (YYYY-MM-DD.md)
"""

from pathlib import Path
from datetime import datetime, timedelta

from clawai.utils.helpers import ensure_dir, today_date


class MemoryStore:
    """Filesystem-backed persistent memory for ClawAI."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory_dir = ensure_dir(workspace / "memory")
        self.long_term_file = self.memory_dir / "MEMORY.md"

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def daily_file(self, date: str | None = None) -> Path:
        """Return path to a daily memory file."""
        return self.memory_dir / f"{date or today_date()}.md"

    # ------------------------------------------------------------------ #
    # Daily Memory
    # ------------------------------------------------------------------ #

    def read_today(self) -> str:
        """Read today's memory notes."""
        path = self.daily_file()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def append_today(self, content: str) -> None:
        """Append content to today's memory file."""
        path = self.daily_file()

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            text = f"{existing}\n\n{content}"
        else:
            text = f"# {today_date()}\n\n{content}"

        path.write_text(text, encoding="utf-8")

    def read_recent_days(self, days: int = 7) -> str:
        """Read daily memories from the last N days."""
        today = datetime.now().date()
        parts: list[str] = []

        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            path = self.daily_file(date_str)

            if path.exists():
                parts.append(path.read_text(encoding="utf-8"))

        return "\n\n---\n\n".join(parts)

    def list_daily_files(self) -> list[Path]:
        """List all daily memory files (newest first)."""
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    # ------------------------------------------------------------------ #
    # Long-term Memory
    # ------------------------------------------------------------------ #

    def read_long_term(self) -> str:
        """Read long-term memory."""
        return (
            self.long_term_file.read_text(encoding="utf-8")
            if self.long_term_file.exists()
            else ""
        )

    def write_long_term(self, content: str) -> None:
        """Overwrite long-term memory."""
        self.long_term_file.write_text(content, encoding="utf-8")

    def append_long_term(self, content: str) -> None:
        """Append content to long-term memory."""
        existing = self.read_long_term()
        text = f"{existing}\n\n{content}" if existing else content
        self.long_term_file.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Agent Context
    # ------------------------------------------------------------------ #

    def get_context(self) -> str:
        """
        Build memory context for prompt injection.

        Includes:
        - Long-term memory
        - Today's notes
        """
        sections: list[str] = []

        long_term = self.read_long_term()
        if long_term:
            sections.append("# Long-term Memory\n\n" + long_term)

        today = self.read_today()
        if today:
            sections.append("# Today's Notes\n\n" + today)

        return "\n\n---\n\n".join(sections)
