"""
ClawAI Context Builder
---------------------
Assembles system prompt and message context for the agent.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional

from clawai.agent.memory import MemoryStore
from clawai.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Responsible for assembling the full prompt context for ClawAI.

    Includes:
    - Agent identity
    - Bootstrap documents
    - Long-term memory
    - Available and active skills
    - Conversation history
    """

    BOOTSTRAP_FILES = (
        "IDENTITY.md",
        "AGENTS.md",
        "TOOLS.md",
        "USER.md",
        "SOUL.md",
    )

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    # ------------------------------------------------------------------ #
    # System Prompt
    # ------------------------------------------------------------------ #

    def build_system_prompt(
        self,
        skill_names: Optional[list[str]] = None,
    ) -> str:
        """Build the complete system prompt."""

        sections: list[str] = []

        sections.append(self._build_identity())

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            sections.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            sections.append(self._section("Memory", memory))

        active_skills = self._load_active_skills()
        if active_skills:
            sections.append(self._section("Active Skills", active_skills))

        skills_index = self.skills.build_skills_summary()
        if skills_index:
            sections.append(self._build_skills_index(skills_index))

        return "\n\n---\n\n".join(sections)

    def _build_identity(self) -> str:
        """Core agent identity block."""
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace = str(self.workspace.expanduser().resolve())

        return f"""# ClawAI 🦾

You are **ClawAI**, a lightweight, action-oriented personal AI assistant.

Your purpose is to:
- Plan tasks
- Execute tools
- Complete real-world actions end-to-end

## Capabilities
You can:
- Read, write, and edit files
- Execute shell commands
- Search and fetch web content
- Send messages to external channels
- Spawn sub-agents for background work

## Runtime Context
**Current Time:** {now}

**Workspace:** {workspace}

- Memory: {workspace}/memory/MEMORY.md
- Daily logs: {workspace}/memory/YYYY-MM-DD.md
- Skills: {workspace}/skills/<skill-name>/SKILL.md

## Operating Rules
- Prefer direct answers when no tools are required
- Use tools only when they help complete the task
- Explain actions briefly and clearly
- Persist long-term knowledge into MEMORY.md
"""

    # ------------------------------------------------------------------ #
    # Bootstrap / Memory / Skills
    # ------------------------------------------------------------------ #

    def _load_bootstrap_files(self) -> str:
        """Load static bootstrap documents."""
        parts = []

        for name in self.BOOTSTRAP_FILES:
            path = self.workspace / name
            if path.exists():
                content = path.read_text(encoding="utf-8")
                parts.append(f"## {name}\n\n{content}")

        return "\n\n".join(parts)

    def _load_active_skills(self) -> str:
        """Load always-enabled skills (full content)."""
        always = self.skills.get_always_skills()
        if not always:
            return ""

        return self.skills.load_skills_for_context(always)

    def _build_skills_index(self, summary: str) -> str:
        """Build discoverable skills index."""
        return f"""# Available Skills

The following skills extend your capabilities.

To use a skill:
1. Read its SKILL.md file using the `read_file` tool
2. Follow the instructions inside

Some skills may require dependencies to be installed first.

{summary}
"""

    @staticmethod
    def _section(title: str, content: str) -> str:
        return f"# {title}\n\n{content}"

    # ------------------------------------------------------------------ #
    # Message Assembly
    # ------------------------------------------------------------------ #

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: Optional[list[str]] = None,
        media: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Assemble messages for an LLM call."""
        messages: list[dict[str, Any]] = []

        messages.append(
            {"role": "system", "content": self.build_system_prompt(skill_names)}
        )

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": self._build_user_content(current_message, media),
            }
        )

        return messages

    def _build_user_content(
        self,
        text: str,
        media: Optional[list[str]],
    ) -> str | list[dict[str, Any]]:
        """Attach images to user message if provided."""
        if not media:
            return text

        parts: list[dict[str, Any]] = []

        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)

            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue

            b64 = base64.b64encode(p.read_bytes()).decode()
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}"
                    },
                }
            )

        if not parts:
            return text

        parts.append({"type": "text", "text": text})
        return parts

    # ------------------------------------------------------------------ #
    # Agent Loop Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def add_assistant_message(
        messages: list[dict[str, Any]],
        content: Optional[str],
        tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or "",
        }

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        return messages

    @staticmethod
    def add_tool_result(
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages
