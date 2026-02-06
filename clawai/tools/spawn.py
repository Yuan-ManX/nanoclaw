"""
Spawn tool for creating background subagents (ClawAI style).
"""

from typing import Any, TYPE_CHECKING

from clawai.tools.base import Tool

if TYPE_CHECKING:
    from clawai.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """
    Spawn a background subagent to execute a task asynchronously.

    The spawned subagent runs independently and reports its result
    back to the originating context when finished.
    """

    def __init__(self, manager: "SubagentManager") -> None:
        self._manager = manager

        # Context of the caller (used for routing callbacks)
        self._origin_channel: str = "cli"
        self._origin_chat_id: str = "direct"

    # =========================
    # Context propagation
    # =========================

    def set_context(self, *, channel: str, chat_id: str) -> None:
        """
        Set origin context for subagent result announcements.
        """
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    # =========================
    # Tool interface
    # =========================

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a background subagent to handle a task asynchronously. "
            "Use this for complex or long-running work that does not need "
            "to block the main agent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the subagent",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Optional short label for the task "
                        "(used for logging or UI display)"
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        *,
        task: str,
        label: str | None = None,
        **_: Any,
    ) -> str:
        """
        Spawn a subagent to execute the given task.
        """
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
