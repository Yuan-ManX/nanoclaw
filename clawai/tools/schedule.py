"""
Cron tool for scheduling reminders and recurring tasks.

ClawAI Agent style:
- Declarative tool metadata
- Structured JSON responses
- Explicit action dispatch
"""

from __future__ import annotations

from typing import Any, Callable

from clawai.tools.base import Tool
from clawai.cron.service import CronService
from clawai.cron.types import CronSchedule


class CronTool(Tool):
    """
    Schedule reminders and recurring background tasks.
    """

    # ---------------------------------------------------------------------
    # Tool declaration
    # ---------------------------------------------------------------------

    name = "cron"
    description = "Schedule reminders and recurring tasks."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove"],
                "description": "Action to perform",
            },
            "message": {
                "type": "string",
                "description": "Reminder message (required for add)",
            },
            "every_seconds": {
                "type": "integer",
                "description": "Run every N seconds (recurring)",
            },
            "cron_expr": {
                "type": "string",
                "description": "Cron expression, e.g. '0 9 * * *'",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID (required for remove)",
            },
        },
        "required": ["action"],
    }

    # ---------------------------------------------------------------------

    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._channel: str | None = None
        self._chat_id: str | None = None

        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "add": self._handle_add,
            "list": self._handle_list,
            "remove": self._handle_remove,
        }

    # ---------------------------------------------------------------------
    # Context
    # ---------------------------------------------------------------------

    def set_context(self, channel: str, chat_id: str) -> None:
        """Bind delivery context for scheduled messages."""
        self._channel = channel
        self._chat_id = chat_id

    def _require_context(self) -> tuple[bool, str]:
        if not self._channel or not self._chat_id:
            return False, "Missing delivery context (channel/chat_id)"
        return True, ""

    # ---------------------------------------------------------------------
    # Tool entry
    # ---------------------------------------------------------------------

    async def execute(
        self,
        action: str,
        **kwargs: Any,
    ) -> str:
        handler = self._handlers.get(action)
        if not handler:
            return self._error(f"Unknown action: {action}")

        try:
            result = handler(**kwargs)
            return self._ok(result)
        except Exception as exc:
            return self._error(str(exc))

    # ---------------------------------------------------------------------
    # Handlers
    # ---------------------------------------------------------------------

    def _handle_add(
        self,
        message: str | None = None,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not message:
            raise ValueError("message is required for add")

        ok, err = self._require_context()
        if not ok:
            raise ValueError(err)

        if every_seconds:
            schedule = CronSchedule(
                kind="every",
                every_ms=every_seconds * 1000,
            )
        elif cron_expr:
            schedule = CronSchedule(
                kind="cron",
                expr=cron_expr,
            )
        else:
            raise ValueError("either every_seconds or cron_expr is required")

        job = self._cron.add_job(
            name=message[:32],
            schedule=schedule,
            message=message,
            deliver=True,
            channel=self._channel,
            to=self._chat_id,
        )

        return {
            "id": job.id,
            "name": job.name,
            "schedule": schedule.kind,
        }

    def _handle_list(self, **_: Any) -> dict[str, Any]:
        jobs = self._cron.list_jobs()
        return {
            "count": len(jobs),
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "schedule": j.schedule.kind,
                }
                for j in jobs
            ],
        }

    def _handle_remove(self, job_id: str | None = None, **_: Any) -> dict[str, Any]:
        if not job_id:
            raise ValueError("job_id is required for remove")

        removed = self._cron.remove_job(job_id)
        if not removed:
            raise ValueError(f"Job not found: {job_id}")

        return {"removed": True, "job_id": job_id}

    # ---------------------------------------------------------------------
    # Response helpers
    # ---------------------------------------------------------------------

    def _ok(self, data: dict[str, Any]) -> str:
        return self.json_response({"ok": True, "data": data})

    def _error(self, message: str) -> str:
        return self.json_response({"ok": False, "error": message})
  
