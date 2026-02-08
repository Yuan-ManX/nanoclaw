"""
Heartbeat runtime service.

Responsible for:
    - Periodic agent wake-up
    - Workspace task sensing
    - Autonomous task triggering
    - Runtime lifecycle integration
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from loguru import logger


# ============================================================
# Constants
# ============================================================

DEFAULT_HEARTBEAT_INTERVAL_S = 30 * 60

HEARTBEAT_FILENAME = "HEARTBEAT.md"

HEARTBEAT_PROMPT = """Read HEARTBEAT.md in your workspace (if it exists).
Follow any instructions or tasks listed there.
If nothing needs attention, reply with just: HEARTBEAT_OK
"""

HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"


# ============================================================
# Types
# ============================================================

HeartbeatCallback = Callable[[str], Awaitable[str]]


# ============================================================
# Utilities
# ============================================================

def is_heartbeat_actionable(content: Optional[str]) -> bool:
    """
    Determine whether HEARTBEAT.md contains actionable tasks.

    Rules:
        - Ignore empty lines
        - Ignore markdown headers
        - Ignore HTML comments
        - Ignore unchecked/checked empty checkboxes
    """
    if not content:
        return False

    skip_patterns = {"- [ ]", "* [ ]", "- [x]", "* [x]"}

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("<!--"):
            continue
        if line in skip_patterns:
            continue
        return True

    return False


# ============================================================
# Config Schema
# ============================================================

@dataclass(slots=True)
class HeartbeatConfig:
    enabled: bool = True
    interval_s: int = DEFAULT_HEARTBEAT_INTERVAL_S
    filename: str = HEARTBEAT_FILENAME


# ============================================================
# Heartbeat Service
# ============================================================

class HeartbeatService:
    """
    Runtime heartbeat service.

    This is NOT a timer utility.

    This is a **runtime-level autonomous scheduler** that periodically wakes
    the agent and injects environment awareness.
    """

    def __init__(
        self,
        workspace: Path,
        callback: HeartbeatCallback,
        config: HeartbeatConfig = HeartbeatConfig(),
    ):
        self.workspace = workspace
        self.callback = callback
        self.config = config

        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / self.config.filename

    # ------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------

    async def start(self) -> None:
        if not self.config.enabled:
            logger.info("Heartbeat service disabled")
            return

        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

        logger.info(
            f"Heartbeat service started (interval={self.config.interval_s}s)"
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        logger.info("Heartbeat service stopped")

    async def trigger_now(self) -> Optional[str]:
        """
        Manually trigger heartbeat tick.
        Useful for:
            - CLI manual trigger
            - Debug
            - Test
        """
        return await self._tick()

    # ------------------------------------------------------------
    # Internal Loop
    # ------------------------------------------------------------

    async def _run_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.config.interval_s)
                if self._running:
                    await self._tick()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Heartbeat loop crashed")

    async def _tick(self) -> Optional[str]:
        """
        Execute one heartbeat cycle.
        """
        content = self._read_heartbeat()

        if not is_heartbeat_actionable(content):
            logger.debug("Heartbeat: no actionable tasks")
            return None

        logger.info("Heartbeat: task detected → waking agent")

        try:
            response = await self.callback(HEARTBEAT_PROMPT)

            if self._is_ok_response(response):
                logger.info("Heartbeat: agent reported OK")
            else:
                logger.success("Heartbeat: task executed")

            return response

        except Exception:
            logger.exception("Heartbeat execution failed")
            return None

    # ------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------

    def _read_heartbeat(self) -> Optional[str]:
        if not self.heartbeat_file.exists():
            return None

        try:
            return self.heartbeat_file.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read HEARTBEAT.md")
            return None

    # ------------------------------------------------------------
    # Semantics
    # ------------------------------------------------------------

    @staticmethod
    def _is_ok_response(response: Optional[str]) -> bool:
        if not response:
            return False
        normalized = response.upper().replace("_", "")
        return HEARTBEAT_OK_TOKEN.replace("_", "") in normalized
  
