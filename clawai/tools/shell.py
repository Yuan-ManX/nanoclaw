"""
Shell execution tool (ClawAI style).

Provides guarded, timeout-limited shell command execution
with workspace-aware safety checks.
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from clawai.tools.base import Tool


class ExecTool(Tool):
    """
    Execute shell commands with safety guards.

    This tool is intentionally conservative and should only be used
    when absolutely necessary.
    """

    def __init__(
        self,
        *,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ) -> None:
        self.timeout = timeout
        self.working_dir = working_dir
        self.restrict_to_workspace = restrict_to_workspace

        # Default denylist: destructive or system-level commands
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fq]\b",
            r"\brmdir\s+/s\b",
            r"\b(format|mkfs|diskpart)\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\b(shutdown|reboot|poweroff)\b",
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]

        # Optional allowlist (if provided, must match at least one)
        self.allow_patterns = allow_patterns or []

    # =========================
    # Tool interface
    # =========================

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Commands are guarded and time-limited."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory override",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        *,
        command: str,
        working_dir: str | None = None,
        **_: Any,
    ) -> str:
        """
        Execute a shell command with safety checks.
        """
        cwd = working_dir or self.working_dir or os.getcwd()

        guard_error = self._check_command_safety(command, cwd)
        if guard_error:
            return guard_error

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: command timed out after {self.timeout} seconds"

            return self._format_output(
                stdout=stdout,
                stderr=stderr,
                returncode=process.returncode,
            )

        except Exception as e:
            return f"Error: failed to execute command: {e}"

    # =========================
    # Safety & helpers
    # =========================

    def _check_command_safety(self, command: str, cwd: str) -> str | None:
        """
        Perform best-effort safety checks on the command.
        """
        cmd = command.strip()
        lower = cmd.lower()

        # Denylist
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: command blocked by safety guard (dangerous pattern detected)"

        # Allowlist (if configured)
        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: command blocked by safety guard (not in allowlist)"

        # Workspace restriction
        if self.restrict_to_workspace:
            if "../" in cmd or "..\\" in cmd:
                return "Error: command blocked (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            paths = (
                re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
                + re.findall(r"/[^\s\"']+", cmd)
            )

            for raw in paths:
                try:
                    resolved = Path(raw).resolve()
                except Exception:
                    continue

                if resolved != cwd_path and cwd_path not in resolved.parents:
                    return "Error: command blocked (path outside working directory)"

        return None

    def _format_output(
        self,
        *,
        stdout: bytes | None,
        stderr: bytes | None,
        returncode: int,
    ) -> str:
        """
        Normalize and truncate process output.
        """
        parts: list[str] = []

        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace"))

        if stderr:
            err = stderr.decode("utf-8", errors="replace").strip()
            if err:
                parts.append(f"STDERR:\n{err}")

        if returncode != 0:
            parts.append(f"Exit code: {returncode}")

        result = "\n".join(parts) if parts else "(no output)"

        # Hard cap output size
        max_len = 10_000
        if len(result) > max_len:
            result = (
                result[:max_len]
                + f"\n... (truncated, {len(result) - max_len} more chars)"
            )

        return result
