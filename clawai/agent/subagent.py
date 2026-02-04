"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from clawai.bus.events import InboundMessage
from clawai.bus.queue import MessageBus
from clawai.providers.base import LLMProvider
from clawai.agent.tools.registry import ToolRegistry
from clawai.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    ListDirTool,
)
from clawai.agent.tools.shell import ExecTool
from clawai.agent.tools.web import WebSearchTool, WebFetchTool


DEFAULT_MAX_ITERATIONS = 15
TASK_ID_LENGTH = 8


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in isolation
    to complete focused tasks and report results back to the main agent.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        brave_api_key: str | None = None,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.brave_api_key = brave_api_key

        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """Spawn a background subagent to execute a task."""
        task_id = self._generate_task_id()
        display_label = label or self._truncate(task)

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task

        bg_task.add_done_callback(
            lambda _: self._running_tasks.pop(task_id, None)
        )

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return (
            f"Subagent [{display_label}] started "
            f"(id: {task_id}). I'll notify you when it completes."
        )

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    # ---------------------------------------------------------------------
    # Subagent lifecycle
    # ---------------------------------------------------------------------

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            tools = self._build_tool_registry()
            system_prompt = self._build_subagent_prompt(task)

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            result = await self._run_agent_loop(
                task_id=task_id,
                messages=messages,
                tools=tools,
            )

            await self._announce_result(
                task_id=task_id,
                label=label,
                task=task,
                result=result,
                origin=origin,
                status="ok",
            )

            logger.info(f"Subagent [{task_id}] completed successfully")

        except Exception as exc:
            logger.exception(f"Subagent [{task_id}] failed")
            await self._announce_result(
                task_id=task_id,
                label=label,
                task=task,
                result=f"Error: {exc}",
                origin=origin,
                status="error",
            )

    # ---------------------------------------------------------------------
    # Agent loop
    # ---------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        task_id: str,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
    ) -> str:
        """Run the LLM + tool loop for a subagent."""
        for iteration in range(1, DEFAULT_MAX_ITERATIONS + 1):
            logger.debug(f"Subagent [{task_id}] iteration {iteration}")

            response = await self.provider.chat(
                messages=messages,
                tools=tools.get_definitions(),
                model=self.model,
            )

            if response.has_tool_calls:
                messages.append(
                    self._format_assistant_tool_message(response)
                )

                for tool_call in response.tool_calls:
                    logger.debug(
                        f"Subagent [{task_id}] executing tool: {tool_call.name}"
                    )
                    result = await tools.execute(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )
                continue

            return response.content or "Task completed with no output."

        return "Task completed but reached the maximum iteration limit."

    # ---------------------------------------------------------------------
    # Tooling & prompt
    # ---------------------------------------------------------------------

    def _build_tool_registry(self) -> ToolRegistry:
        tools = ToolRegistry()
        tools.register(ReadFileTool())
        tools.register(WriteFileTool())
        tools.register(ListDirTool())
        tools.register(ExecTool(working_dir=str(self.workspace)))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())
        return tools

    def _build_subagent_prompt(self, task: str) -> str:
        return f"""# Subagent

You are a focused subagent spawned by the main agent.

## Task
{task}

## Rules
- Complete only the assigned task
- Do not initiate side objectives
- Be concise and factual
- Provide a clear final summary

## Capabilities
- Read and write files in the workspace
- Execute shell commands
- Search and fetch web content

## Restrictions
- No direct user interaction
- No spawning other agents
- No access to main agent conversation history

## Workspace
{self.workspace}

When finished, return a clear summary of your findings or actions.
"""

    # ---------------------------------------------------------------------
    # Messaging
    # ---------------------------------------------------------------------

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        status_text = (
            "completed successfully" if status == "ok" else "failed"
        )

        content = f"""[Task {status_text}]

Task:
{task}

Result:
{result}

Summarize this naturally for the user in 1–2 sentences.
Do not mention internal agent mechanics.
"""

        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=content,
        )

        await self.bus.publish_inbound(msg)

        logger.debug(
            f"Subagent [{task_id}] announced result to "
            f"{origin['channel']}:{origin['chat_id']}"
        )

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _generate_task_id() -> str:
        return str(uuid.uuid4())[:TASK_ID_LENGTH]

    @staticmethod
    def _truncate(text: str, max_len: int = 30) -> str:
        return text if len(text) <= max_len else text[:max_len] + "..."

    @staticmethod
    def _format_assistant_tool_message(response) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ],
        }
