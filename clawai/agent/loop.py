"""
ClawAI Agent Loop
-----------------
The minimal, action-first execution engine.

Responsibilities:
1. Build agent context
2. Call LLM for decision making
3. Execute tool actions
4. Return final result
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger

from clawai.bus.events import InboundMessage, OutboundMessage
from clawai.bus.queue import MessageBus
from clawai.providers.base import LLMProvider
from clawai.agent.context import ContextBuilder
from clawai.agent.tools.registry import ToolRegistry
from clawai.agent.subagent import SubagentManager
from clawai.session.manager import SessionManager


class AgentLoop:
    """Minimal action-oriented agent execution loop."""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: Optional[str] = None,
        max_steps: int = 20,
        brave_api_key: Optional[str] = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_steps = max_steps

        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()

        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
        )

        self._running = False
        self._register_tools(brave_api_key)

    # --------------------------------------------------------------------- #
    # Setup
    # --------------------------------------------------------------------- #

    def _register_tools(self, brave_api_key: Optional[str]) -> None:
        """Register built-in ClawAI tools."""
        from clawai.agent.tools.filesystem import (
            ReadFileTool,
            WriteFileTool,
            EditFileTool,
            ListDirTool,
        )
        from clawai.agent.tools.shell import ExecTool
        from clawai.agent.tools.web import WebSearchTool, WebFetchTool
        from clawai.agent.tools.message import MessageTool
        from clawai.agent.tools.spawn import SpawnTool

        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

        self.tools.register(ExecTool(working_dir=str(self.workspace)))

        self.tools.register(WebSearchTool(api_key=brave_api_key))
        self.tools.register(WebFetchTool())

        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))

    # --------------------------------------------------------------------- #
    # Runtime
    # --------------------------------------------------------------------- #

    async def run(self) -> None:
        """Main event loop."""
        self._running = True
        logger.info("ClawAI agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0,
                )
                response = await self._handle_message(msg)
                if response:
                    await self.bus.publish_outbound(response)

            except asyncio.TimeoutError:
                continue

            except Exception as e:
                logger.exception("Agent loop crashed")
                raise e

    def stop(self) -> None:
        self._running = False
        logger.info("ClawAI agent loop stopped")

    # --------------------------------------------------------------------- #
    # Core Logic
    # --------------------------------------------------------------------- #

    async def _handle_message(
        self, msg: InboundMessage
    ) -> Optional[OutboundMessage]:
        """Unified entry for user / system messages."""

        logger.info(
            f"Incoming message [{msg.channel}] {msg.sender_id}: {msg.content}"
        )

        session = self.sessions.get_or_create(msg.session_key)
        self._update_tool_context(msg.channel, msg.chat_id)

        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media,
        )

        final_answer = await self._agent_loop(messages)

        session.add_message("user", msg.content)
        session.add_message("assistant", final_answer)
        self.sessions.save(session)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_answer,
        )

    async def _agent_loop(self, messages: list[dict]) -> str:
        """Core agent reasoning + execution loop."""

        for step in range(self.max_steps):
            logger.debug(f"Agent step {step + 1}/{self.max_steps}")

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )

            if not response.has_tool_calls:
                return response.content or ""

            messages = self.context.add_assistant_message(
                messages,
                response.content,
                self._format_tool_calls(response),
            )

            for tool_call in response.tool_calls:
                logger.debug(
                    f"Executing tool: {tool_call.name} {tool_call.arguments}"
                )
                result = await self.tools.execute(
                    tool_call.name, tool_call.arguments
                )
                messages = self.context.add_tool_result(
                    messages,
                    tool_call.id,
                    tool_call.name,
                    result,
                )

        logger.warning("Agent loop hit max steps limit")
        return "Task execution stopped after reaching step limit."

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _update_tool_context(self, channel: str, chat_id: str) -> None:
        """Inject routing context into stateful tools."""
        for name in ("message", "spawn"):
            tool = self.tools.get(name)
            if hasattr(tool, "set_context"):
                tool.set_context(channel, chat_id)

    @staticmethod
    def _format_tool_calls(response) -> list[dict]:
        """Convert tool calls to OpenAI-compatible format."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]

    # --------------------------------------------------------------------- #
    # Direct / CLI
    # --------------------------------------------------------------------- #

    async def process_direct(
        self, content: str, session_key: str = "cli:direct"
    ) -> str:
        """Run agent without message bus (CLI / tests)."""
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=content,
        )
        response = await self._handle_message(msg)
        return response.content if response else ""
