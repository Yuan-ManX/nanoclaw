"""
ClawAI message tool.

Provides an explicit communication capability for agents
to send messages to users via configured outbound channels.
"""

from typing import Any, Awaitable, Callable

from clawai.tools.base import Tool
from clawai.bus.events import OutboundMessage


class MessageTool(Tool):
    """
    Tool for sending messages to users.

    This is a side-effect tool used by agents to communicate
    information back to the user via chat channels.
    """

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str | None = None,
        default_chat_id: str | None = None,
    ) -> None:
        self._send_callback = send_callback
        self._channel = default_channel
        self._chat_id = default_chat_id

    # =========================
    # Runtime context
    # =========================

    def set_context(self, channel: str, chat_id: str) -> None:
        """Bind the message context for this tool invocation."""
        self._channel = channel
        self._chat_id = chat_id

    def set_send_callback(
        self,
        callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """Configure the outbound message sender."""
        self._send_callback = callback

    # =========================
    # Tool definition
    # =========================

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user via the active chat channel. "
            "Use this to communicate results, status updates, or errors."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Message content to send to the user",
                },
                "channel": {
                    "type": "string",
                    "description": "Optional override for target channel",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional override for target chat/user ID",
                },
            },
            "required": ["content"],
        }

    # =========================
    # Execution
    # =========================

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        **_: Any,
    ) -> str:
        # Resolve context
        resolved_channel = channel or self._channel
        resolved_chat_id = chat_id or self._chat_id

        if not resolved_channel or not resolved_chat_id:
            return (
                "Error: message context not set "
                "(channel/chat_id missing)"
            )

        if not self._send_callback:
            return (
                "Error: message sender not configured "
                "(send_callback missing)"
            )

        message = OutboundMessage(
            channel=resolved_channel,
            chat_id=resolved_chat_id,
            content=content,
        )

        try:
            await self._send_callback(message)
            return (
                f"Message sent to "
                f"{resolved_channel}:{resolved_chat_id}"
            )
        except Exception as e:
            return f"Error: failed to send message: {e}"
  
