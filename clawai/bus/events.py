"""
Event types for ClawAI message bus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final


# ---------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------

@dataclass(slots=True)
class InboundMessage:
    """
    Message received from an external chat channel.
    """

    channel: str              # telegram / discord / slack / whatsapp / web
    sender_id: str            # User identifier (platform-specific)
    chat_id: str              # Conversation / channel identifier
    content: str              # Message text content

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------

    @property
    def session_key(self) -> str:
        """
        Stable session identifier for routing and memory.
        """
        return f"{self.channel}:{self.chat_id}"


# ---------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------

@dataclass(slots=True)
class OutboundMessage:
    """
    Message to be sent to an external chat channel.
    """

    channel: str
    chat_id: str
    content: str

    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
