"""Base channel abstraction for chat platform integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

from clawai.bus.events import InboundMessage, OutboundMessage
from clawai.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Base abstraction for all chat platform channels.

    Design goals:
        - Unified lifecycle control
        - Platform-agnostic permission filtering
        - Consistent message ingress / egress contract
        - Observability-first architecture
    """

    #: Channel unique identifier
    name: str = "base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running: bool = False

    # =============================
    # Lifecycle
    # =============================

    @abstractmethod
    async def start(self) -> None:
        """
        Start channel runtime.

        This should:
            1. Initialize platform SDK
            2. Establish network connections
            3. Start receiving messages
            4. Block until stopped
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop channel runtime and cleanup all resources.

        This should:
            - Close network connections
            - Terminate background tasks
            - Flush buffers
        """
        ...

    # =============================
    # Outbound
    # =============================

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """
        Send outbound message to platform.

        Contract:
            - Must be non-blocking
            - Must handle retry / exception isolation internally
        """
        ...

    # =============================
    # Inbound handling
    # =============================

    async def handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Unified ingress pipeline for all channels.

        Flow:
            Channel -> handle_message -> MessageBus

        Responsibilities:
            - Permission filtering
            - Message normalization
            - Bus forwarding
        """

        if not self._is_allowed(sender_id):
            self._log_permission_denied(sender_id, chat_id)
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
        )

        await self.bus.publish_inbound(msg)

    # =============================
    # Permission model
    # =============================

    def _is_allowed(self, sender_id: str) -> bool:
        allow_list = getattr(self.config, "allow_from", None)

        # Empty or missing allow list means allow all
        if not allow_list:
            return True

        sender = str(sender_id)

        if sender in allow_list:
            return True

        # Support composite IDs: "xxx|yyy|zzz"
        if "|" in sender:
            return any(part in allow_list for part in sender.split("|") if part)

        return False

    def _log_permission_denied(self, sender_id: str, chat_id: str) -> None:
        logger.warning(
            "Access denied | channel={} sender={} chat={} | "
            "Add sender to allow_from to grant permission",
            self.name,
            sender_id,
            chat_id,
        )

    # =============================
    # Runtime state
    # =============================

    @property
    def is_running(self) -> bool:
        return self._running
