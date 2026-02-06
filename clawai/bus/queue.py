"""
Async message bus for decoupled channel-agent communication.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, DefaultDict
from collections import defaultdict

from loguru import logger

from clawai.bus.events import InboundMessage, OutboundMessage


OutboundCallback = Callable[[OutboundMessage], Awaitable[None]]


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Architecture:
        Channels -> inbound queue -> agent core -> outbound queue -> dispatch -> channels

    Design goals:
        - Fully async
        - Backpressure safe
        - Channel-agnostic
        - Fault isolated dispatch
    """

    # ---------------------------------------------------------------------

    def __init__(
        self,
        inbound_size: int = 0,
        outbound_size: int = 0,
    ):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(
            maxsize=inbound_size
        )
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(
            maxsize=outbound_size
        )

        self._subscribers: DefaultDict[str, list[OutboundCallback]] = defaultdict(list)
        self._running = asyncio.Event()

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    async def start(self) -> None:
        """Start outbound dispatcher loop."""
        if self._running.is_set():
            return
        self._running.set()
        asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop outbound dispatcher loop gracefully."""
        self._running.clear()

    # ---------------------------------------------------------------------
    # Inbound
    # ---------------------------------------------------------------------

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from channel into the agent pipeline."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume next inbound message (blocking)."""
        return await self.inbound.get()

    # ---------------------------------------------------------------------
    # Outbound
    # ---------------------------------------------------------------------

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish agent response to outbound pipeline."""
        await self.outbound.put(msg)

    def subscribe(
        self,
        channel: str,
        callback: OutboundCallback,
    ) -> None:
        """Subscribe a channel handler to outbound messages."""
        self._subscribers[channel].append(callback)

    # ---------------------------------------------------------------------
    # Dispatcher
    # ---------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """
        Dispatch outbound messages to subscribed channels.

        Runs forever until stop() is called.
        """
        logger.info("MessageBus dispatcher started")

        while self._running.is_set():
            try:
                msg = await self.outbound.get()

                callbacks = self._subscribers.get(msg.channel)
                if not callbacks:
                    logger.warning(
                        f"No outbound subscriber for channel: {msg.channel}"
                    )
                    continue

                await self._fanout(msg, callbacks)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"MessageBus dispatch error: {e}")

        logger.info("MessageBus dispatcher stopped")

    async def _fanout(
        self,
        msg: OutboundMessage,
        callbacks: list[OutboundCallback],
    ) -> None:
        """
        Deliver message to all subscribers safely.

        Guarantees:
            - One channel failure does NOT affect others
            - Full async parallel dispatch
        """
        tasks = [self._safe_call(cb, msg) for cb in callbacks]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_call(
        self,
        callback: OutboundCallback,
        msg: OutboundMessage,
    ) -> None:
        try:
            await callback(msg)
        except Exception:
            logger.exception(
                f"Outbound callback failed [{msg.channel}]: {callback}"
            )

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
