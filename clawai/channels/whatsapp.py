"""
WhatsApp channel implementation using Node.js bridge.

This channel communicates with a Node.js service
via WebSocket to integrate WhatsApp Web protocol into ClawAI Agent runtime.

Architecture:
    WhatsApp Web
          ↓
    Node.js Bridge (WebSocket)
          ↓
    WhatsAppChannel (Python)
          ↓
    MessageBus
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from loguru import logger

from clawai.bus.events import OutboundMessage
from clawai.bus.queue import MessageBus
from clawai.channels.base import BaseChannel
from clawai.config.schema import WhatsAppConfig


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel backed by Node.js WebSocket bridge.

    Responsibilities:
        - Maintain persistent WebSocket connection
        - Translate inbound WhatsApp messages into InboundMessage events
        - Deliver outbound messages reliably to the bridge
        - Handle reconnect & runtime fault tolerance
    """

    name = "whatsapp"

    def __init__(self, config: WhatsAppConfig, bus: MessageBus):
        super().__init__(config, bus)

        self.config: WhatsAppConfig = config
        self._ws: Optional[Any] = None
        self._connected: bool = False
        self._reconnect_interval: int = getattr(config, "reconnect_interval", 5)

    # =============================
    # Lifecycle
    # =============================

    async def start(self) -> None:
        """
        Start WhatsApp channel runtime.

        Behavior:
            - Establish WebSocket connection to Node.js bridge
            - Maintain persistent reconnect loop
            - Consume inbound messages and forward into MessageBus
        """
        import websockets

        bridge_url = self.config.bridge_url
        self._running = True

        logger.info("WhatsApp channel starting | bridge={}", bridge_url)

        while self._running:
            try:
                async with websockets.connect(bridge_url) as ws:
                    self._ws = ws
                    self._connected = True

                    logger.success("WhatsApp bridge connected")

                    async for raw in ws:
                        await self._handle_bridge_message(raw)

            except asyncio.CancelledError:
                logger.warning("WhatsApp channel cancelled")
                break

            except Exception as e:
                self._connected = False
                self._ws = None

                logger.error("WhatsApp bridge error | {}", e)

                if self._running:
                    logger.info(
                        "Reconnecting WhatsApp bridge in {}s...",
                        self._reconnect_interval,
                    )
                    await asyncio.sleep(self._reconnect_interval)

        logger.warning("WhatsApp channel stopped")

    async def stop(self) -> None:
        """
        Stop WhatsApp channel runtime.
        """
        self._running = False
        self._connected = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            finally:
                self._ws = None

        logger.info("WhatsApp channel shutdown complete")

    # =============================
    # Outbound
    # =============================

    async def send(self, msg: OutboundMessage) -> None:
        """
        Send outbound message to WhatsApp.

        Contract:
            - Non-blocking
            - Fail-safe: no crash on send failure
        """
        if not self._connected or not self._ws:
            logger.warning("WhatsApp bridge not connected, drop message")
            return

        payload = {
            "type": "send",
            "to": msg.chat_id,
            "text": msg.content,
        }

        try:
            await self._ws.send(json.dumps(payload))
            logger.debug(
                "WhatsApp outbound sent | chat={} len={}",
                msg.chat_id,
                len(msg.content),
            )
        except Exception as e:
            logger.error("WhatsApp send failed | {}", e)

    # =============================
    # Bridge handling
    # =============================

    async def _handle_bridge_message(self, raw: str) -> None:
        """
        Handle inbound messages from Node.js bridge.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WhatsApp bridge: {}", raw[:200])
            return

        msg_type = data.get("type")

        if msg_type == "message":
            await self._handle_inbound_message(data)

        elif msg_type == "status":
            self._handle_status_update(data)

        elif msg_type == "qr":
            logger.info("WhatsApp QR received – scan in Node.js terminal")

        elif msg_type == "error":
            logger.error("WhatsApp bridge error | {}", data.get("error"))

        else:
            logger.debug("Unknown WhatsApp bridge event: {}", data)

    # =============================
    # Message normalization
    # =============================

    async def _handle_inbound_message(self, data: dict) -> None:
        """
        Normalize WhatsApp inbound event → ClawAI InboundMessage.
        """
        sender = data.get("sender", "")
        content = data.get("content", "")

        # JID format: <phone>@s.whatsapp.net
        sender_id = sender.split("@")[0] if "@" in sender else sender

        if content == "[Voice Message]":
            content = "[Voice message received – transcription unsupported]"

        await self.handle_message(
            sender_id=sender_id,
            chat_id=sender,  # preserve full JID for reply routing
            content=content,
            metadata={
                "message_id": data.get("id"),
                "timestamp": data.get("timestamp"),
                "is_group": data.get("isGroup", False),
            },
        )

    def _handle_status_update(self, data: dict) -> None:
        """
        Process bridge connection status update.
        """
        status = data.get("status")

        logger.info("WhatsApp bridge status | {}", status)

        if status == "connected":
            self._connected = True
        elif status == "disconnected":
            self._connected = False
  
