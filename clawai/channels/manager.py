"""Channel runtime orchestrator for ClawAI Agent."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from clawai.bus.events import OutboundMessage
from clawai.bus.queue import MessageBus
from clawai.channels.base import BaseChannel
from clawai.config.schema import Config


class ChannelManager:
    """
    Channel runtime orchestrator.

    Responsibilities:
        - Channel lifecycle orchestration
        - Outbound message dispatch loop
        - Channel registry & status monitoring
    """

    def __init__(self, config: Config, bus: MessageBus):
        self.config = config
        self.bus = bus

        self.channels: Dict[str, BaseChannel] = {}

        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running: bool = False

        self._init_channels()

    # ==========================================================
    # Channel initialization
    # ==========================================================

    def _init_channels(self) -> None:
        """Initialize all enabled channels from config."""
        self._register("telegram", self._init_telegram)
        self._register("whatsapp", self._init_whatsapp)
        self._register("discord", self._init_discord)
        self._register("feishu", self._init_feishu)

        if not self.channels:
            logger.warning("No channels enabled")

    def _register(self, name: str, factory) -> None:
        try:
            channel = factory()
            if channel:
                self.channels[name] = channel
                logger.info("Channel enabled: {}", name)
        except Exception as e:
            logger.warning("Channel {} init failed: {}", name, e)

    def _init_telegram(self) -> Optional[BaseChannel]:
        if not self.config.channels.telegram.enabled:
            return None
        from clawai.channels.telegram import TelegramChannel
        return TelegramChannel(
            self.config.channels.telegram,
            self.bus,
            groq_api_key=self.config.providers.groq.api_key,
        )

    def _init_whatsapp(self) -> Optional[BaseChannel]:
        if not self.config.channels.whatsapp.enabled:
            return None
        from clawai.channels.whatsapp import WhatsAppChannel
        return WhatsAppChannel(self.config.channels.whatsapp, self.bus)

    def _init_discord(self) -> Optional[BaseChannel]:
        if not self.config.channels.discord.enabled:
            return None
        from clawai.channels.discord import DiscordChannel
        return DiscordChannel(self.config.channels.discord, self.bus)

    def _init_feishu(self) -> Optional[BaseChannel]:
        if not self.config.channels.feishu.enabled:
            return None
        from clawai.channels.feishu import FeishuChannel
        return FeishuChannel(self.config.channels.feishu, self.bus)

    # ==========================================================
    # Lifecycle orchestration
    # ==========================================================

    async def start(self) -> None:
        """Start all channels and dispatcher."""
        if not self.channels:
            return

        self._running = True

        logger.info("Starting ChannelManager ...")

        self._dispatcher_task = asyncio.create_task(
            self._dispatch_loop(), name="channel-dispatcher"
        )

        for name, channel in self.channels.items():
            logger.info("Starting channel: {}", name)
            asyncio.create_task(channel.start(), name=f"channel-{name}")

    async def stop(self) -> None:
        """Gracefully shutdown all channels and dispatcher."""
        if not self._running:
            return

        self._running = False

        logger.info("Stopping ChannelManager ...")

        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Channel stopped: {}", name)
            except Exception as e:
                logger.error("Channel stop failed: {} | {}", name, e)

    # ==========================================================
    # Dispatcher
    # ==========================================================

    async def _dispatch_loop(self) -> None:
        """Outbound message dispatch loop."""
        logger.info("Outbound dispatcher started")

        while self._running:
            try:
                msg: OutboundMessage = await asyncio.wait_for(
                    self.bus.consume_outbound(),
                    timeout=1.0,
                )

                await self._dispatch_message(msg)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Outbound dispatcher error: {}", e)

        logger.info("Outbound dispatcher stopped")

    async def _dispatch_message(self, msg: OutboundMessage) -> None:
        channel = self.channels.get(msg.channel)

        if not channel:
            logger.warning("Unknown channel: {}", msg.channel)
            return

        try:
            await channel.send(msg)
        except Exception as e:
            logger.error("Send failed | channel={} error={}", msg.channel, e)

    # ==========================================================
    # Query API
    # ==========================================================

    def get_channel(self, name: str) -> Optional[BaseChannel]:
        return self.channels.get(name)

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get_status(self) -> Dict[str, Any]:
        return {
            name: {
                "enabled": True,
                "running": channel.is_running,
            }
            for name, channel in self.channels.items()
        }
