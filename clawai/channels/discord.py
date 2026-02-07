"""Discord channel implementation using Gateway WebSocket + REST API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import websockets
from loguru import logger

from clawai.bus.events import OutboundMessage
from clawai.bus.queue import MessageBus
from clawai.channels.base import BaseChannel
from clawai.config.schema import DiscordConfig


DISCORD_API_BASE = ""
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024  # 20MB


class DiscordChannel(BaseChannel):
    """
    Discord Gateway + REST dual-stack channel implementation.

    Architecture:
        - Gateway WebSocket: inbound event stream
        - REST API: outbound messaging & typing indicator
        - Heartbeat task: keepalive
        - Auto reconnect loop
    """

    name = "discord"

    def __init__(self, config: DiscordConfig, bus: MessageBus):
        super().__init__(config, bus)

        self.config: DiscordConfig = config

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._seq: Optional[int] = None

        self._http: Optional[httpx.AsyncClient] = None

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._typing_tasks: Dict[str, asyncio.Task] = {}

    # ==========================================================
    # Lifecycle
    # ==========================================================

    async def start(self) -> None:
        if not self.config.token:
            logger.error("Discord token not configured")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30.0)

        logger.info("Discord channel starting...")

        while self._running:
            try:
                await self._connect_gateway()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Gateway error: {}", e)

            if self._running:
                logger.info("Reconnecting to Discord in 5s...")
                await asyncio.sleep(5)

        logger.info("Discord channel stopped")

    async def stop(self) -> None:
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._http:
            await self._http.aclose()
            self._http = None

    # ==========================================================
    # Gateway
    # ==========================================================

    async def _connect_gateway(self) -> None:
        logger.info("Connecting to Discord Gateway...")

        async with websockets.connect(self.config.gateway_url) as ws:
            self._ws = ws
            self._seq = None

            await self._gateway_loop()

    async def _gateway_loop(self) -> None:
        assert self._ws

        async for raw in self._ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid gateway JSON: {}", raw[:200])
                continue

            op = data.get("op")
            event = data.get("t")
            seq = data.get("s")
            payload = data.get("d")

            if seq is not None:
                self._seq = seq

            if op == 10:
                await self._on_hello(payload)
            elif op == 0:
                await self._on_dispatch(event, payload)
            elif op == 7:
                logger.info("Gateway reconnect requested")
                break
            elif op == 9:
                logger.warning("Invalid session")
                break

    async def _on_hello(self, payload: dict) -> None:
        interval_ms = payload.get("heartbeat_interval", 45000)
        await self._start_heartbeat(interval_ms / 1000)
        await self._identify()

    async def _identify(self) -> None:
        if not self._ws:
            return

        payload = {
            "op": 2,
            "d": {
                "token": self.config.token,
                "intents": self.config.intents,
                "properties": {
                    "os": "clawai",
                    "browser": "clawai",
                    "device": "clawai",
                },
            },
        }

        await self._ws.send(json.dumps(payload))

    async def _start_heartbeat(self, interval: float) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        async def loop():
            while self._running and self._ws:
                try:
                    await self._ws.send(json.dumps({"op": 1, "d": self._seq}))
                except Exception as e:
                    logger.warning("Heartbeat failed: {}", e)
                    break
                await asyncio.sleep(interval)

        self._heartbeat_task = asyncio.create_task(loop())

    async def _on_dispatch(self, event: str, payload: dict) -> None:
        if event == "READY":
            logger.info("Discord READY")
        elif event == "MESSAGE_CREATE":
            await self._handle_message_create(payload)

    # ==========================================================
    # Inbound handling
    # ==========================================================

    async def _handle_message_create(self, payload: dict[str, Any]) -> None:
        author = payload.get("author") or {}
        if author.get("bot"):
            return

        sender_id = str(author.get("id", ""))
        channel_id = str(payload.get("channel_id", ""))
        content = payload.get("content") or ""

        if not sender_id or not channel_id:
            return

        content_parts = [content] if content else []
        media_paths: list[str] = []

        await self._download_attachments(payload, content_parts, media_paths)

        reply_to = (payload.get("referenced_message") or {}).get("id")

        await self._start_typing(channel_id)

        await self.handle_message(
            sender_id=sender_id,
            chat_id=channel_id,
            content="\n".join(p for p in content_parts if p) or "[empty]",
            media=media_paths,
            metadata={
                "message_id": str(payload.get("id", "")),
                "guild_id": payload.get("guild_id"),
                "reply_to": reply_to,
            },
        )

    async def _download_attachments(
        self,
        payload: dict,
        content_parts: list[str],
        media_paths: list[str],
    ) -> None:
        if not self._http:
            return

        media_dir = Path.home() / ".clawai" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        for attachment in payload.get("attachments") or []:
            url = attachment.get("url")
            filename = attachment.get("filename") or "file"
            size = attachment.get("size") or 0

            if not url:
                continue

            if size > MAX_ATTACHMENT_BYTES:
                content_parts.append(f"[attachment: {filename} too large]")
                continue

            try:
                resp = await self._http.get(url)
                resp.raise_for_status()

                file_path = media_dir / f"{attachment.get('id','file')}_{filename}"
                file_path.write_bytes(resp.content)

                media_paths.append(str(file_path))
                content_parts.append(f"[attachment: {file_path.name}]")

            except Exception as e:
                logger.warning("Attachment download failed: {}", e)
                content_parts.append(f"[attachment: {filename} failed]")

    # ==========================================================
    # Outbound
    # ==========================================================

    async def send(self, msg: OutboundMessage) -> None:
        if not self._http:
            logger.warning("Discord HTTP client not ready")
            return

        url = f"{DISCORD_API_BASE}/channels/{msg.chat_id}/messages"
        headers = {"Authorization": f"Bot {self.config.token}"}

        payload: dict[str, Any] = {"content": msg.content}

        if msg.reply_to:
            payload["message_reference"] = {"message_id": msg.reply_to}
            payload["allowed_mentions"] = {"replied_user": False}

        for attempt in range(3):
            try:
                resp = await self._http.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    retry = float(resp.json().get("retry_after", 1.0))
                    await asyncio.sleep(retry)
                    continue

                resp.raise_for_status()
                return

            except Exception as e:
                if attempt == 2:
                    logger.error("Discord send failed: {}", e)
                else:
                    await asyncio.sleep(1)
            finally:
                await self._stop_typing(msg.chat_id)

    # ==========================================================
    # Typing indicator
    # ==========================================================

    async def _start_typing(self, channel_id: str) -> None:
        await self._stop_typing(channel_id)

        async def loop():
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/typing"
            headers = {"Authorization": f"Bot {self.config.token}"}
            while self._running:
                try:
                    await self._http.post(url, headers=headers)
                except Exception:
                    pass
                await asyncio.sleep(8)

        self._typing_tasks[channel_id] = asyncio.create_task(loop())

    async def _stop_typing(self, channel_id: str) -> None:
        task = self._typing_tasks.pop(channel_id, None)
        if task:
            task.cancel()
  
