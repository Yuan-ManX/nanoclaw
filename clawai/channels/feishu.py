"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import json
import threading
from collections import OrderedDict
from typing import Any, Optional

from loguru import logger

from clawai.bus.events import OutboundMessage
from clawai.bus.queue import MessageBus
from clawai.channels.base import BaseChannel
from clawai.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        Emoji,
        P2ImMessageReceiveV1,
    )
    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None
    Emoji = None


MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark channel using WebSocket long connection.

    Architecture:
        WS Thread -> Sync Handler -> asyncio loop -> MessageBus
    """

    name = "feishu"

    def __init__(self, config: FeishuConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config = config

        self._client: Optional[Any] = None
        self._ws_client: Optional[Any] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._processed_ids: OrderedDict[str, None] = OrderedDict()
        self._dedup_limit = 1000
        self._dedup_trim = 500

    # =============================
    # Lifecycle
    # =============================

    async def start(self) -> None:
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id/app_secret missing")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        self._init_clients()
        self._start_ws_thread()

        logger.info("Feishu channel started (WebSocket long connection)")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False

        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as e:
                logger.warning(f"Stopping ws client failed: {e}")

        logger.info("Feishu channel stopped")

    # =============================
    # Init
    # =============================

    def _init_clients(self) -> None:
        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        handler = (
            lark.EventDispatcherHandler.builder(
                self.config.encrypt_key or "",
                self.config.verification_token or "",
            )
            .register_p2_im_message_receive_v1(self._on_message_sync)
            .build()
        )

        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

    def _start_ws_thread(self) -> None:
        def runner():
            try:
                self._ws_client.start()
            except Exception as e:
                logger.error(f"Feishu WS runtime error: {e}")

        self._ws_thread = threading.Thread(target=runner, daemon=True)
        self._ws_thread.start()

    # =============================
    # Sending
    # =============================

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client:
            logger.warning("Feishu client not ready")
            return

        try:
            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"

            content = json.dumps({"text": msg.content})

            req = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )

            resp = self._client.im.v1.message.create(req)

            if not resp.success():
                logger.error(
                    f"Feishu send failed: code={resp.code}, msg={resp.msg}, log_id={resp.get_log_id()}"
                )
            else:
                logger.debug(f"Feishu → {msg.chat_id}: {msg.content}")

        except Exception as e:
            logger.exception(f"Feishu send exception: {e}")

    # =============================
    # Incoming Handling
    # =============================

    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        if not self._loop or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender

            if self._is_duplicate(message.message_id):
                return

            if sender.sender_type == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type

            content = self._parse_message_content(msg_type, message.content)
            if not content:
                return

            reply_to = chat_id if chat_type == "group" else sender_id

            await self._add_reaction(message.message_id)

            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "platform": "feishu",
                    "message_id": message.message_id,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                },
            )

        except Exception as e:
            logger.exception(f"Feishu message handling error: {e}")

    # =============================
    # Helpers
    # =============================

    def _is_duplicate(self, message_id: str) -> bool:
        if message_id in self._processed_ids:
            return True

        self._processed_ids[message_id] = None

        if len(self._processed_ids) > self._dedup_limit:
            for _ in range(self._dedup_trim):
                self._processed_ids.popitem(last=False)

        return False

    def _parse_message_content(self, msg_type: str, raw: str) -> str:
        if msg_type == "text":
            try:
                return json.loads(raw).get("text", "")
            except json.JSONDecodeError:
                return raw or ""
        return MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

    async def _add_reaction(self, message_id: str, emoji: str = "THUMBSUP") -> None:
        if not self._client or not Emoji:
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._add_reaction_sync, message_id, emoji)

    def _add_reaction_sync(self, message_id: str, emoji: str) -> None:
        try:
            req = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji).build())
                    .build()
                )
                .build()
            )
            resp = self._client.im.v1.message_reaction.create(req)

            if not resp.success():
                logger.warning(f"Add reaction failed: {resp.code} {resp.msg}")

        except Exception as e:
            logger.warning(f"Add reaction exception: {e}")
  
