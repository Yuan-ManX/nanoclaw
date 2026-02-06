"""
Groq voice transcription provider for ClawAI Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """
    Voice transcription provider backed by Groq Whisper API.

    Designed for fast, async, and lightweight speech-to-text.
    """

    DEFAULT_MODEL: Final[str] = "whisper-large-v3"
    DEFAULT_TIMEOUT: Final[float] = 60.0
    API_URL: Final[str] = ""

    # ---------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        if not self.api_key:
            logger.warning("Groq API key not configured")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file into text.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self.api_key:
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return ""

        try:
            return await self._request_transcription(path)
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            return ""

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    async def _request_transcription(self, path: Path) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with path.open("rb") as f:
                files = {
                    "file": (path.name, f),
                    "model": (None, self.model),
                }

                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                )

        response.raise_for_status()
        payload = response.json()

        text = payload.get("text", "")
        if not text:
            logger.warning("Groq transcription returned empty result")

        return text
