"""
Configuration loading and persistence utilities.

Design goals:
    - Deterministic loading & fallback
    - Forward-compatible migration support
    - Strict schema validation
    - Stable persistence format (camelCase on disk, snake_case in memory)
    - Observability-first logging
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from clawai.config.schema import Config
from clawai.utils.helpers import get_data_path


# =============================
# Paths
# =============================

def get_config_path() -> Path:
    """
    Return default configuration file path.

    Default:
        ~/.clawai/config.json
    """
    return Path.home() / ".clawai" / "config.json"


def get_data_dir() -> Path:
    """
    Return clawai runtime data directory.
    """
    return get_data_path()


# =============================
# Load & Save
# =============================

def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from disk or fallback to defaults.

    Flow:
        1. Read raw JSON
        2. Migrate legacy schema
        3. camelCase → snake_case
        4. Pydantic validation

    Args:
        config_path: Optional explicit path override.

    Returns:
        Validated Config object.
    """
    path = config_path or get_config_path()

    if not path.exists():
        logger.warning("Config file not found, using defaults | path={}", path)
        return Config()

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        migrated = _migrate_config(raw)
        normalized = convert_keys(migrated)

        config = Config.model_validate(normalized)

        logger.success("Config loaded successfully | path={}", path)
        return config

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config | path={} err={}", path, e)

    except Exception as e:
        logger.exception("Failed to load config | path={} err={}", path, e)

    logger.warning("Falling back to default configuration")
    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Persist configuration to disk.

    Behavior:
        - snake_case → camelCase
        - Pretty JSON formatting
        - Atomic overwrite
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = convert_to_camel(config.model_dump())

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.success("Config saved | path={}", path)

    except Exception as e:
        logger.exception("Failed to save config | path={} err={}", path, e)


# =============================
# Migration
# =============================

def _migrate_config(data: dict) -> dict:
    """
    Migrate legacy config schema → latest schema.

    Migration rules:
        - tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    """
    tools = data.get("tools") or {}
    exec_cfg = tools.get("exec") or {}

    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
        logger.info("Migrated legacy config: tools.exec.restrictToWorkspace")

    data["tools"] = tools
    return data


# =============================
# Key Conversion
# =============================

def convert_keys(data: Any) -> Any:
    """
    Convert camelCase → snake_case recursively.
    """
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(x) for x in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """
    Convert snake_case → camelCase recursively.
    """
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(x) for x in data]
    return data


# =============================
# Naming helpers
# =============================

def camel_to_snake(name: str) -> str:
    """
    Convert camelCase → snake_case.

    Example:
        restrictToWorkspace → restrict_to_workspace
    """
    buf = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            buf.append("_")
        buf.append(ch.lower())
    return "".join(buf)


def snake_to_camel(name: str) -> str:
    """
    Convert snake_case → camelCase.

    Example:
        restrict_to_workspace → restrictToWorkspace
    """
    head, *tail = name.split("_")
    return head + "".join(w.capitalize() for w in tail)
