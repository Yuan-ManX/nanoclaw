"""
Configuration schema definitions.

Design principles:
    - Explicit structure
    - Predictable defaults
    - Forward-compatible evolution
    - Environment override support
    - Strong typing + validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# =============================
# Channel Configurations
# =============================

class ChannelBaseConfig(BaseModel):
    """Base configuration for all channels."""
    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)


class WhatsAppConfig(ChannelBaseConfig):
    """WhatsApp channel configuration (Node.js bridge)."""
    bridge_url: str = ""


class TelegramConfig(ChannelBaseConfig):
    """Telegram channel configuration."""
    token: str = ""
    proxy: Optional[str] = None


class FeishuConfig(ChannelBaseConfig):
    """Feishu / Lark channel configuration."""
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""


class DiscordConfig(ChannelBaseConfig):
    """Discord channel configuration."""
    token: str = ""
    gateway_url: str = ""
    intents: int = 37377


class ChannelsConfig(BaseModel):
    """Unified channel configuration root."""
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


# =============================
# Agent Runtime Config
# =============================

class AgentDefaults(BaseModel):
    """Agent runtime default parameters."""
    workspace: str = "~/.clawai/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20


class AgentsConfig(BaseModel):
    """Agent configuration root."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


# =============================
# Provider Config
# =============================

class ProviderConfig(BaseModel):
    """LLM provider credentials."""
    api_key: str = ""
    api_base: Optional[str] = None


class ProvidersConfig(BaseModel):
    """Multi-provider configuration."""
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)


# =============================
# Gateway Config
# =============================

class GatewayConfig(BaseModel):
    """Gateway / API server configuration."""
    host: str = "0.0.0.0"
    port: int = 0


# =============================
# Tools Config
# =============================

class WebSearchConfig(BaseModel):
    """Web search tool configuration."""
    api_key: str = ""
    max_results: int = 3


class WebToolsConfig(BaseModel):
    """Web-related tools."""
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    """Shell execution tool configuration."""
    timeout: int = 60


class ToolsConfig(BaseModel):
    """Unified tool configuration root."""
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False


# =============================
# Root Config
# =============================

class Config(BaseSettings):
    """
    Root configuration schema.

    Priority:
        env > config.json > defaults
    """

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    # -------------------------
    # Runtime helpers
    # -------------------------

    @property
    def workspace_path(self) -> Path:
        """Expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    # -------------------------
    # Provider matching
    # -------------------------

    def _match_provider(self, model: Optional[str] = None) -> Optional[ProviderConfig]:
        """
        Match provider based on model naming conventions.
        """
        model = (model or self.agents.defaults.model).lower()

        routing: Dict[str, ProviderConfig] = {
            "openrouter": self.providers.openrouter,
            "deepseek": self.providers.deepseek,
            "anthropic": self.providers.anthropic,
            "claude": self.providers.anthropic,
            "openai": self.providers.openai,
            "gpt": self.providers.openai,
            "gemini": self.providers.gemini,
            "zhipu": self.providers.zhipu,
            "glm": self.providers.zhipu,
            "zai": self.providers.zhipu,
            "groq": self.providers.groq,
            "moonshot": self.providers.moonshot,
            "kimi": self.providers.moonshot,
            "vllm": self.providers.vllm,
        }

        for keyword, provider in routing.items():
            if keyword in model and provider.api_key:
                return provider
        return None

    def get_api_key(self, model: Optional[str] = None) -> Optional[str]:
        """
        Resolve API key based on model routing or fallback order.
        """
        matched = self._match_provider(model)
        if matched:
            return matched.api_key

        for provider in self.providers.__dict__.values():
            if provider.api_key:
                return provider.api_key

        return None

    def get_api_base(self, model: Optional[str] = None) -> Optional[str]:
        """
        Resolve API base URL dynamically.
        """
        model = (model or self.agents.defaults.model).lower()

        if "openrouter" in model:
            return self.providers.openrouter.api_base or "https://openrouter.ai/api/v1"

        if any(k in model for k in ("zhipu", "glm", "zai")):
            return self.providers.zhipu.api_base

        if "vllm" in model:
            return self.providers.vllm.api_base

        return None

    # -------------------------
    # Env integration
    # -------------------------

    class Config:
        env_prefix = "CLAWAI_"
        env_nested_delimiter = "__"
