"""Builds the right LLMClient for a provider name. This is the package's single entry point."""

from .config import REGISTRY
from .base import LLMClient
from .exceptions import ConfigurationError, ProviderNotFoundError
from .providers.anthropic_client import AnthropicClient
from .providers.openai_compatible import OpenAICompatibleClient


def get_client(provider: str, *, base_url: str | None = None, api_key: str | None = None) -> LLMClient:
    """
    provider: a REGISTRY name ("openai", "xai", "ollama", "anthropic"), or
              "custom" for any other OpenAI-compatible / self-hosted server.
    base_url: overrides the registry default; required when provider="custom".
    api_key:  overrides the registry's env var lookup.
    """
    if provider == "custom":
        if not base_url:
            raise ConfigurationError('base_url is required when provider="custom"')
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key)

    cfg = REGISTRY.get(provider)
    if cfg is None:
        raise ProviderNotFoundError(f"Unknown provider '{provider}'. Known: {list(REGISTRY)}, or 'custom'.")

    if cfg.kind == "anthropic":
        return AnthropicClient(api_key=api_key, env_var=cfg.env_var)

    return OpenAICompatibleClient(base_url=base_url or cfg.base_url, api_key=api_key, env_var=cfg.env_var)
