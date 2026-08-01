"""
Registry of built-in providers. Add an entry here to support a new hosted
backend by name; for one-off or self-hosted servers, callers can instead pass
provider="custom" with an explicit base_url to get_client() - no registry
entry needed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    kind: str  # "openai_compatible" | "anthropic"
    base_url: str | None  # None for providers with no fixed endpoint (e.g. anthropic uses its SDK default)
    env_var: str | None  # env var holding the API key; None if no key is required (e.g. local Ollama)


REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai_compatible", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "xai": ProviderConfig("openai_compatible", "https://api.x.ai/v1", "XAI_API_KEY"),
    "ollama": ProviderConfig("openai_compatible", "http://localhost:11434/v1", None),
    "vllm": ProviderConfig("openai_compatible", "http://localhost:8000/v1", None),
    "trtllm": ProviderConfig("openai_compatible", "http://localhost:8000/v1", None),
    "anthropic": ProviderConfig("anthropic", None, "ANTHROPIC_API_KEY"),
}
