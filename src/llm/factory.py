"""
The one place that decides which model backs the whole system.

Every pipeline stage asks `build_llm_client()` for an `LLMClient` and never
imports a provider directly. Switching providers -- local weights,
Anthropic, Groq, a self-hosted vLLM/Ollama/whatever OpenAI-compatible server
-- is a change to `.env`, never to pipeline code.
"""

import os

from .base import LLMClient

# The local provider's default checkpoint and its step-down ladder. If this
# machine cannot pull LLM_MODEL, it tries these next, in order -- but the
# model actually loaded is printed by every stage report, so a downgrade is
# never silent. This ladder is specific to the local provider: an API
# provider that can't serve the requested model should fail clearly, not
# quietly substitute a different one you didn't ask for and didn't budget for.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3.6-27B-Instruct"
LOCAL_MODEL_FALLBACKS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

_DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.6-flash",
}

_BASE_URL_BY_PROVIDER = {
    "groq": "https://api.groq.com/openai/v1",
    # Gemini's OpenAI-compatible endpoint -- same request/response shape as
    # OpenAI, so it needs no provider class of its own.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

_API_KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_LOCAL_PROVIDER_NAMES = {"local", "huggingface", "transformers"}
_OPENAI_SHAPED_PROVIDERS = {"openai", "groq", "gemini", "vllm", "openai_compatible"}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def build_llm_client(logger=None) -> LLMClient:
    """
    Construct the LLMClient named by LLM_PROVIDER (default: "local").

        LLM_PROVIDER=local              a transformers checkpoint, in-process
        LLM_PROVIDER=anthropic          Claude, via the Anthropic API
        LLM_PROVIDER=openai             GPT models, via the OpenAI API
        LLM_PROVIDER=groq               any model Groq hosts
        LLM_PROVIDER=gemini             Gemini, via its OpenAI-compatible endpoint
        LLM_PROVIDER=vllm               a self-hosted vLLM server
        LLM_PROVIDER=openai_compatible  anything else speaking that API shape
                                         (Ollama, LM Studio, TGI, ...)

    See .env.example for the full set of variables each provider reads.
    """
    provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
    max_image_side = int(os.getenv("MAX_IMAGE_SIDE", "1280"))

    if provider in _LOCAL_PROVIDER_NAMES:
        from .providers.local_transformers import LocalTransformersClient
        return LocalTransformersClient(
            model_id=os.getenv("LLM_MODEL", DEFAULT_LOCAL_MODEL),
            fallback_ids=LOCAL_MODEL_FALLBACKS,
            dtype=os.getenv("LOCAL_MODEL_DTYPE", "bfloat16"),
            use_flash_attention=_env_bool("USE_FLASH_ATTENTION", False),
            device_map=os.getenv("DEVICE_MAP", "auto"),
            hf_token=os.getenv("HF_TOKEN") or None,
            max_image_side=max_image_side,
            logger=logger,
        )

    if provider == "anthropic":
        from .providers.anthropic_client import AnthropicClient
        return AnthropicClient(
            model_id=os.getenv("LLM_MODEL", _DEFAULT_MODEL_BY_PROVIDER["anthropic"]),
            api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY") or None,
            max_image_side=max_image_side,
        )

    if provider in _OPENAI_SHAPED_PROVIDERS:
        from .providers.openai_compatible import OpenAICompatibleClient

        api_key = (os.getenv(_API_KEY_ENV_BY_PROVIDER.get(provider, ""), "")
                  or os.getenv("LLM_API_KEY") or "")
        base_url = os.getenv("LLM_BASE_URL") or _BASE_URL_BY_PROVIDER.get(provider)
        if provider in ("vllm", "openai_compatible") and not base_url:
            raise RuntimeError(
                f"LLM_PROVIDER={provider} needs LLM_BASE_URL set to the server's "
                f"OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")

        model_id = os.getenv("LLM_MODEL", _DEFAULT_MODEL_BY_PROVIDER.get(provider, ""))
        if not model_id:
            raise RuntimeError(f"LLM_PROVIDER={provider} needs LLM_MODEL set — there "
                              f"is no default model for a self-hosted server.")

        return OpenAICompatibleClient(
            model_id=model_id, api_key=api_key, base_url=base_url,
            max_image_side=max_image_side,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected one of: local, anthropic, "
        f"openai, groq, gemini, vllm, openai_compatible.")
