"""
Client for any backend that implements OpenAI's /v1/chat/completions API.

That covers OpenAI and xAI directly, plus effectively every self-hosted
option: Ollama, vLLM, LM Studio, text-generation-webui, etc - point base_url
at the server and it works the same way.
"""

import os

from openai import OpenAI

from ..base import LLMClient
from ..exceptions import ConfigurationError, MissingCredentialsError
from ..types import ChatResponse, Message, Usage


class OpenAICompatibleClient(LLMClient):
    def __init__(self, base_url: str, api_key: str | None = None, env_var: str | None = None):
        if not base_url:
            raise ConfigurationError("base_url is required for an OpenAI-compatible client")

        resolved_key = api_key or (os.environ.get(env_var) if env_var else None)
        if env_var and not resolved_key:
            raise MissingCredentialsError(f"{env_var} is not set")

        self._client = OpenAI(base_url=base_url, api_key=resolved_key or "not-needed")

    def chat(self, model: str, messages: list[Message], **kwargs) -> ChatResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        usage = response.usage
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
            ),
            raw=response,
        )
