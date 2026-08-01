"""Client for Anthropic's native Messages API (not OpenAI-compatible)."""

import os

from anthropic import Anthropic

from ..base import LLMClient
from ..exceptions import MissingCredentialsError
from ..types import ChatResponse, Message, Usage


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str | None = None, env_var: str = "ANTHROPIC_API_KEY"):
        resolved_key = api_key or os.environ.get(env_var)
        if not resolved_key:
            raise MissingCredentialsError(f"{env_var} is not set")
        self._client = Anthropic(api_key=resolved_key)

    def chat(self, model: str, messages: list[Message], **kwargs) -> ChatResponse:
        system = next((m.content for m in messages if m.role == "system"), None)
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        max_tokens = kwargs.pop("max_tokens", 1024)

        response = self._client.messages.create(
            model=model,
            system=system,
            messages=turns,
            max_tokens=max_tokens,
            **kwargs,
        )
        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            raw=response,
        )
