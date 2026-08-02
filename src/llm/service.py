"""Convenience helpers on top of LLMClient - most callers only need these two functions."""

from .base import LLMClient
from .types import ChatResponse, Message


def chat(client: LLMClient, model: str, prompt: str, **kwargs) -> str:
    """Send a single user prompt, return the assistant's reply text."""
    response = client.chat(model, [Message(role="user", content=prompt)], **kwargs)
    return response.content


def chat_with_messages(
    client: LLMClient, model: str, messages: list[Message], **kwargs
) -> ChatResponse:
    """Send a full message history (e.g. including a system prompt), return the full response."""
    return client.chat(model, messages, **kwargs)
