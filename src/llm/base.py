"""The interface every backend (hosted or self-hosted) must implement."""

from abc import ABC, abstractmethod

from .types import ChatResponse, Message


class LLMClient(ABC):
    @abstractmethod
    def chat(self, model: str, messages: list[Message], **kwargs) -> ChatResponse:
        """Send a message history to `model` and return the assistant's reply."""
