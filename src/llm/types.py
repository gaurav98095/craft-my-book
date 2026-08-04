"""Data transfer objects passed between callers, the factory, and provider clients."""

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    # str for plain text; a list of OpenAI-style content parts (see llm.multimodal's
    # text_part/image_part) for multimodal messages.
    content: str | list[dict[str, Any]]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # original SDK response object, kept as an escape hatch
