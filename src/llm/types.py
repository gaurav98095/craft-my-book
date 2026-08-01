"""Data transfer objects passed between callers, the factory, and provider clients."""

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


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
