"""
Helpers for building multimodal (text + image) message content.

Content parts follow OpenAI's chat-completions convention ({"type": "text", ...} /
{"type": "image_url", ...}), which OpenAICompatibleClient passes straight through to
the underlying SDK unchanged - so this covers OpenAI's own vision models and any
OpenAI-compatible server (vLLM, DashScope compatible-mode, OpenRouter, ...), which in
practice is how Qwen-VL gets served. AnthropicClient is not wired up for this - Claude
uses a different content-block schema, and nothing in this project currently needs it.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

_MIME_TYPES = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
}


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = _MIME_TYPES.get(path.suffix.lower(), "jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(image_path: str | Path) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}
