"""Anthropic's Messages API, behind the shared LLMClient interface."""

from typing import Any, Dict, List, Optional

from ..base import LLMClient, ContentBlock, encode_image_base64


class AnthropicClient(LLMClient):
    """Claude models via the `anthropic` SDK."""

    def __init__(self, model_id: str, api_key: Optional[str] = None,
                max_image_side: int = 1_568):
        super().__init__()
        import anthropic  # optional dependency; only needed for this provider

        self.model_id = model_id
        self.max_image_side = max_image_side
        # `api_key=None` falls through to the SDK's own ANTHROPIC_API_KEY lookup.
        self._client = anthropic.Anthropic(api_key=api_key or None)

    def _chat(self, content: List[ContentBlock], system: Optional[str],
             max_tokens: int, temperature: float) -> str:
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if block.get("type") == "image":
                data, mime = encode_image_base64(block["image"], self.max_image_side)
                blocks.append({"type": "image",
                               "source": {"type": "base64", "media_type": mime,
                                         "data": data}})
            else:
                blocks.append({"type": "text", "text": block.get("text", "")})

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": blocks}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)
        return "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
