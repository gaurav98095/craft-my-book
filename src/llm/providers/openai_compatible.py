"""
Any server that speaks the OpenAI chat-completions API, behind the shared
LLMClient interface.

This one class covers more ground than its name suggests: OpenAI itself,
Groq (OpenAI-compatible endpoint), and a self-hosted vLLM / Ollama / LM
Studio / text-generation-webui / TGI server -- anything exposing
`/chat/completions` in the OpenAI shape needs nothing but a `base_url`,
an `api_key` (or none, for most local servers), and a `model` id.
"""

from typing import Any, Dict, List, Optional

from ..base import LLMClient, ContentBlock, encode_image_base64


class OpenAICompatibleClient(LLMClient):
    """OpenAI, Groq, vLLM, or any other OpenAI-compatible chat-completions server."""

    def __init__(self, model_id: str, api_key: Optional[str] = None,
                base_url: Optional[str] = None, max_image_side: int = 1_280):
        super().__init__()
        from openai import OpenAI  # optional dependency; only needed for this provider

        self.model_id = model_id
        self.max_image_side = max_image_side
        # Most self-hosted servers ignore the key entirely, but the SDK
        # requires a non-empty string.
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    def _chat(self, content: List[ContentBlock], system: Optional[str],
             max_tokens: int, temperature: float) -> str:
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if block.get("type") == "image":
                data, mime = encode_image_base64(block["image"], self.max_image_side)
                blocks.append({"type": "image_url",
                               "image_url": {"url": f"data:{mime};base64,{data}"}})
            else:
                blocks.append({"type": "text", "text": block.get("text", "")})

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": blocks})

        response = self._client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()
