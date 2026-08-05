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
    """OpenAI, Groq, Gemini, vLLM, or any other OpenAI-compatible chat-completions server."""

    def __init__(self, model_id: str, api_key: Optional[str] = None,
                base_url: Optional[str] = None, max_image_side: int = 1_280,
                reasoning_effort: Optional[str] = None):
        super().__init__()
        from openai import OpenAI  # optional dependency; only needed for this provider

        self.model_id = model_id
        self.max_image_side = max_image_side
        # A reasoning-capable model (OpenAI's o-series, Gemini 3.x, ...)
        # spends tokens on hidden thinking before any visible output -- the
        # LLMClient token budgets already account for that, but on a task
        # that genuinely needs no deep reasoning (tag extraction, a title,
        # a yes/no check) it's pure latency and cost with no quality benefit.
        # Unset by default: only takes effect if you opt in via
        # LLM_REASONING_EFFORT, and a server that doesn't recognise the
        # field in extra_body simply ignores it.
        self.reasoning_effort = reasoning_effort
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

        kwargs: Dict[str, Any] = {}
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        response = self._client.chat.completions.create(
            model=self.model_id, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        return (response.choices[0].message.content or "").strip()
