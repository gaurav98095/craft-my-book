"""
Any server that speaks the OpenAI chat-completions API, behind the shared
LLMClient interface.

This one class covers more ground than its name suggests: OpenAI itself,
Groq (OpenAI-compatible endpoint), and a self-hosted vLLM / Ollama / LM
Studio / text-generation-webui / TGI server -- anything exposing
`/chat/completions` in the OpenAI shape needs nothing but a `base_url`,
an `api_key` (or none, for most local servers), and a `model` id.
"""

import re
from typing import Any, Dict, List, Optional

from ..base import LLMClient, ContentBlock, encode_image_base64


def _is_token_param_error(exc: Exception, param_name: str) -> bool:
    """
    OpenAI renamed `max_tokens` to `max_completion_tokens` for its newer
    models but not for older ones, and it is not documented anywhere which
    is which -- the server tells you by rejecting the wrong one, by name, in
    a 400. Self-hosted/other OpenAI-compatible servers vary the same way.

    The rejection message names BOTH parameters -- the one that failed and
    the one to use instead ("'max_tokens' is not supported ... Use
    'max_completion_tokens' instead") -- so a plain substring check on
    `param_name` matches either one. This checks specifically that
    `param_name` is the one reported as unsupported, not the suggested
    replacement mentioned alongside it.
    """
    text = str(exc)
    quoted = re.escape(param_name)
    return bool(
        re.search(rf"['\"]?{quoted}['\"]?\s+is not supported", text, re.IGNORECASE)
        or re.search(rf"Unsupported parameter:\s*['\"]?{quoted}\b", text, re.IGNORECASE)
    )


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """
    Some newer models (OpenAI's reasoning family, and evidently this one
    too) accept only the default temperature and reject anything else --
    "'temperature' does not support 0.2 with this model. Only the default
    (1) value is supported." Detected the same way as the token-param
    rename: by the server's own rejection, not guessed up front.
    """
    text = str(exc)
    return bool(re.search(r"['\"]?temperature['\"]?\s+does not support", text, re.IGNORECASE))


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
        # Which token-limit parameter this server wants. `max_tokens` is the
        # long-standing name and what Gemini/Groq/most self-hosted servers'
        # OpenAI-compatible endpoints still expect; only some newer OpenAI
        # models require the renamed `max_completion_tokens`. Corrected on
        # the first call that gets rejected, from whichever name the server
        # names in its error, and cached for every call after -- so a
        # mismatch costs one extra round trip per client instance, not one
        # per call.
        self._token_param = "max_tokens"
        # Some models accept only their default temperature (1) and reject
        # any explicit value -- also discovered from the server's rejection,
        # not guessed up front. True means "send it"; corrected to False on
        # the first call that gets rejected for it, then never sent again.
        self._send_temperature = True
        # Most self-hosted servers ignore the key entirely, but the SDK
        # requires a non-empty string.
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    def _build_kwargs(self, max_tokens: int, temperature: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {self._token_param: max_tokens}
        if self._send_temperature:
            kwargs["temperature"] = temperature
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return kwargs

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

        # The server reports one rejected parameter per response, not every
        # one that's wrong -- a model that dislikes both the token-limit
        # param name AND a custom temperature only says so one at a time.
        # Loop until an attempt teaches us nothing new: each correctable
        # rejection is learned and cached on `self` (see __init__), so a
        # second quirk on the same call still gets fixed, and every call
        # after this one on this client goes straight through.
        last_exc: Optional[Exception] = None
        for _ in range(4):  # 1 original guess + headroom for every known quirk
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id, messages=messages,
                    **self._build_kwargs(max_tokens, temperature))
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_exc = exc
                corrected = False
                if _is_token_param_error(exc, self._token_param):
                    self._token_param = (
                        "max_tokens" if self._token_param == "max_completion_tokens"
                        else "max_completion_tokens")
                    corrected = True
                if self._send_temperature and _is_unsupported_temperature_error(exc):
                    self._send_temperature = False
                    corrected = True
                if not corrected:
                    raise
        raise last_exc
