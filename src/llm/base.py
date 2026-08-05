"""
The provider-agnostic model interface.

Every pipeline stage that touches a language model — Stage 3's figure
description, Pipeline B's tagging/clustering, Pipeline C's five agents —
talks to an `LLMClient`, never to a specific provider's SDK. Swapping the
model backing the whole system (a local transformers checkpoint, Anthropic,
Groq, a self-hosted vLLM server, anything OpenAI-compatible) is a change to
`.env`, not to any of that call-site code. See `factory.build_llm_client`.

A provider only has to implement `_chat`. `generate` and `generate_structured`
— including the JSON-repair retry loop and the transient-error retry below —
are shared here, once, so every provider gets them for free and behaves
identically under retry.
"""

import os
import re
import json
import time
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

_log = logging.getLogger("src.llm")

ContentBlock = Dict[str, Any]


def _is_retryable_error(exc: Exception) -> bool:
    """
    A rate limit or a provider's own "high demand, try again" is not a bug in
    the request -- it clears on its own. A bad API key or an unknown model
    never will, no matter how many times it's retried. Every provider SDK
    used here (openai, anthropic) exposes `.status_code` on its HTTP errors,
    so this needs no provider-specific imports to tell the two apart.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or 500 <= status < 600
    # network-level failures (timeouts, connection resets) carry no status
    # code in any of these SDKs, but are transient by nature.
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """
    Prefer the SERVER'S OWN guidance over our fixed backoff schedule -- a
    generic exponential ladder is a guess; a 429 that says "retry in 51s" or
    "retry in 4s" is not. Checked in order: the standard HTTP `Retry-After`
    header (both the openai and anthropic SDKs expose the raw httpx
    response), then a `retryDelay`-shaped field some providers (Gemini)
    embed in the error body text instead.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        value = headers.get("retry-after")
        if value:
            try:
                return float(value)
            except ValueError:
                pass

    text = str(exc)
    match = (re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
             or re.search(r"retry in (\d+(?:\.\d+)?)s", text, re.IGNORECASE))
    return float(match.group(1)) if match else None


def extract_json_object(raw: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Pull the first balanced JSON object out of a model reply.

    Models wrap JSON in prose, in ``` fences, or both. Naive `json.loads`
    fails on all of it, and a regex for `\\{.*\\}` breaks on nested braces and
    on any brace inside a string. So we scan for the first '{' and walk
    forward counting depth, skipping over string literals and their escapes.

    Returns (object, None) or (None, reason). The reason is fed back to the
    model verbatim on the retry -- an error message is a far better repair
    instruction than "try again".
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return None, "the reply contained no '{'"

    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1]), None
                    except json.JSONDecodeError as exc:
                        return None, f"JSON was malformed: {exc.msg} at char {exc.pos}"
    return None, "braces never balanced - the reply was probably truncated"


def load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """Accept a path or an already-open PIL image; always return one that's loaded."""
    if isinstance(image, Image.Image):
        return image
    img = Image.open(str(image))
    img.load()
    return img


def fit_image(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest edge is at most `max_side`. Every provider pays
    per pixel in one way or another, and no vision encoder needs more than this."""
    side = max(image.size)
    if side <= max_side:
        return image
    scale = max_side / side
    return image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.LANCZOS,
    )


def encode_image_base64(image: Union[str, Path, Image.Image], max_side: int) -> Tuple[str, str]:
    """Returns (base64_data, mime_type) for an API that wants images inline."""
    import base64

    img = fit_image(load_image(image).convert("RGB"), max_side)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


class LLMClient(ABC):
    """
    The single interface every pipeline stage generates through.

    `model_id` is set by the implementation once it knows exactly which
    model answered -- a local loader may step down a fallback ladder, so the
    id actually running is not always the id requested. Every stage report
    prints `model_id`, so a downgrade is never silent.
    """

    model_id: str

    def __init__(self) -> None:
        self.calls = 0
        self.structured_repairs = 0
        self.structured_failures = 0
        # A single flaky call should not lose a multi-hour run: Pipeline B's
        # tag extraction alone is one sequential call per chunk, and any
        # hosted provider returns a 429/503 under load sooner or later.
        # Overridable per machine without touching provider config.
        self.max_retries = int(os.getenv("LLM_RETRY_ATTEMPTS", "5"))
        self.retry_backoff_seconds = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "2"))
        # A provider's own suggested wait is honoured over our exponential
        # schedule (see _retry_after_seconds) -- but a daily/monthly quota
        # can suggest a wait far longer than any per-call retry should ever
        # block for. Past this cap, give up with a clear message instead of
        # hanging: that is a billing/plan problem, not a transient one, and
        # no amount of waiting inside this process fixes it.
        self.max_retry_wait_seconds = float(os.getenv("LLM_MAX_RETRY_WAIT_SECONDS", "120"))

    # ------------------------------------------------------- provider hook --
    @abstractmethod
    def _chat(self, content: List[ContentBlock], system: Optional[str],
             max_tokens: int, temperature: float) -> str:
        """
        One turn, provider-specific. `content` is text/image blocks, in order:

            {"type": "text",  "text": "..."}
            {"type": "image", "image": <path | PIL.Image>}
        """

    # ------------------------------------------------- transient-error retry --
    def _chat_with_retry(self, content: List[ContentBlock], system: Optional[str],
                         max_tokens: int, temperature: float) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._chat(content, system, max_tokens, temperature)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt == self.max_retries:
                    raise

                suggested = _retry_after_seconds(exc)
                if suggested is not None and suggested > self.max_retry_wait_seconds:
                    _log.error(
                        f"  [{self.model_id}] the provider asked to wait "
                        f"{suggested:.0f}s before retrying -- longer than the "
                        f"{self.max_retry_wait_seconds:.0f}s cap (LLM_MAX_RETRY_WAIT_SECONDS). "
                        f"This usually means a daily/monthly quota, not a momentary "
                        f"rate limit; giving up rather than blocking the run: {exc}")
                    raise
                delay = (suggested if suggested is not None
                        else self.retry_backoff_seconds * (2 ** (attempt - 1)))

                _log.warning(
                    f"  [{self.model_id}] transient error (attempt "
                    f"{attempt}/{self.max_retries}), retrying in {delay:.0f}s"
                    f"{' (provider-suggested)' if suggested is not None else ''}: "
                    f"{type(exc).__name__}: {exc}")
                time.sleep(delay)
        raise last_exc  # pragma: no cover — loop always returns or raises above

    # ------------------------------------------------------------ free text --
    def generate(self, system: Optional[str], user: str, max_tokens: int = 1_024,
                temperature: float = 0.2, images: Optional[List[Any]] = None) -> str:
        content: List[ContentBlock] = [
            {"type": "image", "image": im} for im in (images or [])
        ]
        content.append({"type": "text", "text": user})
        self.calls += 1
        return self._chat_with_retry(content, system, max_tokens, temperature)

    # ------------------------------------------------------------ structured --
    def generate_structured(
        self, system: Optional[str], user: Union[str, List[ContentBlock]],
        schema: Dict[str, Any], max_tokens: int = 2_048, temperature: float = 0.0,
        max_attempts: int = 3, images: Optional[List[Any]] = None,
    ) -> Optional[Dict]:
        """
        Ask for one JSON object matching `schema`, repairing on parse failure.

        `user` is either a plain prompt string, or a ready-made list of
        content blocks (Stage 3 needs this: describing several figures in one
        call means images and their captions have to stay interleaved with
        the surrounding document text, not flattened into one string).

        On a parse failure we re-prompt with the PARSER'S OWN ERROR appended.
        "braces never balanced - the reply was probably truncated" is a far
        more useful instruction than "try again", and it usually succeeds on
        the second attempt.

        Returns the object, or None once the attempts are exhausted. Callers
        must handle None -- a missing result is recoverable, a fabricated one
        is not.
        """
        base: List[ContentBlock] = [
            {"type": "image", "image": im} for im in (images or [])
        ]
        base.extend([{"type": "text", "text": user}] if isinstance(user, str) else user)

        instruction = (
            "Reply with ONE JSON object and nothing else: no prose, "
            "no markdown fence, no explanation.\n"
            "It must match this shape exactly:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        base = base + [{"type": "text", "text": instruction}]
        repair_note: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            blocks = base if repair_note is None else base + [{
                "type": "text",
                "text": f"YOUR PREVIOUS REPLY COULD NOT BE PARSED: {repair_note}\n"
                        f"Reply with only the JSON object.",
            }]

            self.calls += 1
            raw = self._chat_with_retry(blocks, system, max_tokens, temperature)
            obj, reason = extract_json_object(raw)

            if obj is not None:
                missing = [k for k in schema if k not in obj]
                if not missing:
                    return obj
                reason = f"missing required key(s): {missing}"

            _log.warning(
                f"  [{self.model_id}] structured attempt {attempt}/{max_attempts}: "
                f"{reason} — raw reply: {raw[:300]!r}")
            self.structured_repairs += 1
            repair_note = reason

        self.structured_failures += 1
        _log.error(
            f"  [{self.model_id}] structured generation failed after "
            f"{max_attempts} attempts")
        return None

    # ------------------------------------------------------------- teardown --
    def cleanup(self) -> None:
        """Release provider-held resources (a loaded model, an open session).
        A no-op for API-backed providers; overridden where there is state to free."""
