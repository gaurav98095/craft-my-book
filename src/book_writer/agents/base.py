"""
Phase 4.1 — BaseAgent.

All five agents share one execution path: build a prompt, call the model, get
either free text or a validated JSON object back.

Structured calls go through `generate_structured`, which re-prompts with the
parser's own error on a malformed reply -- so a stray brace costs a retry,
not the whole generation.
"""

from typing import Any, Dict, List, Optional, Tuple

from ...ingestion.setup import count_tokens
from ..setup import WriterConfig, writer_log


class BaseAgent:
    """Common execution for every agent."""

    name = "BaseAgent"

    def __init__(self, llm, cfg: WriterConfig):
        self.llm = llm
        self.cfg = cfg
        self.calls = 0
        self.failures = 0

    def _execute_step(
        self,
        prompt_text: str,
        system_prompt: str,
        structured_schema: Optional[Dict] = None,
        max_tokens: int = 1_200,
        temperature: Optional[float] = None,
        images: Optional[List[Any]] = None,
    ) -> Any:
        self.calls += 1
        tokens_in = count_tokens(system_prompt) + count_tokens(prompt_text)
        if tokens_in > 200_000:
            writer_log.warning(
                f"[{self.name}] prompt is {tokens_in:,} tokens — "
                f"the assembler's budgets are being exceeded upstream"
            )

        try:
            if structured_schema is not None:
                result = self.llm.generate_structured(
                    system_prompt,
                    prompt_text,
                    structured_schema,
                    max_tokens=max_tokens,
                    temperature=(
                        self.cfg.temperature_structured
                        if temperature is None
                        else temperature
                    ),
                )
                if result is None:
                    self.failures += 1
                return result
            kwargs: Dict[str, Any] = {}
            if images:
                # the Writer is a multimodal model: attach the actual figure
                # crops so it reads the diagram, not a summary of the diagram
                kwargs["images"] = images
            return self.llm.generate(
                system_prompt,
                prompt_text,
                max_tokens=max_tokens,
                temperature=(
                    self.cfg.temperature_prose if temperature is None else temperature
                ),
                **kwargs,
            )
        except Exception as exc:
            self.failures += 1
            writer_log.error(f"[{self.name}] step failed: {exc}")
            raise


def render_context(blocks: Dict[str, Any], include: Tuple[str, ...]) -> str:
    """Stitch selected assembler blocks into one prompt body, in a fixed order."""
    titles = {
        "constitution": "",  # already self-labelled
        "book_spine": "",
        "ledger_slice": "=== WHAT THE BOOK ALREADY KNOWS ===",
        "neighbors": "=== SURROUNDING SECTIONS ===",
        "source": "=== SOURCE MATERIAL ===",
        "figures_text": "=== FIGURES AVAILABLE ===",
    }
    parts = []
    for key in include:
        body = blocks.get(key)
        if isinstance(body, str) and body.strip():
            head = titles.get(key, f"=== {key.upper()} ===")
            parts.append(f"{head}\n{body}".strip())
    return "\n\n".join(parts)
