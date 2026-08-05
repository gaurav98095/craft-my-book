"""
Step B0.2 — The model wrapper.

Not a second model. A thin adapter that gives Pipeline B's steps a
`(system, user)` calling convention over the SAME transformers model Stage 3
loaded -- so the model that described the diagrams is the model that names
the chapters, exactly as the design intends.

    "The model that describes a diagram during preprocessing is the same
     model that later writes the chapter about it. That alone does more for
     consistency than any amount of prompt tuning."

It also inherits Stage 3's careful loading: the resolved dtype keyword, the
attention-backend ladder, and the audible fallback if BOOK_MODEL itself could
not be pulled.
"""

import json
from typing import Any, Dict, Optional

from ..ingestion.stage3_figures import BookModel, Stage3Config, extract_json_object
from .setup import PipelineBConfig, toc_log


class BookLLM:
    """Pipeline B's interface to the book's model."""

    def __init__(
        self, cfg: PipelineBConfig, existing_model: Optional[BookModel] = None
    ):
        self.cfg = cfg
        self.model = self._acquire(existing_model)
        self.model_id = self.model.model_id
        self.calls = 0
        self.structured_repairs = 0
        self.structured_failures = 0

    # ---------------------------------------------------------------- setup --
    def _acquire(self, existing_model: Optional[BookModel]) -> BookModel:
        """Reuse Stage 3's model if one was passed in; otherwise load one."""
        if (
            existing_model is not None
            and getattr(existing_model, "model", None) is not None
        ):
            toc_log.info(
                f"Reusing the model Stage 3 already loaded: {existing_model.model_id}"
            )
            return existing_model

        toc_log.warning(
            "Stage 3's model was not passed in - loading a fresh copy of "
            f"{self.cfg.model_name}. Passing the Stage 3 BookModel in "
            "avoids a second multi-gigabyte load."
        )
        return BookModel(Stage3Config(model_name=self.cfg.model_name))

    # ------------------------------------------------------------ generation --
    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 1_024,
        temperature: Optional[float] = None,
        images=None,
    ) -> str:
        """
        One turn. Prompts are rendered by the model's own chat template.

        `images` (PIL objects) pass straight through to the multimodal model.
        Pipeline B never uses this; Pipeline C's Writer does, to attach the
        actual figure crops next to their descriptions.
        """
        temperature = self.cfg.temperature if temperature is None else temperature
        self.calls += 1
        return self.model.generate(
            prompt=user,
            system=system,
            images=images,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------ structured --
    def generate_structured(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_tokens: int = 2_048,
        temperature: float = 0.0,
    ) -> Optional[Dict]:
        """
        One JSON object matching `schema`, with the same repair loop Stage 3
        uses: on a parse failure, re-prompt with the parser's own error.

        "braces never balanced - the reply was probably truncated" is a far more
        useful instruction than "try again", and it usually succeeds on the
        second attempt. Throwing away a good generation over a stray brace is an
        expensive way to lose a chunk's tags.

        Returns None once the attempts are exhausted. Callers must handle None --
        a missing result is recoverable, a fabricated one is not.
        """
        instruction = (
            "\n\nReply with ONE JSON object and nothing else: no prose, "
            "no markdown fence, no explanation.\nIt must match this shape "
            f"exactly:\n{json.dumps(schema, indent=2)}"
        )
        repair = None

        for attempt in range(1, self.cfg.structured_max_attempts + 1):
            message = user + instruction
            if repair:
                message += (
                    f"\n\nYOUR PREVIOUS REPLY COULD NOT BE PARSED: {repair}\n"
                    f"Reply with only the JSON object."
                )

            raw = self.generate(
                system, message, max_tokens=max_tokens, temperature=temperature
            )
            obj, reason = extract_json_object(raw)
            if obj is not None:
                missing = [k for k in schema if k not in obj]
                if not missing:
                    return obj
                reason = f"missing required key(s): {missing}"

            toc_log.warning(f"  structured attempt {attempt}: {reason}")
            self.structured_repairs += 1
            repair = reason

        self.structured_failures += 1
        return None
