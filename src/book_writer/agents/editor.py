"""
Phase 4.4 — Code masking, and the Editor.

    "The biggest risk in letting a model 'improve flow' is that it silently
     edits a code sample and breaks it. Do not rely on an instruction for
     this -- make it mechanically impossible."

The Editor sees `[[CODE_BLOCK_0]]`. It can move it, keep it, or write a
better sentence leading into it -- but it cannot touch a character inside it.
"""

import re
from typing import Dict, List, Optional, Tuple

from ..constitution import Constitution
from .base import BaseAgent

FENCE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def mask_code(text: str) -> Tuple[str, List[str]]:
    blocks = FENCE.findall(text)
    for i, _ in enumerate(blocks):
        text = FENCE.sub(f"[[CODE_BLOCK_{i}]]", text, count=1)
    return text, blocks


def unmask_code(text: str, blocks: List[str]) -> str:
    for i, block in enumerate(blocks):
        # a function replacement keeps backslashes in the code literal
        text = re.sub(re.escape(f"[[CODE_BLOCK_{i}]]"), lambda _m, b=block: b, text)
    return text


EDITOR_SCHEMA = {
    "content": "string — the finished section as markdown",
    "opening_line": "string — the section's first sentence, verbatim",
    "closing_line": "string — the section's final sentence, verbatim",
    "doubts_woven": ["string — each Student doubt folded into the prose"],
    "notes": "string — anything the Editor could not resolve, for the log",
}

EDITOR_SYSTEM = (
    "You are the Editor. You receive the raw output of a teaching session — "
    "several drafted steps plus clarifications the writer added when a reader "
    "got confused — and you turn it into one continuous section of a book.\n\n"
    "YOU MAY: reorder sentences, rewrite transitions, merge or split paragraphs, "
    "fold a clarification into the passage it clarifies, write a real opening "
    "and closing sentence, cut redundancy across steps.\n\n"
    "YOU MAY NOT: add a technical fact, definition, claim, statistic, or forward "
    "reference that is not already present in the input. You may not alter code "
    "blocks in any way — they arrive as placeholders and must be returned "
    "unchanged, in their original order. You may not remove a concept the input "
    "teaches. If something reads badly and you cannot fix it without inventing "
    "content, leave it and say so in `notes`."
)


class EditorAgent(BaseAgent):
    """Melts the seams between teaching steps. Rewrites prose. Never invents content."""

    name = "Editor"

    @staticmethod
    def _render_steps(raw_steps: List[Dict]) -> str:
        parts = []
        for i, step in enumerate(raw_steps, 1):
            parts.append(f"--- STEP {i}: {step['title']} ---\n{step['prose']}")
            for doubt in step.get("clarifications", []):
                parts.append(
                    f"--- CLARIFICATION added after step {i} "
                    f"(reader asked: \"{doubt['question']}\") ---\n"
                    f"{doubt['answer']}"
                )
        return "\n\n".join(parts)

    def smooth_section(
        self,
        section: Dict,
        raw_steps: List[Dict],
        constitution: Constitution,
        prev_tail: str = "",
    ) -> Optional[Dict]:
        task = f"""SECTION: {section['title']}  ({section['section_id']})
TARGET LENGTH: {section.get('estimated_word_count', 800)} words (±15%)

{constitution.get_style_injection()}

HOW THE PREVIOUS SECTION ENDED (open so this follows on, do not restate it):
{prev_tail or '(this is the opening section of the book)'}

RAW TEACHING OUTPUT — steps in order, with the doubts each one triggered:
{self._render_steps(raw_steps)}

Produce the finished section.
"""
        return self._execute_step(
            task, EDITOR_SYSTEM, EDITOR_SCHEMA, max_tokens=self.cfg.editor_max_tokens
        )
