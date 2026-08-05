"""
Phase 4.2 — The Writer.

The Writer plans the section, drafts each step as ACTUAL BOOK PROSE, and
clarifies when the Student is confused.

It never "chats about" the topic: every generation is text that could appear
in the book unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent, render_context

PLAN_SCHEMA = {
    "steps": [
        {
            "title": "string — the teaching move, not a heading",
            "topic": "string — what this step establishes, one sentence",
            "uses_chunks": ["string — chunk ids this step draws on"],
        }
    ],
    "teaches": ["string — the concepts this section will define or explain"],
    "assumes": ["string — concepts used but NOT defined here"],
    "opening_promise": "string — what the reader will be able to do afterwards",
}

WRITER_SYSTEM = """You are the Writer of a technical book. You produce finished book prose, not notes and not chat.

RULES THAT OVERRIDE EVERYTHING ELSE:
1. Use the book's canonical terms exactly as the ledger gives them. If a term is listed with
   variants to avoid, never use a variant.
2. Never redefine a concept the ledger says is already defined. Reference it and build on it.
3. Never re-explain source material marked "already used in <section>". Build on it instead.
4. Honour the running examples. Do not invent a new one.
5. Write in the book's voice, for the book's stated reader.
6. Never write a phrase on the forbidden list."""


class WriterAgent(BaseAgent):
    name = "Writer"

    def create_teaching_plan(
        self, section: Dict, context: Dict, feedback: str = ""
    ) -> Optional[Dict]:
        """Plan the section as a sequence of teaching moves."""
        body = render_context(
            context,
            (
                "constitution",
                "book_spine",
                "ledger_slice",
                "neighbors",
                "source",
                "figures_text",
            ),
        )
        target = section.get("estimated_word_count", self.cfg.default_word_count)
        n_min, n_max = self.cfg.steps_per_section_min, self.cfg.steps_per_section_max

        task = f"""{body}

=== YOUR TASK ===
Plan the section "{section['title']}" ({section['section_id']}, chapter {section['chapter_id']}).
It should teach: {', '.join(section.get('tags', [])) or '(see the source)'}
Target length: {target} words total, so plan {n_min}-{n_max} steps of roughly
{target // max(1, n_min):d}-{target // max(1, n_max):d} words each.

A step is a teaching MOVE, not a heading: "show why the naive loop fails" beats "Overview".
Plan only what the source material can actually support.
"""
        if feedback:
            task += f"\n=== YOUR PREVIOUS PLAN WAS REJECTED ===\n{feedback}\nAddress this.\n"

        return self._execute_step(
            task, WRITER_SYSTEM, PLAN_SCHEMA, max_tokens=self.cfg.plan_max_tokens
        )

    def write_step(
        self,
        section: Dict,
        plan: Dict,
        step_index: int,
        context: Dict,
        drafted_so_far: str,
        prev_tail: str,
        conversation: str = "",
    ) -> str:
        """Draft one step as finished book prose, with the figures attached."""
        step = plan["steps"][step_index]
        body = render_context(
            context, ("constitution", "ledger_slice", "source", "figures_text")
        )

        # The image half of figure delivery: load the crops that exist and
        # pass them to the multimodal model alongside the prompt.
        images: List[Any] = []
        for fig in context.get("figures") or []:
            path = fig.get("path")
            if path and Path(path).exists():
                try:
                    from PIL import Image

                    img = Image.open(path)
                    img.load()
                    images.append(img)
                except Exception:
                    pass

        dialogue = ""
        if conversation:
            dialogue = (
                "\n=== THE TEACHING DIALOGUE SO FAR (doubts already "
                "resolved — do not re-trigger them) ===\n"
                f"{conversation[-6000:]}\n"
            )
        target = section.get("estimated_word_count", self.cfg.default_word_count)
        per_step = max(120, target // max(1, len(plan["steps"])))

        continuity = ""
        if step_index == 0 and prev_tail:
            continuity = (
                f"\n=== HOW THE PREVIOUS SECTION ENDED (continue from this "
                f"voice; do not restate it) ===\n{prev_tail}\n"
            )
        elif drafted_so_far:
            continuity = (
                f"\n=== WHAT YOU HAVE WRITTEN IN THIS SECTION SO FAR ===\n"
                f"{drafted_so_far[-2500:]}\n"
            )

        task = f"""{body}
{continuity}{dialogue}
=== THE PLAN FOR THIS SECTION ===
{chr(10).join(f"{i + 1}. {s['title']} — {s.get('topic', '')}"
              for i, s in enumerate(plan['steps']))}

=== YOUR TASK ===
Write step {step_index + 1} of {len(plan['steps'])}: "{step['title']}"
What it must establish: {step.get('topic', '')}
Length: about {per_step} words.

Write finished book prose. No headings, no "In this step", no meta-commentary about
what you are about to do. Code goes in fenced blocks and must follow the book's
code conventions. Do not summarise or conclude — later steps continue from here.
"""
        return self._execute_step(
            task,
            WRITER_SYSTEM,
            max_tokens=self.cfg.step_max_tokens,
            images=images or None,
        )

    def clarify(self, section: Dict, step_prose: str, doubt: str, context: Dict) -> str:
        """Answer a Student doubt — in book prose, not in a Q&A voice."""
        task = f"""{render_context(context, ("constitution", "ledger_slice"))}

=== THE PROSE A READER FOUND UNCLEAR ===
{step_prose}

=== WHAT CONFUSED THEM ===
{doubt}

=== YOUR TASK ===
Write the additional book prose that resolves this. One or two short paragraphs.
Write it as it would appear in the book — do NOT address the reader's question
directly, do not write "You might wonder", and do not repeat what is already above.
The Editor will fold this into the passage where the confusion arises.
"""
        return self._execute_step(
            task, WRITER_SYSTEM, max_tokens=self.cfg.step_max_tokens // 2
        )
