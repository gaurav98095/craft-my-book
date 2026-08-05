"""
Phase 4.3 — The Reviewer and the Student.

The Reviewer judges the PLAN, before any prose exists -- catching a problem
here costs one cheap call instead of one expensive generation plus a rewrite.

The Student judges the PROSE, and sees nothing else: no plan, no source, no
ledger. It has to react the way a real reader would, and a real reader does
not have the source material open beside them.
"""

import json
from typing import Dict

from ..setup import writer_log
from ..constitution import Constitution
from .base import BaseAgent

REVIEW_SCHEMA = {
    "approved": "boolean",
    "feedback": "string — what to change, concrete, empty if approved",
    "concerns": ["string — specific pedagogical or grounding problems"],
}

REVIEWER_SYSTEM = """You are the Reviewer. You judge a teaching plan before any prose is written.

Approve or reject on three grounds only:
1. PEDAGOGY — do the steps build in an order a reader can follow? Is anything assumed too early?
2. GROUNDING — can the assigned source material actually support every step, or is the plan
   inventing content the sources do not contain?
3. FIT — does the plan match the section's stated scope and length, without sprawling into
   what neighbouring sections cover?

You are not judging prose quality; none exists yet. Be decisive: approve a workable plan
rather than holding out for a perfect one. Reject only for a concrete, fixable problem."""

STUDENT_SCHEMA = {
    "verdict": "UNDERSTOOD|DOUBT",
    "doubt": "string — the single most blocking question, empty if UNDERSTOOD",
    "can_wait": "boolean — true if this is better answered in a later section",
}

STUDENT_SYSTEM = """You are a target reader of this book, reading a passage for the first time.

You have the book's assumed prerequisites and nothing else. You cannot see the source
material, the plan, or any other section.

Read the passage and answer honestly:
- UNDERSTOOD — you followed it and could explain it back.
- DOUBT — something specific blocks you. State the ONE most blocking question.

Raise a doubt only for something genuinely unclear or unexplained: a term used but never
defined, a leap in reasoning, a claim with no support. Do NOT raise a doubt because the
passage is incomplete — passages continue in later steps. Do not ask for more examples
out of politeness. Most well-written passages are UNDERSTOOD."""


class ReviewerAgent(BaseAgent):
    name = "Reviewer"

    def review_plan(
        self, section: Dict, plan: Dict, context: Dict, constitution: Constitution
    ) -> Dict:
        source_excerpt = (context.get("source") or "")[:12_000]
        task = f"""{constitution.get_style_injection()}

=== THE SECTION ===
{section['section_id']}: "{section['title']}" (chapter {section['chapter_id']})
Concepts it should teach: {', '.join(section.get('tags', []))}
Target length: {section.get('estimated_word_count', 700)} words

=== THE PROPOSED PLAN ===
{json.dumps(plan, indent=2)}

=== THE SOURCE MATERIAL AVAILABLE (excerpt) ===
{source_excerpt}

=== YOUR TASK ===
Approve or reject this plan.
"""
        result = self._execute_step(
            task, REVIEWER_SYSTEM, REVIEW_SCHEMA, max_tokens=self.cfg.review_max_tokens
        )
        # A failed review must not silently block the book.
        if result is None:
            writer_log.warning("  reviewer returned nothing — approving by default")
            return {
                "approved": True,
                "feedback": "",
                "concerns": ["reviewer call failed"],
            }
        return result


class StudentAgent(BaseAgent):
    name = "Student"

    def evaluate(self, prose: str) -> Dict:
        task = f"""=== THE PASSAGE ===
{prose}

=== YOUR TASK ===
Did you follow it? Answer UNDERSTOOD, or state the one thing that blocks you.
"""
        result = self._execute_step(
            task, STUDENT_SYSTEM, STUDENT_SCHEMA, max_tokens=self.cfg.student_max_tokens
        )
        if result is None:
            return {"verdict": "UNDERSTOOD", "doubt": "", "can_wait": False}
        # normalise a chatty verdict
        verdict = str(result.get("verdict", "")).strip().upper()
        result["verdict"] = "DOUBT" if verdict.startswith("DOUBT") else "UNDERSTOOD"
        if result["verdict"] == "DOUBT" and not str(result.get("doubt", "")).strip():
            result["verdict"] = "UNDERSTOOD"  # a doubt with no question is noise
        return result
