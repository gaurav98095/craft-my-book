"""
Phase 1.2 — Layer 0: The Constitution.

A small, fixed JSON file loaded into every single prompt. It never changes
during a run -- that is the point.

Why each block earns its place (design section 9):

  canonical_examples  stops failure #7. If every prompt says "the book's
                      running example is SupportBot", sections stop inventing
                      a fresh FooBot each time.
  code_conventions    is what makes 175 code blocks look like one engineer
                      wrote them. Style drift in code is more visible to a
                      reader than style drift in prose.
  forbidden_patterns  costs about 40 tokens and removes the tics that make
                      generated text instantly recognisable.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from .setup import writer_log

DEFAULT_CONSTITUTION = {
    "book_identity": {
        "title": "The Future of AI Agents",
        "subtitle": "A Production-Grade Implementation Guide",
        "target_audience": {
            "level": "Intermediate Developer",
            "prerequisites": ["Python", "Basic LLM knowledge"],
            "persona_description": "Software engineers building autonomous systems.",
        },
    },
    "style_guide": {
        "depth": "applied",
        "tone": "rigorous_accessible",
        "teaching_approach": "Code-First",
        "practicality_level": "Production-Grade",
        "engagement_style": "Socratic",
        "analogy_density": "Moderate",
    },
    # the running examples the whole book shares
    "canonical_examples": [
        {
            "name": "SupportBot",
            "description": "A customer-support agent we build across the book",
            "introduced_in_chapter": "ch02",
            "purpose": "The single thread readers follow end to end",
        }
    ],
    # mechanical conventions, so all the code looks like one author
    "code_conventions": {
        "language": "python",
        "style": "type-hinted, dataclasses for config, logging not print",
        "error_handling": "explicit try/except with logged context",
        "naming": "snake_case functions, PascalCase classes",
    },
    # house rules
    "forbidden_patterns": [
        "In today's fast-paced world",
        "It is important to note that",
        "Let's dive in",
        "In conclusion",
        "Furthermore,",
        "delve into",
    ],
}


class Constitution:
    """Layer 0. Fixed for the whole run; injected into every prompt."""

    def __init__(self, path: Path, default: Dict[str, Any]):
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(default, indent=2), encoding="utf-8")
            writer_log.info(
                f"Constitution written to {path} (edit it before a real run)"
            )
        self.data = json.loads(path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- reads --
    @property
    def title(self) -> str:
        return self.data["book_identity"]["title"]

    @property
    def audience(self) -> str:
        return self.data["book_identity"]["target_audience"]["level"]

    def examples(self) -> List[Dict[str, Any]]:
        return self.data.get("canonical_examples", [])

    # ------------------------------------------------------------ injection --
    def get_prompt_injection(self) -> str:
        """The full constitution, for the Writer. Roughly 700 tokens."""
        identity = self.data["book_identity"]
        audience = identity["target_audience"]
        style = self.data["style_guide"]
        code = self.data["code_conventions"]

        lines = [
            "=== THE BOOK ===",
            f"Title: {identity['title']} — {identity.get('subtitle', '')}",
            f"Reader: {audience['level']}. {audience.get('persona_description', '')}",
            f"Assumed knowledge: {', '.join(audience.get('prerequisites', []))}",
            "",
            "=== HOW IT SOUNDS ===",
        ]
        lines += [f"{k}: {v}" for k, v in style.items()]

        if self.examples():
            lines += [
                "",
                "=== RUNNING EXAMPLES (use these, do not invent new ones) ===",
            ]
            for ex in self.examples():
                lines.append(
                    f"- {ex['name']}: {ex['description']} "
                    f"(introduced in {ex.get('introduced_in_chapter', '?')})"
                )

        lines += ["", "=== CODE CONVENTIONS ==="]
        lines += [f"{k}: {v}" for k, v in code.items()]

        forbidden = self.data.get("forbidden_patterns", [])
        if forbidden:
            lines += [
                "",
                "=== NEVER WRITE THESE PHRASES ===",
                "; ".join(f'"{p}"' for p in forbidden),
            ]
        return "\n".join(lines)

    def get_style_injection(self) -> str:
        """The shorter form, for the Editor and Reviewer."""
        style = self.data["style_guide"]
        forbidden = self.data.get("forbidden_patterns", [])
        parts = ["=== STYLE ===", "; ".join(f"{k}: {v}" for k, v in style.items())]
        if forbidden:
            parts.append("Never write: " + "; ".join(f'"{p}"' for p in forbidden))
        return "\n".join(parts)
