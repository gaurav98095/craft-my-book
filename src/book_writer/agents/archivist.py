"""
Phase 4.5 — The Archivist: the write-back loop.

Everything in Phases 1-3 is inert without this. The Archivist is the arrow
that points from a finished section back into memory.

It reads the EDITED text, not the raw steps, because that is what readers
will read -- and it is told to be literal: "Record what the text actually
says, not what it should have said."
"""

import re
from typing import Dict, List, Optional

from .base import BaseAgent

ARCHIVIST_SCHEMA = {
    "summary": {
        "abstract": "string — 60-100 words, what this section actually taught",
        "teaches": ["string — concepts introduced or explained here"],
        "assumes": ["string — concepts used but not defined here"],
        "closing_line": "string — the section's final sentence, verbatim",
    },
    "concepts_defined": [
        {"term": "string", "definition": "string — one sentence, as written"}
    ],
    "concepts_referenced": ["string"],
    "claims": [
        {
            "claim_id": "string",
            "text": "string",
            "confidence": "strong|moderate|tentative",
        }
    ],
    "promises_made": [
        {"promise_id": "string", "text": "string", "target_hint": "string chapter id"}
    ],
    "promises_fulfilled": ["string — promise_ids this section delivered on"],
    "example_states": {"ExampleName": "string — its state after this section"},
    "chunks_used": {"chunk_id": "primary|supporting|mentioned"},
}

ARCHIVIST_SYSTEM = (
    "You are the Archivist. You do not write or edit — you catalogue.\n"
    "You read a finished book section and extract a precise, structured record "
    "of what it defined, asserted, promised, demonstrated, and consumed.\n"
    "Be literal. Record what the text actually says, not what it should have said."
)


class ArchivistAgent(BaseAgent):
    """Reads a finished section and harvests it into the Book Ledger."""

    name = "Archivist"

    def harvest(
        self, section: Dict, content: str, open_promises: List[Dict]
    ) -> Optional[Dict]:
        promise_list = (
            "\n".join(f"  {p['promise_id']}: {p['text']}" for p in open_promises[:30])
            or "  (none)"
        )
        task = f"""SECTION: {section['title']}  ({section['section_id']})
SOURCE CHUNKS THIS SECTION WAS GIVEN: {', '.join(section['chunk_ids']) or '(none)'}

PROMISES CURRENTLY OPEN IN THE BOOK:
{promise_list}

FINISHED SECTION TEXT:
{content}

Catalogue this section:
1. A 60-100 word abstract of what it TAUGHT (not what it was about).
2. Every term it DEFINED, with the definition as written.
3. Every previously-defined term it USED.
4. Every claim a later section could contradict.
5. Every FORWARD PROMISE it made ("we'll cover X later").
6. Any promise from the list above that this section FULFILLED (give its promise_id).
7. The state of any running example after this section.
8. Which given chunks it actually used, and how deeply.
"""
        return self._execute_step(
            task,
            ARCHIVIST_SYSTEM,
            ARCHIVIST_SCHEMA,
            max_tokens=self.cfg.archivist_max_tokens,
        )

    @staticmethod
    def fallback_update(section: Dict, content: str) -> Dict:
        """
        What to write into the ledger when the Archivist call fails outright.

        A missing harvest is recoverable; an empty one that looks complete is
        not. So the fallback records the mechanical facts we can be certain
        of -- the section exists, it is this long, it consumed these chunks --
        and deliberately claims no concepts, no definitions and no promises.
        """
        words = content.split()
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", content.strip()) if s.strip()
        ]
        return {
            "summary": {
                "abstract": f"[ARCHIVIST FAILED] {' '.join(words[:60])}…",
                "teaches": [],
                "assumes": [],
                "closing_line": sentences[-1] if sentences else "",
                "word_count": len(words),
            },
            "concepts_defined": [],
            "concepts_referenced": [],
            "claims": [],
            "promises_made": [],
            "promises_fulfilled": [],
            "example_states": {},
            "chunks_used": {cid: "mentioned" for cid in section["chunk_ids"]},
            "_degraded": True,
        }
