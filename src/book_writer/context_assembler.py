"""Phase 3.1 — Layer 4: The Context Assembler.

Turns the whole book memory into one prompt for ONE section. Budgets are
explicit so nothing silently overflows.
"""

import re
from typing import Any, Dict, List, Optional

from ..ingestion.setup import count_tokens
from .setup import writer_log
from .constitution import Constitution
from .ledger import BookLedger
from .draft_store import DraftStore
from .source_memory import SourceMemory


class ContextAssembler:
    """
    Turns the whole book memory into one prompt for ONE section.
    Budgets are explicit so nothing silently overflows.
    """

    BUDGETS = {  # tokens
        "constitution": 700,
        "book_spine": 6_000,  # chapter rollups + all section titles
        "ledger_slice": 12_000,  # only the facts relevant to this section
        "neighbors": 14_000,  # previous section + 3 nearest by meaning
        "source": 60_000,  # assigned chunks + neighbourhoods + related
        "figures": 4_000,  # 2-3 actual images
        "conversation": 12_000,  # this section's dialogue so far
        "draft_so_far": 6_000,
    }

    MIN_LEDGER_CONCEPTS = 5  # below this, fall back to similarity
    MAX_SLICE_CONCEPTS = 15
    MAX_FIGURES = 3

    def __init__(
        self,
        constitution: Constitution,
        ledger: BookLedger,
        drafts: DraftStore,
        source: SourceMemory,
        toc: Dict[str, Any],
    ):
        self.constitution = constitution
        self.ledger = ledger
        self.drafts = drafts
        self.source = source
        self.toc = toc
        self.sections_by_id = {s["section_id"]: s for s in toc["sections"]}
        self.truncations: List[Dict] = []

    # ------------------------------------------------------------- budgeting --
    def _fits(self, text: str, block: str, margin: int = 0) -> bool:
        return count_tokens(text) + margin <= self.BUDGETS[block]

    def _truncate(self, text: str, block: str) -> str:
        """Overflow is a logged truncation, never a silent one."""
        budget = self.BUDGETS[block]
        if count_tokens(text) <= budget:
            return text
        # ~4 chars per token, trimmed conservatively then verified
        keep = budget * 4
        cut = text[:keep]
        while count_tokens(cut) > budget and len(cut) > 100:
            cut = cut[: int(len(cut) * 0.9)]
        self.truncations.append(
            {"block": block, "budget": budget, "was": count_tokens(text)}
        )
        writer_log.warning(
            f"  context: '{block}' truncated "
            f"{count_tokens(text)} -> {budget} tokens"
        )
        return cut + f"\n\n[... {block} truncated to fit its budget ...]"

    # ------------------------------------------------------------ 2. the spine --
    def _render_spine(self, chapter_id: str) -> str:
        """Where we are in the whole arc: rollups elsewhere, titles everywhere."""
        lines = [
            "=== THE BOOK'S ARC ===",
            f"\"{self.toc['book_title']}\" — {len(self.toc['chapters'])} chapters, "
            f"{len(self.toc['sections'])} sections.",
        ]

        rollups = self.ledger._cache["chapter_rollups"]
        for chapter in self.toc["chapters"]:
            cid = chapter["chapter_id"]
            marker = "  <-- YOU ARE HERE" if cid == chapter_id else ""
            lines.append(f"\n{cid}. {chapter['title']}{marker}")
            if cid in rollups:
                lines.append(f"    {rollups[cid]}")
            titles = [s for s in self.toc["sections"] if s["chapter_id"] == cid]
            if cid == chapter_id:
                # our own chapter: titles plus abstracts of what is already written
                for s in titles:
                    summary = self.ledger._cache["section_summaries"].get(
                        s["section_id"]
                    )
                    mark = "written" if summary else "not yet written"
                    lines.append(f"    {s['section_id']}: {s['title']}  [{mark}]")
                    if summary and summary.get("abstract"):
                        lines.append(f"        {summary['abstract']}")
            else:
                lines.append("    " + " · ".join(s["title"] for s in titles))
        return "\n".join(lines)

    # ------------------------------------------------- 3. the ledger slice ----
    def _render_ledger_slice(
        self, section: Dict, tags: List[str], chapter_id: str
    ) -> str:
        """
        The judgement call that makes or breaks this layer.

        Selects by tag overlap, with the design's stated fallback: if that
        yields too few concepts, fall back to embedding similarity against
        the section title, because thin or wrong tags would otherwise leave
        the Writer with nothing.
        """
        lines: List[str] = []
        relevant = self.ledger.concepts_for_tags(tags)
        selection = "tag overlap"

        # A section's own prerequisites are concepts it will certainly use,
        # even though they are not among its tags. Their definitions and
        # aliases have to travel with them or the Writer can reintroduce them
        # under a variant name.
        seen = {c["canonical_name"] for c in relevant}
        prereqs = [
            c
            for c in self.ledger.defined_prerequisites(tags)
            if c["canonical_name"] not in seen
        ]
        if prereqs:
            relevant = relevant + prereqs
            selection += f" + {len(prereqs)} defined prerequisite(s)"

        if len(relevant) < self.MIN_LEDGER_CONCEPTS:
            extra = self._concepts_by_similarity(section, exclude=relevant)
            if extra:
                relevant = relevant + extra
                selection = f"tag overlap + similarity fallback ({len(extra)} added)"

        # (a) concepts this section will use, with their agreed definitions
        if relevant:
            lines.append("ALREADY DEFINED — use these exact terms, do not redefine:")
            for c in relevant[: self.MAX_SLICE_CONCEPTS]:
                lines.append(
                    f"  • {c['canonical_name']} "
                    f"(defined in {c['defined_in']}): {c['definition']}"
                )
                if c["aliases"]:
                    lines.append(
                        f"      avoid the variants: " f"{', '.join(c['aliases'][:4])}"
                    )

        # (b) concepts this section needs that nothing has defined yet
        missing = self.ledger.undefined_prerequisites(tags)
        if missing:
            lines.append(
                "\nNOT YET DEFINED ANYWHERE — define briefly if you lean on them:"
            )
            lines.append("  " + ", ".join(missing[:10]))

        # (c) promises this section is expected to keep
        for p in self.ledger.open_promises_for(chapter_id):
            lines.append(
                f"\nOPEN PROMISE from {p['made_in']}: \"{p['text']}\" — "
                f"fulfil it here if it fits."
            )

        # (d) claims that constrain what we may now assert
        want = {t.lower() for t in tags}
        claims = [
            c
            for c in self.ledger._cache["claims"][-40:]
            if {t.lower() for t in c.get("tags", [])} & want
        ]
        if claims:
            lines.append("\nTHE BOOK HAS ALREADY ASSERTED — do not contradict:")
            for c in claims[:8]:
                lines.append(f"  • {c['text']}  ({c['section_id']})")

        # (e) where the running examples stand right now
        for name, ex in self.ledger._cache["examples"].items():
            lines.append(
                f"\nRUNNING EXAMPLE '{name}' currently: {ex.get('current_state', '?')} "
                f"(last touched {ex.get('last_touched', '?')})"
            )

        if not lines:
            return "(The book has not yet defined anything relevant to this section.)"
        return f"[ledger slice selected by: {selection}]\n" + "\n".join(lines)

    def _concepts_by_similarity(self, section: Dict, exclude: List[Dict]) -> List[Dict]:
        """Fallback when tags are thin: cheap lexical overlap on the title."""
        seen = {c["canonical_name"] for c in exclude}
        words = {w for w in re.findall(r"[a-z]{4,}", section["title"].lower())}
        scored = []
        for c in self.ledger._cache["concepts"].values():
            if not c.get("definition") or c["canonical_name"] in seen:
                continue
            name_words = set(re.findall(r"[a-z]{4,}", c["canonical_name"].lower()))
            overlap = len(words & name_words)
            if overlap:
                scored.append((overlap, c))
        scored.sort(key=lambda kv: -kv[0])
        return [c for _, c in scored[: self.MIN_LEDGER_CONCEPTS]]

    # -------------------------------------------------------- 4. neighbours ---
    def _render_neighbors(self, section: Dict, prev_section_id: Optional[str]) -> str:
        """Previous section in full, plus the nearest written sections by meaning."""
        parts: List[str] = []

        if prev_section_id:
            prev_text = self.drafts.get_full(prev_section_id)
            if prev_text:
                prev_title = self.sections_by_id.get(prev_section_id, {}).get(
                    "title", ""
                )
                parts.append(
                    f"=== THE PREVIOUS SECTION ({prev_section_id}: "
                    f"{prev_title}) — IN FULL ===\n{prev_text}"
                )

        query = f"{section['title']} {' '.join(section.get('tags', []))}"
        for sim in self.drafts.find_similar(query, k=3, exclude=section["section_id"]):
            if sim["section_id"] == prev_section_id:
                continue
            body = self.drafts.get_full(sim["section_id"])
            if body:
                parts.append(
                    f"=== NEAREST BY MEANING: {sim['section_id']} "
                    f"({sim['title']}, {sim['similarity']:.0%} similar) ===\n{body}"
                )

        return (
            "\n\n".join(parts)
            if parts
            else "(Nothing written yet — this is the opening.)"
        )

    # ----------------------------------------------------------- 6. figures ---
    def _select_figures(self, chunk_ids: List[str], limit: int = 3) -> List[Dict]:
        """
        Two or three actual images, preferring ones no section has leaned on
        yet. Usage lives in the LEDGER (book state), not in Pipeline A's
        read-only chunk metadata -- the orchestrator records it after each
        section ships.
        """
        figures = self.source.get_figures(chunk_ids)
        figures.sort(key=lambda f: (self.ledger.figure_use_count(f["id"]), f["id"]))
        return figures[:limit]

    def _render_figures_text(self, figures: List[Dict]) -> str:
        """
        The textual half of figure delivery: id, kind and description, so the
        Writer can reference a figure by id even when the image itself cannot
        be attached. The image half rides along in write_step.
        """
        if not figures:
            return ""
        lines = [
            "These figures from the source are available. Reference one in "
            "prose as (see Figure: <id>). The image itself is attached when "
            "the crop exists on disk."
        ]
        for f in figures:
            note = (
                " [already shown earlier in the book — do not re-explain it]"
                if self.ledger.figure_use_count(f["id"])
                else ""
            )
            lines.append(
                f"- {f['id']} ({f['kind']}, from {f['chunk_id']}): "
                f"{(f.get('description') or '')[:300]}{note}"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------- assemble --
    def assemble(self, section: Dict, prev_section_id: Optional[str]) -> Dict[str, Any]:
        chapter_id = section["chapter_id"]
        tags = section.get("tags", [])
        blocks: Dict[str, Any] = {}

        # 1. Constitution, always, in full.
        blocks["constitution"] = self.constitution.get_prompt_injection()

        # 2. Where we are in the whole arc.
        blocks["book_spine"] = self._truncate(
            self._render_spine(chapter_id), "book_spine"
        )

        # 3. Only the ledger facts that touch this section.
        blocks["ledger_slice"] = self._truncate(
            self._render_ledger_slice(section, tags, chapter_id), "ledger_slice"
        )

        # 4. The previous section, plus the nearest written sections by meaning.
        blocks["neighbors"] = self._truncate(
            self._render_neighbors(section, prev_section_id), "neighbors"
        )

        # 5. Source: assigned first, related material to fill what is left.
        assigned = self.source.get_assigned(section["chunk_ids"])
        if self._fits(assigned, "source", margin=2_000):
            assigned += self.source.find_related(
                f"{section['title']} {' '.join(tags)}",
                exclude=section["chunk_ids"],
                k=2,
            )
        blocks["source"] = self._truncate(assigned, "source")

        # 6. Two or three actual images, plus their textual half.
        blocks["figures"] = self._select_figures(section["chunk_ids"], self.MAX_FIGURES)
        blocks["figures_text"] = self._render_figures_text(blocks["figures"])

        blocks["_tokens"] = {
            k: count_tokens(v) for k, v in blocks.items() if isinstance(v, str)
        }
        blocks["_tokens"]["TOTAL"] = sum(blocks["_tokens"].values())
        return blocks
