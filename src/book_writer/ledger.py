"""
Phase 1.3 — Layer 1: The Book Ledger.

The book's memory of itself: what it has defined, claimed, promised,
demonstrated, and consumed. Read by every section, written by the Archivist.

Not a prose summary -- a database of small typed facts, so the Continuity
Gate can check it mechanically and the Context Assembler can slice it cheaply.
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ingestion.setup import PATHS
from .setup import BOOK, writer_log

EMPTY_LEDGER = {
    "version": 0,
    "last_section": None,
    "concepts": {},  # prevents drift and redefinition (#2, #5)
    "claims": [],  # prevents contradiction (#6)
    "promises": [],  # prevents broken promises and phantoms (#3, #4)
    "examples": {},  # prevents orphaned examples (#7)
    "coverage": {},  # prevents repetition (#1)
    "section_summaries": {},  # the book's memory of itself
    "chapter_rollups": {},  # keeps the ledger small in the prompt
    "figures_used": {},  # figure_id -> sections that leaned on it
}

CONCEPT_DEPTHS = ("unwritten", "mentioned", "introduced", "explained")


class BookLedger:
    """Layer 1."""

    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(EMPTY_LEDGER, indent=2), encoding="utf-8")
        self._cache = json.loads(path.read_text(encoding="utf-8"))
        self.ledger_diff_log = BOOK.ledger_diffs
        # tolerate a ledger written by an older version of this notebook
        for key, blank in EMPTY_LEDGER.items():
            self._cache.setdefault(
                key, blank if not isinstance(blank, (dict, list)) else type(blank)()
            )

    # ---------------------------------------------------------------- writes --
    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.path)  # atomic: a crash mid-write cannot corrupt it

    @staticmethod
    def key_for(term: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", term.lower().strip()).strip("_") or "concept"

    # ----------------------------------------------------------------- reads --
    def resolve_alias(self, term: str) -> Optional[str]:
        """'thought-action cycle' -> 'react_loop'. This is the drift detector."""
        t = term.lower().strip()
        for key, c in self._cache["concepts"].items():
            if t == c["canonical_name"].lower() or t in [
                a.lower() for a in c["aliases"]
            ]:
                return key
        return None

    def undefined_prerequisites(self, concept_keys: List[str]) -> List[str]:
        """Concepts this section needs that the book has not defined yet (#5)."""
        missing = []
        for term in concept_keys:
            key = self.resolve_alias(term) or self.key_for(term)
            c = self._cache["concepts"].get(key)
            if not c:
                continue
            for pre in c.get("prerequisites", []):
                pkey = self.resolve_alias(pre) or self.key_for(pre)
                p = self._cache["concepts"].get(pkey)
                if p and p["depth"] in ("unwritten", "mentioned"):
                    missing.append(p["canonical_name"])
        return sorted(set(missing))

    def defined_prerequisites(self, concept_keys: List[str]) -> List[Dict]:
        """
        Concepts this section depends on that the book HAS already defined.

        Tag overlap alone is not enough. A section tagged only `transformers`
        will certainly USE self-attention -- the prerequisite relation says so
        -- but self-attention is not one of its tags, so the Writer would
        never be shown its agreed definition or its alias list, and could
        reintroduce it under a variant name. That is failure #2 arriving
        through the front door.
        """
        out: Dict[str, Dict] = {}
        for term in concept_keys:
            key = self.resolve_alias(term) or self.key_for(term)
            c = self._cache["concepts"].get(key)
            if not c:
                continue
            for pre in c.get("prerequisites", []):
                pkey = self.resolve_alias(pre) or self.key_for(pre)
                p = self._cache["concepts"].get(pkey)
                if p and p.get("definition"):
                    out[pkey] = p
        return list(out.values())

    def open_promises_for(self, chapter_id: str) -> List[Dict]:
        return [
            p
            for p in self._cache["promises"]
            if p["status"] == "open" and p.get("target_hint") == chapter_id
        ]

    def all_open_promises(self) -> List[Dict]:
        return [p for p in self._cache["promises"] if p["status"] == "open"]

    def chunk_already_used(self, chunk_id: str) -> List[str]:
        return self._cache["coverage"].get(chunk_id, {}).get("used_by", [])

    def concepts_for_tags(self, tags: List[str]) -> List[Dict]:
        """The defined concepts whose source tags overlap this section's tags."""
        want = {t.lower() for t in tags}
        return [
            c
            for c in self._cache["concepts"].values()
            if c.get("definition")
            and (
                {s.lower() for s in c.get("source_tags", [])} & want
                or c["canonical_name"].lower() in want
            )
        ]

    def stats(self) -> Dict[str, int]:
        concepts = self._cache["concepts"].values()
        return {
            "version": self._cache["version"],
            "concepts": len(self._cache["concepts"]),
            "concepts_defined": sum(1 for c in concepts if c.get("definition")),
            "claims": len(self._cache["claims"]),
            "promises_open": sum(
                1 for p in self._cache["promises"] if p["status"] == "open"
            ),
            "promises_fulfilled": sum(
                1 for p in self._cache["promises"] if p["status"] == "fulfilled"
            ),
            "examples": len(self._cache["examples"]),
            "chunks_used": len(self._cache["coverage"]),
            "sections_written": len(self._cache["section_summaries"]),
        }

    # ------------------------------------------------------------- seeding ---
    def seed_from_toc_pipeline(
        self,
        normalized_tags: Dict[str, List[str]],
        tag_relationships: Dict[str, List[str]],
    ) -> int:
        """
        Pre-populate the concept registry before any writing begins.

        "The drift detector works from the first section rather than slowly
         learning the vocabulary as it goes."

        Idempotent: re-seeding an existing ledger enriches aliases rather than
        wiping definitions the Archivist has already harvested.
        """
        added = 0
        for canonical, raw_variants in normalized_tags.items():
            key = self.key_for(canonical)
            entry = self._cache["concepts"].get(key)
            if entry is None:
                self._cache["concepts"][key] = {
                    "canonical_name": canonical,
                    "aliases": sorted(set(raw_variants)),
                    "definition": None,
                    "defined_in": None,
                    "depth": "unwritten",
                    "referenced_in": [],
                    "source_tags": [canonical],
                    "prerequisites": tag_relationships.get(canonical, []),
                }
                added += 1
            else:
                for alias in raw_variants:
                    if (
                        alias != entry["canonical_name"]
                        and alias not in entry["aliases"]
                    ):
                        entry["aliases"].append(alias)
                for pre in tag_relationships.get(canonical, []):
                    if pre not in entry["prerequisites"]:
                        entry["prerequisites"].append(pre)
        self._flush()
        return added

    # ----------------------------- writes (the Archivist is the only caller) --
    def apply_archivist_update(
        self, update: Dict[str, Any], section_id: str
    ) -> Dict[str, Any]:
        """
        Fold one harvested section into the ledger, and record the diff.

        The design's first stated risk is that the Archivist can be wrong and
        its errors compound. A diff log is what makes a bad harvest traceable
        rather than mysterious.
        """
        diff = {
            "section_id": section_id,
            "at": datetime.now().isoformat(timespec="seconds"),
            "concepts_new": [],
            "concepts_enriched": [],
            "aliases_added": [],
            "claims": 0,
            "promises_made": [],
            "promises_fulfilled": [],
            "examples_touched": [],
            "chunks": [],
        }

        # ---- concepts defined ---------------------------------------------
        for c in update.get("concepts_defined", []):
            term = str(c.get("term", "")).strip()
            if not term:
                continue
            key = self.resolve_alias(term) or self.key_for(term)
            entry = self._cache["concepts"].get(key)
            if entry is None:
                entry = {
                    "canonical_name": term,
                    "aliases": [],
                    "definition": None,
                    "defined_in": None,
                    "depth": "unwritten",
                    "referenced_in": [],
                    "source_tags": [],
                    "prerequisites": [],
                }
                self._cache["concepts"][key] = entry
                diff["concepts_new"].append(key)

            # enrich in place, never duplicate
            if entry["definition"] is None:
                entry["definition"] = str(c.get("definition", "")).strip() or None
                entry["defined_in"] = section_id
                entry["depth"] = "explained"
                diff["concepts_enriched"].append(key)
            if term != entry["canonical_name"] and term not in entry["aliases"]:
                entry["aliases"].append(term)
                diff["aliases_added"].append(f"{key}<-{term}")

        # ---- concepts referenced ------------------------------------------
        for term in update.get("concepts_referenced", []):
            key = self.resolve_alias(str(term))
            if not key:
                continue
            entry = self._cache["concepts"][key]
            if section_id not in entry["referenced_in"]:
                entry["referenced_in"].append(section_id)
            if entry["depth"] == "unwritten":
                entry["depth"] = "mentioned"

        # ---- claims --------------------------------------------------------
        for claim in update.get("claims", []):
            claim = dict(claim)
            claim.setdefault("claim_id", f"c_{len(self._cache['claims']) + 1:04d}")
            claim["section_id"] = section_id
            claim.setdefault("tags", update.get("summary", {}).get("teaches", []))
            self._cache["claims"].append(claim)
            diff["claims"] += 1

        # ---- promises ------------------------------------------------------
        for promise in update.get("promises_made", []):
            promise = dict(promise)
            promise.setdefault(
                "promise_id", f"p_{len(self._cache['promises']) + 1:04d}"
            )
            promise["made_in"] = section_id
            promise["status"] = "open"
            self._cache["promises"].append(promise)
            diff["promises_made"].append(promise["promise_id"])

        for pid in update.get("promises_fulfilled", []):
            for p in self._cache["promises"]:
                if p["promise_id"] == pid and p["status"] == "open":
                    p["status"] = "fulfilled"
                    p["fulfilled_in"] = section_id
                    diff["promises_fulfilled"].append(pid)

        # ---- running examples ----------------------------------------------
        for name, state in update.get("example_states", {}).items():
            ex = self._cache["examples"].setdefault(
                name, {"introduced_in": section_id, "files_shown": []}
            )
            ex["current_state"] = state
            ex["last_touched"] = section_id
            diff["examples_touched"].append(name)

        # ---- coverage ------------------------------------------------------
        for cid, depth in update.get("chunks_used", {}).items():
            cov = self._cache["coverage"].setdefault(
                cid, {"used_by": [], "depth": depth}
            )
            if section_id not in cov["used_by"]:
                cov["used_by"].append(section_id)
            # a chunk used as primary anywhere is primary
            if depth == "primary":
                cov["depth"] = "primary"
            diff["chunks"].append(f"{cid}:{depth}")

        # ---- summary + bookkeeping -----------------------------------------
        summary = update.get("summary") or {}
        self._cache["section_summaries"][section_id] = summary
        self._cache["version"] += 1
        self._cache["last_section"] = section_id
        self._flush()

        with self.ledger_diff_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(diff, ensure_ascii=False) + "\n")

        return diff

    def figure_use_count(self, figure_id: str) -> int:
        return len(self._cache.get("figures_used", {}).get(figure_id, []))

    def record_figures_used(self, figure_ids: List[str], section_id: str) -> None:
        """The design budgets 2-3 figures per section and tracks usage so the
        same diagram is not leaned on twice. This is the tracking half."""
        if not figure_ids:
            return
        used = self._cache.setdefault("figures_used", {})
        for fid in figure_ids:
            entries = used.setdefault(fid, [])
            if section_id not in entries:
                entries.append(section_id)
        self._flush()

    def add_chapter_rollup(self, chapter_id: str, text: str) -> None:
        self._cache["chapter_rollups"][chapter_id] = text
        self._flush()


# ---------------------------------------------------------------------------
# Phase 1.4 — Seed the ledger from Pipeline B, and load the TOC
# ---------------------------------------------------------------------------
#     "Before section 1 is written, the Book Ledger is PRE-LOADED with every
#      canonical concept and every alias the sources use for it."
#
# This is the join between Pipeline B and Pipeline C. If it fails, the drift
# detector cold-starts and failure #2 goes uncovered for the first several
# chapters, so it fails loudly rather than quietly.


def load_toc() -> Dict[str, Any]:
    if not BOOK.toc.exists():
        raise FileNotFoundError(
            f"{BOOK.toc} not found — run the TOC Generation section first."
        )
    toc = json.loads(BOOK.toc.read_text(encoding="utf-8"))

    required = {
        "section_id",
        "chapter_id",
        "title",
        "tags",
        "chunk_ids",
        "estimated_word_count",
    }
    for section in toc["sections"]:
        missing = required - set(section)
        if missing:
            raise ValueError(
                f"{section.get('section_id', '?')} is missing {missing} — "
                f"toc.json does not match the design's contract."
            )
    return toc


def seed_ledger_from_pipeline_b(ledger: BookLedger) -> Dict[str, int]:
    tag_file, rel_file = PATHS.normalized_tags, PATHS.tag_relationships
    if not tag_file.exists():
        raise FileNotFoundError(
            f"{tag_file} not found. Pipeline B writes it, and the ledger is seeded "
            f"from it — without it the drift detector starts empty."
        )

    normalized_tags = json.loads(tag_file.read_text(encoding="utf-8"))
    relationships = (
        json.loads(rel_file.read_text(encoding="utf-8")) if rel_file.exists() else {}
    )
    if not rel_file.exists():
        writer_log.warning(
            f"{rel_file} not found — concepts will be seeded with no "
            f"prerequisites, so the Continuity Gate's check #3 cannot fire."
        )

    added = ledger.seed_from_toc_pipeline(normalized_tags, relationships)
    return {
        "canonical_concepts": len(normalized_tags),
        "newly_added": added,
        "with_prerequisites": len(relationships),
        "total_aliases": sum(len(v) for v in normalized_tags.values()),
    }
