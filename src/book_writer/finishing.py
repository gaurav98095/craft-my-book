"""Phase 6 — The Finishing Passes."""

import re
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from ..llm import LLMClient
from .setup import BOOK, writer_log
from .ledger import BookLedger
from .draft_store import DraftStore
from .source_memory import SourceMemory

CONTRADICTION_SCHEMA = {
    "conflict": "boolean",
    "explanation": "string — empty if no conflict",
}


# ---------------------------------------------------------------------------
# Passes 1 and 2: promises, glossary, index
# ---------------------------------------------------------------------------


def resolve_promises(
    ledger: BookLedger, drafts: DraftStore, similarity_threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Pass 1. Before reporting a promise as broken, search the book for it --
    a section may have delivered without the Archivist noticing.
    """
    resolved, gaps, orphaned = [], [], []

    for p in ledger.all_open_promises():
        candidates = drafts.find_similar(p["text"], k=3)
        if candidates and candidates[0]["similarity"] > similarity_threshold:
            p["status"] = "fulfilled"
            p["fulfilled_in"] = candidates[0]["section_id"]
            p["fulfilled_by"] = "resolution_pass"
            resolved.append(p)
        elif p.get("target_hint"):
            gaps.append(p)  # a human decides: write it, or cut it
        else:
            p["status"] = "orphaned"  # the promise sentence gets edited out
            orphaned.append(p)

    ledger._flush()
    return {"auto_resolved": resolved, "gaps": gaps, "orphaned": orphaned}


def generate_glossary(ledger: BookLedger) -> str:
    """Pass 2a. The concept registry already IS a glossary."""
    lines = ["# Glossary\n"]
    for c in sorted(
        ledger._cache["concepts"].values(), key=lambda x: x["canonical_name"].lower()
    ):
        if not c.get("definition"):
            continue
        refs = ", ".join(c["referenced_in"][:5])
        entry = f"**{c['canonical_name']}** — {c['definition']}  \n*Introduced in {c['defined_in']}"
        if refs:
            entry += f". Also discussed in {refs}"
        entry += ".*\n"
        if c["aliases"]:
            entry += f"*Also written as: {', '.join(c['aliases'][:5])}.*\n"
        lines.append(entry)
    return "\n".join(lines)


def generate_index(ledger: BookLedger) -> str:
    """Pass 2b. Every concept, and every section that touches it."""
    lines = ["# Index\n"]
    entries = []
    for c in ledger._cache["concepts"].values():
        places = ([c["defined_in"]] if c["defined_in"] else []) + c["referenced_in"]
        if places:
            entries.append((c["canonical_name"], sorted(set(places))))
    for name, places in sorted(entries, key=lambda kv: kv[0].lower()):
        lines.append(f"**{name}** — {', '.join(places)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Passes 3, 4, 5: transitions, contradictions, coverage
# ---------------------------------------------------------------------------


def find_rough_seams(
    toc: Dict, ledger: BookLedger, drafts: DraftStore, max_report: int = 25
) -> List[Dict]:
    """
    Pass 3. Read consecutive section pairs -- the closing_line of one and the
    opening of the next -- and flag the seams where a bridge is needed.

    Deliberately a REPORT, not an automatic rewrite: inserting generated
    sentences between finished sections is the one finishing pass that can
    make the prose worse, and it should be a human decision.
    """
    order = [s["section_id"] for s in toc["sections"]]
    summaries = ledger._cache["section_summaries"]
    seams = []

    for prev_id, next_id in zip(order, order[1:]):
        prev_sum = summaries.get(prev_id)
        if not prev_sum or next_id not in summaries:
            continue
        closing = (prev_sum.get("closing_line") or "").strip()
        opening = drafts.get_full(next_id)[:400].strip()
        if not closing or not opening:
            continue

        # A seam is rough when the closing line does not set up the opening
        # at all: no shared vocabulary, and no forward gesture.
        closing_words = set(re.findall(r"[a-z]{4,}", closing.lower()))
        opening_words = set(re.findall(r"[a-z]{4,}", opening.lower()))
        gestures_forward = bool(
            re.search(
                r"\bnext\b|\bnow\b|\bfollow|\bturn to\b|\bbut\b|\byet\b",
                closing.lower(),
            )
        )
        if not (closing_words & opening_words) and not gestures_forward:
            seams.append(
                {
                    "from": prev_id,
                    "to": next_id,
                    "closing_line": closing,
                    "opening": opening[:160],
                }
            )
    return seams[:max_report]


def contradiction_sweep(ledger: BookLedger, llm: LLMClient, max_pairs: int = 40) -> List[Dict]:
    """
    Pass 4. Cluster the claims log by tag; within each cluster ask whether
    any pair conflicts. Because claims are short and typed, this is a few
    dozen tiny calls rather than a re-read of the whole book.
    """
    by_tag: Dict[str, List[Dict]] = defaultdict(list)
    for claim in ledger._cache["claims"]:
        for tag in claim.get("tags", []) or ["_untagged"]:
            by_tag[tag].append(claim)

    checked, conflicts, seen_pairs = 0, [], set()
    for tag, claims in by_tag.items():
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                a, b = claims[i], claims[j]
                if a["section_id"] == b["section_id"]:
                    continue
                pair = tuple(
                    sorted((a.get("claim_id", str(i)), b.get("claim_id", str(j))))
                )
                if pair in seen_pairs or checked >= max_pairs:
                    continue
                seen_pairs.add(pair)
                checked += 1

                verdict = llm.generate_structured(
                    "You check whether two statements from the same book contradict "
                    "each other. Differences of emphasis or scope are NOT contradictions.",
                    f"A ({a['section_id']}): {a['text']}\nB ({b['section_id']}): {b['text']}\n\n"
                    f"Do these contradict?",
                    CONTRADICTION_SCHEMA,
                    # See the reasoning-budget note in book_writer/setup.py.
                    max_tokens=1_500,
                )
                if verdict and verdict.get("conflict"):
                    conflicts.append(
                        {
                            "tag": tag,
                            "a": a,
                            "b": b,
                            "explanation": verdict.get("explanation", ""),
                        }
                    )
    return conflicts


def coverage_report(ledger: BookLedger, source: SourceMemory) -> Dict[str, Any]:
    """Pass 5. Source material that never made it into the book."""
    all_ids = source.all_chunk_ids()
    unused = [cid for cid in all_ids if not ledger.chunk_already_used(cid)]

    by_document: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"used": 0, "unused": 0}
    )
    for cid in all_ids:
        doc = source.meta.get(cid, {}).get("source_document", "unknown")
        by_document[doc]["unused" if cid in set(unused) else "used"] += 1

    # A document that contributed nothing at all is the signal worth acting
    # on: usually a topic the TOC clustering dropped entirely.
    silent = [doc for doc, counts in by_document.items() if counts["used"] == 0]
    return {
        "chunks_total": len(all_ids),
        "chunks_unused": len(unused),
        "coverage_pct": round(
            100 * (len(all_ids) - len(unused)) / max(1, len(all_ids)), 1
        ),
        "by_document": dict(by_document),
        "documents_never_used": silent,
        "unused_sample": unused[:20],
    }


# ---------------------------------------------------------------------------
# Assemble the manuscript and the completeness report
# ---------------------------------------------------------------------------


def assemble_manuscript(toc: Dict, drafts: DraftStore, ledger: BookLedger) -> str:
    """Stitch the sections into one manuscript, in TOC order."""
    lines = [f"# {toc['book_title']}", ""]
    for chapter in toc["chapters"]:
        lines.append(f"\n# {chapter['order']}. {chapter['title']}\n")
        rollup = ledger._cache["chapter_rollups"].get(chapter["chapter_id"])
        if rollup:
            lines.append(f"> {rollup}\n")
        for section in toc["sections"]:
            if section["chapter_id"] != chapter["chapter_id"]:
                continue
            body = drafts.get_full(section["section_id"])
            if body:
                lines.append(body.rstrip() + "\n")
            else:
                lines.append(
                    f"## {section['title']}\n\n*[NOT YET WRITTEN — "
                    f"{section['section_id']}]*\n"
                )
    return "\n".join(lines)


def run_finishing_passes(
    toc: Dict,
    ledger: BookLedger,
    drafts: DraftStore,
    source: SourceMemory,
    llm: LLMClient,
    run_contradictions: bool = True,
) -> Dict[str, Any]:
    """All five passes, then write the book directory."""
    BOOK.book.mkdir(parents=True, exist_ok=True)
    writer_log.info("=" * 70)
    writer_log.info("FINISHING PASSES")
    writer_log.info("=" * 70)

    writer_log.info("Pass 1: promise resolution")
    promises = resolve_promises(ledger, drafts)

    writer_log.info("Pass 2: glossary and index")
    (BOOK.book / "glossary.md").write_text(generate_glossary(ledger), encoding="utf-8")
    (BOOK.book / "index.md").write_text(generate_index(ledger), encoding="utf-8")

    writer_log.info("Pass 3: transitions")
    seams = find_rough_seams(toc, ledger, drafts)

    conflicts: List[Dict] = []
    if run_contradictions:
        writer_log.info("Pass 4: contradiction sweep")
        conflicts = contradiction_sweep(ledger, llm)

    writer_log.info("Pass 5: coverage")
    coverage = coverage_report(ledger, source)

    manuscript = assemble_manuscript(toc, drafts, ledger)
    (BOOK.book / "manuscript.md").write_text(manuscript, encoding="utf-8")

    written = set(ledger._cache["section_summaries"])
    planned = [s["section_id"] for s in toc["sections"]]
    report = {
        "book_title": toc["book_title"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sections_written": len(written),
        "sections_planned": len(planned),
        "sections_missing": [s for s in planned if s not in written],
        "words": len(manuscript.split()),
        "estimated_pages": round(len(manuscript.split()) / 350),
        "promises": {
            "auto_resolved": len(promises["auto_resolved"]),
            "gaps": [
                {
                    "promise_id": p["promise_id"],
                    "text": p["text"],
                    "made_in": p["made_in"],
                    "target": p.get("target_hint"),
                }
                for p in promises["gaps"]
            ],
            "orphaned": len(promises["orphaned"]),
        },
        "rough_seams": seams,
        "contradictions": conflicts,
        "coverage": coverage,
        "ledger": ledger.stats(),
    }
    (BOOK.book / "completeness_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# Completeness report — {toc['book_title']}",
        "",
        f"Sections written: {report['sections_written']} / {report['sections_planned']}",
        f"Length: {report['words']:,} words ≈ {report['estimated_pages']} pages",
        "",
    ]
    if report["sections_missing"]:
        lines += (
            ["## Sections never written", ""]
            + [f"- {s}" for s in report["sections_missing"]]
            + [""]
        )
    if report["promises"]["gaps"]:
        lines += (
            ["## Promises the book did not keep", ""]
            + [
                f"- **{p['text']}** — promised in {p['made_in']}, due in {p['target']}"
                for p in report["promises"]["gaps"]
            ]
            + [""]
        )
    if conflicts:
        lines += (
            ["## Possible contradictions", ""]
            + [
                f"- {c['a']['section_id']}: \"{c['a']['text']}\"  \n"
                f"  vs {c['b']['section_id']}: \"{c['b']['text']}\"  \n"
                f"  → {c['explanation']}"
                for c in conflicts
            ]
            + [""]
        )
    if seams:
        lines += (
            ["## Rough seams between sections", ""]
            + [
                f"- {s['from']} → {s['to']}: ends \"{s['closing_line'][:80]}…\""
                for s in seams
            ]
            + [""]
        )
    lines += [
        "## Source coverage",
        "",
        f"- {coverage['coverage_pct']}% of chunks used "
        f"({coverage['chunks_unused']} never used)",
    ]
    if coverage["documents_never_used"]:
        lines += [
            f"- **Documents that contributed nothing:** "
            f"{', '.join(coverage['documents_never_used'])}"
        ]
    (BOOK.book / "completeness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    return report
