"""Step 8 — Building toc.json."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

from .setup import BookSizeConfig, PipelineBConfig, toc_log
from ..llm import LLMClient


def rank_chunks_for_tags(
    tags: List[str],
    tag_to_chunks: Dict[str, List[str]],
    chunk_tags: Dict[str, List[str]],
    cfg: PipelineBConfig,
) -> List[str]:
    """
    Pick the chunks a section should be written from.

    Score = how many of the section's tags the chunk contains.
    Tie-break = how concentrated those tags are in the chunk, so a focused chunk
    beats a sprawling one that mentions the topic in passing.
    """
    overlap: Counter = Counter()
    for tag in tags:
        for chunk_id in tag_to_chunks.get(tag, []):
            overlap[chunk_id] += 1

    def sort_key(item):
        chunk_id, score = item
        density = score / max(1, len(chunk_tags.get(chunk_id, [])))
        return (-score, -density, chunk_id)

    ranked = [cid for cid, _ in sorted(overlap.items(), key=sort_key)]
    return ranked[: cfg.max_chunks_per_section]


def allocate_word_counts(sections: List[Dict], size: BookSizeConfig) -> None:
    """
    Give every section a word budget, in place.

    Weighted by how many concepts a section teaches, so a section covering seven
    concepts is allowed to be longer than one covering two -- then clamped to the
    design's 600-800 band (900 ceiling for the densest) so no section becomes an
    unreadable slab.
    """
    if not sections:
        return
    mean_tags = sum(len(s["tags"]) for s in sections) / len(sections) or 1
    base = size.total_words / len(sections)

    for section in sections:
        weight = len(section["tags"]) / mean_tags if mean_tags else 1.0
        words = base * (0.6 + 0.4 * weight)  # damped, not proportional
        words = max(size.min_section_words, min(size.max_section_words, words))
        section["estimated_word_count"] = int(round(words / 25) * 25)

    # The clamp keeps any single section readable, but it also means that if the
    # pipeline produced far fewer sections than the budget assumed, every one of
    # them pins to the maximum and the book quietly comes out short. Say so.
    allocated = sum(s["estimated_word_count"] for s in sections)
    drift = abs(allocated - size.total_words) / max(1, size.total_words)
    if drift > 0.10:
        toc_log.warning(
            f"  word budget: {len(sections)} sections x ~{base:.0f} words wanted "
            f"{size.total_words:,}, but clamping to "
            f"{size.min_section_words}-{size.max_section_words} allocated "
            f"{allocated:,} ({allocated / size.words_per_page:.0f} pages vs "
            f"{size.target_pages} requested)"
        )


def build_final_toc(
    ordered_chapters: List[Dict],
    chunk_tags: Dict[str, List[str]],
    llm: LLMClient,
    cfg: PipelineBConfig,
) -> Dict[str, Any]:
    """
    Produce toc.json in the exact shape Pipeline C reads.

        {"section_id": "sec_07_02", "chapter_id": "ch07", "title": "...",
         "tags": [...], "chunk_ids": [...], "estimated_word_count": 800}
    """
    tag_to_chunks: Dict[str, List[str]] = defaultdict(list)
    for chunk_id, tags in chunk_tags.items():
        for tag in tags:
            tag_to_chunks[tag].append(chunk_id)

    chapters_out: List[Dict[str, Any]] = []
    sections_out: List[Dict[str, Any]] = []

    for chapter_number, chapter in enumerate(ordered_chapters, 1):
        chapter_id = f"ch{chapter_number:02d}"
        chapters_out.append(
            {
                "chapter_id": chapter_id,
                "title": chapter["title"],
                "description": chapter.get("description", ""),
                "order": chapter_number,
                "section_count": len(chapter["sections"]),
            }
        )

        for section_number, section in enumerate(chapter["sections"], 1):
            section_id = f"sec_{chapter_number:02d}_{section_number:02d}"
            sections_out.append(
                {
                    "section_id": section_id,
                    "chapter_id": chapter_id,
                    "title": section["title"],
                    "tags": section["tags"],
                    "chunk_ids": rank_chunks_for_tags(
                        section["tags"], tag_to_chunks, chunk_tags, cfg
                    ),
                    "estimated_word_count": 0,  # filled in below
                    "order": section_number,
                }
            )

    allocate_word_counts(sections_out, cfg.size)

    # ---- book title -------------------------------------------------------
    titles = "\n".join(f"- {c['title']}" for c in chapters_out)
    try:
        title = (
            llm.generate(
                "You are an expert at creating book titles. Reply with the title only.",
                f"Based on these chapter titles, give this book a concise, professional "
                f"title:\n\n{titles}\n\nReply with just the title.",
                # The visible answer is a few words, but a reasoning-capable
                # model spends part of this budget on hidden thinking first
                # -- see the note in toc/setup.py. 40 tokens left it none.
                max_tokens=1_000,
                temperature=0.3,
            )
            .strip()
            .strip("\"'")
            .split("\n")[0]
        )
    except Exception as exc:
        toc_log.warning(f"Title generation failed: {exc}")
        title = "Technical Curriculum"

    used_chunks = {cid for s in sections_out for cid in s["chunk_ids"]}
    total_words = sum(s["estimated_word_count"] for s in sections_out)

    toc = {
        "book_title": title or "Technical Curriculum",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": llm.model_id,
        "chapters": chapters_out,
        "sections": sections_out,
        "stats": {
            "chapters": len(chapters_out),
            "sections": len(sections_out),
            "target_sections": cfg.size.target_sections,
            "total_estimated_words": total_words,
            "estimated_pages": round(total_words / cfg.size.words_per_page),
            "chunks_assigned": len(used_chunks),
            "chunks_total": len(chunk_tags),
            "chunks_unused": len(chunk_tags) - len(used_chunks),
            "avg_chunks_per_section": round(
                sum(len(s["chunk_ids"]) for s in sections_out)
                / max(1, len(sections_out)),
                1,
            ),
            "sections_with_no_chunks": sum(
                1 for s in sections_out if not s["chunk_ids"]
            ),
        },
    }
    return toc


def validate_toc(toc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Four structural checks over the finished TOC. No model, no analysis of
    meaning -- just the things that make a curriculum mechanically unusable.

        "Nothing here fixes a bad table of contents... That is a signal to fix
         the TOC, not to keep writing."
    """
    sections = toc["sections"]

    no_source = [s["section_id"] for s in sections if not s["chunk_ids"]]
    no_tags = [s["section_id"] for s in sections if not s["tags"]]

    seen: Dict[str, str] = {}
    duplicate_titles = []
    for section in sections:
        key = section["title"].strip().lower()
        if key in seen:
            duplicate_titles.append(
                {
                    "title": section["title"],
                    "sections": [seen[key], section["section_id"]],
                }
            )
        else:
            seen[key] = section["section_id"]

    with_sections = {s["chapter_id"] for s in sections}
    empty_chapters = [
        c["chapter_id"] for c in toc["chapters"] if c["chapter_id"] not in with_sections
    ]

    return {
        "sections": len(sections),
        "sections_without_source": no_source,
        "sections_without_tags": no_tags,
        "duplicate_titles": duplicate_titles,
        "chapters_without_sections": empty_chapters,
        "passed": not (no_source or no_tags or duplicate_titles or empty_chapters),
    }
