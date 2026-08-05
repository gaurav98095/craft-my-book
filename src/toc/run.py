"""Pipeline B — Run: the eight steps, in order."""

import json
import time
from typing import Dict, List, Tuple

from ..ingestion.setup import PATHS
from .setup import PipelineBConfig, toc_log
from ..llm import LLMClient
from .checkpoints import StepCheckpoints
from .step2_tag_extraction import run_step2_extraction
from .step3_tag_normalization import run_step3_normalization
from .curriculum_structure import (
    ThemeDiscovery,
    TagAssigner,
    allocate_section_quota,
    SectionFormer,
    CurriculumOrderer,
)
from .toc_builder import build_final_toc, validate_toc
from tqdm.auto import tqdm


def run_pipeline_b(
    chunks: List[Dict], llm: LLMClient, cfg: PipelineBConfig, ckpt: StepCheckpoints
) -> Tuple[Dict, Dict]:
    """The eight steps, in order."""
    toc_log.info("=" * 70)
    toc_log.info("PIPELINE B: TABLE OF CONTENTS GENERATION")
    toc_log.info("=" * 70)
    started = time.time()

    # -- Steps 2 & 3 --------------------------------------------------------
    chunk_tags, relationships, summaries = run_step2_extraction(chunks, llm, cfg, ckpt)
    normalized_chunk_tags, normalized_relationships, aliases = run_step3_normalization(
        chunk_tags, relationships, llm, cfg, ckpt
    )

    # -- Steps 4-7 ----------------------------------------------------------
    cached = ckpt.load("step4to7_clustering")
    if cached:
        ordered_chapters = cached["ordered_chapters"]
    else:
        toc_log.info(">>> STEP 4: THEME DISCOVERY")
        themes = ThemeDiscovery(llm, cfg).discover(normalized_chunk_tags)
        toc_log.info(f"  {len(themes)} themes: {[t['title'] for t in themes]}")

        toc_log.info(">>> STEP 5: TAG ASSIGNMENT")
        tags_by_theme = TagAssigner(llm, cfg).assign(themes, normalized_chunk_tags)
        for theme in themes:
            toc_log.info(
                f"  {theme['id']}: {len(tags_by_theme.get(theme['id'], []))} concepts"
            )

        # Chapters that received nothing are not chapters.
        themes = [t for t in themes if tags_by_theme.get(t["id"])]

        toc_log.info(">>> STEP 6: SECTION FORMATION")
        quota = allocate_section_quota(
            {t["id"]: len(tags_by_theme[t["id"]]) for t in themes}, cfg.size
        )
        toc_log.info(
            f"  section budget {cfg.size.target_sections} allocated as: "
            f"{ {t['title'][:24]: quota[t['id']] for t in themes} }"
        )

        former = SectionFormer(llm, cfg)
        chapters: List[Dict] = []
        for theme in tqdm(themes, desc="Forming sections"):
            tags = tags_by_theme[theme["id"]]
            sections = former.form(theme, tags, quota[theme["id"]])
            for i, section in enumerate(sections, 1):
                section["id"] = f"{theme['id']}_s{i:02d}"
            chapters.append({**theme, "tags": tags, "sections": sections})

        toc_log.info(">>> STEP 7: CURRICULUM ORDERING")
        orderer = CurriculumOrderer(llm, cfg)
        ordered_chapters = orderer.order(chapters, "chapters")
        for chapter in ordered_chapters:
            chapter["sections"] = orderer.order(
                chapter["sections"], f"sections of {chapter['title'][:28]}"
            )

        ckpt.save("step4to7_clustering", {"ordered_chapters": ordered_chapters})

    # -- Step 8 -------------------------------------------------------------
    toc_log.info(">>> STEP 8: BUILD toc.json")
    toc = build_final_toc(ordered_chapters, normalized_chunk_tags, llm, cfg)
    PATHS.toc.parent.mkdir(parents=True, exist_ok=True)
    PATHS.toc.write_text(
        json.dumps(toc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # A human-readable companion, for reading the curriculum by eye.
    lines = [f"# {toc['book_title']}", ""]
    for chapter in toc["chapters"]:
        lines.append(f"\n## {chapter['order']}. {chapter['title']}")
        for section in toc["sections"]:
            if section["chapter_id"] == chapter["chapter_id"]:
                lines.append(
                    f"{chapter['order']}.{section['order']} {section['title']}"
                    f"  *({section['estimated_word_count']} words, "
                    f"{len(section['chunk_ids'])} chunks)*"
                )
    (PATHS.toc.parent / "toc.md").write_text("\n".join(lines), encoding="utf-8")

    # -- Validation ---------------------------------------------------------
    toc_log.info(">>> VALIDATION")
    report = validate_toc(toc)
    (PATHS.toc.parent / "toc_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    toc_log.info("-" * 70)
    toc_log.info(f"  chapters : {toc['stats']['chapters']}")
    toc_log.info(
        f"  sections : {toc['stats']['sections']} "
        f"(target {toc['stats']['target_sections']})"
    )
    toc_log.info(
        f"  words    : {toc['stats']['total_estimated_words']:,} "
        f"≈ {toc['stats']['estimated_pages']} pages"
    )
    toc_log.info(f"  elapsed  : {(time.time() - started) / 60:.1f} min")

    return toc, report
