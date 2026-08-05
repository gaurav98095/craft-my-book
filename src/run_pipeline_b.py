"""
Pipeline B — Autonomous Table of Contents Generation, run end to end.

Requires Pipeline A to have produced data/chunk_metadata.json and
data/chunks/*.txt (run_pipeline_a.py). Produces storage/toc.json,
storage/toc.md, and the tag artifacts Pipeline C's Book Ledger seeds from.

    python -m src.run_pipeline_b
"""

import json
from pathlib import Path

from .ingestion.setup import PATHS
from .toc.setup import PipelineBConfig, log_config, toc_log
from .toc.checkpoints import StepCheckpoints
from .toc.step1_load_chunks import load_chunks_from_pipeline_a
from .toc.run import run_pipeline_b
from .llm import build_llm_client


def report(toc, validation) -> None:
    """
    Four numbers decide whether the TOC is usable:

      * sections vs target      -- the book's actual length
      * chunks unused           -- source material no section will ever see
      * sections with no source -- sections that would be written from nothing
      * duplicate titles        -- the same thing taught twice

    And then read storage/toc.md. Nothing in this pipeline can tell you that
    two chapters overlap or that a chapter title is vague; two minutes of
    reading can.

        "Nothing here fixes a bad table of contents... That is a signal to
         fix the TOC, not to keep writing."
    """
    s = toc["stats"]

    print("=" * 72)
    print(f"PIPELINE B COMPLETE — \"{toc['book_title']}\"")
    print("=" * 72)
    print(f"Model                    : {toc['model']}")
    print(f"Chapters                 : {s['chapters']}")
    print(
        f"Sections                 : {s['sections']}   (target {s['target_sections']}, "
        f"design band 120-250)"
    )
    print(
        f"Estimated length         : {s['total_estimated_words']:,} words "
        f"≈ {s['estimated_pages']} pages"
    )
    print()
    print(f"Chunks assigned          : {s['chunks_assigned']} / {s['chunks_total']}")
    print(
        f"  never assigned         : {s['chunks_unused']}"
        "   (still reachable via find_related)"
    )
    print(f"  avg per section        : {s['avg_chunks_per_section']}")

    print()
    print("Structural validation:")
    print(
        f"  sections with NO source: {len(validation['sections_without_source'])}"
        f"{'   <-- these would be written from nothing' if validation['sections_without_source'] else ''}"
    )
    print(f"  sections with NO tags  : {len(validation['sections_without_tags'])}")
    print(f"  duplicate titles       : {len(validation['duplicate_titles'])}")
    for d in validation["duplicate_titles"][:3]:
        print(f"    \"{d['title']}\"  in {d['sections']}")
    print(f"  empty chapters         : {len(validation['chapters_without_sections'])}")
    print(
        f"  verdict                : {'PASS' if validation['passed'] else 'NEEDS ATTENTION'}"
    )

    print()
    print("Artifacts for Pipeline C:")
    for label, path in [
        ("toc.json", PATHS.toc),
        ("normalized_tags.json", PATHS.normalized_tags),
        ("tag_relationships.json", PATHS.tag_relationships),
        ("chunk_tags.json", PATHS.chunk_tags),
    ]:
        exists = "ok" if Path(path).exists() else "MISSING"
        print(f"  {label:24s} -> {path}   [{exists}]")

    aliases = json.loads(PATHS.normalized_tags.read_text(encoding="utf-8"))
    prereqs = json.loads(PATHS.tag_relationships.read_text(encoding="utf-8"))
    print(
        f"\n  concepts with aliases  : {sum(1 for v in aliases.values() if v)} "
        f"/ {len(aliases)}      (seeds drift detection)"
    )
    print(
        f"  concepts with prereqs  : {len(prereqs)}"
        "                (seeds the Continuity Gate in Pipeline C)"
    )
    print("=" * 72)


def main() -> None:
    bcfg = PipelineBConfig()
    llm = build_llm_client(logger=toc_log)
    log_config(bcfg, llm.model_id)

    ckpt = StepCheckpoints(PATHS.checkpoints, enabled=bcfg.resume)

    chunks = load_chunks_from_pipeline_a(bcfg)
    print(f"Step 1: {len(chunks)} chunks loaded (chunking happened once, in Stage 4)")

    toc, validation = run_pipeline_b(chunks, llm, bcfg, ckpt)
    report(toc, validation)
    llm.cleanup()


if __name__ == "__main__":
    main()
