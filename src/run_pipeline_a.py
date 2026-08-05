"""
Pipeline A — Document Preprocessing, run end to end.

Drop your sources into data/raw_sources/ and run:

    python -m src.run_pipeline_a

Each stage skips cleanly if its inputs are missing or its optional
dependencies (faster-whisper, ffmpeg, chromadb, ...) are not installed, so
this can be re-run at any point in the corpus's life.
"""

import json

from .ingestion.setup import PATHS
from .ingestion.stage1_speech import Stage1Config, run_stage1_speech
from .ingestion.stage2_parsing import (
    Stage2Config,
    run_stage2_parsing,
    analyse_parsed_outputs,
)
from .ingestion.stage3_figures import (
    Stage3Config,
    FallbackDescriber,
    run_stage3,
    stage3_log,
)
from .ingestion.stage4_chunking import (
    Stage4Config,
    run_stage4,
    get_chunk_neighbourhood,
)
from .ingestion.stage5_source_index import build_source_index
from .llm import LLMClient, build_llm_client


def report_stage2(summary) -> None:
    print("=" * 70)
    print("STAGE 2 OUTPUT")
    print("=" * 70)
    print(f"Documents parsed : {summary['documents']}")
    print(f"Content items    : {summary['items']:,}")
    print(f"By source type   : {summary['by_source_type']}")

    print("\nContent types (what Stage 3 will have to deal with):")
    for t, n in sorted(summary["types"].items(), key=lambda kv: -kv[1]):
        share = 100 * n / summary["items"] if summary["items"] else 0
        note = ""
        if t == "discarded":
            note = "   <-- layout furniture; Stage 3 drops these"
        elif t == "image":
            note = "   <-- each of these is a potential VLM call"
        print(f"  {t:12s} {n:6,}  ({share:4.1f}%){note}")

    print(f"\nImage crops resolved : {summary['images_resolved']}")
    print(
        f"Image crops MISSING  : {summary['images_missing']}"
        f"{'   <-- investigate before running Stage 3' if summary['images_missing'] else ''}"
    )

    print("\nLargest documents:")
    for name, n in summary["largest"][:5]:
        print(f"  {n:6,} items   {name}")
    print("=" * 70)


def report_stage3(stage3_summary, model: LLMClient) -> None:
    t = stage3_summary["totals"]
    kept = t["figures_kept"] + t["tables_kept"] + t["equations_kept"]
    seen = kept + t["decorative"]

    print("=" * 70)
    print("STAGE 3 REPORT")
    print("=" * 70)
    print(f"Model actually used          : {stage3_summary['model']}")

    print(
        f"\nLayout items dropped         : {t['dropped_layout']}"
        "   (headers, footers, page numbers)"
    )
    print(f"Multimodal items seen        : {seen}")
    print(
        f"  described and kept         : {kept}"
        f"  ({t['figures_kept']} figures, {t['tables_kept']} tables, "
        f"{t['equations_kept']} equations)"
    )
    print(f"  decorative / declined      : {t['decorative']}")

    print(
        f"\nDescribed at document level  : {t['document_level']}   "
        "(the design's path: whole document, one call)"
    )
    print(f"Fell back to per-figure      : {t['fallback_used']}")
    print(f"  fallback cache hits        : {stage3_summary['fallback']['cache_hits']}")

    print(f"\nModel calls total            : {stage3_summary['model_calls']}")
    print(f"  JSON repairs needed        : {stage3_summary['structured_repairs']}")
    print(
        f"  JSON failures (items lost) : {stage3_summary['structured_failures']}"
        f"{'   <-- check the log' if stage3_summary['structured_failures'] else ''}"
    )

    print(f"\nFigure crops on disk         : {len(list(PATHS.figures.glob('*')))}")
    print("=" * 70)


def report_stage4(stage4_summary) -> None:
    c = stage4_summary.get("chunking", {})
    print("=" * 70)
    print("STAGE 4 REPORT")
    print("=" * 70)
    print(
        f"Documents              : {stage4_summary['documents']}  "
        f"{stage4_summary['by_source_type']}"
    )
    print(
        f"Corpus                 : {stage4_summary['words']:,} words / "
        f"{stage4_summary['corpus_tokens']:,} tokens / "
        f"{stage4_summary['megabytes']} MB"
    )
    print(f"Figures referenced     : {stage4_summary['figures_referenced']}")
    print()
    print(f"Chunks                 : {c.get('chunks', 0)}")
    print(
        f"  token range          : {c.get('min_tokens')} - {c.get('median_tokens')} "
        f"(median) - {c.get('max_tokens')}"
    )
    print(
        f"  inside 1500-2000     : {c.get('in_target_band', 0)} / {c.get('chunks', 0)}"
        "   <-- the design's band; the coverage map's resolution"
    )
    print(f"  carrying figures     : {c.get('chunks_with_figures', 0)}")
    print(f"  carrying timestamps  : {c.get('chunks_with_timestamps', 0)}")
    if c.get("oversized"):
        print(
            f"  OVER max_chunk_tokens: {len(c['oversized'])}   "
            f"(indivisible blocks: {c['oversized'][:3]})"
        )
    if stage4_summary["issues"]:
        print("\nValidation issues:")
        for issue in stage4_summary["issues"]:
            print(f"  - {issue}")
    print("=" * 70)

    # Show one chunk's Layer 3 record and one neighbourhood, so the output of
    # this stage is something you have actually looked at rather than trusted.
    meta = json.loads(PATHS.chunk_metadata.read_text(encoding="utf-8"))
    if meta:
        with_figure = next((k for k, v in meta.items() if v["figures"]), None)
        sample = with_figure or next(iter(meta))
        print(f"\nSAMPLE chunk_metadata['{sample}']:")
        record = dict(meta[sample])
        record["figures"] = [
            {"id": f["id"], "kind": f["kind"], "path": f["path"]}
            for f in record["figures"]
        ]
        print(json.dumps(record, indent=2)[:900])
        print(f"\nNeighbourhood of {sample} (what Layer 3 will retrieve):")
        print(get_chunk_neighbourhood(sample, window=1, metadata=meta)[:600])


def main() -> None:
    # -- Stage 1: speech to text ---------------------------------------------
    stage1_summary = run_stage1_speech(Stage1Config())
    print(stage1_summary)

    # -- Stage 2: layout parsing ----------------------------------------------
    stage2_summary = run_stage2_parsing(Stage2Config())
    print(stage2_summary)
    report_stage2(analyse_parsed_outputs())

    # -- Stage 3: describing figures in context --------------------------------
    stage3_cfg = Stage3Config()
    llm = build_llm_client(logger=stage3_log)
    fallback_describer = FallbackDescriber(llm, stage3_cfg)
    stage3_summary = run_stage3(llm, fallback_describer, stage3_cfg)
    report_stage3(stage3_summary, llm)
    llm.cleanup()

    # -- Stage 4: consolidation and chunking -----------------------------------
    stage4_summary = run_stage4(Stage4Config())
    report_stage4(stage4_summary)

    # -- Stage 5: the source index ----------------------------------------------
    source_index_summary = build_source_index()
    print("=" * 70)
    print("STAGE 5 — SOURCE INDEX")
    print("=" * 70)
    for k, v in source_index_summary.items():
        print(f"  {k:18s}: {v}")
    print("=" * 70)

    print()
    print("=" * 70)
    print("PIPELINE A COMPLETE")
    print("=" * 70)
    print("Artifacts for the rest of the system:")
    print(f"  corpus          -> {PATHS.consolidated_text}")
    print(f"  chunks          -> {PATHS.chunks}/chunk_0001.txt ...")
    print(f"  chunk metadata  -> {PATHS.chunk_metadata}          (Layer 3)")
    print(f"  source index    -> {PATHS.source_index}/           (Layer 3, Chroma)")
    print(f"  figure store    -> {PATHS.figure_store}")
    print(
        f"  figure crops    -> {PATHS.figures}/                (passed to the Writer)"
    )
    print(f"  document index  -> {PATHS.document_index}")
    print(f"  transcripts     -> {PATHS.transcripts}/            (raw + clean)")
    print("=" * 70)


if __name__ == "__main__":
    main()
