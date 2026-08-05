"""Stage 4 — Consolidation and Chunking."""

import re
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from .setup import PATHS, make_logger, count_tokens
from .stage3_figures import FIGURE_MARKER_RE

stage4_log = make_logger("stage4.consolidate", "stage4_consolidation.log")


@dataclass
class Stage4Config:
    """Configuration for Stage 4 - Consolidation and chunking."""

    # -- Consolidation -------------------------------------------------------
    sort_order: str = "alphabetical"  # "alphabetical" | "custom" | "type"
    custom_order_file: Optional[str] = None
    add_document_boundaries: bool = True
    boundary_rule: str = "=" * 78
    blank_lines_between_docs: int = 2
    create_backup: bool = True

    # -- Chunking ------------------------------------------------------------
    # "Keep chunks moderate: roughly 1,500-2,000 tokens." Chunk size is the
    # resolution of the coverage map, which is what detects repetition.
    target_chunk_tokens: int = 1_750
    min_chunk_tokens: int = 1_200
    max_chunk_tokens: int = 2_200

    # Zero on purpose. The design handles mid-thought fragments with
    # neighbourhood retrieval at read time, not with overlap at write time --
    # overlapping chunks would double-count in the coverage map.
    overlap_tokens: int = 0

    write_chunk_files: bool = True  # data/chunks/chunk_0001.txt
    verify_offsets: bool = True  # assert corpus[start:end] == chunk


def discover_converted() -> List[Path]:
    return sorted(
        p for p in PATHS.converted.glob("*.txt") if not p.name.startswith("SAMPLE_")
    )


def load_sidecar(txt_path: Path) -> Dict[str, Any]:
    sidecar = txt_path.parent / f"{txt_path.name}.stats.json"
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            stage4_log.warning(f"Unreadable sidecar for {txt_path.name}: {exc}")
    return {}


def order_documents(paths: List[Path], cfg: Stage4Config) -> List[Path]:
    """
    Decide the order documents appear in the corpus.

    This is not the book's ordering -- Pipeline B's CurriculumOrderer decides
    that. It only has to be STABLE, so re-running the pipeline does not shuffle
    every character offset and invalidate a partially written book.
    """
    if cfg.sort_order == "custom" and cfg.custom_order_file:
        try:
            wanted = json.loads(Path(cfg.custom_order_file).read_text(encoding="utf-8"))
            rank = {name: i for i, name in enumerate(wanted)}
            return sorted(
                paths, key=lambda p: (rank.get(p.name, 10**9), p.name.lower())
            )
        except Exception as exc:
            stage4_log.warning(f"Custom order unusable ({exc}); using alphabetical")

    if cfg.sort_order == "type":
        return sorted(
            paths,
            key=lambda p: (load_sidecar(p).get("source_type", "zz"), p.name.lower()),
        )
    return sorted(paths, key=lambda p: p.name.lower())


def build_boundary(sidecar: Dict[str, Any], filename: str, cfg: Stage4Config) -> str:
    """Machine-readable, so the corpus is self-describing even without the index."""
    lines = [
        cfg.boundary_rule,
        f"DOCUMENT: {sidecar.get('source_document', filename)}",
        f"SOURCE_TYPE: {sidecar.get('source_type', 'unknown')}",
    ]
    if sidecar.get("duration_seconds"):
        lines.append(f"DURATION_MINUTES: {sidecar['duration_seconds'] / 60:.1f}")
    if sidecar.get("figure_ids"):
        lines.append(f"FIGURES: {len(sidecar['figure_ids'])}")
    lines.append(cfg.boundary_rule)
    return "\n".join(lines)


def consolidate(
    paths: List[Path], cfg: Stage4Config
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build the corpus and its index in one pass.

    `emit()` is the only thing that touches the buffer and the only thing that
    moves the cursor, so there is no separate arithmetic that can drift out of
    step with the string being built.
    """
    buffer: List[str] = []
    cursor = 0
    index: List[Dict[str, Any]] = []

    def emit(chunk: str) -> None:
        nonlocal cursor
        if chunk:
            buffer.append(chunk)
            cursor += len(chunk)

    separator = "\n" * (cfg.blank_lines_between_docs + 1)

    for i, path in enumerate(tqdm(paths, desc="Consolidating")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            stage4_log.error(f"Could not read {path.name}: {exc}")
            continue

        sidecar = load_sidecar(path)
        if i > 0:
            emit(separator)
        if cfg.add_document_boundaries:
            emit(build_boundary(sidecar, path.name, cfg))
            emit("\n\n")

        start = cursor
        emit(content)
        end = cursor

        index.append(
            {
                "order": i + 1,
                "filename": path.name,
                "doc_slug": path.stem,
                "source_document": sidecar.get("source_document", path.name),
                "source_type": sidecar.get("source_type", "unknown"),
                "start": start,
                "end": end,
                "length": end - start,
                "figure_ids": sidecar.get("figure_ids", []),
                "timestamp_index": sidecar.get("timestamp_index", []),
                "timestamp_index_exact": sidecar.get("timestamp_index_exact", True),
            }
        )

    text = "".join(buffer)

    if cfg.verify_offsets:
        for record, path in zip(index, paths):
            if text[record["start"] : record["end"]] != path.read_text(
                encoding="utf-8"
            ):
                raise AssertionError(
                    f"Offset mismatch for {record['filename']}: recorded "
                    f"[{record['start']}:{record['end']}] does not match the file. "
                    f"Provenance would be wrong for the whole book."
                )
        stage4_log.info(f"Document offsets verified for all {len(index)} documents")

    return text, index


def validate_corpus(
    text: str, index: List[Dict[str, Any]], cfg: Stage4Config
) -> List[str]:
    issues: List[str] = []

    if len(text.strip()) < 100:
        issues.append("corpus is essentially empty")

    if cfg.add_document_boundaries:
        # Count real boundary lines. text.count("DOCUMENT:") would be inflated
        # by any document that merely mentions the word.
        found = len(re.findall(r"^DOCUMENT: ", text, flags=re.MULTILINE))
        if found != len(index):
            issues.append(f"expected {len(index)} boundaries, found {found}")

    if PATHS.figure_store.exists():
        stored = {
            f["id"]
            for f in json.loads(PATHS.figure_store.read_text(encoding="utf-8"))[
                "figures"
            ]
        }
        referenced = {m.group(2) for m in FIGURE_MARKER_RE.finditer(text)}
        orphans = referenced - stored
        if orphans:
            issues.append(
                f"{len(orphans)} figure markers have no store entry "
                f"(e.g. {sorted(orphans)[:3]})"
            )

    junk = text.count("[UNKNOWN CONTENT TYPE:")
    if junk:
        issues.append(
            f"{junk} '[UNKNOWN CONTENT TYPE:]' markers - stale Stage 3 "
            f"output, delete data/converted and re-run"
        )

    return issues


def split_into_blocks(text: str) -> List[Tuple[int, int, str]]:
    """
    Split into paragraph-sized blocks, keeping exact offsets.

    A block is a run of lines with no blank line inside it -- which means a
    figure marker and its description, written as

        [IMAGE fig_x_003]
        A diagram of the encoder stack...

    stay together automatically. Splitting those apart would leave a chunk
    holding a description with no marker, and the Layer 3 figure join would miss
    it.
    """
    return [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]*)*", text)
    ]


def split_oversized_block(
    start: int, block: str, cfg: Stage4Config
) -> List[Tuple[int, int, str]]:
    """A single block larger than max_chunk_tokens, split on sentence ends."""
    pieces, current_start, current_end = [], None, None
    for m in re.finditer(r"[^.!?]*[.!?]+[\s]*|[^.!?]+$", block):
        if not m.group(0).strip():
            continue
        if current_start is None:
            current_start, current_end = m.start(), m.end()
        else:
            current_end = m.end()
        if count_tokens(block[current_start:current_end]) >= cfg.target_chunk_tokens:
            pieces.append(
                (
                    start + current_start,
                    start + current_end,
                    block[current_start:current_end],
                )
            )
            current_start = None
    if current_start is not None:
        pieces.append(
            (
                start + current_start,
                start + current_end,
                block[current_start:current_end],
            )
        )
    return pieces or [(start, start + len(block), block)]


def chunk_document(doc_text: str, cfg: Stage4Config) -> List[Tuple[int, int]]:
    """
    Greedily pack blocks into chunks of roughly `target_chunk_tokens`.

    Returns (start, end) offsets RELATIVE to the document. Chunking runs per
    document, so no chunk ever straddles a document boundary and every chunk has
    exactly one source_document.
    """

    def _chunk_tokens(span: Tuple[int, int]) -> int:
        return count_tokens(doc_text[span[0] : span[1]])

    blocks: List[Tuple[int, int, str]] = []
    for start, end, block in split_into_blocks(doc_text):
        if count_tokens(block) > cfg.max_chunk_tokens:
            blocks.extend(split_oversized_block(start, block, cfg))
        else:
            blocks.append((start, end, block))

    chunks: List[Tuple[int, int]] = []
    current_start = current_end = None
    current_tokens = 0

    for start, end, block in blocks:
        block_tokens = count_tokens(block)
        combined = current_tokens + block_tokens

        # Two independent reasons to close the current chunk:
        #
        #   1. HARD CEILING. Adding this block would break max_chunk_tokens.
        #      This check must NOT also require that we have reached the
        #      minimum -- otherwise a chunk sitting just under min swallows an
        #      arbitrarily large next block, roughly doubling the design's
        #      band and halving the coverage map's resolution.
        #
        #   2. SOFT TARGET. We are past target and already large enough to be
        #      a useful chunk on its own.
        over_ceiling = current_start is not None and combined > cfg.max_chunk_tokens
        past_target = (
            current_start is not None
            and combined > cfg.target_chunk_tokens
            and current_tokens >= cfg.min_chunk_tokens
        )

        if over_ceiling or past_target:
            chunks.append((current_start, current_end))
            current_start, current_tokens = None, 0

        if current_start is None:
            current_start = start
        current_end = end
        current_tokens += block_tokens

    if current_start is not None:
        # A short tail is merged back rather than shipped as a stub -- a
        # 90-token chunk pollutes the coverage map without teaching anything.
        # But not if the merge would itself breach the ceiling: an undersized
        # tail is a smaller problem than an oversized chunk.
        merged_tokens = (
            count_tokens("")
            if not chunks
            else current_tokens + _chunk_tokens(chunks[-1])
        )
        if (
            chunks
            and current_tokens < cfg.min_chunk_tokens // 2
            and merged_tokens <= cfg.max_chunk_tokens
        ):
            chunks[-1] = (chunks[-1][0], current_end)
        else:
            chunks.append((current_start, current_end))

    return chunks


def timestamp_for(doc: Dict[str, Any], rel_start: int, rel_end: int) -> Optional[Dict]:
    """For a transcript chunk, the seconds range it covers."""
    anchors = doc.get("timestamp_index") or []
    hits = [
        a for a in anchors if a["char_end"] >= rel_start and a["char_start"] <= rel_end
    ]
    if not hits:
        return None
    return {
        "start_seconds": round(min(h["t_start"] for h in hits), 1),
        "end_seconds": round(max(h["t_end"] for h in hits), 1),
        "approximate": not doc.get("timestamp_index_exact", True),
    }


def build_chunks_and_metadata(
    corpus: str, index: List[Dict[str, Any]], cfg: Stage4Config
) -> Dict[str, Any]:
    """
    Produce data/chunks/chunk_NNNN.txt and data/chunk_metadata.json.

    The metadata is exactly the Layer 3 shape from the design document:

        "chunk_0212": {
          "source_document": "lecture_04_slides.pptx",
          "source_type": "slides",
          "timestamp": null,
          "tags": [],
          "figures": [ {...} ]
        }

    `tags` is deliberately empty: Pipeline B's FineGrainedTagExtractor fills it.
    This function's job is the part only Pipeline A can know -- where the text
    came from.

    Figures are joined from the inline markers, which are inside the chunk text,
    so no position arithmetic is involved for them at all.
    """
    figures_by_id: Dict[str, Dict] = {}
    if PATHS.figure_store.exists():
        for figure in json.loads(PATHS.figure_store.read_text(encoding="utf-8"))[
            "figures"
        ]:
            figures_by_id[figure["id"]] = figure

    for stale in PATHS.chunks.glob("chunk_*.txt"):
        stale.unlink()

    metadata: Dict[str, Any] = {}
    counter = 0
    token_counts: List[int] = []
    oversized: List[Tuple[str, int]] = []

    for doc in tqdm(index, desc="Chunking"):
        doc_text = corpus[doc["start"] : doc["end"]]

        for rel_start, rel_end in chunk_document(doc_text, cfg):
            counter += 1
            chunk_id = f"chunk_{counter:04d}"

            abs_start = doc["start"] + rel_start
            abs_end = doc["start"] + rel_end
            chunk_text = corpus[abs_start:abs_end]

            if cfg.verify_offsets and chunk_text != doc_text[rel_start:rel_end]:
                raise AssertionError(f"{chunk_id}: chunk offsets do not resolve")

            if cfg.write_chunk_files:
                (PATHS.chunks / f"{chunk_id}.txt").write_text(
                    chunk_text, encoding="utf-8"
                )

            figure_ids = [m.group(2) for m in FIGURE_MARKER_RE.finditer(chunk_text)]
            tokens = count_tokens(chunk_text)
            token_counts.append(tokens)

            # After the ceiling fix, the only way to land here is a single
            # block over max_chunk_tokens that split_oversized_block could not
            # divide -- i.e. a paragraph with no sentence-ending punctuation
            # anywhere in it. In a real corpus that means a base64 blob, a
            # dumped table, or ASCII art. Shipping it whole is the right call;
            # shipping it whole and silently is not, because an oversized chunk
            # blunts the coverage map exactly where the design says it must be
            # sharp.
            if tokens > cfg.max_chunk_tokens:
                oversized.append((chunk_id, tokens))
                stage4_log.warning(
                    f"  {chunk_id}: {tokens} tokens exceeds "
                    f"max_chunk_tokens={cfg.max_chunk_tokens} - one "
                    f"indivisible block (no sentence breaks)"
                )

            metadata[chunk_id] = {
                "source_document": doc["source_document"],
                "source_type": doc["source_type"],
                "timestamp": timestamp_for(doc, rel_start, rel_end),
                "tags": [],  # Pipeline B fills this in
                "figures": [figures_by_id[f] for f in figure_ids if f in figures_by_id],
                # -- extras Layer 3 and the finishing passes make use of --
                "doc_slug": doc["doc_slug"],
                "char_span": [abs_start, abs_end],
                "token_count": tokens,
                "index_in_document": len(
                    [
                        c
                        for c in metadata.values()
                        if c["source_document"] == doc["source_document"]
                    ]
                ),
            }

    PATHS.chunk_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    token_counts.sort()
    stats = {
        "chunks": len(metadata),
        "chunks_with_figures": sum(1 for c in metadata.values() if c["figures"]),
        "chunks_with_timestamps": sum(1 for c in metadata.values() if c["timestamp"]),
        "median_tokens": token_counts[len(token_counts) // 2] if token_counts else 0,
        "min_tokens": token_counts[0] if token_counts else 0,
        "max_tokens": token_counts[-1] if token_counts else 0,
        "in_target_band": sum(1 for t in token_counts if 1_500 <= t <= 2_000),
        "oversized": oversized,
    }
    stage4_log.info(
        f"Chunked into {stats['chunks']} chunks "
        f"(median {stats['median_tokens']} tokens)"
    )
    return stats


def get_chunk_neighbourhood(
    chunk_id: str, window: int = 1, metadata: Optional[Dict] = None
) -> str:
    """
    "When a search hits chunk_211, return chunk_210 through chunk_212."

    Layer 3's SourceMemory calls this instead of reading a lone chunk. The
    refinement the design implies: a neighbour from a DIFFERENT source document
    is not a neighbour, it is a stranger that happens to sit next in the
    numbering, so the window stops at document boundaries.
    """
    metadata = metadata or json.loads(PATHS.chunk_metadata.read_text(encoding="utf-8"))
    if chunk_id not in metadata:
        return ""

    home = metadata[chunk_id]["source_document"]
    number = int(chunk_id.split("_")[1])

    parts = []
    for n in range(number - window, number + window + 1):
        neighbour = f"chunk_{n:04d}"
        record = metadata.get(neighbour)
        if not record or record["source_document"] != home:
            continue
        path = PATHS.chunks / f"{neighbour}.txt"
        if path.exists():
            marker = " (requested)" if neighbour == chunk_id else ""
            parts.append(
                f"--- {neighbour}{marker} | {record['source_document']} ---\n"
                f"{path.read_text(encoding='utf-8')}"
            )
    return "\n\n".join(parts)


def run_stage4(cfg: Stage4Config) -> Dict[str, Any]:
    stage4_log.info("=" * 70)
    stage4_log.info("STAGE 4: CONSOLIDATION AND CHUNKING")
    stage4_log.info("=" * 70)

    paths = discover_converted()
    if not paths:
        stage4_log.error("Nothing in data/converted - run Stage 3 first.")
        return {"error": "no converted documents"}

    ordered = order_documents(paths, cfg)
    stage4_log.info(f"Consolidating {len(ordered)} documents ({cfg.sort_order} order)")
    for p in ordered[:10]:
        stage4_log.info(f"  {p.name}")
    if len(ordered) > 10:
        stage4_log.info(f"  ... and {len(ordered) - 10} more")

    corpus, index = consolidate(ordered, cfg)

    if cfg.create_backup and PATHS.consolidated_text.exists():
        backup = PATHS.consolidated_text.with_suffix(".txt.backup")
        shutil.copy2(PATHS.consolidated_text, backup)
        stage4_log.info(f"Previous corpus backed up to {backup.name}")

    PATHS.consolidated_text.write_text(corpus, encoding="utf-8")
    PATHS.document_index.write_text(
        json.dumps({"documents": index}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    issues = validate_corpus(corpus, index, cfg)
    for issue in issues:
        stage4_log.warning(f"  validation: {issue}")
    if not issues:
        stage4_log.info("  validation: clean")

    chunk_stats = build_chunks_and_metadata(corpus, index, cfg)

    words = len(corpus.split())
    summary = {
        "documents": len(index),
        "characters": len(corpus),
        "words": words,
        "corpus_tokens": count_tokens(corpus),
        "megabytes": round(len(corpus.encode("utf-8")) / (1024 * 1024), 2),
        "figures_referenced": len(
            set(m.group(2) for m in FIGURE_MARKER_RE.finditer(corpus))
        ),
        "by_source_type": {},
        "chunking": chunk_stats,
        "issues": issues,
    }
    for record in index:
        st = record["source_type"]
        summary["by_source_type"][st] = summary["by_source_type"].get(st, 0) + 1

    stage4_log.info("-" * 70)
    stage4_log.info(
        f"  documents  : {summary['documents']}  {summary['by_source_type']}"
    )
    stage4_log.info(f"  tokens     : {summary['corpus_tokens']:,}")
    stage4_log.info(
        f"  chunks     : {chunk_stats['chunks']} "
        f"(median {chunk_stats['median_tokens']} tokens)"
    )
    return summary
