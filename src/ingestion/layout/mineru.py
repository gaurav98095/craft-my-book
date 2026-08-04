"""
ingestion.layout.mineru - Module 1.3: Layout Parsing (MinerU).

A PDF is a visual layout, not a logical document: a collection of positioned text
spans, images, and lines with no inherent concept of "heading", "paragraph",
"caption", or "figure", and no guarantee reading order matches storage order (e.g.
two-column papers). MinerU reconstructs that structure deterministically - layout
detection, reading-order reconstruction, table/equation recognition - before any LLM
sees the document, so downstream stages get clean structured blocks instead of having
to infer structure from a flat text dump.

This stage only *finds* figures/tables/equations as first-class objects; it does not
*understand* them - that's ingestion.vision's job (Module 1.4, model configured
separately in config.yaml). Keeping "find" and "understand" as separate stages is a
deliberate separation of responsibilities.

Pipeline: PDF/DOCX/PPTX/XLSX -> `mineru` CLI (pipeline backend, local/deterministic)
-> {stem}_content_list.json -> IngestedDocument, one IngestedSegment per block, in
reading order.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from config import INGESTION

from ..base import Ingestor, IngestedDocument, IngestedSegment

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})


def run_mineru(
    source_path: str | Path,
    out_dir: str | Path,
    backend: str = INGESTION.layout.backend,
) -> Path:
    """
    Run the `mineru` CLI against one source file and return the path to the
    `{stem}_content_list.json` it produces.

    MinerU nests its output under out_dir/<stem>/<method>/ (the exact subdirectory
    name depends on version/method), so we search for the file by name rather than
    hardcoding that layout.
    """
    source_path = Path(source_path)
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported source extension '{source_path.suffix}'. Expected one of: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["mineru", "-p", str(source_path), "-o", str(out_dir), "-b", backend],
        check=True,
        capture_output=True,
        text=True,
    )

    matches = sorted(out_dir.rglob(f"{source_path.stem}_content_list.json"))
    if not matches:
        raise RuntimeError(
            f"mineru produced no _content_list.json for '{source_path.name}' under {out_dir}"
        )
    return matches[-1]


def _block_text(block: dict) -> str:
    """Best-effort plain text for a content_list.json block, regardless of type.

    `chart` blocks come back with an empty `content` field under the pipeline backend -
    deterministic parsing finds the chart but doesn't describe it; that gap is exactly
    what the later ingestion.vision stage (Module 1.4) fills in.
    """
    block_type = block.get("type")
    if block_type == "table":
        return block.get("table_body", "")
    if block_type == "code":
        return block.get("code_body", "")
    if block_type == "list":
        return "\n".join(block.get("list_items", []))
    return block.get("text") or block.get("content") or ""


def parse_content_list(path: str | Path) -> list[IngestedSegment]:
    """Turn a MinerU `_content_list.json` (flat array, already in reading order) into
    IngestedSegments - one per block, page/bbox/block-type preserved in metadata.

    `img_path` (image/table/chart/equation blocks all carry one) is resolved to an
    absolute path here, relative to where the content list itself lives - so anything
    downstream (e.g. the vision stage) can open the file without knowing MinerU's
    output-directory conventions.
    """
    path = Path(path)
    base_dir = path.parent
    blocks = json.loads(path.read_text())

    segments = []
    for i, block in enumerate(blocks):
        block_type = block.get("type", "text")
        extra = {
            k: v
            for k, v in block.items()
            if k not in {"type", "page_idx", "bbox", "text"}
        }
        if extra.get("img_path"):
            extra["img_path"] = str((base_dir / extra["img_path"]).resolve())
        segments.append(
            IngestedSegment(
                text=_block_text(block),
                order=i,
                page=block.get("page_idx"),
                metadata={"block_type": block_type, "bbox": block.get("bbox"), **extra},
            )
        )
    return segments


# --------------------------------------------------------------------------
# Strategy - the Ingestor this module contributes to the registry
# --------------------------------------------------------------------------


class LayoutIngestor(Ingestor):
    """Ingestion strategy for PDF/DOCX/PPTX/XLSX: MinerU layout parsing.

    Purely deterministic - no LLM call happens in this strategy. Figures and tables
    come out as first-class blocks (type + bbox + page); a later strategy is
    responsible for describing what's actually inside them.
    """

    source_type = "layout"
    supported_extensions = SUPPORTED_EXTENSIONS

    def __init__(
        self,
        out_dir: str | Path = INGESTION.layout.work_dir,
        backend: str = INGESTION.layout.backend,
    ):
        self.out_dir = Path(out_dir)
        self.backend = backend

    def ingest(self, source_path: str | Path) -> IngestedDocument:
        content_list_path = run_mineru(source_path, self.out_dir, backend=self.backend)
        segments = parse_content_list(content_list_path)
        page_count = (
            max((s.page for s in segments if s.page is not None), default=-1) + 1
        )
        return IngestedDocument(
            source=Path(source_path).name,
            source_type=self.source_type,
            segments=segments,
            metadata={
                "page_count": page_count,
                "mineru_output": str(content_list_path),
            },
        )
