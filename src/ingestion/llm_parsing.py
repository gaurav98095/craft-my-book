"""
Stage 2, alternative backend — document parsing via the shared model's
vision capability, instead of a local layout-detection tool (MinerU/Docling).

MinerU and Docling both run real ML models locally — layout detection, OCR,
sometimes table-structure recognition — which is exactly what makes them
heavy: on a laptop with no GPU to spare, that can hang the machine. This
module replaces that local pipeline with the same `LLMClient` every other
stage already uses: render each PDF page to an image with PyMuPDF, ask the
model to transcribe it into structured blocks, and crop figures/tables out
of the page image using the bounding boxes it returns.

Bounding-box grounding quality depends entirely on the provider. Gemini and
Qwen-VL are both trained for this and give usable boxes; a provider without
grounding-tuned training may return imprecise or no boxes at all. A missing
or degenerate box is never a crash and never a silently dropped figure — the
block survives as a text description instead, and Stage 3 sees it either way.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..llm import LLMClient

PAGE_SCHEMA = {
    "blocks": [
        {
            "type": "text|figure|table|equation",
            "text": "string — transcribed body text for 'text'; a one-sentence "
                    "caption of what it shows for 'figure'; markdown or a "
                    "caption for 'table'; the LaTeX for 'equation'",
            "bbox": "[left, top, right, bottom], each 0-1000, normalized to "
                    "this image's width/height — required for figure and "
                    "table, omit otherwise",
        }
    ]
}

PAGE_SYSTEM = (
    "You transcribe a scanned document page into structured blocks, in "
    "reading order, top to bottom, left to right within a column.\n"
    "Every block is one of:\n"
    "  text     — a paragraph, heading, or list item. Transcribe verbatim.\n"
    "  figure   — a chart, diagram, or photo. Do not transcribe its content; "
    "write a one-sentence caption of what it shows.\n"
    "  table    — transcribe as markdown if the text is legible, otherwise a "
    "one-sentence caption.\n"
    "  equation — its LaTeX, if you can read it.\n"
    "For figure and table blocks, also return a bounding box "
    "[left, top, right, bottom], each 0-1000, normalized to this page "
    "image's width and height.\n"
    "Skip page headers, footers, and page numbers entirely — do not emit a "
    "block for them."
)


def _render_pdf_pages(file_path: Path, dpi: int = 150) -> List[Image.Image]:
    import fitz  # PyMuPDF — optional dependency, only needed for this backend

    doc = fitz.open(str(file_path))
    try:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        return [
            Image.frombytes(
                "RGB", (pix.width, pix.height), pix.samples
            )
            for pix in (page.get_pixmap(matrix=matrix) for page in doc)
        ]
    finally:
        doc.close()


def _crop_from_bbox(page_image: Image.Image, bbox: Any) -> Optional[Image.Image]:
    """
    Best-effort crop from a normalized [left, top, right, bottom] (0-1000)
    box. Returns None — never raises — for a missing, malformed, or
    degenerate box, so a bad grounding call loses one figure, not the page.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in bbox):
        return None

    w, h = page_image.size
    left, top, right, bottom = bbox
    left, right = sorted((max(0, min(1000, left)), max(0, min(1000, right))))
    top, bottom = sorted((max(0, min(1000, top)), max(0, min(1000, bottom))))
    if right - left < 10 or bottom - top < 10:  # under 1% of the page edge
        return None

    px = (left / 1000 * w, top / 1000 * h, right / 1000 * w, bottom / 1000 * h)
    return page_image.crop(px)


def parse_document_with_llm(
    file_path: Path, output_dir: Path, llm: LLMClient, dpi: int = 150,
) -> List[Dict[str, Any]]:
    """
    The parser contract Stage 2 expects (`parser.parse_document(...)`),
    implemented with the shared model instead of a local layout tool.
    Returns a content_list in the same shape MinerU/Docling produce.
    """
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"the LLM parser backend currently only handles PDFs, got "
            f"{file_path.suffix} ({file_path.name})")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pages = _render_pdf_pages(file_path, dpi=dpi)
    content_list: List[Dict[str, Any]] = []
    fig_counter = 0

    for page_no, page_image in enumerate(pages, start=1):
        reply = llm.generate_structured(
            system=PAGE_SYSTEM,
            user=[
                {"type": "text",
                 "text": f"Page {page_no} of {len(pages)}. Transcribe it."},
                {"type": "image", "image": page_image},
            ],
            schema=PAGE_SCHEMA,
            # See the reasoning-budget note in toc/setup.py -- a whole page of
            # blocks plus bounding boxes is verbose, and a reasoning-capable
            # model spends part of this same budget thinking before any of
            # it becomes visible.
            max_tokens=8_000,
            temperature=0.0,
        )
        blocks = (reply or {}).get("blocks", [])

        for block in blocks:
            kind = str(block.get("type", "text")).strip().lower()
            text = str(block.get("text", "")).strip()

            if kind in ("figure", "table"):
                crop = _crop_from_bbox(page_image, block.get("bbox"))
                if crop is not None:
                    fig_counter += 1
                    img_path = images_dir / f"page{page_no:03d}_{fig_counter:02d}.png"
                    crop.save(img_path)
                    item: Dict[str, Any] = {
                        "type": "image" if kind == "figure" else "table",
                        "img_path": str(img_path),
                        "caption": text or None,
                    }
                    if kind == "table":
                        item["table_body"] = text
                    content_list.append(item)
                elif text:
                    # No usable box: keep the model's own description as text
                    # rather than lose the block entirely.
                    content_list.append({
                        "type": "text",
                        "text": f"[{kind.upper()}, page {page_no}] {text}",
                    })
                continue

            if kind == "equation":
                if text:
                    content_list.append({"type": "equation", "latex": text})
                continue

            if text:
                content_list.append({"type": "text", "text": text})

    return content_list
