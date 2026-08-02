"""
ingestion.vision.describe - Module 1.4: Figure Understanding.

Not an Ingestor (it doesn't turn a raw source file into an IngestedDocument) - this
is an enrichment pass over a document ingestion.layout already produced. MinerU
*finds* every image/chart/table/equation as a first-class block; this stage
*understands* what's inside them, which is a deliberately separate responsibility
(see ingestion/layout/mineru.py).

No model provider is hardcoded here - this calls whatever vision-capable model
config.yaml's `ingestion.vision.vlm` points at (via the generic `llm` package, same as
every other component), whether that's Qwen-VL served through xAI/OpenRouter, a
self-hosted OpenAI-compatible endpoint, or a hosted vision model directly. Swapping
providers is a config.yaml edit, never a code change.

Key design choices, straight from the Module 1.4 brief:
- Never describe a visual in isolation. Each call gets the surrounding page text plus
  every other visual on that same page, so e.g. a figure can be described with
  awareness of a table right next to it, and the model can use the page's prose for
  context it has no other way to get from a cropped image.
- Batching granularity ("page" vs "document") is a config choice
  (config.INGESTION.vision.batch_by), not hardcoded - see config.yaml for the tradeoff.
- The description REPLACES `segment.text` (so downstream text-only stages, e.g.
  Pipeline B, can treat it like any other paragraph) but the raw MinerU extraction
  (LaTeX for an equation, HTML for a table) and the original image path are preserved
  in `segment.metadata` - never deleted, matching "keep both the image and the
  description" from the brief.
"""

from __future__ import annotations

import json
from collections import defaultdict

from config import INGESTION, load_prompt
from llm import LLMClient, Message, chat_with_messages, image_part, text_part

from ..base import IngestedDocument, IngestedSegment

_SYSTEM_PROMPT = load_prompt("vision_description.txt")

_CAPTION_KEYS = ("image_caption", "table_caption", "chart_caption")


def _is_visual(segment: IngestedSegment) -> bool:
    return segment.metadata.get(
        "block_type"
    ) in INGESTION.vision.visual_block_types and segment.metadata.get("img_path")


def _caption(segment: IngestedSegment) -> str | None:
    for key in _CAPTION_KEYS:
        value = segment.metadata.get(key)
        if value:
            return "; ".join(value) if isinstance(value, list) else str(value)
    return None


def _group_key(segment: IngestedSegment) -> int | str:
    if INGESTION.vision.batch_by == "document":
        return "__all__"
    return segment.page if segment.page is not None else -1


def _build_message(context_text: str, visuals: list[IngestedSegment]) -> list[Message]:
    system = Message(
        role="system",
        content=_SYSTEM_PROMPT.format(context_text=context_text or "(none)"),
    )

    content = [text_part(f"Visual elements on this page: {len(visuals)}")]
    for i, seg in enumerate(visuals, start=1):
        label = f"[VISUAL {i}] type={seg.metadata['block_type']}"
        caption = _caption(seg)
        if caption:
            label += f", caption={caption}"
        content.append(text_part(label))
        content.append(image_part(seg.metadata["img_path"]))

    return [system, Message(role="user", content=content)]


def _describe_group(
    context_text: str, visuals: list[IngestedSegment], client: LLMClient, model: str
) -> list[str]:
    messages = _build_message(context_text, visuals)
    response = chat_with_messages(client, model, messages)
    descriptions = json.loads(response.content)
    if not isinstance(descriptions, list) or len(descriptions) != len(visuals):
        raise ValueError(
            f"Expected {len(visuals)} descriptions back, got: {response.content!r}"
        )
    return descriptions


def describe_visuals(
    document: IngestedDocument,
    client: LLMClient,
    model: str = INGESTION.vision.vlm.model,
) -> IngestedDocument:
    """
    Enrich every image/chart/table/equation block in `document` with a semantic
    description, grouped per config.INGESTION.vision.batch_by so visuals on the same
    page/document are described together rather than in N isolated calls.

    Mutates and returns `document`. Segments with no visual blocks are untouched.
    """
    groups: dict[int | str, list[IngestedSegment]] = defaultdict(list)
    for seg in document.segments:
        groups[_group_key(seg)].append(seg)

    for segs in groups.values():
        visuals = [s for s in segs if _is_visual(s)]
        if not visuals:
            continue

        context_text = "\n\n".join(s.text for s in segs if s.text and not _is_visual(s))
        descriptions = _describe_group(context_text, visuals, client, model)

        for seg, description in zip(visuals, descriptions):
            seg.metadata["raw_text"] = (
                seg.text
            )  # preserve MinerU's extraction (LaTeX/HTML/empty)
            seg.metadata["description"] = description
            seg.text = description

    return document
