"""Stage 3 — Describing Figures, In Context."""

import re
import json
import time
import shutil
import hashlib
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm.auto import tqdm

from .setup import PATHS, make_logger, doc_slug, normalise_text, \
    load_vocabulary_from_pipeline_b
from ..llm import LLMClient

stage3_log = make_logger("stage3.describe", "stage3_describe.log")


@dataclass
class Stage3Config:
    """
    Configuration for Stage 3 - Describing figures in context.

    Which model answers these calls -- local weights, Anthropic, Groq, a
    self-hosted server -- is not configured here. It is decided once, for
    the whole system, by `.env` (see `src.llm.build_llm_client`).
    """

    # -- Whole-document description (the design's primary path) --------------
    describe_whole_document: bool = True
    max_figures_per_call: int = 8  # images per call; lower this if VRAM is tight
    max_document_chars: int = 60_000  # how much marker text travels with each call
    # A reasoning-capable model spends part of this SAME budget on hidden
    # thinking before it writes a visible token -- see the identical note in
    # toc/setup.py. Sized with that margin; a non-reasoning model just stops
    # sooner and never touches the ceiling.
    document_call_max_tokens: int = 8_000  # the structured reply can be long

    # -- Per-figure fallback (used when a document-level id comes back empty) -
    context_window_items: int = 6  # text items either side of the figure
    context_max_chars: int = 2_000
    figure_max_tokens: int = 2_000
    temperature: float = 0.2  # descriptions should be boring and stable

    # -- Structured output ---------------------------------------------------
    structured_max_attempts: int = 3  # re-prompt with the parser's own error

    # -- The cheap decorative pre-filter -------------------------------------
    min_image_width: int = 96
    min_image_height: int = 96
    min_image_pixels: int = 20_000
    max_aspect_ratio: float = 12.0  # a 600x20 strip is a divider, not a figure

    # -- Item types ----------------------------------------------------------
    drop_types: Tuple[str, ...] = ("discarded",)

    # -- Transcript cleanup --------------------------------------------------
    clean_transcripts: bool = True
    transcript_window_chars: int = 4_000
    transcript_length_tolerance: float = 0.25

    # -- Housekeeping --------------------------------------------------------
    skip_existing: bool = True
    force_reconvert: bool = False
    continue_on_error: bool = True
    log_vlm_responses: bool = False


# ---------------------------------------------------------------------------
# The document view, and describing every figure in one call
# ---------------------------------------------------------------------------

# Verbatim from the design document. Every clause is load-bearing; the last
# sentence is what stops a newsletter signup box becoming 400 tokens of analysis.
DESCRIBE_SYSTEM = (
    "You are preparing source material for a technical book. For each figure, table, "
    "or equation given, write a description that could stand in for it in running text: "
    "what it shows, what the surrounding text is using it to demonstrate, and any values, "
    "labels, or axis meanings a reader would need.\n"
    "Do not invent labels that are not visible. If a figure is decorative or unreadable, "
    "say so and write nothing further."
)

# The design's schema. `decorative` is a legitimate answer, not an error.
DESCRIBE_SCHEMA = {
    "figures": [
        {
            "id": "string",
            "description": "string - empty if decorative",
            "kind": "figure|table|equation|decorative",
        }
    ]
}

FIGURE_MARKER_RE = re.compile(r"\[(IMAGE|TABLE|EQUATION) (fig_[a-z0-9_]+)\]")
KIND_BY_ITEM_TYPE = {"image": "figure", "table": "table", "equation": "equation"}


def looks_decorative(image: Image.Image, cfg: Stage3Config) -> Optional[str]:
    """
    The cheap filter, applied before any GPU time is spent.

    An avatar is 48x48. A follow button is 200x40. A horizontal rule is 700x3.
    None is worth a place in the model's context window. Returns a reason, or
    None if the image deserves a real look.
    """
    w, h = image.size
    if w < cfg.min_image_width or h < cfg.min_image_height:
        return f"too small ({w}x{h})"
    if w * h < cfg.min_image_pixels:
        return f"too few pixels ({w * h})"
    ratio = max(w, h) / max(1, min(w, h))
    if ratio > cfg.max_aspect_ratio:
        return f"extreme aspect ratio ({ratio:.1f}:1)"
    return None


def build_document_view(
    parsed: Dict[str, Any], cfg: Stage3Config
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build the design's `text_with_markers`: the full document as running text,
    with every figure replaced by its id.

    This single string is what lets the model notice that figure 4 refines
    figure 2 -- it can read the sentences between them.

    Returns (marker_text, figures) where each figure entry carries its id, the
    parsed item, and its pre-filter verdict.
    """
    slug = parsed.get("doc_slug") or doc_slug(parsed["filename"])
    parts: List[str] = []
    figures: List[Dict[str, Any]] = []
    counter = 0

    for index, item in enumerate(parsed["content_list"]):
        item_type = item.get("type", "unknown")

        if item_type in cfg.drop_types:  # layout furniture, never seen again
            continue

        if item_type == "text":
            text = normalise_text(item.get("text", ""))
            if text:
                parts.append(text)
            continue

        if item_type not in KIND_BY_ITEM_TYPE:
            salvage = normalise_text(str(item.get("text", "")))
            if salvage:
                parts.append(salvage)
            continue

        counter += 1
        fig_id = f"fig_{slug}_{counter:03d}"

        caption = item.get("caption") or item.get("image_caption") or []
        if isinstance(caption, list):
            caption = " ".join(str(c) for c in caption)
        # A newline inside a caption would break the one-line <<marker>> that
        # convert_document's substitution regex matches, leaving the caption
        # text stranded in the corpus. Collapse all whitespace to spaces.
        caption = re.sub(r"\s+", " ", str(caption)).strip()

        entry: Dict[str, Any] = {
            "id": fig_id,
            "kind": KIND_BY_ITEM_TYPE[item_type],
            "item": item,
            "item_index": index,
            "caption": caption or None,
            "image": None,
            "prefilter": None,
            "payload": None,
        }

        # Load the crop if there is one, and apply the cheap filter now so a
        # button never occupies a slot in the model's context.
        img_path = item.get("img_path")
        if img_path and Path(img_path).exists():
            try:
                image = Image.open(img_path)
                image.load()
                reason = looks_decorative(image, cfg)
                if reason:
                    entry["prefilter"] = reason
                else:
                    entry["image"] = image
            except Exception as exc:
                entry["prefilter"] = f"unreadable ({exc})"
        elif img_path:
            entry["prefilter"] = "crop missing on disk"

        # Items with no usable crop still carry markup worth describing.
        if entry["image"] is None:
            if item_type == "equation":
                entry["payload"] = str(item.get("latex") or item.get("text") or "")
            elif item_type == "table":
                entry["payload"] = str(
                    item.get("table_body") or item.get("table_data") or ""
                )

        figures.append(entry)

        marker = f"<<{fig_id}>>"
        if caption:
            marker += f"  (printed caption: {caption})"
        parts.append(marker)

    return "\n\n".join(parts), figures


def describe_document(
    parsed: Dict[str, Any], model: LLMClient, cfg: Stage3Config
) -> Tuple[Dict[str, Dict[str, str]], str, List[Dict[str, Any]]]:
    """
    The design's Stage 3: describe every figure in a document in one call.

        "With a 256K window you can pass an entire document and all its figures
         at once, rather than making one isolated call per image. Fewer calls,
         better descriptions, and the model can notice when figure 4 is a
         refinement of figure 2."

    One practical departure, stated rather than hidden: a document with 95
    figures will not fit any single GPU's activation memory, so figures are sent
    in batches of `max_figures_per_call`. Every batch still receives the WHOLE
    marker text, so cross-figure reasoning survives; only the images are split.
    With `max_figures_per_call` above the document's figure count it is exactly
    one call, as the design describes.

    Returns ({fig_id: {"description", "kind"}}, marker_text, figure_entries).
    """
    marker_text, figures = build_document_view(parsed, cfg)
    results: Dict[str, Dict[str, str]] = {}

    # Anything the pre-filter already rejected is settled without a call.
    describable = []
    for entry in figures:
        if entry["prefilter"]:
            results[entry["id"]] = {
                "kind": "decorative",
                "description": "",
                "why": entry["prefilter"],
            }
        elif entry["image"] is not None or entry["payload"]:
            describable.append(entry)
        else:
            results[entry["id"]] = {
                "kind": "decorative",
                "description": "",
                "why": "nothing to describe",
            }

    if not describable:
        return results, marker_text, figures

    view = marker_text[: cfg.max_document_chars]
    if len(marker_text) > cfg.max_document_chars:
        view += "\n\n[document truncated for this call]"

    batches = [
        describable[i : i + cfg.max_figures_per_call]
        for i in range(0, len(describable), cfg.max_figures_per_call)
    ]

    stage3_log.info(
        f"  {parsed['filename']}: {len(describable)} describable items "
        f"in {len(batches)} call(s)"
    )

    for batch in batches:
        # This mirrors the design's `describe_document` content layout exactly.
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"FULL DOCUMENT TEXT (figures marked by id):\n{view}",
            },
            {
                "type": "text",
                "text": (
                    "Describe ONLY the items listed below. Use the document text "
                    "above to say what each one is being used to demonstrate, and "
                    "to note when one item refines or repeats an earlier one."
                ),
            },
        ]

        for entry in batch:
            if entry["image"] is not None:
                content.append({"type": "image", "image": entry["image"]})
                content.append(
                    {
                        "type": "text",
                        "text": f"^ figure id: {entry['id']}  "
                        f"(kind: {entry['kind']})",
                    }
                )
            else:
                content.append(
                    {
                        "type": "text",
                        "text": f"figure id: {entry['id']}  "
                        f"(kind: {entry['kind']})\n"
                        f"{entry['kind'].upper()} SOURCE:\n"
                        f"{entry['payload'][:4000]}",
                    }
                )

        reply = model.generate_structured(
            system=DESCRIBE_SYSTEM,
            user=content,
            schema=DESCRIBE_SCHEMA,
            max_tokens=cfg.document_call_max_tokens,
            temperature=0.0,
            max_attempts=cfg.structured_max_attempts,
        )

        if not reply:
            continue

        for record in reply.get("figures", []):
            fig_id = str(record.get("id", "")).strip()
            if fig_id not in {e["id"] for e in batch}:
                continue  # the model invented an id; ignore it
            kind = str(record.get("kind", "figure")).strip().lower()
            description = str(record.get("description", "")).strip()
            if kind == "decorative" or len(description) < 40:
                results[fig_id] = {
                    "kind": "decorative",
                    "description": "",
                    "why": "model marked it decorative",
                }
            else:
                results[fig_id] = {"kind": kind, "description": description}

    return results, marker_text, figures


# ---------------------------------------------------------------------------
# The per-figure fallback
# ---------------------------------------------------------------------------
# The document-level call is the design's path and handles the normal case.
# Two things can still leave a figure without a description: the structured
# reply omitted an id, or `describe_whole_document` is switched off because
# the GPU is small. For those, we fall back to the design's other rule --
# "describe the figure with the page around it" -- calling once per figure
# with a window of the surrounding text. It is strictly worse than the
# document-level call, because the model cannot see that figure 4 refines
# figure 2, so it is a fallback and the report counts how often it fires.
#
# The description cache is keyed on the SHA-256 of the image bytes, not the
# filename, so repeated crops (e.g. the same author avatar) are caught even
# under different filenames.


class FallbackDescriber:
    """Per-figure description with page context, plus a content-hash cache."""

    def __init__(self, model: LLMClient, cfg: Stage3Config):
        self.model = model
        self.cfg = cfg
        self._cache: Dict[str, Optional[str]] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "declined": 0, "errors": 0}

    def page_context(self, content_list: List[Dict[str, Any]], index: int) -> str:
        """Collect the text immediately before and after an item."""
        cfg = self.cfg
        before, after = [], []

        for j in range(index - 1, max(-1, index - 1 - cfg.context_window_items), -1):
            if content_list[j].get("type") == "text":
                before.append(content_list[j].get("text", "").strip())
        before.reverse()

        for j in range(
            index + 1, min(len(content_list), index + 1 + cfg.context_window_items)
        ):
            if content_list[j].get("type") == "text":
                after.append(content_list[j].get("text", "").strip())

        half = cfg.context_max_chars // 2
        parts = []
        if before:
            parts.append(
                "TEXT IMMEDIATELY BEFORE THIS ITEM:\n"
                + " ".join(t for t in before if t)[-half:]
            )
        if after:
            parts.append(
                "TEXT IMMEDIATELY AFTER THIS ITEM:\n"
                + " ".join(t for t in after if t)[:half]
            )
        return "\n\n".join(parts)

    def describe(
        self, entry: Dict[str, Any], content_list: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Return {"kind", "description"} -- kind may be 'decorative'."""
        decorative = {
            "kind": "decorative",
            "description": "",
            "why": "fallback declined",
        }

        digest = None
        if entry["image"] is not None:
            try:
                digest = hashlib.sha256(
                    Path(entry["item"]["img_path"]).read_bytes()
                ).hexdigest()
            except Exception:
                digest = None

        if digest and digest in self._cache:
            self.stats["cache_hits"] += 1
            cached = self._cache[digest]
            return (
                {"kind": entry["kind"], "description": cached} if cached else decorative
            )

        context = self.page_context(content_list, entry["item_index"])
        prompt_parts = [f"ITEM TYPE: {entry['kind']}"]
        if entry["caption"]:
            prompt_parts.append(f"CAPTION AS PRINTED: {entry['caption']}")
        if context:
            prompt_parts.append(context)
        if entry["payload"]:
            prompt_parts.append(
                f"{entry['kind'].upper()} SOURCE:\n{entry['payload'][:4000]}"
            )
        prompt_parts.append(
            "Describe this item so it could stand in for the original in running text, "
            "given what the surrounding text uses it to demonstrate. "
            "If it is decorative or unreadable, reply with exactly DECORATIVE."
        )

        try:
            reply = self.model.generate(
                system=DESCRIBE_SYSTEM,
                user="\n\n".join(prompt_parts),
                images=[entry["image"]] if entry["image"] is not None else None,
                max_tokens=self.cfg.figure_max_tokens,
                temperature=self.cfg.temperature,
            )
            self.stats["calls"] += 1
        except Exception as exc:
            stage3_log.error(f"  fallback failed on {entry['id']}: {exc}")
            self.stats["errors"] += 1
            return decorative

        cleaned = reply.strip()
        # A two-word answer is the model shrugging: treat it as decorative.
        if cleaned.upper().startswith("DECORATIVE") or len(cleaned) < 40:
            self.stats["declined"] += 1
            if digest:
                self._cache[digest] = None
            return decorative

        if digest:
            self._cache[digest] = cleaned
        return {"kind": entry["kind"], "description": cleaned}


# ---------------------------------------------------------------------------
# Figures stay figures: writing the document and the figure store
# ---------------------------------------------------------------------------
#     "A figure becomes a description, but the image is NOT thrown away."
#
# Every figure produces two artifacts:
#
#   * a line in the text        ->  [IMAGE fig_slug_003]
#                                   <the description>
#   * a record in figure_store  ->  {"id": ..., "kind": ..., "path": ...,
#                                    "description": ..., "used_in_sections": []}
#
# The inline marker is what makes the Layer 3 join possible: after Stage 4
# chunks the corpus, a chunk's figure list is exactly the marker ids inside
# its span. Without the marker, the figure store is a folder of PNGs nothing
# points at.
#
# Decorative items are recorded too, with `kind: "decorative"` and no
# description. They never enter the corpus text, but they stay countable -- a
# filter you cannot audit is a filter you cannot tune.


def convert_document(
    parsed: Dict[str, Any],
    model: LLMClient,
    fallback: FallbackDescriber,
    cfg: Stage3Config,
) -> Tuple[str, List[Dict], Dict[str, Any]]:
    """Turn one Stage 2 document into (text, figure_records, stats)."""
    stats = {
        "total_items": len(parsed["content_list"]),
        "text_blocks": 0,
        "figures_kept": 0,
        "tables_kept": 0,
        "equations_kept": 0,
        "decorative": 0,
        "dropped_layout": 0,
        "document_level": 0,
        "fallback_used": 0,
        "started": time.time(),
    }

    stats["dropped_layout"] = sum(
        1 for item in parsed["content_list"] if item.get("type") in cfg.drop_types
    )

    if cfg.describe_whole_document:
        described, marker_text, figures = describe_document(parsed, model, cfg)
        stats["document_level"] = sum(
            1 for v in described.values() if v.get("description")
        )
    else:
        marker_text, figures = build_document_view(parsed, cfg)
        described = {}

    # Any figure the document-level pass did not settle goes to the fallback.
    for entry in figures:
        if entry["id"] in described:
            continue
        if entry["prefilter"]:
            described[entry["id"]] = {
                "kind": "decorative",
                "description": "",
                "why": entry["prefilter"],
            }
            continue
        described[entry["id"]] = fallback.describe(entry, parsed["content_list"])
        stats["fallback_used"] += 1

    # ---- build the final text by substituting each marker ------------------
    figure_records: List[Dict[str, Any]] = []
    by_id = {e["id"]: e for e in figures}

    def render(match: re.Match) -> str:
        fig_id = match.group(1)
        entry = by_id.get(fig_id)
        outcome = described.get(fig_id, {"kind": "decorative", "description": ""})
        kind = outcome.get("kind", "figure")
        description = outcome.get("description", "")

        stored_path = None
        if description:
            # Keep the crop: the Writer in Pipeline C is multimodal and reads
            # the diagram itself, not somebody's summary of it.
            source = (entry or {}).get("item", {}).get("img_path")
            if source and Path(source).exists():
                dest = PATHS.figures / f"{fig_id}{Path(source).suffix or '.png'}"
                try:
                    shutil.copy2(source, dest)
                    stored_path = str(dest)
                except Exception as exc:
                    stage3_log.warning(f"  could not store crop for {fig_id}: {exc}")

        figure_records.append(
            {
                "id": fig_id,
                "kind": kind,
                "path": stored_path,
                "caption": (entry or {}).get("caption"),
                "description": description,
                "source_document": parsed["filename"],
                "source_type": parsed.get("source_type", "unknown"),
                "used_in_sections": [],
                "skipped_reason": outcome.get("why") if not description else None,
            }
        )

        if not description:
            stats["decorative"] += 1
            return ""  # "write nothing further"

        stats[
            {
                "figure": "figures_kept",
                "table": "tables_kept",
                "equation": "equations_kept",
            }.get(kind, "figures_kept")
        ] += 1

        word = {"figure": "IMAGE", "table": "TABLE", "equation": "EQUATION"}.get(
            kind, "IMAGE"
        )
        caption = (entry or {}).get("caption")
        header = f"[{word} {fig_id}: {caption}]" if caption else f"[{word} {fig_id}]"

        # Equations keep their LaTeX: it is ground truth, and the explanation is
        # a gloss on it rather than a replacement for it.
        payload = (entry or {}).get("payload")
        if kind == "equation" and payload:
            return f"{header}\n{payload}\n\n{description}"
        return f"{header}\n{description}"

    text = re.sub(
        r"<<(fig_[a-z0-9_]+)>>(?:  \(printed caption: [^\n]*\))?", render, marker_text
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    stats["text_blocks"] = marker_text.count("\n\n") + 1
    stats["elapsed"] = round(time.time() - stats.pop("started"), 2)
    stats["output_chars"] = len(text)
    stats["source_document"] = parsed["filename"]
    stats["source_type"] = parsed.get("source_type", "unknown")
    stats["figure_ids"] = [f["id"] for f in figure_records if f["description"]]

    return text, figure_records, stats


# ---------------------------------------------------------------------------
# Conservative transcript cleanup
# ---------------------------------------------------------------------------
#     "The same model also cleans transcripts... Do this as a separate and
#      explicitly conservative pass -- the instruction is FIX TERMS AND
#      PUNCTUATION, CHANGE NOTHING ELSE -- and keep the raw transcript alongside
#      the cleaned one. Cleaning is the step most likely to quietly delete
#      content."
#
# Two guards make that instruction enforceable rather than aspirational:
#
#   1. The transcript is cleaned in windows, so no single generation holds an
#      hour of speech, and no single bad generation can lose an hour of it.
#   2. If a cleaned window's length differs from the raw window by more than
#      `transcript_length_tolerance`, the RAW window is kept. A model that
#      summarised instead of correcting fails this immediately.

TRANSCRIPT_CLEAN_SYSTEM = (
    "You repair automatic speech transcripts of technical lectures.\n"
    "You may: fix mis-transcribed technical terms, fix punctuation and "
    "capitalisation, insert paragraph breaks at topic changes, and mark a "
    "passage where the speaker is plainly reading text off a slide by putting "
    "[READING SLIDE] on its own line before it.\n"
    "You may NOT: summarise, shorten, reword, reorder, add commentary, or remove "
    "anything the speaker said, including repetitions and false starts.\n"
    "Return only the corrected transcript text. No preamble, no explanation."
)


def clean_transcript_text(
    raw_text: str, vocabulary: List[str], model: LLMClient, cfg: Stage3Config
) -> Tuple[str, Dict[str, int]]:
    """Clean window by window, falling back to raw on any doubt."""
    guard = {"windows": 0, "kept_clean": 0, "kept_raw": 0}

    paragraphs = raw_text.split("\n\n")
    windows, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) > cfg.transcript_window_chars:
            windows.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        windows.append(current)

    vocab_hint = ", ".join(vocabulary[:60])
    cleaned_windows: List[str] = []

    for window in tqdm(windows, desc="  cleaning transcript", leave=False):
        guard["windows"] += 1
        prompt = (
            f"KNOWN TECHNICAL TERMS IN THIS CORPUS: {vocab_hint}\n\n"
            f"TRANSCRIPT SEGMENT:\n{window}"
        )
        try:
            cleaned = model.generate(
                system=TRANSCRIPT_CLEAN_SYSTEM,
                user=prompt,
                max_tokens=int(len(window) / 2.5) + 128,
                temperature=0.0,
            ).strip()
        except Exception as exc:
            stage3_log.warning(f"  transcript cleanup failed on a window: {exc}")
            cleaned = ""

        ratio = len(cleaned) / max(1, len(window))
        if cleaned and abs(1 - ratio) <= cfg.transcript_length_tolerance:
            cleaned_windows.append(cleaned)
            guard["kept_clean"] += 1
        else:
            if cleaned:
                stage3_log.warning(
                    f"  cleanup changed length by {abs(1 - ratio):.0%} "
                    f"- keeping raw"
                )
            cleaned_windows.append(window)
            guard["kept_raw"] += 1

    return "\n\n".join(cleaned_windows), guard


def convert_transcripts(model: LLMClient, cfg: Stage3Config) -> List[Dict[str, Any]]:
    """
    Turn every Stage 1 transcript into a converted text file with a
    character -> timestamp index attached.

    Transcripts skip Stage 2 entirely: a layout parser has nothing to say about
    speech. They join the corpus here.
    """
    from .stage1_speech import segments_to_text

    raws = sorted(PATHS.transcripts.glob("*.raw.json"))
    if not raws:
        return []

    vocabulary = load_vocabulary_from_pipeline_b(PATHS.root / "normalized_tags.json")
    results = []

    for raw_path in raws:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        stem = raw_path.name.replace(".raw.json", "")
        out_txt = PATHS.converted / f"{stem}.txt"

        if cfg.skip_existing and not cfg.force_reconvert and out_txt.exists():
            stage3_log.info(f"Transcript already converted: {stem}")
            continue

        # Normalise each segment BEFORE building the index. Normalising the
        # flattened text afterwards shifts character offsets ("…" -> "...",
        # collapsed runs of spaces), which would quietly falsify the
        # `timestamp_index_exact: True` claim for uncleaned transcripts.
        segments = [
            {**seg, "text": normalise_text(seg.get("text", ""))}
            for seg in data["segments"]
        ]
        flat, ts_index = segments_to_text(segments)

        guard = {"windows": 0, "kept_clean": 0, "kept_raw": 0}
        index_exact = True
        text = flat

        if cfg.clean_transcripts:
            stage3_log.info(f"Cleaning transcript: {stem} ({len(flat):,} chars)")
            text, guard = clean_transcript_text(flat, vocabulary, model, cfg)
            # Cleaning shifts character positions, so the index built from the
            # raw text no longer lines up exactly. We keep it and mark it
            # approximate: minute-level provenance survives a few hundred
            # characters of drift, and that is all it is used for.
            index_exact = False
            (PATHS.transcripts / f"{stem}.clean.md").write_text(text, encoding="utf-8")

        out_txt.write_text(text, encoding="utf-8")

        stats = {
            "source_document": data["source_document"],
            "source_type": "transcript",
            "output_chars": len(text),
            "figures": [],
            "figure_ids": [],
            "duration_seconds": data.get("duration_seconds"),
            "language": data.get("language"),
            "timestamp_index": ts_index,
            "timestamp_index_exact": index_exact,
            "cleanup": guard,
        }
        (PATHS.converted / f"{stem}.txt.stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )

        stage3_log.info(
            f"  {stem}: {len(text):,} chars, {len(ts_index)} timestamp anchors, "
            f"{guard['kept_raw']}/{guard['windows']} windows kept raw"
        )
        results.append(stats)

    return results


def run_stage3(
    model: LLMClient, fallback: FallbackDescriber, cfg: Stage3Config
) -> Dict[str, Any]:
    stage3_log.info("=" * 70)
    stage3_log.info("STAGE 3: DESCRIBING FIGURES IN CONTEXT")
    stage3_log.info("=" * 70)

    parsed_files = sorted(PATHS.parsed.glob("*.json"))
    stage3_log.info(f"Found {len(parsed_files)} parsed documents")

    all_figures: List[Dict[str, Any]] = []
    processed, skipped, failed = [], [], []
    totals = {
        "figures_kept": 0,
        "tables_kept": 0,
        "equations_kept": 0,
        "decorative": 0,
        "dropped_layout": 0,
        "document_level": 0,
        "fallback_used": 0,
    }
    t0 = time.time()

    for json_path in tqdm(parsed_files, desc="Describing documents"):
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            stage3_log.error(f"Unreadable {json_path.name}: {exc}")
            failed.append(json_path.name)
            continue

        if parsed.get("status") != "success":
            skipped.append(json_path.name)
            continue

        slug = parsed.get("doc_slug") or doc_slug(parsed["filename"])
        out_txt = PATHS.converted / f"{slug}.txt"
        sidecar = PATHS.converted / f"{slug}.txt.stats.json"

        if cfg.skip_existing and not cfg.force_reconvert and out_txt.exists():
            skipped.append(json_path.name)
            if sidecar.exists():  # keep figure_store complete across resumes
                all_figures.extend(
                    json.loads(sidecar.read_text(encoding="utf-8")).get("figures", [])
                )
            continue

        try:
            text, figures, stats = convert_document(parsed, model, fallback, cfg)
            out_txt.write_text(text, encoding="utf-8")
            stats["figures"] = figures
            sidecar.write_text(
                json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            all_figures.extend(figures)
            processed.append(json_path.name)
            for key in totals:
                totals[key] += stats.get(key, 0)

            stage3_log.info(
                f"  {parsed['filename']}: {stats['output_chars']:,} chars | "
                f"kept {stats['figures_kept']}f/{stats['tables_kept']}t/"
                f"{stats['equations_kept']}e | {stats['decorative']} decorative | "
                f"{stats['dropped_layout']} layout items dropped"
            )
        except Exception as exc:
            stage3_log.error(f"Conversion failed for {json_path.name}: {exc}")
            stage3_log.debug(traceback.format_exc())
            failed.append(json_path.name)
            if not cfg.continue_on_error:
                raise

    transcript_stats = convert_transcripts(model, cfg)

    PATHS.figure_store.write_text(
        json.dumps(
            {"model": model.model_id, "figures": all_figures},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "model": model.model_id,
        "documents_converted": len(processed),
        "documents_skipped": len(skipped),
        "documents_failed": len(failed),
        "transcripts_converted": len(transcript_stats),
        "figures_in_store": len(all_figures),
        "figures_described": sum(1 for f in all_figures if f["description"]),
        "totals": totals,
        "model_calls": model.calls,
        "structured_repairs": model.structured_repairs,
        "structured_failures": model.structured_failures,
        "fallback": dict(fallback.stats),
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    }

    stage3_log.info("-" * 70)
    stage3_log.info(f"  model       : {summary['model']}")
    stage3_log.info(
        f"  converted   : {summary['documents_converted']} documents, "
        f"{summary['transcripts_converted']} transcripts"
    )
    stage3_log.info(
        f"  described   : {summary['figures_described']} of "
        f"{summary['figures_in_store']} items"
    )
    stage3_log.info(f"  model calls : {summary['model_calls']}")
    stage3_log.info(f"  elapsed     : {summary['elapsed_minutes']} min")
    return summary
