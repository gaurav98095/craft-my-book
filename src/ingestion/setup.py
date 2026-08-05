"""
Stage 0 — Paths, logging, and the domain vocabulary.

One place that decides where everything lives. Every stage in Pipeline A
imports PATHS from here rather than constructing its own directory strings,
so re-running one stage after another can never point them at different
directories.
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()          # reads .env in the cwd (or a parent dir) if present
except ImportError:
    pass                    # python-dotenv is optional; real env vars still work


# ---------------------------------------------------------------- paths -----
@dataclass
class PipelinePaths:
    """
    The on-disk layout for Pipeline A, mirroring `Files on Disk` in the design doc.

    `root` and `storage` default from DATA_ROOT / STORAGE_ROOT so the same
    corpus tree can live somewhere other than the repo (a scratch disk, a
    mounted volume) without editing code.
    """

    root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data")))

    @property
    def raw_sources(self) -> Path:  # what you drop in: PDFs, slides, lectures
        return self.root / "raw_sources"

    @property
    def transcripts(self) -> Path:  # Stage 1: Whisper output, timestamps kept
        return self.root / "transcripts"

    @property
    def parsed(self) -> Path:  # Stage 2: layout JSON, one dir per document
        return self.root / "parsed"

    @property
    def converted(self) -> Path:  # Stage 3: plain text, one file per document
        return self.root / "converted"

    @property
    def figures(self) -> Path:  # Stage 3b: the retained image crops
        return self.root / "figures"

    @property
    def chunks(self) -> Path:  # Stage 4: chunk_0001.txt ... (Layer 3)
        return self.root / "chunks"

    @property
    def source_index(self) -> Path:  # Stage 5: Chroma over the chunks (Layer 3)
        return self.root / "source_index"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def consolidated_text(self) -> Path:
        return self.root / "raw_consolidated_text.txt"

    @property
    def document_index(self) -> Path:
        return self.root / "document_index.json"

    @property
    def chunk_metadata(self) -> Path:  # L3 - provenance, timestamps, tags, figures
        return self.root / "chunk_metadata.json"

    # -- Pipeline B artifacts. The design's `Files on Disk` puts the tag files
    # -- under data/ (they describe the sources) and toc.json under storage/
    # -- (it describes the book).
    @property
    def normalized_tags(self) -> Path:  # canonical -> aliases, seeds the ledger
        return self.root / "normalized_tags.json"

    @property
    def tag_relationships(self) -> Path:  # canonical -> prerequisites
        return self.root / "tag_relationships.json"

    @property
    def chunk_tags(self) -> Path:  # chunk -> canonical tags
        return self.root / "chunk_tags.json"

    @property
    def storage(self) -> Path:
        return Path(os.getenv("STORAGE_ROOT", "./storage"))

    @property
    def toc(self) -> Path:
        return self.storage / "toc.json"

    @property
    def figure_store(self) -> Path:
        return self.root / "figure_store.json"

    def mkdirs(self) -> None:
        for p in (
            self.raw_sources,
            self.transcripts,
            self.parsed,
            self.converted,
            self.figures,
            self.chunks,
            self.checkpoints,
            self.storage,
        ):
            p.mkdir(parents=True, exist_ok=True)


PATHS = PipelinePaths()
PATHS.mkdirs()


# -------------------------------------------------------------- logging -----
def make_logger(name: str, logfile: str) -> logging.Logger:
    """
    One logger per stage, each with its own file.

    Jupyter re-runs cells, and `logging.basicConfig` is a no-op the second time
    it is called — which is why per-stage log files can quietly all go to
    whichever file was configured first. Building the logger explicitly and
    clearing old handlers avoids that.
    """
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


# ------------------------------------------------- the domain vocabulary ----
# Seed Whisper with the terms you already know are in the corpus. Anything you
# leave out, Whisper will guess at phonetically.
#
# After Pipeline B has run once, replace this list with the canonical tag
# vocabulary from `normalized_tags.json` -- it is a strictly better seed and it
# costs nothing, because that LLM bill has already been paid.
DOMAIN_VOCABULARY: List[str] = [
    "transformer",
    "attention",
    "self-attention",
    "multi-head attention",
    "KV cache",
    "tokenizer",
    "embedding",
    "positional encoding",
    "RAG",
    "retrieval augmented generation",
    "vector database",
    "chunking",
    "reranker",
    "LLM",
    "fine-tuning",
    "LoRA",
    "quantization",
    "ReAct loop",
    "agent",
    "tool calling",
    "function calling",
    "PyTorch",
    "CUDA",
    "Hugging Face",
    "prompt engineering",
]


def load_vocabulary_from_pipeline_b(normalized_tags_path: Path) -> List[str]:
    """
    If a previous run produced Pipeline B's canonical tag vocabulary, use it.
    Falls back to the hand-written list above.
    """
    if not normalized_tags_path.exists():
        return DOMAIN_VOCABULARY
    try:
        tags = json.loads(normalized_tags_path.read_text(encoding="utf-8"))
        canonical = list(tags.keys()) if isinstance(tags, dict) else list(tags)
        return sorted(set(DOMAIN_VOCABULARY) | set(canonical))
    except Exception:
        return DOMAIN_VOCABULARY


# ------------------------------------------------- source type detection ----
# `source_type` is provenance, and provenance changes how the Writer phrases
# things. The design doc puts it well: a reader can tell the difference between
# "as Vaswani et al. put it" and "as the lecture demonstrated on the whiteboard"
# -- but only if the Writer knows which is which.
SOURCE_TYPE_BY_EXT: Dict[str, str] = {
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".md": "text",
    ".txt": "text",
    ".pptx": "slides",
    ".ppt": "slides",
    ".xlsx": "spreadsheet",
    ".xls": "spreadsheet",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
    ".mp4": "transcript",
    ".mkv": "transcript",
    ".mov": "transcript",
    ".avi": "transcript",
    ".webm": "transcript",
    ".mp3": "transcript",
    ".wav": "transcript",
    ".m4a": "transcript",
    ".flac": "transcript",
    ".ogg": "transcript",
}

MEDIA_EXTENSIONS = {e for e, t in SOURCE_TYPE_BY_EXT.items() if t == "transcript"}
DOCUMENT_EXTENSIONS = {e for e, t in SOURCE_TYPE_BY_EXT.items() if t != "transcript"}


def source_type_for(path: Path) -> str:
    return SOURCE_TYPE_BY_EXT.get(path.suffix.lower(), "unknown")


def doc_slug(name: str) -> str:
    """
    A short, stable, filesystem-safe id derived from a filename.

    Truncation alone is not enough. "...Advanced RAG (Part 10)..." and
    "...(Part 11)..." share their first 40 characters, so a truncated slug
    silently MERGES the two documents: the second one looks "already parsed",
    gets skipped, and simply never enters the corpus. A short hash of the FULL
    filename keeps every id distinct while staying readable and stable.
    """
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    return f"{(slug[:33] or 'doc')}_{digest}"


# --------------------------------------------------- text normalisation -----
_SMART_QUOTES = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": " -- ",
    "…": "...",
    " ": " ",
    "ﬁ": "fi",
    "ﬂ": "fl",
}


def normalise_text(text: str) -> str:
    """
    Conservative typographic cleanup.

    Deliberately does NOT touch wording, line structure, or anything a language
    model would have an opinion about. Smart quotes and ligatures are the ones
    worth fixing here because they break naive string matching later -- the
    ledger's alias resolution in Pipeline C is exact string comparison.
    """
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ------------------------------------------------------------ the model ----
# The design document (section 1, "The models") specifies TWO models and one of
# them does everything that is not speech:
#
#   | Speech to text                      | Whisper large-v3 |
#   | Everything else - figure description, tagging, planning, writing,
#     editing, cataloguing                | one shared LLMClient               |
#
# Using one language model for the whole pipeline is a deliberate choice:
#
#   "The model that describes a diagram during preprocessing is the same model
#    that later writes the chapter about it. That alone does more for
#    consistency than any amount of prompt tuning."
#
# Which model that actually is -- local weights, Anthropic, Groq, a
# self-hosted server -- is decided once, for the whole system, by .env.
# See `src.llm.build_llm_client`.
WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "large-v3")


# ------------------------------------------------------ token accounting ----
# Chunk sizes are specified in tokens, not characters, so we need a counter.
# tiktoken's cl100k_base is not Qwen's tokenizer, but it is close enough for
# deciding whether a chunk is 1,500 or 2,500 tokens, and it needs no GPU.
try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def count_tokens(text: str) -> int:
    """Token count, falling back to the ~4-characters-per-token approximation."""
    if _ENCODING is not None:
        return len(_ENCODING.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)
