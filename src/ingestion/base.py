"""
ingestion.base - the contract every ingestion strategy implements.

Pipeline A takes many kinds of source material - lecture audio/video today, PDFs,
DOCX, PPTX, and slide images as more strategies get added (MinerU/Docling for
documents, Qwen2-VL for images, ...). Whatever consumes this output downstream
(chunking, tag extraction, TOC generation) shouldn't have to know which strategy
produced it, so every strategy is required to produce the same shape: an
IngestedDocument made of ordered IngestedSegments.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class IngestedSegment:
    """One addressable unit of a document - a transcript sentence, a PDF paragraph, a
    slide's speaker notes, whatever the source naturally breaks into.

    start/end and page are optional because they only make sense for some sources
    (time-based vs paginated); anything strategy-specific that doesn't fit the common
    shape (word-level timestamps, a bounding box, a slide number) goes in `metadata`.
    """

    text: str
    order: int
    start: float | None = None
    end: float | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestedDocument:
    source: str
    source_type: str
    segments: list[IngestedSegment]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "IngestedDocument":
        data = json.loads(Path(path).read_text())
        data["segments"] = [IngestedSegment(**s) for s in data["segments"]]
        return cls(**data)


class Ingestor(ABC):
    """One strategy for turning a raw source file into an IngestedDocument.

    Subclasses declare `source_type` and `supported_extensions`; `supports()` lets a
    registry pick the right strategy for a file without the caller knowing whether
    Whisper, MinerU, Qwen2-VL, or something else lives behind the interface.
    """

    source_type: ClassVar[str]
    supported_extensions: ClassVar[frozenset[str]]

    @classmethod
    def supports(cls, source_path: str | Path) -> bool:
        return Path(source_path).suffix.lower() in cls.supported_extensions

    @abstractmethod
    def ingest(self, source_path: str | Path) -> IngestedDocument:
        """Convert one source file into an IngestedDocument."""


def save_document(document: IngestedDocument, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(document.source).stem}.json"
    document.to_json(out_path)
    return out_path
