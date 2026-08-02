"""
ingestion.registry - pick the right Ingestor strategy for a source file by extension.

Adding a new strategy (PDF, DOCX, PPTX, slide images, ...) means writing a class and
adding one line to STRATEGIES - nothing that dispatches to an Ingestor needs to change.
"""

from __future__ import annotations

from pathlib import Path

from .base import Ingestor
from .speech import SpeechIngestor

STRATEGIES: tuple[type[Ingestor], ...] = (SpeechIngestor,)


def get_ingestor_class(source_path: str | Path) -> type[Ingestor]:
    for strategy in STRATEGIES:
        if strategy.supports(source_path):
            return strategy
    raise ValueError(
        f"No ingestion strategy registered for '{Path(source_path).suffix}' "
        f"({Path(source_path).name}). Registered: {[s.source_type for s in STRATEGIES]}"
    )
