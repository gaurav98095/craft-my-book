"""
ingestion - Pipeline A: turn raw source material into IngestedDocuments.

base.py / registry.py are the shared contract every modality plugs into; each
modality then lives in its own subpackage (speech/, layout/, vision/, ...) so adding
one doesn't touch the others. See ingestion.base.Ingestor for the strategy interface
and ingestion.registry.get_ingestor_class for how a source file gets routed to one.
"""

from .base import Ingestor, IngestedDocument, IngestedSegment, save_document
from .layout import LayoutIngestor, parse_content_list, run_mineru
from .registry import STRATEGIES, get_ingestor_class
from .speech import (
    Segment,
    SpeechIngestor,
    Transcript,
    Word,
    bootstrap_vocab_from_audio,
    clean_transcript,
    extract_audio,
    extract_vocab,
    transcribe_audio,
    vocab_material_from_filenames,
)
from .vision import describe_visuals

__all__ = [
    # strategy contract
    "Ingestor",
    "IngestedDocument",
    "IngestedSegment",
    "save_document",
    # dispatch
    "STRATEGIES",
    "get_ingestor_class",
    # speech modality
    "SpeechIngestor",
    "Word",
    "Segment",
    "Transcript",
    "extract_audio",
    "transcribe_audio",
    "clean_transcript",
    "extract_vocab",
    "vocab_material_from_filenames",
    "bootstrap_vocab_from_audio",
    # layout modality
    "LayoutIngestor",
    "run_mineru",
    "parse_content_list",
    # vision modality (enrichment pass, not a registry strategy)
    "describe_visuals",
]
