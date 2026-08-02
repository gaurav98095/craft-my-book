from .base import Ingestor, IngestedDocument, IngestedSegment, save_document
from .registry import STRATEGIES, get_ingestor_class
from .speech import (
    Segment,
    SpeechIngestor,
    Transcript,
    Word,
    clean_transcript,
    extract_audio,
    transcribe_audio,
)
from .vocab import (
    bootstrap_vocab_from_audio,
    extract_vocab,
    vocab_material_from_filenames,
)

__all__ = [
    # strategy contract
    "Ingestor",
    "IngestedDocument",
    "IngestedSegment",
    "save_document",
    # dispatch
    "STRATEGIES",
    "get_ingestor_class",
    # speech strategy
    "SpeechIngestor",
    "Word",
    "Segment",
    "Transcript",
    "extract_audio",
    "transcribe_audio",
    "clean_transcript",
    # vocab
    "extract_vocab",
    "vocab_material_from_filenames",
    "bootstrap_vocab_from_audio",
]
