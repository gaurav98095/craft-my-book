"""ingestion.speech - Module 1.2: lecture audio/video -> transcript (Whisper)."""

from .vocab import bootstrap_vocab_from_audio, extract_vocab, vocab_material_from_filenames
from .whisper import (
    Segment,
    SpeechIngestor,
    Transcript,
    Word,
    clean_transcript,
    extract_audio,
    transcribe_audio,
)

__all__ = [
    "Word",
    "Segment",
    "Transcript",
    "SpeechIngestor",
    "extract_audio",
    "transcribe_audio",
    "clean_transcript",
    "extract_vocab",
    "vocab_material_from_filenames",
    "bootstrap_vocab_from_audio",
]
