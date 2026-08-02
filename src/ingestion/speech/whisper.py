"""
ingestion.speech.whisper - Module 1.2: Speech Processing (Whisper).

Turns lecture audio/video into a structured, timestamped, domain-aware transcript,
suitable for downstream tag extraction / TOC generation / retrieval.

Pipeline: video -> audio (ffmpeg) -> raw transcript (faster-whisper) -> cleaned
transcript (conservative LLM pass) -> Transcript (internal) -> IngestedDocument.

The low-level functions (extract_audio, transcribe_audio, clean_transcript) are
useful on their own for inspecting intermediate output; SpeechIngestor wraps them
into the Ingestor strategy (see ingestion.base) that ingestion.registry dispatches
to for audio/video sources.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

from config import INGESTION, load_prompt
from llm import LLMClient, chat

from ..base import Ingestor, IngestedDocument, IngestedSegment

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    source: str  # original video/audio filename, kept for provenance
    language: str
    duration: float
    segments: list[Segment]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Transcript":
        data = json.loads(Path(path).read_text())
        data["segments"] = [
            Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]})
            for s in data["segments"]
        ]
        return cls(**data)


# --------------------------------------------------------------------------
# Stage 1a - audio extraction
# --------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio(source_path: str | Path, out_path: str | Path | None = None) -> Path:
    """
    Normalize a lecture source - video OR audio - to mono 16kHz WAV via ffmpeg.

    ffmpeg decodes whichever container/codec the source uses and -vn drops any video
    stream (a no-op when the source is already audio-only), so the same call handles
    both a lecture video and a standalone audio recording (e.g. a podcast-style talk).
    """
    source_path = Path(source_path)
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported source extension '{source_path.suffix}'. Expected one of: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    # with_suffix(".wav") would collide with the input when the source is already a .wav
    out_path = (
        Path(out_path)
        if out_path
        else source_path.with_name(f"{source_path.stem}.16k.wav")
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path


# --------------------------------------------------------------------------
# Stage 1b - transcription
# --------------------------------------------------------------------------

_MODEL_CACHE: dict[str, WhisperModel] = {}


def _get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    key = f"{model_size}:{device}:{compute_type}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _MODEL_CACHE[key]


def transcribe_audio(
    audio_path: str | Path,
    vocab: list[str] | None = None,
    model_size: str = INGESTION.speech.whisper.model_size,
    device: str = INGESTION.speech.whisper.device,
    compute_type: str = INGESTION.speech.whisper.compute_type,
) -> Transcript:
    """
    Domain-primed, VAD-filtered transcription with word-level timestamps.

    - initial_prompt seeds Whisper with domain vocabulary so terms like "CUDA" / "LoRA" /
      "KV Cache" aren't misheard as near-homophones; that error would otherwise propagate
      into tag extraction and pollute the vocabulary Pipeline B builds.
    - vad_filter drops silence so segment boundaries stay clean and runtime stays down.
    - word_timestamps is kept for provenance: any sentence in the book can be traced back
      to a (source, start, end) triple.
    - condition_on_previous_text=False avoids repetition loops that faster-whisper can fall
      into on long (1-3h) lectures, trading a little cross-segment continuity for robustness.
    """
    model = _get_model(model_size, device, compute_type)

    raw_segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        word_timestamps=True,
        initial_prompt=", ".join(vocab) if vocab else None,
        condition_on_previous_text=False,
    )

    segments = [
        Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
            words=[
                Word(start=w.start, end=w.end, text=w.word.strip())
                for w in (seg.words or [])
            ],
        )
        for seg in raw_segments
    ]

    return Transcript(
        source=Path(audio_path).name,
        language=info.language,
        duration=info.duration,
        segments=segments,
    )


# --------------------------------------------------------------------------
# Stage 1c - conservative cleaning pass
# --------------------------------------------------------------------------

_CLEAN_SYSTEM_PROMPT = load_prompt("speech_cleaning.txt")


def clean_segment(text: str, client: LLMClient, model: str) -> str:
    prompt = f"{_CLEAN_SYSTEM_PROMPT}\n\nRaw segment:\n{text}\n\nCleaned segment:"
    return chat(client, model, prompt).strip()


def clean_transcript(
    transcript: Transcript, client: LLMClient, model: str
) -> Transcript:
    """Conservative per-segment LLM pass: fix terms/punctuation, change nothing else.

    Segment start/end timestamps are untouched, so provenance still holds after cleaning.
    """
    for seg in transcript.segments:
        seg.text = clean_segment(seg.text, client, model)
    return transcript


# --------------------------------------------------------------------------
# Strategy - the Ingestor this module contributes to the registry
# --------------------------------------------------------------------------


class SpeechIngestor(Ingestor):
    """Ingestion strategy for lecture audio/video: ffmpeg -> Whisper -> LLM cleanup.

    `vocab` should come from ingestion.vocab.extract_vocab() (or bootstrap_vocab_from_audio),
    run per corpus - there's no single vocabulary that fits every domain a lecture might cover.
    """

    source_type = "speech"
    supported_extensions = frozenset(SUPPORTED_EXTENSIONS)

    def __init__(
        self,
        client: LLMClient,
        clean_model: str,
        vocab: list[str] | None = None,
        whisper_model_size: str = INGESTION.speech.whisper.model_size,
    ):
        self.client = client
        self.clean_model = clean_model
        self.vocab = vocab
        self.whisper_model_size = whisper_model_size

    def ingest(self, source_path: str | Path) -> IngestedDocument:
        audio_path = extract_audio(source_path)
        transcript = transcribe_audio(
            audio_path, vocab=self.vocab, model_size=self.whisper_model_size
        )
        transcript = clean_transcript(transcript, self.client, self.clean_model)
        return self._to_document(source_path, transcript)

    @classmethod
    def _to_document(
        cls, source_path: str | Path, transcript: Transcript
    ) -> IngestedDocument:
        segments = [
            IngestedSegment(
                text=seg.text,
                order=i,
                start=seg.start,
                end=seg.end,
                metadata={"words": [asdict(w) for w in seg.words]} if seg.words else {},
            )
            for i, seg in enumerate(transcript.segments)
        ]
        return IngestedDocument(
            source=Path(source_path).name,
            source_type=cls.source_type,
            segments=segments,
            metadata={"language": transcript.language, "duration": transcript.duration},
        )
