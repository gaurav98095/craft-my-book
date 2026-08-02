"""
ingestion.speech - Module 1.2: Speech Processing (Whisper).

Turns lecture audio/video into a structured, timestamped, domain-aware transcript,
suitable for downstream tag extraction / TOC generation / retrieval.

Pipeline: video -> audio (ffmpeg) -> raw transcript (faster-whisper) -> cleaned
transcript (conservative LLM pass) -> Transcript JSON.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

from config import TRANSCRIPTS_OUT_DIR, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE
from llm import LLMClient, chat

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
    out_path = Path(out_path) if out_path else source_path.with_name(f"{source_path.stem}.16k.wav")

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
    model_size: str = WHISPER_MODEL_SIZE,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
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

_CLEAN_SYSTEM_PROMPT = """You clean raw lecture transcript segments for a technical book pipeline.

Fix ONLY:
- misheard technical terms (e.g. "could a" -> "CUDA", "laura" -> "LoRA")
- punctuation and sentence breaks
- obvious ASR artifacts (stutters, false starts, filler words like "um"/"uh")

Do NOT:
- summarize, paraphrase, or reword sentences
- add information that wasn't said, or remove information that was said
- change the meaning or the speaker's intent

Return only the cleaned segment text, nothing else."""


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
# Orchestration
# --------------------------------------------------------------------------


def process_source(
    source_path: str | Path,
    vocab: list[str] | None,
    client: LLMClient,
    clean_model: str,
    whisper_model_size: str = WHISPER_MODEL_SIZE,
    out_dir: str | Path = TRANSCRIPTS_OUT_DIR,
) -> Transcript:
    """Video or audio source -> normalized audio -> raw transcript -> cleaned transcript -> saved JSON.

    `vocab` should come from ingestion.vocab.extract_vocab() (or similar), run per corpus -
    there's no single vocabulary that fits every domain a lecture might cover.
    """
    audio_path = extract_audio(source_path)
    transcript = transcribe_audio(
        audio_path, vocab=vocab, model_size=whisper_model_size
    )
    transcript = clean_transcript(transcript, client, clean_model)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript.to_json(out_dir / f"{Path(source_path).stem}.json")

    return transcript
