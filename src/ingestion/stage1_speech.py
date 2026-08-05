"""Stage 1 — Speech to Text (Whisper large-v3)."""

import json
import time
import subprocess
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from .setup import (
    PATHS,
    make_logger,
    load_vocabulary_from_pipeline_b,
    normalise_text,
    MEDIA_EXTENSIONS,
)

stage1_log = make_logger("stage1.speech", "stage1_speech.log")


@dataclass
class Stage1Config:
    """Configuration for Stage 1 - Speech to Text."""

    # Model
    model_size: str = "large-v3"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    compute_type: str = "float16"  # "float16" on GPU, "int8" on CPU

    # Decoding
    vad_filter: bool = True  # drop silence; large speedup on lectures
    word_timestamps: bool = True  # provenance, not readability
    condition_on_previous_text: bool = False  # stops repetition loops on long files
    beam_size: int = 5
    language: Optional[str] = None  # None = autodetect

    # Audio extraction
    sample_rate: int = 16_000
    keep_extracted_wav: bool = False  # WAVs are large; delete once transcribed

    # Resume
    skip_existing: bool = True


def extract_audio(media_path: Path, out_dir: Path, sample_rate: int = 16_000) -> Path:
    """
    Pull a mono 16 kHz WAV out of any media container using ffmpeg.

    Whisper resamples internally anyway, but doing it once here means we do not
    pay for it on every retry, and it makes an MP4 and an MP3 look identical to
    everything downstream.

        -ac 1     mono         (Whisper is mono; stereo just doubles the bytes)
        -ar 16000 16 kHz       (Whisper's native rate)
        -vn       no video     (we only want the audio stream)
        -y        overwrite
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{media_path.stem}.wav"

    if wav_path.exists():
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(wav_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # ffmpeg writes everything to stderr, including normal progress output,
        # so only the tail is worth showing.
        raise RuntimeError(
            f"ffmpeg failed for {media_path.name}:\n{proc.stderr[-800:]}"
        )

    return wav_path


_whisper_model = None  # loaded lazily; the model is ~3 GB


def get_whisper_model(cfg: Stage1Config):
    """Load Whisper once and reuse it. Returns None if faster-whisper is absent."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        stage1_log.warning("faster-whisper is not installed - Stage 1 will be skipped.")
        return None

    device = cfg.device
    compute_type = cfg.compute_type
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu" and compute_type == "float16":
        # float16 on CPU is either unsupported or catastrophically slow.
        compute_type = "int8"

    stage1_log.info(f"Loading Whisper {cfg.model_size} on {device} ({compute_type})...")
    t0 = time.time()
    _whisper_model = WhisperModel(
        cfg.model_size, device=device, compute_type=compute_type
    )
    stage1_log.info(f"Whisper loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def transcribe_media(
    media_path: Path, cfg: Stage1Config, vocabulary: List[str]
) -> Dict[str, Any]:
    """
    Transcribe one audio or video file into the structure Pipeline A expects.

    Returns a dict with the *segments preserved*. The flat text is derived from
    them, never stored instead of them.
    """
    model = get_whisper_model(cfg)
    if model is None:
        raise RuntimeError("Whisper model unavailable")

    wav = extract_audio(media_path, PATHS.transcripts / "_wav", cfg.sample_rate)

    stage1_log.info(f"Transcribing {media_path.name} ...")
    t0 = time.time()

    segments_iter, info = model.transcribe(
        str(wav),
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        beam_size=cfg.beam_size,
        language=cfg.language,
        initial_prompt=", ".join(vocabulary),  # <-- the vocabulary prior
        condition_on_previous_text=cfg.condition_on_previous_text,
    )

    # faster-whisper streams segments lazily; iterating is what does the work.
    segments = []
    for s in segments_iter:
        segments.append(
            {
                "id": len(segments),
                "start": round(float(s.start), 2),
                "end": round(float(s.end), 2),
                "text": s.text.strip(),
            }
        )

    elapsed = time.time() - t0
    duration = float(getattr(info, "duration", 0.0) or 0.0)

    if not cfg.keep_extracted_wav:
        wav.unlink(missing_ok=True)

    result = {
        "source_document": media_path.name,
        "source_type": "transcript",
        "language": getattr(info, "language", None),
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0) or 0.0), 3
        ),
        "duration_seconds": round(duration, 2),
        "transcribe_seconds": round(elapsed, 2),
        "model": f"whisper-{cfg.model_size}",
        "vocabulary_terms": len(vocabulary),
        "segments": segments,
    }

    stage1_log.info(
        f"  {len(segments)} segments | {duration/60:.1f} min audio "
        f"| {elapsed/60:.1f} min compute | lang={result['language']}"
    )
    return result


def segments_to_text(
    segments: List[Dict[str, Any]], paragraph_gap_seconds: float = 2.0
) -> Tuple[str, List[Dict]]:
    """
    Flatten segments into readable text AND build a character -> timestamp index.

    The index is what lets a chunk 40,000 characters into the corpus report
    "this came from lecture_04.mp4 at 34:12". Without it, timestamps are a
    number in a JSON file that nothing can ever use.

    A pause longer than `paragraph_gap_seconds` is treated as a paragraph break.
    It is a crude heuristic, but it is the only structural signal speech has,
    and it beats one 60,000-character wall of text.
    """
    parts: List[str] = []
    index: List[Dict[str, Any]] = []
    cursor = 0
    prev_end = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        if prev_end is not None:
            sep = "\n\n" if (seg["start"] - prev_end) > paragraph_gap_seconds else " "
            parts.append(sep)
            cursor += len(sep)

        index.append(
            {
                "char_start": cursor,
                "char_end": cursor + len(text),
                "t_start": seg["start"],
                "t_end": seg["end"],
            }
        )
        parts.append(text)
        cursor += len(text)
        prev_end = seg["end"]

    return "".join(parts), index


def run_stage1_speech(cfg: Stage1Config) -> Dict[str, Any]:
    """
    Transcribe every audio/video file in data/raw_sources/.

    Skips cleanly (rather than crashing) when there is no media, or when
    faster-whisper / ffmpeg are not installed -- so the rest of the pipeline
    still runs on a document-only corpus.
    """
    stage1_log.info("=" * 70)
    stage1_log.info("STAGE 1: SPEECH TO TEXT")
    stage1_log.info("=" * 70)

    media_files = sorted(
        p
        for p in PATHS.raw_sources.rglob("*")
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )

    summary: Dict[str, Any] = {
        "media_found": len(media_files),
        "transcribed": [],
        "skipped": [],
        "failed": [],
        "total_audio_minutes": 0.0,
    }

    if not media_files:
        stage1_log.info("No audio or video files found - nothing to do.")
        stage1_log.info(
            "(This is fine. Stage 1 only runs if your corpus has recordings.)"
        )
        return summary

    vocabulary = load_vocabulary_from_pipeline_b(PATHS.root / "normalized_tags.json")
    stage1_log.info(f"Priming Whisper with {len(vocabulary)} domain terms")

    if get_whisper_model(cfg) is None:
        stage1_log.warning("Skipping Stage 1 - install faster-whisper to enable it.")
        summary["skipped"] = [p.name for p in media_files]
        return summary

    for media in tqdm(media_files, desc="Transcribing"):
        out_json = PATHS.transcripts / f"{media.stem}.raw.json"

        if cfg.skip_existing and out_json.exists():
            stage1_log.info(f"Skipping (already transcribed): {media.name}")
            summary["skipped"].append(media.name)
            continue

        try:
            result = transcribe_media(media, cfg, vocabulary)
            out_json.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # Also write the flat text now, so a run can be inspected without
            # waiting for Stage 3. Stage 3 will overwrite it with a cleaned
            # version, and keep this one alongside.
            flat, _ = segments_to_text(result["segments"])
            (PATHS.transcripts / f"{media.stem}.raw.txt").write_text(
                normalise_text(flat), encoding="utf-8"
            )

            summary["transcribed"].append(media.name)
            summary["total_audio_minutes"] += result["duration_seconds"] / 60.0

        except Exception as exc:
            stage1_log.error(f"Failed on {media.name}: {exc}")
            stage1_log.debug(traceback.format_exc())
            summary["failed"].append({"file": media.name, "error": str(exc)})

    stage1_log.info("-" * 70)
    stage1_log.info(f"Transcribed : {len(summary['transcribed'])}")
    stage1_log.info(f"Skipped     : {len(summary['skipped'])}")
    stage1_log.info(f"Failed      : {len(summary['failed'])}")
    stage1_log.info(f"Audio       : {summary['total_audio_minutes']:.1f} minutes")
    return summary
