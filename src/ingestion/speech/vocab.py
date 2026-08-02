"""
ingestion.speech.vocab - automatic domain-vocabulary extraction.

Whisper's initial_prompt (Module 1.2) works best when primed with the domain's own
technical vocabulary, but that vocabulary isn't knowable ahead of time - the corpus
could be about transformers this run and control theory the next. Rather than hand-
authoring a vocab list, extract candidate terms from whatever material already exists
around the recording (slide titles, filenames, headings, abstracts) via an LLM pass,
mirroring what Pipeline A's own PDFs/slides already carry.
"""

from __future__ import annotations

from pathlib import Path

from config import INGESTION, load_prompt
from llm import LLMClient, chat

from .whisper import transcribe_audio

_EXTRACT_PROMPT = load_prompt("vocab_extraction.txt")


def extract_vocab(
    material: str,
    client: LLMClient,
    model: str,
    max_terms: int = INGESTION.speech.max_vocab_terms,
) -> list[str]:
    """Extract a domain vocabulary list from source material (slide titles, filenames,
    headings, abstracts, etc.) so Whisper can be primed without a hardcoded term list.
    """
    if not material.strip():
        return []

    response = chat(client, model, _EXTRACT_PROMPT.format(material=material))
    terms = [t.strip() for t in response.split(",") if t.strip()]

    seen: set[str] = set()
    deduped = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped[:max_terms]


def vocab_material_from_filenames(paths: list[str | Path]) -> str:
    """Cheap fallback source material: filenames/slide titles often already carry domain terms."""
    return "\n".join(Path(p).stem.replace("_", " ").replace("-", " ") for p in paths)


def bootstrap_vocab_from_audio(
    audio_path: str | Path,
    client: LLMClient,
    model: str,
    draft_model_size: str = INGESTION.speech.draft_whisper_model_size,
    max_terms: int = INGESTION.speech.max_vocab_terms,
) -> list[str]:
    """
    Fallback for when audio is the *only* source - no slides, no filenames worth reading.

    Runs a fast, unprimed draft transcription (small Whisper model, cheap) and extracts
    candidate domain terms from its rough text. The draft may still mishear some terms,
    but an LLM skimming the draft catches far more real vocabulary than no priming at
    all - and the real transcription pass then gets primed with what it needs to self-correct.
    """
    draft = transcribe_audio(audio_path, vocab=None, model_size=draft_model_size)
    material = "\n".join(seg.text for seg in draft.segments)
    return extract_vocab(material, client, model, max_terms=max_terms)
