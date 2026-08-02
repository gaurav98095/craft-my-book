"""
Loads config/config.yaml into typed, per-component config objects.

Nested by pipeline component (top-level key) so each part of the pipeline - and each
model it calls - can be tuned independently. e.g. INGESTION.speech has its own
cleaning_llm and vocab_llm; they don't have to be the same provider/model as each
other, let alone as some other component added later (toc, writer, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_raw = yaml.safe_load(_CONFIG_PATH.read_text())

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt template from config/prompts/<name> (e.g. "speech_cleaning.txt")."""
    return (_PROMPTS_DIR / name).read_text()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class SpeechConfig:
    whisper: WhisperConfig
    draft_whisper_model_size: str
    cleaning_llm: LLMConfig
    vocab_llm: LLMConfig
    max_vocab_terms: int


@dataclass(frozen=True)
class IngestionConfig:
    output_dir: str
    speech: SpeechConfig


def _load_llm(d: dict) -> LLMConfig:
    return LLMConfig(provider=d["provider"], model=d["model"])


def _load_ingestion(d: dict) -> IngestionConfig:
    speech = d["speech"]
    return IngestionConfig(
        output_dir=d["output_dir"],
        speech=SpeechConfig(
            whisper=WhisperConfig(**speech["whisper"]),
            draft_whisper_model_size=speech["draft_whisper"]["model_size"],
            cleaning_llm=_load_llm(speech["cleaning_llm"]),
            vocab_llm=_load_llm(speech["vocab_llm"]),
            max_vocab_terms=speech["vocab_llm"]["max_terms"],
        ),
    )


INGESTION = _load_ingestion(_raw["ingestion"])

__all__ = [
    "INGESTION",
    "IngestionConfig",
    "SpeechConfig",
    "WhisperConfig",
    "LLMConfig",
    "load_prompt",
]
