"""Loads config/config.yaml once and exposes its values as module-level constants."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_raw = yaml.safe_load(_CONFIG_PATH.read_text())

LLM_PROVIDER: str = _raw["llm"]["provider"]
LLM_MODEL: str = _raw["llm"]["model"]

WHISPER_MODEL_SIZE: str = _raw["whisper"]["model_size"]
WHISPER_DEVICE: str = _raw["whisper"]["device"]
WHISPER_COMPUTE_TYPE: str = _raw["whisper"]["compute_type"]
DRAFT_WHISPER_MODEL_SIZE: str = _raw["whisper"]["draft_model_size"]

MAX_VOCAB_TERMS: int = _raw["vocab"]["max_terms"]

TRANSCRIPTS_OUT_DIR: str = _raw["output"]["transcripts_dir"]

__all__ = [
    "LLM_PROVIDER",
    "LLM_MODEL",
    "WHISPER_MODEL_SIZE",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "DRAFT_WHISPER_MODEL_SIZE",
    "MAX_VOCAB_TERMS",
    "TRANSCRIPTS_OUT_DIR",
]
