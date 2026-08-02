"""
Central place for pipeline choices - model sizes, providers, output locations.

Change a value here instead of hunting through notebook cells / function signatures.
"""

# --- LLM (transcript cleaning, vocab extraction/bootstrapping, drafting) ---
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"

# --- Whisper (speech transcription, Module 1.2) ---
# "small" while iterating locally - fully cached after the first run, fast, no multi-GB
# download. Switch to "large-v3" for real lecture-quality transcription once you have a
# stable connection (and ideally a GPU) - see ingestion/speech.py for why it matters.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "auto"

# Fast draft pass used only to bootstrap vocab from audio when no other material exists
# (ingestion/vocab.py: bootstrap_vocab_from_audio). Deliberately small/cheap.
DRAFT_WHISPER_MODEL_SIZE = "small"
MAX_VOCAB_TERMS = 60

# --- Output locations ---
TRANSCRIPTS_OUT_DIR = "output/transcripts"
