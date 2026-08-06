"""
Step B0 — Setup: configuration and logging for Pipeline B.

Pipeline B reads what Pipeline A produced. It does not re-chunk, it does not
invent its own directory layout, and it does not load its own model --
which model answers its calls is decided once, for the whole system, by
`.env` (see `src.llm.build_llm_client`), and the same client is reused here.
"""

import logging
from dataclasses import dataclass, field

# Sets RUN_ID and forces DATA_ROOT / STORAGE_ROOT / OUTPUT_ROOT / LOGS_ROOT to
# runs/<run_id>/... (a no-op if ingestion.setup already ran it this process).
from ..run_context import LOGS_ROOT


def make_toc_logger(name: str, logfile: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for handler in (
        logging.FileHandler(LOGS_ROOT / logfile, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(fmt)
        lg.addHandler(handler)
    return lg


toc_log = make_toc_logger("pipelineB.toc", "pipelineB_toc.log")


@dataclass
class BookSizeConfig:
    """
    The design's arithmetic, made explicit.

        "A book of this size is not one piece of writing. Let us do the
         arithmetic, because the whole design falls out of it."

    Everything below -- how many chapters, how many sections per chapter, how
    many words each section is budgeted -- is derived from `target_pages`.
    Change that one number and the whole TOC resizes.
    """

    target_pages: int = 350  # design: 250-500
    words_per_page: int = 350  # technical prose with code
    words_per_section: int = 700  # design: 600-800

    min_section_words: int = 600
    max_section_words: int = 900

    target_chapters_min: int = 8
    target_chapters_max: int = 14
    min_sections_per_chapter: int = 4
    max_sections_per_chapter: int = 24

    @property
    def total_words(self) -> int:
        return self.target_pages * self.words_per_page

    @property
    def target_sections(self) -> int:
        return max(1, round(self.total_words / self.words_per_section))


@dataclass
class PipelineBConfig:
    """Configuration for Pipeline B - TOC generation."""

    # -- Model ---------------------------------------------------------------
    # No model settings here: which model backs `LLMClient` is decided once,
    # for the whole system, by .env (see src.llm.build_llm_client).
    temperature: float = 0.3
    structured_max_attempts: int = 3

    # -- Extraction ----------------------------------------------------------
    min_tags_per_chunk: int = 10  # these are now actually USED in the prompt
    max_tags_per_chunk: int = 20
    max_relationships_per_chunk: int = 8
    # A reasoning-capable model (Gemini 3.x, o-series, ...) spends part of
    # this SAME budget on hidden thinking before it writes a visible token,
    # so a budget sized only for the visible JSON truncates the reply before
    # it starts -- every attempt, since retrying does not change the budget.
    # Set generously; it is a ceiling; a non-reasoning model simply stops
    # sooner and never touches it.
    max_tokens_extraction: int = 6_000

    # Generation is sequential, so extraction over a few thousand chunks is the
    # long pole of the whole pipeline. Save partial results this often, so a
    # crash costs minutes rather than the whole step.
    extraction_checkpoint_every: int = 50

    # -- Normalization -------------------------------------------------------
    normalization_batch_size: int = 100
    max_tokens_normalization: int = 6_000

    # -- Clustering ----------------------------------------------------------
    theme_sample_size: int = 200
    assignment_batch_size: int = 60
    max_tokens_structure: int = 6_000

    # -- Chunk mapping -------------------------------------------------------
    # The design's example section carries THREE chunk ids. Taking every chunk
    # that shares any tag would hand a section hundreds of them and blow the
    # Context Assembler's 60K source budget.
    max_chunks_per_section: int = 8
    min_chunks_per_section: int = 1

    # -- Book size -----------------------------------------------------------
    size: BookSizeConfig = field(default_factory=BookSizeConfig)

    # -- Housekeeping --------------------------------------------------------
    # Off by default. A "full run" with this True silently processes 20 chunks.
    quick_test_mode: bool = False
    quick_test_chunks: int = 20
    resume: bool = True


def log_config(bcfg: PipelineBConfig, model_id: str) -> None:
    toc_log.info("Pipeline B configuration:")
    toc_log.info(f"  model            : {model_id}")
    toc_log.info(f"  target pages     : {bcfg.size.target_pages}")
    toc_log.info(f"  → total words    : {bcfg.size.total_words:,}")
    toc_log.info(
        f"  → target sections: {bcfg.size.target_sections}" f"  (design band: 120-250)"
    )
    toc_log.info(f"  quick test mode  : {bcfg.quick_test_mode}")

    if not 120 <= bcfg.size.target_sections <= 250:
        toc_log.warning(
            f"  target_sections={bcfg.size.target_sections} is outside the "
            f"design's 120-250 band - check target_pages"
        )
