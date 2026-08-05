"""
Phase 1.1 — Paths, config, and the shared model.

Pipeline C writes into the tree the design specifies in `Files on Disk`:

    storage/
      schemas/constitution.json     L0 - fixed, hand-authored
      toc.json                      from Pipeline B
      book_ledger.json              L1 - the central file
      checkpoints/work_queue.json
    output/
      sections/sec_01_01.md         the book, one file per section
      draft_index/                  L2 - Chroma: section abstracts
      raw_steps/                    pre-Editor drafts, for raw->edited diffs
      sessions/                     L5 - per-section classroom state
      history/                      L6 - per-section conversations
      book/                         manuscript, glossary, index, report

Everything is plain JSON plus one Chroma collection. It survives a crash, it
is diffable in git, and you can open book_ledger.json and read exactly what
the book believes about itself.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class BookPaths:
    """
    The design's `Files on Disk`, for the writing side.

    `storage` defaults from STORAGE_ROOT (the same tree Pipeline B writes
    toc.json into) and `output` from OUTPUT_ROOT, so the manuscript can be
    written somewhere other than the repo without editing code.
    """

    storage: Path = field(default_factory=lambda: Path(os.getenv("STORAGE_ROOT", "./storage")))
    output: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_ROOT", "./output")))

    # -- storage ------------------------------------------------------------
    @property
    def constitution(self) -> Path:
        return self.storage / "schemas" / "constitution.json"

    @property
    def toc(self) -> Path:
        return self.storage / "toc.json"

    @property
    def ledger(self) -> Path:
        return self.storage / "book_ledger.json"

    @property
    def checkpoints(self) -> Path:
        return self.storage / "checkpoints"

    @property
    def work_queue(self) -> Path:
        return self.checkpoints / "work_queue.json"

    # -- output -------------------------------------------------------------
    @property
    def sections(self) -> Path:
        return self.output / "sections"

    @property
    def draft_index(self) -> Path:
        return self.output / "draft_index"

    @property
    def raw_steps(self) -> Path:
        return self.output / "raw_steps"

    @property
    def sessions(self) -> Path:
        return self.output / "sessions"

    @property
    def history(self) -> Path:
        return self.output / "history"

    @property
    def book(self) -> Path:
        return self.output / "book"

    @property
    def ledger_diffs(self) -> Path:
        return self.output / "ledger_diffs.jsonl"

    def mkdirs(self) -> None:
        for p in (
            self.storage / "schemas",
            self.checkpoints,
            self.sections,
            self.raw_steps,
            self.sessions,
            self.history,
            self.book,
        ):
            p.mkdir(parents=True, exist_ok=True)


BOOK = BookPaths()
BOOK.mkdirs()


def make_writer_logger(name: str, logfile: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )
    for h in (logging.FileHandler(logfile, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


writer_log = make_writer_logger("pipelineC.writer", "pipelineC_writer.log")


@dataclass
class WriterConfig:
    """Configuration for Pipeline C."""

    # -- loop limits ---------------------------------------------------------
    max_plan_revisions: int = 3
    max_doubt_rounds: int = 2  # per teaching step
    max_editor_retries: int = 1

    # -- generation ----------------------------------------------------------
    # A reasoning-capable model spends part of this SAME budget on hidden
    # thinking before it writes a visible token, so these are sized with
    # that margin built in -- see the identical note in toc/setup.py. A
    # non-reasoning model just stops sooner and never touches the ceiling.
    plan_max_tokens: int = 4_000
    step_max_tokens: int = 5_000
    review_max_tokens: int = 3_000
    student_max_tokens: int = 2_000
    editor_max_tokens: int = 7_000
    archivist_max_tokens: int = 5_000
    temperature_prose: float = 0.7  # writing wants some life
    temperature_structured: float = 0.0

    # -- section shape -------------------------------------------------------
    default_word_count: int = 700
    steps_per_section_min: int = 3
    steps_per_section_max: int = 5

    # -- memory --------------------------------------------------------------
    tail_chars: int = 800  # verbatim ending of the previous section
    compaction_threshold: int = 12  # exchanges before Layer 6 compacts
    repetition_threshold: float = 0.85

    # -- housekeeping --------------------------------------------------------
    resume: bool = True
    stop_on_error: bool = False
    save_raw_steps: bool = True
