"""Phase 4 — the five agents, one model: Writer, Reviewer, Student, Editor, Archivist."""

from .base import BaseAgent, render_context
from .writer import WriterAgent
from .reviewer_student import ReviewerAgent, StudentAgent
from .editor import EditorAgent, mask_code, unmask_code
from .archivist import ArchivistAgent

__all__ = [
    "BaseAgent",
    "render_context",
    "WriterAgent",
    "ReviewerAgent",
    "StudentAgent",
    "EditorAgent",
    "mask_code",
    "unmask_code",
    "ArchivistAgent",
]
