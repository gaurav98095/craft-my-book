"""The sections still to write, and what happened to the ones that are done."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .setup import BOOK, writer_log


@dataclass
class WorkQueue:
    """The sections still to write, and what happened to the ones that are done."""

    pending: List[str] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    failed: List[Dict] = field(default_factory=list)
    prev_section_id: Optional[str] = None
    started_at: str = ""

    @classmethod
    def build(cls, toc: Dict[str, Any], resume: bool) -> "WorkQueue":
        path = BOOK.work_queue
        order = [s["section_id"] for s in toc["sections"]]

        if resume and path.exists():
            try:
                q = cls(**json.loads(path.read_text(encoding="utf-8")))
                # trust the disk for what is DONE, the TOC for what remains
                done = set(q.completed)
                q.pending = [sid for sid in order if sid not in done]
                writer_log.info(
                    f"Resuming: {len(q.completed)} written, "
                    f"{len(q.pending)} remaining"
                )
                return q
            except Exception as exc:
                writer_log.warning(f"Unreadable work queue ({exc}); starting fresh")

        return cls(
            pending=order, started_at=datetime.now().isoformat(timespec="seconds")
        )

    def save(self) -> None:
        BOOK.work_queue.parent.mkdir(parents=True, exist_ok=True)
        BOOK.work_queue.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
