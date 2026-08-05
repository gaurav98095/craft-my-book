"""
Phases 3.2-3.4 — Layer 5 (Classroom Memory), Layer 6 (Conversation Memory),
and the MemoryManager facade.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .setup import BookPaths, writer_log
from .ledger import BookLedger
from .constitution import Constitution
from .draft_store import DraftStore
from .source_memory import SourceMemory
from .context_assembler import ContextAssembler


class ClassroomMemory:
    """
    Layer 5. Per-section, disposable -- but not before close_session runs.

    One section's working state: which version of the teaching plan is
    current, how many times the Writer drifted from it, which student doubts
    were resolved and which were deferred. Thrown away when the section ships.
    """

    def __init__(self, paths: BookPaths):
        self.dir = paths.sessions
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def start_session(self, section_id: str, section: Dict) -> Dict[str, Any]:
        self.sessions[section_id] = {
            "section_id": section_id,
            "title": section["title"],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "plan": None,
            "plan_version": 0,
            "plan_rejections": [],
            "drift_count": 0,
            "steps_completed": 0,
            "student_interactions": {"doubts_resolved": [], "doubts_deferred": []},
        }
        return self.sessions[section_id]

    def get_session(self, section_id: str) -> Dict[str, Any]:
        return self.sessions.setdefault(
            section_id,
            {
                "section_id": section_id,
                "plan": None,
                "plan_version": 0,
                "plan_rejections": [],
                "drift_count": 0,
                "steps_completed": 0,
                "student_interactions": {"doubts_resolved": [], "doubts_deferred": []},
            },
        )

    def set_plan(
        self, section_id: str, plan: Dict, rejected_feedback: str = ""
    ) -> None:
        s = self.get_session(section_id)
        s["plan"] = plan
        s["plan_version"] += 1
        if rejected_feedback:
            s["plan_rejections"].append(rejected_feedback)

    def record_doubt(
        self,
        section_id: str,
        question: str,
        resolution: str,
        deferred: bool = False,
        suggested_chapter: Optional[str] = None,
    ) -> None:
        s = self.get_session(section_id)
        bucket = "doubts_deferred" if deferred else "doubts_resolved"
        s["student_interactions"][bucket].append(
            {
                "question": question,
                "resolution": resolution,
                "suggested_chapter": suggested_chapter,
            }
        )

    def close_session(self, section_id: str, ledger: BookLedger) -> List[Dict]:
        """
        A deferred doubt is a promise in disguise. Promote it to the ledger so
        the book actually owes the reader an answer later.

        "A deferred doubt is the most honest signal available about a real gap
         in the book: a simulated reader hit something confusing and the
         Writer said 'later'. If 'later' never arrives, that is failure #3."
        """
        session = self.get_session(section_id)
        promoted = []
        for doubt in session["student_interactions"]["doubts_deferred"]:
            promise = {
                "promise_id": f"p_{ledger._cache['version']}_{len(promoted)}",
                "text": f"Address the open question: {doubt['question']}",
                "made_in": section_id,
                "target_hint": doubt.get("suggested_chapter"),
                "status": "open",
                "origin": "deferred_doubt",
            }
            ledger._cache["promises"].append(promise)
            promoted.append(promise)
        if promoted:
            ledger._flush()
            writer_log.info(
                f"  [{section_id}] promoted {len(promoted)} deferred doubt(s) "
                f"to open promises"
            )

        # archive, then discard
        session["closed_at"] = datetime.now().isoformat(timespec="seconds")
        (self.dir / f"{section_id}.json").write_text(
            json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.sessions.pop(section_id, None)
        return promoted


COMPACTION_TEMPLATE = """Summarize this teaching conversation using EXACTLY this structure.
Be terse. Omit any section that has no content.

## Decisions Made
- (structural or pedagogical choices that must persist)

## Definitions Given
- term: definition

## Student Doubts — Resolved
- doubt → how it was answered

## Student Doubts — Deferred
- doubt (and where it should be answered)

## Open Threads
- anything the next step must pick up

CONVERSATION:
{conversation_text}
"""


class ConversationMemory:
    """
    Layer 6. Per-section dialogue, with anchored compaction.

    Anchored compaction: keep the most recent exchanges word for word, and
    summarize everything older into one structured block.
    """

    def __init__(
        self, paths: BookPaths, llm, threshold: int = 12, keep_recent: int = 6
    ):
        self.dir = paths.history
        self.dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.histories: Dict[str, Dict[str, Any]] = {}
        self.compactions = 0

    def _history(self, section_id: str) -> Dict[str, Any]:
        return self.histories.setdefault(
            section_id, {"exchanges": [], "compressed_context": ""}
        )

    def add(self, section_id: str, role: str, content: str) -> None:
        h = self._history(section_id)
        h["exchanges"].append(
            {
                "role": role,
                "content": content,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if len(h["exchanges"]) >= self.threshold:
            self._compact(section_id)

    def _compact(self, section_id: str) -> None:
        """
        Rewrite the summary, do not stack it.

        The naive approach appends each new summary to the old one, which
        grows without bound and turns into a game of telephone. Instead, feed
        the old summary AND the new exchanges to the model and ask for one
        merged summary in the same template. Fixed size, no drift.
        """
        h = self._history(section_id)
        older = h["exchanges"][: -self.keep_recent]
        if not older:
            return

        text = "\n\n".join(f"[{e['role'].upper()}]\n{e['content']}" for e in older)
        prompt = COMPACTION_TEMPLATE.format(
            conversation_text=(
                (
                    f"[EXISTING SUMMARY]\n{h['compressed_context']}\n\n"
                    f"[NEW EXCHANGES]\n{text}"
                )
                if h["compressed_context"]
                else text
            )
        )

        try:
            h["compressed_context"] = self.llm.generate(
                "You compress teaching conversations into a fixed structure. "
                "Return only the structured summary.",
                prompt,
                max_tokens=600,
                temperature=0.0,
            )
            h["exchanges"] = h["exchanges"][-self.keep_recent :]
            self.compactions += 1
            writer_log.info(
                f"  [{section_id}] conversation compacted "
                f"({len(older)} exchanges -> summary)"
            )
        except Exception as exc:
            # Losing recent turns is worse than a long prompt: keep them.
            writer_log.warning(
                f"  [{section_id}] compaction failed ({exc}); keeping raw"
            )

    def render(self, section_id: str) -> str:
        h = self._history(section_id)
        parts = []
        if h["compressed_context"]:
            parts.append(
                f"=== EARLIER IN THIS SECTION (summarized) ===\n"
                f"{h['compressed_context']}"
            )
        if h["exchanges"]:
            parts.append(
                "=== RECENT EXCHANGES ===\n"
                + "\n\n".join(
                    f"[{e['role'].upper()}]\n{e['content']}" for e in h["exchanges"]
                )
            )
        return "\n\n".join(parts)

    def close(self, section_id: str) -> None:
        """Write the dialogue to disk for debugging, then clear it."""
        h = self.histories.pop(section_id, None)
        if h:
            (self.dir / f"{section_id}_chat.json").write_text(
                json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8"
            )


class MemoryManager:
    """
    The single object the orchestrator talks to. Everything above stays
    addressable (`memory.ledger`, `memory.drafts`, ...) because the agents
    genuinely need different layers -- but the orchestrator should not have
    to wire seven of them together on every call.
    """

    def __init__(
        self,
        constitution: Constitution,
        ledger: BookLedger,
        drafts: DraftStore,
        source: SourceMemory,
        assembler: ContextAssembler,
        classroom: ClassroomMemory,
        conversation: ConversationMemory,
    ):
        self.constitution = constitution  # L0
        self.ledger = ledger  # L1
        self.drafts = drafts  # L2
        self.source = source  # L3
        self.assembler = assembler  # L4
        self.classroom = classroom  # L5
        self.conversation = conversation  # L6

    def build_context(self, section: Dict, prev_section_id: Optional[str]) -> Dict:
        return self.assembler.assemble(section, prev_section_id)

    def previous_tail(self, prev_section_id: Optional[str], n: int = 800) -> str:
        return self.drafts.get_tail(prev_section_id, n)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ledger": self.ledger.stats(),
            "sections_on_disk": len(self.drafts.written_ids()),
            "compactions": self.conversation.compactions,
            "truncations": len(self.assembler.truncations),
        }
